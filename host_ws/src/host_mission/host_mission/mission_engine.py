"""Path-mode mission engine (ROS-free): wires the mission state machine to the world model,
the path commander, the supervisor and the pick monitor. The ROS node only feeds inputs and
publishes ``commands()`` / the returned events. One ``tick(now)`` (``now`` = host monotonic
clock; freshness is judged against it):

    world -> arrival verdicts -> supervisor (health + recovery actions)
          -> MissionStateMachine.tick -> pick monitor -> PathCommander.update

Operator rules: a running trip is never changed by the operator; only a stop (pause or
e-stop) interrupts it.
  * A part that is not delivered (pick failed, trip abandoned) leaves the queue at once;
    the event names it. cancel:<id> removes a waiting part (not one of the running trip);
    clear_queue removes the waiting parts only.
  * abandon only while stopped and never during a pick: drops the trip and targets home.
  * A pick is never cut short: it ends by its result or its own deadline (``pick_dwell``).
  * Nothing to do (IDLE / DONE after a mission has run) -> both rovers drive home; before the
    first mission ``idle_go_home`` decides. ``align`` starts that drive for rovers whose
    heading is not aligned: they go one at a time (leader first).
  * An engaged e-stop wins over everything. Release alone moves nothing; a 'resume' received
    while engaged is kept for ``estop_resume_window`` s and applied on the release.
  * Resume and every target change re-plan from the current pose.
Every refused command emits 'control_rejected' with a reason."""
import math

import numpy as np
from host_planning.geometry import obstacle_array, point_shapes, shape_gaps
from host_planning.region import DropRegion

from host_mission.arrival import ARRIVED_STATES, answers, judge_arrival
from host_mission.goals import GoalPose
from host_mission.idle_turn import IdleTurns
from host_mission.mission_queue import Item, MissionQueueModel
from host_mission.obstacle_mapper import front_stop_rect, rect_shape
from host_mission.path_commander import Desired, PathCommander
from host_mission.pick_monitor import PickMonitor
from host_mission.supervisor import Supervisor
from host_mission.task_sequencer import MissionStateMachine
from host_mission.world_model import WorldModel

_IDLE_STATES = ('IDLE', 'DONE')
_PICK_PHASES_UNGATED = ('TO_PART', 'PICKING')


class MissionEngine:
    def __init__(self, settings, scenario, goals, epoch, planner=None, plan_runner=None):
        self.settings = settings
        self.scenario = scenario
        self.robots = settings.robot_names
        self.world = WorldModel(pose_timeout=settings.freshness.pose_timeout, goals=goals)
        self.queue = MissionQueueModel()
        self.seq = MissionStateMachine(
            self.queue, scenario.leader_standby, scenario.follower_unload,
            scenario.capacity, pick_dwell=scenario.pick_dwell,
            phase_timeout=scenario.phase_timeout, follower_home_node=scenario.follower_home,
            order_cost=self._goal_distance, cooldown=scenario.pick_cooldown,
            dwell_is_failure=scenario.pick_timeout_is_failure,
            next_trip_from_home=settings.idle_heading is not None)
        self.commander = PathCommander(settings, epoch, planner, plan_runner)
        self.supervisor = Supervisor(settings.supervisor, settings.freshness)
        self.pick = PickMonitor(settings.pick.picking_states, settings.pick.error_states)
        self.arrived = {r: False for r in self.robots}
        self._latch = {}            # robot -> cmd_id whose goal was reached
        self._estop = False
        self._pending_since = None
        self._task_seq = 0
        self._pick_key = None       # (item id) the pick monitor is armed for
        self._pick_failures = 0
        self._events = []           # raised between ticks (operator inputs)
        self._now = 0.0             # time of the last tick (default receive time)
        self._has_run = False       # a mission has started since boot (home rule)
        self._align_asked = False   # the operator asked for the alignment drive (same rule)
        self._resume_asked = None   # time of a 'resume' received while the e-stop was engaged
        self.idle_turns = IdleTurns(settings.idle_heading, settings.robots,
                                    settings.rotation_centre_error)
        self._turn_wait_since = None  # when a dispatch began to wait for an idle turn

    def dispatch(self, part_name):
        """Queue one fetch of ``part_name``. False (with an event) when it cannot run."""
        part = self.scenario.parts.get(part_name)
        if part is None:
            return self._reject('dispatch_rejected', f'unknown part {part_name!r}')
        missing = [g for g in self.scenario.goal_names(part) if self.world.goals.get(g) is None]
        if missing:
            return self._reject('dispatch_rejected', f'{part_name}: goals not taught {missing}')
        self._task_seq += 1
        item = Item(id=f'fetch_{part_name}#{self._task_seq}', part_id=part_name,
                    x=part.x, y=part.y, lane_node=part.leader_goal,
                    leader_node=part.leader_goal, follower_node=part.follower_goal)
        self.queue.enqueue(item)
        self._events.append(('enqueued', item.id))
        return True

    def control(self, command, now):
        """Operator command: pause | resume | abandon | clear_queue | cancel:<id> | align."""
        if command == 'pause':
            self.seq.pause()
        elif command == 'resume':
            if self._estop:                     # applied if the release follows shortly
                self._resume_asked = now
                self._events.append(('resume_deferred', 'e-stop still engaged'))
                return True
            self._resume()
        elif command == 'clear_queue':
            self.queue.clear_waiting()
        elif command == 'abandon':
            refusal = self.seq.abandon_refusal()
            if refusal:
                return self._reject('control_rejected', f'abandon: {refusal}')
            self._abandon()
        elif command == 'align':
            refusal = self._align_refusal()
            if refusal:
                return self._reject('control_rejected', f'align: {refusal}')
            self._align_asked = True
        elif command.startswith('cancel:'):
            item_id = command.split(':', 1)[1]
            refusal = self.seq.cancel_refusal(item_id)
            if refusal:
                return self._reject('control_rejected', f'cancel: {refusal}')
            self.queue.cancel(item_id)
        else:
            return self._reject('control_rejected', f'unknown command {command!r}')
        self._events.append(('control', command))
        return True

    def _align_refusal(self):
        """Why the alignment drive may not start now (None = may)."""
        if self._estop:
            return 'e-stop engaged'
        if self.seq.state not in _IDLE_STATES:
            return 'a mission is running'
        if all(self.world.robot(robot).heading_aligned for robot in self.robots):
            return 'the headings are aligned'
        return None

    def offered_controls(self):
        """Operator commands that would be accepted now (the dashboard offers exactly these)."""
        offered = []
        if self.seq.abandon_refusal() is None:
            offered.append('abandon')
        if self._align_refusal() is None:
            offered.append('align')
        return offered

    def estop(self, engaged, now=None):
        """E-stop line; release never resumes by itself, except a 'resume' received just before."""
        now = self._now if now is None else now
        engaged = bool(engaged)
        if engaged != self._estop:
            self._events.append(('estop', 'engaged' if engaged else 'released'))
        self._estop = engaged
        self.seq.estop(engaged)
        asked, self._resume_asked = self._resume_asked, None
        if not engaged and asked is not None \
                and now - asked <= self.settings.estop_resume_window:
            self._resume()
            self._events.append(('control', 'resume (received before the e-stop release)'))

    def unload_done(self):
        """Operator 'unload done'; accepted only while the follower waits at the unload goal."""
        if self.seq.system_state() != 'UNLOADING':
            return self._reject('unload_rejected', f'state {self.seq.system_state()}')
        self.seq.unload_done()
        self._events.append(('unload_done', ''))
        return True

    def set_obstacles(self, circles):
        """Replace the operator obstacle list; re-plans a moving rover whose path it blocks."""
        if self.world.set_manual_obstacles(circles):
            self._events.append(('obstacles', f'{len(circles)} manual'))
            self._check_moving_paths()

    def set_mapped_obstacles(self, circles):
        """Replace the mapped obstacle list; re-plans a moving rover whose path it blocks."""
        if self.world.set_mapped_obstacles(circles):
            self._events.append(('obstacles', f'{len(circles)} mapped'))
            self._check_moving_paths()

    def teach_goal(self, robot, name, now):
        """Store the rover's current pose as goal ``name`` and persist the goal file."""
        if robot not in self.settings.robots or not name:
            return self._reject('teach_rejected', f'{robot}:{name}: unknown robot or empty name')
        pose, reason = self.world.usable_pose(robot, now)
        if pose is None:
            return self._reject('teach_rejected', f'{robot}:{name}: {reason}')
        self.world.goals.set(name, GoalPose(*pose))
        if self.world.goals.path:
            self.world.goals.save()
        self._events.append(('goal_saved', f'{name} = ({pose[0]:.3f}, {pose[1]:.3f}, '
                             f'{pose[2]:.3f}) from {robot}'))
        return True

    def on_work_state(self, state):
        self.pick.on_work_state(state)

    def on_pick_done(self, ok):
        self.pick.on_done(ok)

    def commands(self):
        return dict(self.commander.cmds)

    def tick(self, now):
        """Advance the mission; returns [(kind, detail)] events for logs / dashboard."""
        events, self._events = self._events, []
        self._now = now
        self.world.expire_obstacles(now)
        self._maybe_start(now, events)
        verdicts = self._judge(now)
        blocked = {r: self.commander.blocked(r) for r in self.robots}
        verdict = self.supervisor.evaluate(self.world, self.commander.cmds, verdicts,
                                           self.arrived, blocked, now)
        events += verdict.events
        self._apply_actions(verdict.actions, now, events)
        self._feed_pick_result()
        targets, modes, seq_events = self.seq.tick(now, self.at_target(), verdict.health)
        self._on_seq_events(seq_events, events)
        self._arm_pick()
        events += self.commander.update(self._desired(targets, modes, events), self.world,
                                        self.arrived, now)
        return events

    def _maybe_start(self, now, events):
        if self.seq.state not in _IDLE_STATES or not self.queue.pending():
            self._pending_since = None
        elif self._pending_since is None:
            self._pending_since = now
        elif now - self._pending_since >= self.settings.start_grace \
                and not self._waits_for_idle_turn(now, events):
            self._pending_since = None
            started = self.seq.start(now)
            self._has_run = self._has_run or bool(started)
            events += started

    def _waits_for_idle_turn(self, now, events):
        """True while a rover still turns to the idle heading (at most ``idle_turn_timeout``)."""
        turning = [r for r in self.robots
                   if self.idle_turns.under_way(r, self._at_goal_key(r)) or self.idle_turns.queued(r)]
        if not turning:
            self._turn_wait_since = None
            return False
        if self._turn_wait_since is None:
            self._turn_wait_since = now
            events.append(('start_waits_for_turn', ', '.join(turning)))
        if now - self._turn_wait_since < self.settings.idle_turn_timeout:
            return True
        events.append(('start_after_turn_timeout', f'{", ".join(turning)} not turned after '
                                                   f'{self.settings.idle_turn_timeout:.0f} s'))
        self._turn_wait_since = None
        return False

    def _at_goal_key(self, robot):
        """Goal key of the rover's current command once it has arrived there, else None."""
        cmd = self.commander.cmds.get(robot)
        if cmd is None or not self.arrived.get(robot) or self._latch.get(robot) != cmd.cmd_id:
            return None
        return cmd.goal_key

    def _held_for_alignment(self):
        """Rovers waiting their turn in the idle drive home (one at a time while a heading is
        unaligned)."""
        if all(self.world.robot(robot).heading_aligned for robot in self.robots):
            return ()
        waiting = [robot for robot in self.robots
                   if not (self.world.robot(robot).heading_aligned and self.arrived.get(robot))]
        return tuple(waiting[1:])

    def _judge(self, now):
        fresh = self.settings.freshness
        verdicts = {}
        for robot in self.robots:
            cfg = self.settings.robots[robot]
            cmd = self.commander.cmds.get(robot)
            # never stricter than the tolerance the rover was sent (an idle turn step's)
            pos_tol = max(cfg.arrive_pos_tol, cmd.goal_pos_tol if cmd is not None else 0.0)
            verdicts[robot] = judge_arrival(
                self.world.robot(robot), cmd, now, pos_tol, cfg.arrive_yaw_tol,
                fresh.pose_timeout, fresh.feedback_timeout)
            self.arrived[robot] = self._latched(robot, cmd, verdicts[robot])
        return verdicts

    def at_target(self):
        """{robot: arrived at the goal the mission wants NOW}: a command still serving an earlier
        goal says nothing about the new one."""
        cmds = self.commander.cmds
        return {robot: bool(self.arrived[robot]) and robot in cmds
                and cmds[robot].goal_key == self.seq.targets.get(robot)
                for robot in self.robots}

    def _latched(self, robot, cmd, verdict):
        """Arrival is latched per command: once reached it holds while the rover keeps
        reporting HOLDING/PARKED for it (pose noise must not un-arrive a parked rover)."""
        if cmd is None:
            return False
        if verdict.arrived:
            self._latch[robot] = cmd.cmd_id
            return True
        nav = self.world.robot(robot).nav
        return self._latch.get(robot) == cmd.cmd_id and answers(nav, cmd) \
            and nav.state in ARRIVED_STATES

    def _apply_actions(self, actions, now, events):
        for action in actions:
            if action.kind == 'add_obstacle':
                if not self._register_blocker(action, now):
                    continue            # same obstacle, expiry extended: nothing to report
            elif action.kind == 'halt':
                self.seq.halt()
            else:
                self.commander.request_replan(action.robot, reentry=action.kind == 'reenter')
            events.append((action.kind, f'{action.robot}: {action.detail}'))

    def _register_blocker(self, action, now):
        """Front stop -> assumed box behind the face point, as one exact planner box (the assumed
        size is the margin). True when it is a new obstacle."""
        cfg = self.settings.supervisor
        if _near_fixture(action.point, self.settings.fixtures, cfg.fixture_match_dist):
            return False
        # The other rover: every plan has it already, and an assumed box on it also covers the
        # room beside it. The rover reports the point on its sensor axis; what stopped it is
        # anywhere across the sensed band (plan.front_zone half width).
        zone = self.settings.robots[action.robot].plan.front_zone
        across = max(cfg.fixture_match_dist, zone[2] if zone else 0.0)
        if _near_fixture(action.point, self.commander.peer_footprints(action.robot, self.world),
                         across):
            return False
        rect = front_stop_rect(action.point, action.heading, cfg.assumed_obstacle_depth,
                               cfg.assumed_obstacle_width)
        return self.world.add_detected_region([rect_shape(rect, 0.0)], action.point,
                                              now + cfg.lidar_obstacle_ttl)

    def _on_seq_events(self, seq_events, events):
        for kind, detail in seq_events:
            events.append((kind, detail))
            if kind == 'loaded':
                self._pick_failures = 0
            elif kind == 'pick_failed':
                self._count_pick_failure(events)

    def _count_pick_failure(self, events):
        self._pick_failures += 1
        limit = self.settings.supervisor.max_consecutive_pick_failures
        if self._pick_failures >= limit and self.seq.state == 'RUNNING':   # not when stopped
            self.seq.halt()
            events.append(('halt', f'{self._pick_failures} consecutive pick failures'))

    def _feed_pick_result(self):
        result = self.pick.poll()
        if result is not None:
            self.seq.pick_complete(result)

    def _arm_pick(self):
        """Arm the pick monitor when the mission enters PICKING for a new item; only
        results observed from then on count (latched replays are rejected)."""
        picking = self.seq.phase == 'PICKING' and self.seq.state in ('RUNNING', 'PAUSED')
        key = self.seq.active_id if picking else None
        if key == self._pick_key:
            return
        self._pick_key = key
        if key:
            self.pick.begin()
        else:
            self.pick.cancel()

    def _desired(self, targets, modes, events):
        idle = self.seq.state in _IDLE_STATES
        turn_goals = {}             # goal key of an idle turn step -> its pose
        if idle:                    # nothing to do: home (whatever the last trip targeted)
            targets = {'leader': self.seq.standby, 'follower': self.seq.follower_home}
            modes = {r: 'GOTO' for r in targets}
            homes = dict(targets)
            for robot, home in list(targets.items()):
                other = next((self.world.goals.get(h) for r, h in homes.items() if r != robot), None)
                key, goal = self.idle_turns.wanted(robot, home, self.world.goals.get(home),
                                                   self._at_goal_key(robot), events,
                                                   self.world.robot(robot).pose,
                                                   None if other is None else (other.x, other.y))
                targets[robot], turn_goals[key] = key, goal
        else:
            self.idle_turns.clear()
        active = self.seq.state == 'RUNNING' or \
            (idle and (self._has_run or self._align_asked or self.settings.idle_go_home))
        held = self._held_for_alignment() if idle else ()
        desired = {}
        for robot in self.robots:
            key = targets.get(robot)
            follow = active and not self._estop and modes.get(robot) == 'GOTO' \
                and robot not in held
            goal = turn_goals.get(key) or self.world.goals.get(key)
            # a turn's end position follows the real rotation centre: judged at the host's arrival
            # tolerance, not the rover's tighter goal tolerance
            tol = self.idle_turns.step_tolerance(
                robot, self.settings.robots[robot].arrive_pos_tol) if key in turn_goals else 0.0
            desired[robot] = Desired(key, goal, follow, self._pick_at_goal(robot),
                                     self._load_region(robot, key), tol)
        return desired

    def _load_region(self, robot, key):
        """Load goal of a cargo rover as a region: every pose whose cargo area holds the
        picker's drop point at that part's pick goal (None = use the named goal)."""
        cargo, drop = self.settings.robots[robot].cargo, self.settings.pick.drop_point
        if cargo is None or drop is None:
            return None
        part = next((p for p in self.scenario.parts.values() if p.follower_goal == key), None)
        pick = self.world.goals.get(part.leader_goal) if part is not None else None
        if pick is None:
            return None
        c, s = math.cos(pick.theta), math.sin(pick.theta)
        point = (pick.x + c * drop[0] - s * drop[1], pick.y + s * drop[0] + c * drop[1])
        return DropRegion(point, pick.theta, tuple(cargo), self.settings.pick.drop_margin)

    def _pick_at_goal(self, robot):
        """HOLDING (which triggers the leader's pick) is requested only at a pick goal:
        with the gate, not before both rovers stand ready (PICKING phase)."""
        pick = self.settings.pick
        if robot != pick.picker or self.seq.state != 'RUNNING':
            return False
        if pick.gate_on_phase:
            return self.seq.phase == 'PICKING'
        return self.seq.phase in _PICK_PHASES_UNGATED

    def _resume(self):
        self.supervisor.reset()
        self._pick_failures = 0
        self.seq.resume()

    def _abandon(self):
        dropped = self.seq.abandon()
        self._events.append(('abandoned', 'not delivered: ' + (', '.join(dropped) or 'none')))
        self.supervisor.reset()
        self.commander.clear_blocked()
        self._pick_failures = 0

    def _check_moving_paths(self):
        for robot, cmd in self.commander.cmds.items():
            if cmd.mode == 'FOLLOW' and not self.arrived.get(robot):
                self.commander.check_path(robot)

    def _goal_distance(self, key_a, key_b):
        a, b = self.world.goals.get(key_a), self.world.goals.get(key_b)
        if a is None or b is None:
            return math.inf
        return math.hypot(a.x - b.x, a.y - b.y)

    def _reject(self, kind, detail):
        self._events.append((kind, detail))
        return False


def _near_fixture(point, fixtures, dist):
    """True when ``point`` (x, y) lies within ``dist`` of a known fixture's surface."""
    if not fixtures or point is None:
        return False
    gaps = shape_gaps(point_shapes(np.array([point], dtype=float)), obstacle_array(fixtures))
    return bool(gaps.min() <= dist)
