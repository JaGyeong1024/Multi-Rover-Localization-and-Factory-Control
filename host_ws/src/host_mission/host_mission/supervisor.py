"""Host share of the exception handling (ROS-free): stop when unsure, resume by itself when
the cause is gone, bound every retry, hand over to the operator when the budget is spent.

  * STALE_POSE / STALE_LEAD / quality LOST lasting ``hold_pause_s``, or no path -> health
    LOST for that rover (the sequencer pauses and auto-resumes).
  * OFF_PATH / STUCK -> re-plan (``off_path_max_replans`` / ``stuck_max_replans``).
  * LIDAR_STOP lasting ``lidar_block_s`` (not at a goal) -> register the area behind the
    reported face point as an expiring obstacle (``obstacle_mapper.front_stop_rect``) and
    re-plan; repeated while the stop lasts, also while paused, so it never expires under
    a stopped rover.
  * Goal missed -> re-entry path (``goal_miss_max_reentries``).
Budget exceeded -> ``halt`` action. ``evaluate`` only decides; the engine applies it."""
from dataclasses import dataclass, field
from typing import Optional, Tuple

from host_mission.arrival import ARRIVED_STATES, feedback_for

GOOD = 'GOOD'
LOST = 'LOST'
STALE_POSE = 'STALE_POSE'
STALE_LEAD = 'STALE_LEAD'
QUALITY_LOST = 'QUALITY_LOST'
NO_PATH = 'NO_PATH'
OFF_PATH = 'OFF_PATH'
HEADING_VALID = 'HEADING_VALID'     # re-plan reason: the alignment creep is over
STUCK = 'STUCK'
LIDAR_STOP = 'LIDAR_STOP'
GOAL_MISSED = 'GOAL_MISSED'
_FOLLOW = 'FOLLOW'


@dataclass(frozen=True)
class Action:

    kind: str                       # replan | reenter | add_obstacle | halt
    robot: str
    detail: str = ''
    point: Optional[Tuple[float, float]] = None   # add_obstacle: face point
    heading: float = 0.0                          # add_obstacle: rover heading (rad)


@dataclass
class Verdict:

    health: dict = field(default_factory=dict)    # robot -> GOOD | LOST
    actions: list = field(default_factory=list)
    events: list = field(default_factory=list)    # (kind, detail)


class Supervisor:
    def __init__(self, settings, freshness):
        self.cfg = settings
        self.fresh = freshness
        self.reset()

    def reset(self):
        self._since = {}            # (robot, cause) -> first time seen
        self._holding = set()       # (robot, cause) currently pausing the mission
        self._watch = {}            # (robot, kind) -> first time seen (goal miss, LiDAR)
        self._count = {}            # (robot, kind) -> recoveries spent on this goal
        self._handled = {}          # (robot, kind) -> cmd_id already acted on
        self._goal = {}             # robot -> goal_key the counters belong to
        self._halted = False

    def evaluate(self, world, cmds, verdicts, arrived, blocked, now):
        """Health and recovery actions for this tick; ``blocked`` keys are the supervised rovers."""
        out = Verdict()
        for robot in sorted(blocked):
            causes = self._hold_causes(robot, world, blocked[robot], now)
            out.health[robot] = LOST if self._update_holds(robot, causes, now, out) else GOOD
            if not self._halted and robot in cmds:
                self._recover(robot, world, cmds[robot], verdicts.get(robot),
                              arrived.get(robot, False), now, out)
            self._check_lidar(robot, world, cmds.get(robot), arrived.get(robot, False), now, out)
        return out

    def _hold_causes(self, robot, world, blocked, now):
        reasons = self._safety_reasons(world, robot, now)
        delay = self.cfg.hold_pause_s
        causes = {}
        if not world.pose_fresh(robot, now) or STALE_POSE in reasons:
            causes[STALE_POSE] = delay
        if STALE_LEAD in reasons:
            causes[STALE_LEAD] = delay
        if world.robot(robot).quality == LOST:
            causes[QUALITY_LOST] = delay
        if blocked:
            causes[NO_PATH] = 0.0
        return causes

    def _update_holds(self, robot, causes, now, out):
        """Track cause timers; True while any cause has outlasted its delay."""
        for key in [k for k in self._since if k[0] == robot and k[1] not in causes]:
            del self._since[key]
            if key in self._holding:
                self._holding.discard(key)
                out.events.append(('hold_cleared', f'{robot}: {key[1]}'))
        for cause, delay in causes.items():
            key = (robot, cause)
            start = self._since.setdefault(key, now)
            if now - start >= delay and key not in self._holding:
                self._holding.add(key)
                out.events.append(('hold', f'{robot}: {cause}'))
        return any(key[0] == robot for key in self._holding)

    def _safety_reasons(self, world, robot, now):
        safety = world.robot(robot).safety
        if safety is None or now - safety.time > self.fresh.safety_timeout:
            return ()
        return safety.reasons

    def _recover(self, robot, world, cmd, verdict, arrived, now, out):
        if cmd.mode != _FOLLOW:
            return
        self._track_goal(robot, cmd, arrived)
        nav = feedback_for(world.robot(robot), cmd, now, self.fresh.feedback_timeout)
        safety = self._safety_reasons(world, robot, now)
        stopped = self._stop_reasons(nav, safety)
        if OFF_PATH in stopped and cmd.provisional:
            self._creep_over(robot, cmd, out)
        elif OFF_PATH in stopped:
            self._spend(robot, OFF_PATH, cmd, self.cfg.off_path_max_replans, 'replan', out)
        if STUCK in stopped:
            self._spend(robot, STUCK, cmd, self.cfg.stuck_max_replans, 'replan', out)
        self._check_goal_miss(robot, cmd, verdict, arrived, now, out)

    @staticmethod
    def _stop_reasons(nav, safety):
        """Why the tracker stopped on THIS command (NavFeedback STOPPED, same cmd_id)."""
        if nav is None or nav.state != 'STOPPED':
            return set()
        detail = nav.detail.upper()
        return {token for token in (OFF_PATH, STUCK) if token in detail or token in safety}

    def _creep_over(self, robot, cmd, out):
        """Off a provisional path: the alignment creep is over, one re-plan per command."""
        key = (robot, HEADING_VALID)
        if self._handled.get(key) != cmd.cmd_id:
            self._handled[key] = cmd.cmd_id
            out.actions.append(Action('replan', robot, HEADING_VALID))

    def _track_goal(self, robot, cmd, arrived):
        if self._goal.get(robot) != cmd.goal_key or arrived:
            self._goal[robot] = cmd.goal_key
            for key in [k for k in self._count if k[0] == robot]:
                del self._count[key]

    def _spend(self, robot, kind, cmd, limit, action, out):
        """Act once per command on ``kind``; halt when the budget is used up."""
        key = (robot, kind)
        if self._handled.get(key) == cmd.cmd_id:
            return
        self._handled[key] = cmd.cmd_id
        used = self._count.get(key, 0)
        if used >= limit:
            self._halted = True
            out.actions.append(Action('halt', robot, f'{kind} again after {used} recoveries'))
            return
        self._count[key] = used + 1
        out.actions.append(Action(action, robot, f'{kind} ({used + 1}/{limit})'))

    def _check_goal_miss(self, robot, cmd, verdict, arrived, now, out):
        """Re-enter on a reported goal miss, or a host-side pose disagreement before arrival."""
        key = (robot, GOAL_MISSED)
        if verdict is None or not verdict.missed or (arrived and not verdict.reported):
            self._watch.pop(key, None)
            return
        start = self._watch.setdefault(key, now)
        if verdict.reported or now - start >= self.cfg.goal_miss_confirm_s:
            self._spend(robot, GOAL_MISSED, cmd, self.cfg.goal_miss_max_reentries,
                        'reenter', out)

    def _check_lidar(self, robot, world, cmd, arrived, now, out):
        """Register the blocker every ``lidar_block_s`` of front stop and re-plan a moving rover
        (not one that has arrived facing a wall or shelf)."""
        key = (robot, LIDAR_STOP)
        state = world.robot(robot)
        stopped = LIDAR_STOP in self._safety_reasons(world, robot, now)
        nav = feedback_for(state, cmd, now, self.fresh.feedback_timeout)
        at_goal = arrived or (nav is not None and nav.state in ARRIVED_STATES)
        if not stopped or at_goal or state.safety.blocked is None or state.pose is None:
            self._watch.pop(key, None)
            return
        if now - self._watch.setdefault(key, now) < self.cfg.lidar_block_s:
            return
        self._watch[key] = now      # re-arm: act again only after another full period
        out.actions.append(Action('add_obstacle', robot, LIDAR_STOP, state.safety.blocked,
                                  state.pose[2]))
        if not self._halted and cmd is not None and cmd.mode == _FOLLOW:
            out.actions.append(Action('replan', robot, LIDAR_STOP))
