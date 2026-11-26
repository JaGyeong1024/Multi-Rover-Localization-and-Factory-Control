"""Path command generation (ROS-free): turns "rover R should be at goal G" into PathCommand
contents using the ``host_planning`` library.

  * ``cmd_id`` (per rover) increases whenever the path or mode changes; the node re-publishes
    the unchanged command at 2 Hz. ``epoch`` is fixed per process.
  * The platoon lead is planned first; the follower path is laid on the lead's remaining path
    (``lead_robot`` + ``gap``) when it can share, else it is stand-alone. The other rover is
    an obstacle where it stands, or at its goal when on its way.
  * A stand-alone follower path is released only when it keeps ``conflict_dist`` from the
    lead's remaining path (spatial test, no timing assumption) and the follower parked at
    its goal keeps the lead's clearance from it; until then it yields (STOP / YIELD).
  * ``platoon.dock_turn`` / ``platoon.hold_ride``: a follower that would have to wait rides
    the lead path to a turn-in / hold point and stops there (a ``via`` command, never
    "arrived"), then plans normally.
  * No path -> STOP + ``blocked`` (the supervisor pauses the mission), retried when the
    obstacle set changes or every ``plan_retry_period``. ``check_path`` re-plans a rover on
    its way only when the rest of its path no longer keeps clearance.
  * A stale pose is never planned from; an unaligned heading is planned from a provisional
    heading so the tracker can run its alignment creep.
  * Leading pivot (``plan.pivot_step`` > 0): an in-place rotation before the path
    (``pivot_first`` + ``pivot_heading``) whose swept footprint keeps ``clearance`` from
    obstacles, fixtures and the other rover (also over its own remaining turn).
  * ``fixture_ahead``: a fixture lies within ``front_exempt_reach`` ahead of the goal; the
    rover then exempts the final run-in from its front stop. No path pose may lie inside the
    zone where the rover's gate stops (``plan.front_zone``) except that run-in."""
import copy
import dataclasses
import math

from host_planning import api as planning_api
from host_planning.pivot import pivot_clear, search_departure, swept_poses
from host_planning.region import feasible_goals
from host_planning.types import Pose2D

from host_mission import plan_jobs
from host_mission.arrival import ARRIVED_STATES, GOAL_MISSED, STOPPED_STATE, feedback_for
from host_mission.goals import GoalPose
from host_mission.path_geometry import (faces_fixture, footprint_box,  # noqa: F401 (re-export)
                                        grown, nearest_index, paths_conflict,
                                        pick_goals_without_fixture,  # noqa: F401 (re-export)
                                        planner_shape, rotation_centre, spread_goals,
                                        turn_in_point)  # noqa: F401 (re-export)
from host_mission.path_types import (FOLLOW, LEAD_TURNING, NO_PATH, PIVOT_DETAILS,  # noqa: F401
                                     STOP, TRACKING_STATE, YIELD, Desired, Hold, Job, PathCmd,
                                     Plan, on_the_way, past_the_turn, serves)
from host_mission.peer_shapes import PeerShapes
from host_mission.platoon_rules import PlatoonRules

_GUESS_TURNS = (math.pi / 3, -math.pi / 3, 2 * math.pi / 3, -2 * math.pi / 3)
_MIN_SHARED_POSES = 2
_REGION_TRIES = 3            # distinct region poses planned before giving up (NO_PATH)

# Names the scenario tools and tests reach through this module.
_planner_shape = planner_shape
_nearest_index = nearest_index


class PathCommander(PeerShapes, PlatoonRules):
    """Plans and issues one PathCmd per rover; ``update`` is called every tick."""

    def __init__(self, settings, epoch, planner=None, runner=None):
        self.settings = settings
        self.epoch = int(epoch)
        self.planner = planner or planning_api
        self.runner = runner or plan_jobs.Inline()
        self._jobs = {}
        self._asked = set()
        self.cmds = {}
        self._next_id = {}
        self._replan = {}           # robot -> re-entry flag (True = longer run-in)
        self._recheck = set()
        self._holds = {}            # robot -> Hold (NO_PATH / YIELD: throttled re-plan)
        self._waiting = {}
        self._desired = {}

    def request_replan(self, robot, reentry=False):
        self._replan[robot] = bool(reentry) or self._replan.get(robot, False)

    def check_path(self, robot):
        """The obstacle set changed: re-plan ``robot`` on the next update only if its path no
        longer keeps clearance."""
        self._recheck.add(robot)

    def blocked(self, robot):
        hold = self._holds.get(robot)
        return hold is not None and hold.reason == NO_PATH

    def _yielding(self, robot):
        hold = self._holds.get(robot)
        return hold is not None and hold.reason == YIELD

    def clear_blocked(self):
        self._holds.clear()

    def update(self, desired, world, arrived, now):
        events = []
        platoon = self.settings.platoon
        self._desired = dict(desired)
        self._asked = set()
        arrived = dict(arrived)     # judged before this update: void for re-issued paths
        lead_changed = lead_stopped = False
        for robot in self.settings.robot_names:
            before = self.cmds.get(robot)
            if lead_changed and on_the_way(before, arrived.get(robot)):
                self.request_replan(robot)
            elif lead_stopped and on_the_way(before, arrived.get(robot)) \
                    and (not before.lead_robot or self._yielding(platoon.lead_robot)):
                # stand-alone: re-plan around the lead's goal, not where it stops; shared: a
                # lead that yields to me never moves on (waiting for my gap would be mutual)
                self.request_replan(robot)
            hold = self._holds.get(robot)
            if lead_changed and hold is not None:  # the lead path is a plan input too: re-test
                self._holds[robot] = dataclasses.replace(hold, retry_at=now)
            self._update_robot(robot, desired.get(robot, Desired()), world, arrived,
                               now, events)
            after = self.cmds.get(robot)
            moved = after is not None and after.mode == FOLLOW \
                and (before is None or before.poses != after.poses)
            if moved:
                arrived[robot] = False
            lead_changed = lead_changed or (moved and robot == platoon.lead_robot)
            lead_stopped = lead_stopped or (
                robot == platoon.lead_robot and after is not before and after is not None
                and after.mode == STOP and before is not None and before.mode == FOLLOW)
        for robot in set(self._jobs) - self._asked:
            self._drop_job(robot)
        return events

    def _update_robot(self, robot, want, world, arrived, now, events):
        recheck = robot in self._recheck
        self._recheck.discard(robot)
        if not want.follow or want.goal is None:
            self._issue_stop(robot, now)
            self._probe_blocked(robot, want, world, arrived, now, events)
            return
        cur = self.cmds.get(robot)
        if recheck and on_the_way(cur, arrived.get(robot)) \
                and not self._rest_still_clear(robot, cur, world, arrived, now):
            self.request_replan(robot)
        aligned_now = world.usable_pose(robot, now)[0] is not None
        if cur is not None and cur.provisional and aligned_now:
            self.request_replan(robot)
        if cur is not None and cur.via and self._at_path_end(robot, cur, world, now):
            self.request_replan(robot)
        if serves(cur, want) and robot not in self._replan:
            self._refresh_flags(cur, want, arrived, now)
        elif self._may_plan(robot, world, now):
            self._plan_and_issue(robot, want, world, arrived, now, events)

    def _at_path_end(self, robot, cmd, world, now):
        nav = feedback_for(world.robots.get(robot), cmd, now,
                           self.settings.freshness.feedback_timeout)
        return nav is not None and (
            nav.state in ARRIVED_STATES
            or (nav.state == STOPPED_STATE and GOAL_MISSED in nav.detail.upper()))

    def _refresh_flags(self, cur, want, arrived, now):
        """Same path, but pick flag or gap request changed -> new cmd_id, same poses."""
        lead = cur.lead_robot if cur.lead_robot and not arrived.get(cur.lead_robot) else ''
        if lead == cur.lead_robot and want.pick_at_goal == cur.pick_at_goal:
            return
        # The same path goes out again with its pivot fields; the rover judges whether it is
        # already past the departure point. Only an arrived rover's one-pose path has no pivot.
        keep = {} if not arrived.get(cur.robot) else dict(
            poses=cur.poses[-1:], pivot_first=False, pivot_heading=0.0)
        self._store(dataclasses.replace(
            cur, cmd_id=self._take_id(cur.robot), pick_at_goal=want.pick_at_goal,
            lead_robot=lead, gap=self.settings.platoon.gap if lead else 0.0,
            issued_at=now, **keep))

    def _may_plan(self, robot, world, now):
        hold = self._holds.get(robot)
        return hold is None or world.obstacle_version != hold.obstacle_version \
            or now >= hold.retry_at

    def _probe_blocked(self, robot, want, world, arrived, now, events):
        if not self.blocked(robot) or want.goal is None \
                or not self._may_plan(robot, world, now):
            return
        done = self._planned(robot, want, world, arrived, now)
        if done is None:
            return
        plan, reason, planned_on, _took = done
        if plan is None and reason == NO_PATH:
            self._holds[robot] = self._hold(NO_PATH, planned_on, now)
        else:
            del self._holds[robot]
            events.append(('plan_recovered', robot))

    def _plan_and_issue(self, robot, want, world, arrived, now, events):
        done = self._planned(robot, want, world, arrived, now)
        if done is None:
            return
        plan, reason, planned_on, took = done
        if plan is None:
            self._issue_stop(robot, now)
            self._note_failure(robot, want, reason, planned_on, now, events)
            return
        if self.blocked(robot):
            events.append(('plan_recovered', robot))
        self._holds.pop(robot, None)
        self._waiting.pop(robot, None)
        self._replan.pop(robot, None)
        self._issue_follow(robot, want, plan, now)
        if planned_on.obstacle_version != world.obstacle_version:
            self.check_path(robot)
        shared = f' behind {plan.lead_robot}' if plan.lead_robot else ''
        if plan.via:
            shared += ' to its hold point' if plan.hold else ' to its turn-in point'
        # planning time is logged: a path that arrives seconds late starts behind the rover
        events.append(('path_issued', f'{robot} -> {want.goal_key} '
                       f'(cmd {self.cmds[robot].cmd_id}, {len(plan.poses)} poses{shared}, '
                       f'planned in {took:.1f} s)'))

    def _note_failure(self, robot, want, reason, world, now, events):
        if reason == NO_PATH and not self.blocked(robot):
            events.append(('plan_failed', f'{robot} -> {want.goal_key}'))
        elif reason != NO_PATH and self._waiting.get(robot) != reason:
            self._waiting[robot] = reason
            events.append(('plan_waiting', f'{robot}: {reason}'))
        if reason in (NO_PATH, YIELD):
            self._holds[robot] = self._hold(reason, world, now)
        else:
            self._holds.pop(robot, None)

    def _hold(self, reason, world, now):
        retry = self.settings.supervisor.plan_retry_period
        return Hold(reason, world.obstacle_version, now + retry)

    def _planned(self, robot, want, world, arrived, now):
        """(plan, reason, world planned on, seconds it took) once the plan for ``want`` is
        computed, else None. Runs on ``self.runner`` on a frozen copy, one job at a time; a job
        whose request changed is dropped and redone."""
        self._asked.add(robot)
        key = self._job_key(robot, want)
        job = self._jobs.get(robot)
        if job is not None and job.key != key:
            self._drop_job(robot)
            job = None
        if job is None:
            if self._jobs:
                return None
            twin, frozen = self._frozen(world)
            job = Job(key, frozen, self.runner.submit(
                twin._compute, robot, want, frozen, dict(arrived), now), now)
            self._jobs[robot] = job
        if not job.result.done():
            return None
        del self._jobs[robot]
        plan, reason = job.result.result()
        return plan, reason, job.world, now - job.asked

    def _job_key(self, robot, want):
        # of the other rover's command a plan reads only the path it is driving
        others = tuple((name, cmd.poses) for name, cmd in self.cmds.items()
                       if name != robot and cmd.mode == FOLLOW)
        return (want.goal_key, want.goal, want.region, self._replan.get(robot), others)

    def _drop_job(self, robot):
        self._jobs.pop(robot).result.cancel()

    def _frozen(self, world):
        """(commander, world) copies a plan computation may read while this one moves on."""
        twin = copy.copy(self)
        twin.cmds, twin._replan = dict(self.cmds), dict(self._replan)
        twin._desired, twin._holds = dict(self._desired), dict(self._holds)
        twin._jobs = {}
        return twin, copy.deepcopy(world)

    def _compute(self, robot, want, world, arrived, now):
        start, provisional, reason = self._start_pose(robot, want.goal, world, now)
        if start is None:
            return None, reason
        if self._waits_for_lead_turn(robot, want, world):
            return None, LEAD_TURNING
        margin = self._platoon_margin(robot)
        if margin > 0.0:
            plan, reason = self._compute_from(robot, want, world, arrived, now, start,
                                              provisional, margin)
            if plan is not None:
                return dataclasses.replace(plan, margin=margin), reason
        return self._compute_from(robot, want, world, arrived, now, start, provisional, 0.0)

    def _platoon_margin(self, robot):
        """How much wider the lead plans than its own footprint needs while the follower is wanted
        to drive, so the follower on the lead path keeps its own clearance from mapped
        obstacles."""
        platoon = self.settings.platoon
        want = self._desired.get(platoon.follower)
        if not platoon.enabled or robot != platoon.lead_robot or want is None or not want.follow:
            return 0.0
        lead, follow = (self.settings.robots[r].plan for r in (robot, platoon.follower))
        if lead.footprint is None or follow.footprint is None:
            return 0.0
        return max(0.0, max(f + follow.clearance - (l + lead.clearance)
                            for f, l in zip(follow.footprint, lead.footprint)))

    def _compute_from(self, robot, want, world, arrived, now, start, provisional, margin):
        params = self._plan_params(robot)
        obstacles = self.obstacles_for(robot, world, arrived, margin)
        if want.region is None:
            goal = Pose2D(want.goal.x, want.goal.y, want.goal.theta)
            if not provisional and self._at_goal(robot, start, goal, want):
                return Plan((goal,), '', False, goal), ''
            if not provisional and self._turn_at_goal(
                    robot, start, goal, want, obstacles,
                    self._goal_params(robot, params, goal), world):
                plan = Plan((goal,), '', False, goal, pivot_heading=goal.theta)
                return self._lead_waits(robot, plan, world, arrived, now)
            goals = [goal]
        else:
            if not provisional and not self._replan.get(robot) and want.region.holds(
                    start, self.settings.robots[robot].goal_yaw_tol):
                return Plan((start,), '', False, start), ''
            goals = spread_goals(self.planner_region(want.region, obstacles, params),
                            _REGION_TRIES)
        for goal in goals:
            plan, reason = self._plan_to(robot, start, goal, provisional, obstacles,
                                         self._goal_params(robot, params, goal), world,
                                         arrived, now)
            if plan is not None or reason != NO_PATH:
                return plan, reason
        if goals and self._blocked_only_by_peer(robot, start, goals[0], provisional, world,
                                                self._goal_params(robot, params, goals[0])):
            return None, YIELD
        return None, NO_PATH


    def planner_region(self, region, obstacles, params):
        return feasible_goals(region, obstacles, self.settings.bounds, params)

    def _plan_to(self, robot, start, goal, provisional, obstacles, params, world,
                 arrived, now):
        """Forward plan, or a pivot-first one when none exists, it is cheaper, or the forward one
        would wait."""
        smooth = []
        forward, reason = self._plan_forward(robot, start, goal, provisional, obstacles,
                                             params, world, arrived, now, smooth)
        if provisional and reason == NO_PATH:
            forward, reason = self._other_guess(robot, start, goal, obstacles, params, world,
                                                arrived, now)
        if provisional or params.pivot_step <= 0.0 or reason not in ('', NO_PATH, YIELD):
            return (forward, reason) if forward is None else \
                self._lead_waits(robot, forward, world, arrived, now)
        if reason in (NO_PATH, YIELD) \
                and self._parks_in_lead_way(robot, goal, world, arrived, now):
            # every path to this goal waits, from every heading: ride to the hold point meanwhile,
            # else wait as before
            plan = self._ride_to_hold(robot, start, goal, obstacles, params, world, arrived)
            if plan is not None:
                return plan, ''

        reasons = {reason}

        def plan_from(pose):
            plan, why = self._plan_forward(robot, pose, goal, False, obstacles, params,
                                           world, arrived, now, smooth)
            reasons.add(why)
            return plan

        plan, heading = search_departure(
            start, (goal.x, goal.y), plan_from,
            lambda p: planning_api.path_length(p.poses),
            obstacles + self._peers_beside_a_turn(robot, world), self.settings.bounds, params,
            forward, goal.theta)
        if plan is None and YIELD in reasons and not smooth:
            # no path behind the lead with a smooth exit: try the ride to the turn-in point
            plan = self._ride_to_turn_in(robot, start, goal, obstacles, params, world,
                                         arrived, now)
            if plan is not None:
                return plan, ''
        if plan is None:
            return None, YIELD if YIELD in reasons else reason
        if heading is not None:
            plan = dataclasses.replace(plan, pivot_heading=heading)
        return self._lead_waits(robot, plan, world, arrived, now)

    def _other_guess(self, robot, start, goal, obstacles, params, world, arrived, now):
        """Plan from another guessed start heading (a guess without a path says nothing about the
        world)."""
        for turn in _GUESS_TURNS:
            theta = start.theta + turn
            pose = Pose2D(start.x, start.y, math.atan2(math.sin(theta), math.cos(theta)))
            plan, reason = self._plan_forward(robot, pose, goal, True, obstacles, params,
                                              world, arrived, now)
            if plan is not None:
                return plan, reason
        return None, NO_PATH

    def _lead_waits(self, robot, plan, world, arrived, now):
        """The lead's path must keep ``conflict_dist`` from a follower path driven ahead of it."""
        platoon = self.settings.platoon
        peer = self.cmds.get(platoon.follower)
        if robot != platoon.lead_robot or peer is None or peer.mode != FOLLOW \
                or peer.lead_robot or arrived.get(platoon.follower) \
                or not serves(peer, self._desired.get(platoon.follower, Desired())):
            return plan, ''
        rest = list(peer.poses)
        pose, _reason = world.usable_pose(platoon.follower, now, need_heading=False)
        if pose is not None:
            rest = rest[nearest_index(rest, pose):]
        if paths_conflict(list(plan.poses), rest, platoon.conflict_dist):
            return None, YIELD
        return plan, ''

    def _plan_forward(self, robot, start, goal, provisional, obstacles, params, world,
                      arrived, now, smooth=None):
        """(plan, reason) from ``start``. ``smooth`` (a list) gets an entry when a path behind the
        lead with a smooth exit to the goal exists, released or not."""
        bounds = self.settings.bounds
        lead_path = self._lead_path(robot, world, arrived, now)
        if lead_path is not None:
            ignore = self._lead_zone_ignore(params, world, arrived)
            extra = {'zone_ignore': ignore} if ignore else {}   # test planners lack it
            shared = self.planner.plan_follower_path(start, lead_path, goal, obstacles,
                                                     bounds, params, **extra)
            if smooth is not None and shared is not None and shared.shared:
                smooth.append(True)
            return self._follower_plan(shared, lead_path, provisional, goal)
        path = self.planner.plan_path(start, goal, obstacles, bounds, params)
        if path is None:
            return None, NO_PATH
        return Plan(tuple(path), '', provisional, goal), ''

    def obstacles_for(self, robot, world, arrived, margin=0.0):
        obstacles = [grown(planner_shape(o), margin) if margin else planner_shape(o)
                     for o in world.obstacles()]
        obstacles += list(self.settings.fixtures)
        return obstacles + self._parked_peers(robot, world, arrived)


    def _follower_plan(self, shared, lead_path, provisional, goal):
        platoon = self.settings.platoon
        if shared is None:
            return None, NO_PATH
        if not self._parks_clear_of_lead(goal, lead_path):
            return None, YIELD
        if shared.shared:
            return Plan(tuple(shared.path), platoon.lead_robot, provisional, goal), ''
        if paths_conflict(shared.path, lead_path, platoon.conflict_dist):
            return None, YIELD
        return Plan(tuple(shared.path), '', provisional, goal), ''


    def _start_pose(self, robot, goal, world, now):
        pose, reason = world.usable_pose(robot, now)
        if pose is not None:
            return Pose2D(*pose), False, ''
        if reason != 'NOT_ALIGNED':
            return None, False, reason
        x, y, _yaw = world.robots[robot].pose
        if math.hypot(goal.x - x, goal.y - y) <= self.settings.robots[robot].arrive_pos_tol:
            # Standing on the goal the bearing to it is pose noise: guess the goal heading
            # reversed (a full loop, long enough for the rover's creep); see _other_guess.
            return Pose2D(x, y, math.atan2(-math.sin(goal.theta), -math.cos(goal.theta))), True, ''
        return Pose2D(x, y, math.atan2(goal.y - y, goal.x - x)), True, ''

    def _at_goal(self, robot, start, goal, want):
        cfg = self.settings.robots[robot]
        if self._replan.get(robot):
            return False
        if math.hypot(start.x - goal.x, start.y - goal.y) > self._pos_tol(robot, want):
            return False
        yaw_error = math.atan2(math.sin(start.theta - goal.theta),
                               math.cos(start.theta - goal.theta))
        return cfg.goal_yaw_tol <= 0.0 or abs(yaw_error) <= cfg.goal_yaw_tol

    def _turn_at_goal(self, robot, start, goal, want, obstacles, params, world):
        """True when the way to ``goal`` is a turn in place (same rotation centre within
        ``goal_pos_tol``, only the heading outside ``goal_yaw_tol``) that the rover makes
        (``pivot_step`` > 0, within ``max_pivot``, clear swept footprint); not at a pick /
        fixture goal."""
        cfg = self.settings.robots[robot]
        if params.pivot_step <= 0.0 or cfg.goal_yaw_tol <= 0.0 or want.pick_at_goal \
                or faces_fixture(cfg, self.settings.fixtures, goal):
            return False
        tol = self._pos_tol(robot, want)
        if math.dist(rotation_centre(start, params), rotation_centre(goal, params)) > tol:
            return False
        turn = math.atan2(math.sin(goal.theta - start.theta),
                          math.cos(goal.theta - start.theta))
        if abs(turn) <= cfg.goal_yaw_tol or abs(turn) > params.max_pivot:
            return False
        end = swept_poses(start, turn, params)[-1]
        if math.hypot(end[0] - goal.x, end[1] - goal.y) > tol:
            return False
        return pivot_clear(start, turn, obstacles + self._peers_beside_a_turn(robot, world),
                           self.settings.bounds, params)

    def _pos_tol(self, robot, want):
        return want.goal_pos_tol or self.settings.robots[robot].goal_pos_tol

    def _plan_params(self, robot):
        params = self.settings.robots[robot].plan
        if self._replan.get(robot):
            run_in = max(params.approach_len, self.settings.supervisor.reentry_approach_len)
            params = dataclasses.replace(params, approach_len=run_in)
        if params.front_zone_runin > 0.0:
            params = dataclasses.replace(params, front_zone_runin=0.0)
        return params

    def _goal_params(self, robot, params, goal, fixture_ahead=None):
        runin = self.settings.robots[robot].plan.front_zone_runin
        if fixture_ahead is None:
            fixture_ahead = faces_fixture(self.settings.robots[robot], self.settings.fixtures,
                                          goal)
        if runin <= 0.0 or not fixture_ahead:
            return params
        return dataclasses.replace(params, front_zone_runin=runin)

    def _issued_params(self, robot, cmd):
        params = self._plan_params(robot)
        if cmd is None or not cmd.poses:
            return params
        return self._goal_params(robot, params, cmd.poses[-1], cmd.fixture_ahead)

    def _lead_path(self, robot, world, arrived, now):
        platoon = self.settings.platoon
        lead = self.cmds.get(platoon.lead_robot)
        if not platoon.enabled or robot != platoon.follower or lead is None \
                or lead.mode != FOLLOW or arrived.get(platoon.lead_robot):
            return None
        poses = list(lead.poses)
        pose, _reason = world.usable_pose(platoon.lead_robot, now, need_heading=False)
        if pose is not None:
            poses = poses[nearest_index(poses, pose):]
        return list(poses) if len(poses) >= _MIN_SHARED_POSES else None

    def _take_id(self, robot):
        cmd_id = self._next_id.get(robot, 1)
        self._next_id[robot] = cmd_id + 1
        return cmd_id

    def _store(self, cmd):
        self.cmds[cmd.robot] = cmd

    def _rest_still_clear(self, robot, cmd, world, arrived, now):
        params = self._issued_params(robot, cmd)
        obstacles = self.obstacles_for(robot, world, arrived, cmd.margin)
        poses = list(cmd.poses)
        pose, _reason = world.usable_pose(robot, now, need_heading=False)
        first = 0 if pose is None else nearest_index(poses, pose)
        ignore = self._lead_zone_ignore(params, world, arrived) if cmd.lead_robot else []
        if not planning_api.keeps_clearance(poses[first:], obstacles, params, ignore):
            return False
        heading = world.usable_pose(robot, now)[0]
        if not cmd.pivot_first or first > 0 or heading is None:
            return True
        yaw = heading[2]
        turn = math.atan2(math.sin(cmd.pivot_heading - yaw), math.cos(cmd.pivot_heading - yaw))
        return pivot_clear(Pose2D(poses[0].x, poses[0].y, yaw), turn,
                           obstacles + self._peers_beside_a_turn(robot, world), None, params)

    def _issue_stop(self, robot, now):
        cur = self.cmds.get(robot)
        if cur is not None and cur.mode == STOP:
            return
        self._replan.pop(robot, None)
        self._store(PathCmd(robot=robot, epoch=self.epoch, cmd_id=self._take_id(robot),
                            mode=STOP, issued_at=now))

    def _issue_follow(self, robot, want, plan, now):
        cfg = self.settings.robots[robot]
        pivot = plan.pivot_heading
        self._store(PathCmd(
            robot=robot, epoch=self.epoch, cmd_id=self._take_id(robot), mode=FOLLOW,
            poses=plan.poses, pivot_first=pivot is not None,
            pivot_heading=0.0 if pivot is None else float(pivot),
            fixture_ahead=not plan.via and faces_fixture(cfg, self.settings.fixtures,
                                                         plan.goal),
            via=plan.via, hold=plan.hold, margin=plan.margin,
            speed_cap=cfg.speed_cap, goal_pos_tol=self._pos_tol(robot, want),
            goal_yaw_tol=cfg.goal_yaw_tol, pick_at_goal=want.pick_at_goal,
            lead_robot=plan.lead_robot,
            gap=self.settings.platoon.gap if plan.lead_robot else 0.0,
            goal_key=want.goal_key, issued_at=now, provisional=plan.provisional,
            goal=want.goal if want.region is None else GoalPose(
                plan.goal.x, plan.goal.y, plan.goal.theta),
            region=want.region))
