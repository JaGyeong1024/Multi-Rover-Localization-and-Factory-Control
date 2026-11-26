"""Path command generation: epoch / cmd_id rules, platoon sharing, plan failures."""
import copy
import dataclasses
import math

import pytest

from host_planning import api as planning
from host_planning.types import Box, Circle, PlatoonPlan, Pose2D

from host_mission.goals import GoalPose
from host_mission import plan_jobs
from host_mission.path_commander import (Desired, PathCmd, PathCommander, footprint_box,
                                         turn_in_point,
                                         paths_conflict, pick_goals_without_fixture)
from host_mission.world_model import WorldModel

import helpers

EPOCH = helpers.EPOCH
NONE_ARRIVED = {'leader': False, 'follower': False}


class NoPathPlanner:
    """Planner stand-in that can be switched to 'no path exists'."""

    def __init__(self):
        self.fail = True
        self.calls = 0

    def plan_path(self, *args, **kwargs):
        self.calls += 1
        return None if self.fail else planning.plan_path(*args, **kwargs)

    def plan_follower_path(self, *args, **kwargs):
        self.calls += 1
        return None if self.fail else planning.plan_follower_path(*args, **kwargs)


def _setup(planner=None, leader=(2.6, 3.3, 0.0), follower=(1.9, 3.3, 0.0)):
    world = WorldModel(pose_timeout=0.5, goals=helpers.goals())
    helpers.place(world, 'leader', *leader, now=0.0)
    helpers.place(world, 'follower', *follower, now=0.0)
    return PathCommander(helpers.settings(), EPOCH, planner), world


def _want(world, name, pick=False, follow=True):
    return Desired(name, world.goals.get(name), follow, pick)


def _both(world, leader_goal='pick_gear', follower_goal='load_gear'):
    return {'leader': _want(world, leader_goal), 'follower': _want(world, follower_goal)}


def test_first_command_path_ends_at_goal():
    commander, world = _setup()
    events = commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 0.0)
    cmd = commander.cmds['leader']
    goal = world.goals.get('pick_gear')
    assert (cmd.mode, cmd.epoch, cmd.cmd_id) == ('FOLLOW', EPOCH, 1)
    assert cmd.poses[-1] == Pose2D(goal.x, goal.y, goal.theta)
    assert math.hypot(cmd.poses[0].x - 2.6, cmd.poses[0].y - 3.3) < 1e-6
    steps = [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(cmd.poses, cmd.poses[1:])]
    assert max(steps) <= 0.05 + 1e-6  # ~0.05 m spacing
    cfg = helpers.settings().robots['leader']
    assert (cmd.goal_pos_tol, cmd.goal_yaw_tol) == (cfg.goal_pos_tol, cfg.goal_yaw_tol)
    assert cmd.lead_robot == '' and not cmd.pick_at_goal
    assert events[0][0] == 'path_issued'
    assert commander.cmds['follower'].mode == 'STOP'               # nothing wanted -> STOP


def test_unchanged_goal_keeps_cmd_id():
    commander, world = _setup()
    want = _both(world)
    commander.update(want, world, NONE_ARRIVED, 0.0)
    before = dict(commander.cmds)
    for now in (0.2, 0.4, 0.6):
        helpers.place(world, 'leader', 2.7, 3.3, 0.0, now)
        helpers.place(world, 'follower', 2.0, 3.3, 0.0, now)
        assert commander.update(want, world, NONE_ARRIVED, now) == []
    assert commander.cmds == before


def test_cmd_id_increases_on_changes():
    commander, world = _setup()
    seen = []

    def update(want, arrived=NONE_ARRIVED):
        commander.update({'leader': want}, world, arrived, 0.0)
        cmd = commander.cmds['leader']
        assert cmd.epoch == EPOCH
        seen.append(cmd.cmd_id)
        return cmd

    update(_want(world, 'pick_gear'))
    update(_want(world, 'pick_bolt'))
    assert update(_want(world, 'pick_bolt', follow=False)).mode == 'STOP'
    assert update(_want(world, 'pick_bolt')).mode == 'FOLLOW'
    path = commander.cmds['leader'].poses
    flagged = update(_want(world, 'pick_bolt', pick=True))         # only the pick flag
    assert flagged.pick_at_goal and flagged.poses == path
    assert seen == [1, 2, 3, 4, 5]


def test_flag_change_after_arrival_goal_only():
    commander, world = _setup()
    commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 0.0)
    commander.update({'leader': _want(world, 'pick_gear', pick=True)}, world,
                     {'leader': True}, 1.0)
    cmd = commander.cmds['leader']
    assert cmd.pick_at_goal and cmd.cmd_id == 2
    assert cmd.poses == (Pose2D(4.9, 2.9, 0.0),)


def test_follower_shares_leader_path():
    commander, world = _setup()
    commander.update(_both(world), world, NONE_ARRIVED, 0.0)
    leader, follower = commander.cmds['leader'], commander.cmds['follower']
    assert follower.lead_robot == 'leader'
    assert follower.gap == helpers.settings().platoon.gap
    lead_xy = [(p.x, p.y) for p in leader.poses]
    on_lead = [p for p in follower.poses
               if min(math.hypot(p.x - x, p.y - y) for x, y in lead_xy) < 1e-3]
    assert len(on_lead) >= 10
    goal = world.goals.get('load_gear')
    assert follower.poses[-1] == Pose2D(goal.x, goal.y, goal.theta)
    assert leader.lead_robot == '' and leader.gap == 0.0


def test_follower_gap_ends_on_leader_arrival():
    commander, world = _setup()
    want = _both(world)
    commander.update(want, world, NONE_ARRIVED, 0.0)
    shared = commander.cmds['follower']
    commander.update(want, world, {'leader': True, 'follower': False}, 5.0)
    released = commander.cmds['follower']
    assert released.lead_robot == '' and released.gap == 0.0
    assert released.cmd_id == shared.cmd_id + 1 and released.poses == shared.poses


def test_obstacle_change_keeps_clear_path():
    """A changed obstacle set re-plans only a path whose part still ahead of the
    rover loses the planner's clearance (a new cmd_id makes the tracker re-load)."""
    commander, world = _setup()
    want = {'leader': _want(world, 'pick_gear')}
    commander.update(want, world, NONE_ARRIVED, 0.0)
    path = commander.cmds['leader'].poses
    here = path[len(path) // 2]
    helpers.place(world, 'leader', here.x, here.y, here.theta, 0.2)
    behind, ahead = path[len(path) // 5], path[len(path) * 3 // 4]
    world.set_mapped_obstacles([(behind.x, behind.y, 0.10)])
    commander.check_path('leader')
    assert commander.update(want, world, NONE_ARRIVED, 0.2) == []
    assert commander.cmds['leader'].cmd_id == 1
    world.set_mapped_obstacles([(ahead.x, ahead.y, 0.10)])
    commander.check_path('leader')
    commander.update(want, world, NONE_ARRIVED, 0.4)
    cmd = commander.cmds['leader']
    assert cmd.cmd_id == 2 and cmd.poses != path
    assert planning.keeps_clearance(cmd.poses, [Circle(ahead.x, ahead.y, 0.10)],
                                    helpers.settings().robots['leader'].plan)


def test_follower_replans_on_leader_change():
    commander, world = _setup()
    want = _both(world)
    commander.update(want, world, NONE_ARRIVED, 0.0)
    first = commander.cmds['follower'].cmd_id
    helpers.place(world, 'leader', 3.0, 3.4, 0.2, 1.0)
    helpers.place(world, 'follower', 2.3, 3.3, 0.0, 1.0)
    commander.request_replan('leader')
    commander.update(want, world, NONE_ARRIVED, 1.0)
    assert commander.cmds['leader'].cmd_id == 2
    assert commander.cmds['follower'].cmd_id == first + 1


def test_follower_falls_back_to_solo_path():
    commander, world = _setup()
    settings = commander.settings
    object.__setattr__(settings.platoon, 'enabled', False)
    commander.update(_both(world), world, NONE_ARRIVED, 0.0)
    follower = commander.cmds['follower']
    assert follower.mode == 'FOLLOW' and follower.lead_robot == '' and follower.gap == 0.0


def test_follower_yields_when_crossing_leader():
    # side by side at the pick spot, both leaving: the follower cannot share and waits
    commander, world = _setup(leader=(4.9, 2.9, 0.0), follower=(4.9, 2.35, 0.0))
    want = _both(world, 'leader_standby', 'follower_unload')
    events = commander.update(want, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['leader'].mode == 'FOLLOW'
    assert commander.cmds['follower'].mode == 'STOP'
    # waiting: for the leader's turn out (LEAD_TURNING) or its way (YIELD)
    assert {('plan_waiting', 'follower: YIELD_TO_LEAD'),
            ('plan_waiting', 'follower: LEAD_TURNING')} & set(events)
    assert not commander.blocked('follower')
    helpers.place(world, 'leader', 2.6, 3.3, 0.0, 60.0)
    lead = commander.cmds['leader']                                # ... and says so: its turn
    world.update_nav('leader', 'PARKED', '', lead.epoch, lead.cmd_id, now=60.0)
    helpers.place(world, 'follower', 4.9, 2.35, 0.0, 60.0)
    commander.update(want, world, {'leader': True, 'follower': False}, 60.0)
    assert commander.cmds['follower'].mode == 'FOLLOW'
    clearance = min(math.hypot(p.x - 2.6, p.y - 3.3) for p in commander.cmds['follower'].poses)
    assert clearance >= commander.settings.platoon.peer_radius


def test_new_lead_path_rechecks_moving_follower():
    """Leader re-enters at standby while the released follower drives to unload: the follower's path must be checked against the NEW leader path, not the old one."""
    commander, world = _setup(leader=(2.6, 3.3, 0.0), follower=(4.9, 2.35, 0.0))
    want = _both(world, 'leader_standby', 'follower_unload')
    commander.update(want, world, {'leader': True, 'follower': False}, 0.0)
    released = commander.cmds['follower']
    assert released.mode == 'FOLLOW'
    helpers.place(world, 'leader', 2.5, 3.32, 0.0, 1.0)
    helpers.place(world, 'follower', 4.9, 2.35, 0.0, 1.0)
    commander.request_replan('leader', reentry=True)
    commander.update(want, world, NONE_ARRIVED, 1.0)
    follower, leader = commander.cmds['follower'], commander.cmds['leader']
    assert leader.cmd_id > 1 and follower.cmd_id > released.cmd_id
    assert follower.mode == 'STOP' or not paths_conflict(
        follower.poses, leader.poses, commander.settings.platoon.conflict_dist)


def test_follower_never_parks_on_lead_path():
    """The lead is planned first in a tick, around the follower where it stands; the
    follower's new goal is issued after it. Rule: the follower waits until the lead's
    remaining path is clear of its goal (without the ride to a hold point: platoon.hold_ride)."""
    commander, world = _setup(leader=(4.4, 3.0, math.pi), follower=(4.9, 3.0, math.pi))
    commander.settings = dataclasses.replace(commander.settings, platoon=dataclasses.replace(
        commander.settings.platoon, hold_ride=False))
    want = {'leader': Desired('leader_standby', GoalPose(2.0, 3.0, math.pi), True),
            'follower': Desired('follower_unload', GoalPose(3.0, 3.0, math.pi), True)}
    events = commander.update(want, world, NONE_ARRIVED, 0.0)
    leader = commander.cmds['leader']
    assert leader.mode == 'FOLLOW' and commander.cmds['follower'].mode == 'STOP'
    assert ('plan_waiting', 'follower: YIELD_TO_LEAD') in events
    commander.check_path('leader')                  # e.g. a mapped box moved elsewhere
    commander.update(want, world, NONE_ARRIVED, 0.2)
    assert commander.cmds['leader'] == leader       # its path stays valid: no re-plan
    helpers.place(world, 'leader', 2.4, 3.0, math.pi, 3.0)
    helpers.place(world, 'follower', 4.9, 3.0, math.pi, 3.0)
    commander.update(want, world, NONE_ARRIVED, 3.0)
    follower = commander.cmds['follower']
    assert follower.mode == 'FOLLOW' and follower.poses[-1] == Pose2D(3.0, 3.0, math.pi)


def test_lead_replan_avoids_follower_position():
    """The lead never waits for a follower on its way, and was planned against it only at its
    goal. Rule: the follower occupies where it stands until it has left, as well as its goal."""
    commander, world = _setup(leader=(4.4, 3.0, math.pi), follower=(2.9, 2.3, math.pi))
    want = {'leader': Desired('leader_standby', GoalPose(2.0, 3.0, math.pi), True),
            'follower': Desired('follower_unload', GoalPose(2.0, 2.3, math.pi), True)}
    commander.update(want, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['follower'].mode == 'FOLLOW'          # on its way to (2.0, 2.3)
    helpers.place(world, 'leader', 4.4, 3.0, math.pi, 1.0)
    helpers.place(world, 'follower', 3.4, 3.0, math.pi, 1.0)
    commander.request_replan('leader')
    commander.update(want, world, NONE_ARRIVED, 1.0)
    leader = commander.cmds['leader']
    settings = commander.settings
    standing = footprint_box((3.4, 3.0, math.pi), settings.robots['follower'].plan.footprint)
    assert leader.mode == 'STOP' or planning.keeps_clearance(
        list(leader.poses), [standing], settings.robots['leader'].plan), leader.poses


class PeerBoxPlanner:
    """Real planner, except that a plan fails while the footprint of a rover named in
    ``boxed_by`` stands among the obstacles at one of the poses given for it (the
    planning rover is boxed in by that peer standing there)."""

    def __init__(self, settings, boxed_by):
        self.boxes = [footprint_box(pose, settings.robots[name].plan.footprint)
                      for name, poses in boxed_by.items() for pose in poses]

    def _boxed(self, obstacles):
        return any(isinstance(o, Box) and math.hypot(o.x - b.x, o.y - b.y) < 1e-6
                   and (o.width, o.height) == (b.width, b.height)
                   for o in obstacles for b in self.boxes)

    def plan_path(self, start, goal, obstacles, *args, **kwargs):
        return None if self._boxed(obstacles) else \
            planning.plan_path(start, goal, obstacles, *args, **kwargs)

    def plan_follower_path(self, start, lead_path, goal, obstacles, *args, **kwargs):
        return None if self._boxed(obstacles) else \
            planning.plan_follower_path(start, lead_path, goal, obstacles, *args, **kwargs)


AT_PICK = {'leader': (4.9, 2.9, 0.0), 'follower': (4.9, 2.35, 0.0)}   # pick / load poses


def _boxed_in(boxed_by, extra=None):
    """Leader at the pick pose, follower at the load pose, both wanted home; each rover in
    ``boxed_by`` boxes the other in where it stands (plus the ``extra`` poses)."""
    poses = {name: [AT_PICK[name]] + list((extra or {}).get(name, ())) for name in boxed_by}
    commander, world = _setup(PeerBoxPlanner(helpers.settings(), poses), **AT_PICK)
    want = _both(world, 'leader_standby', 'follower_unload')
    return commander, commander.update(want, world, NONE_ARRIVED, 0.0)


def test_yielding_lead_releases_waiting_follower():
    """A follower on a shared path waits for its gap to the lead; once the lead yields to it
    (re-plan boxed in by where the follower stands), the follower must get a path of its own,
    or both stand for good."""
    standing = (4.4, 2.35, 0.0)
    commander, world = _setup(PeerBoxPlanner(helpers.settings(), {'follower': [standing]}),
                              leader=(3.0, 3.3, 0.0), follower=(2.3, 3.3, 0.0))
    want = _both(world, 'pick_gear', 'load_gear')
    commander.update(want, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['follower'].lead_robot == 'leader'
    helpers.place(world, 'leader', 4.0, 2.9, 0.0, 1.0)
    helpers.place(world, 'follower', *standing, 1.0)
    commander.request_replan('leader')
    events = commander.update(want, world, NONE_ARRIVED, 1.0)
    assert ('plan_waiting', 'leader: YIELD_TO_LEAD') in events
    follower = commander.cmds['follower']
    assert follower.mode == 'FOLLOW' and follower.lead_robot == '', follower


def test_boxed_in_lead_waits_for_follower():
    commander, events = _boxed_in(boxed_by=('follower',))
    assert ('plan_waiting', 'leader: YIELD_TO_LEAD') in events
    assert not commander.blocked('leader')
    follower = commander.cmds['follower']
    assert follower.mode == 'FOLLOW' and follower.lead_robot == ''


def test_waiting_lead_not_reasked_same_map():
    """The lead waits at its pick for the follower to leave. While it stands there and the
    map is the same, a re-test that finds no path even with the follower at its goal is pose
    noise: it keeps waiting. On a new map the question is asked again."""
    settings = helpers.settings()
    planner = PeerBoxPlanner(settings, {'follower': [AT_PICK['follower']]})
    commander, world = _setup(planner, **AT_PICK)
    want = _both(world, 'leader_standby', 'follower_unload')
    events = commander.update(want, world, NONE_ARRIVED, 0.0)
    assert ('plan_waiting', 'leader: YIELD_TO_LEAD') in events
    on_its_way = (3.6, 2.5, math.pi)
    unload = helpers.goals().get('follower_unload')
    planner.boxes += [footprint_box(pose, settings.robots['follower'].plan.footprint)
                      for pose in (on_its_way, (unload.x, unload.y, unload.theta))]
    helpers.place(world, 'leader', *AT_PICK['leader'], 3.0)
    helpers.place(world, 'follower', *on_its_way, 3.0)
    events = commander.update(want, world, NONE_ARRIVED, 3.0)
    assert not commander.blocked('leader')
    assert not [e for e in events if e[0] == 'plan_failed']
    world.set_mapped_obstacles([(2.0, 4.2, 0.10)])
    helpers.place(world, 'leader', *AT_PICK['leader'], 3.2)
    helpers.place(world, 'follower', *on_its_way, 3.2)
    events = commander.update(want, world, NONE_ARRIVED, 3.2)
    assert commander.blocked('leader') and ('plan_failed', 'leader -> leader_standby') in events


def test_lead_not_waiting_for_boxing_goal():
    """Waiting is worth it only when the lead has a path once the follower has left, i.e. with
    the follower standing at its goal."""
    unload = helpers.goals().get('follower_unload')
    commander, events = _boxed_in(
        boxed_by=('follower',), extra={'follower': [(unload.x, unload.y, unload.theta)]})
    assert ('plan_waiting', 'leader: YIELD_TO_LEAD') not in events
    assert commander.blocked('leader') and ('plan_failed', 'leader -> leader_standby') in events


def test_mutually_boxed_never_both_wait():
    """Tie-break: the follower yields to the lead; the lead yields only to a follower
    that can leave with the lead where it stands. Each boxed in by the other -> the lead
    reports NO_PATH (mission pauses for the operator) instead of an endless mutual wait."""
    commander, events = _boxed_in(boxed_by=('leader', 'follower'))
    assert commander.blocked('leader') and ('plan_failed', 'leader -> leader_standby') in events
    assert commander.blocked('follower')        # the boxed lead will not move: surfaced, not a wait
    assert {cmd.mode for cmd in commander.cmds.values()} == {'STOP'}


class HeadingPlanner:
    """Real plan_path; the follower plan depends only on the start heading: from
    ``facing`` (the heading the rover stands at) only a stand-alone path that crosses the
    lead's path, from any other heading a path sharing the lead's path."""

    def __init__(self, facing, crossing):
        self.facing, self.crossing = facing, crossing

    def plan_path(self, *args, **kwargs):
        return planning.plan_path(*args, **kwargs)

    def plan_follower_path(self, start, lead_path, goal, *args, **kwargs):
        if abs(math.remainder(start.theta - self.facing, 2.0 * math.pi)) < 1e-6:
            return PlatoonPlan([start, self.crossing, Pose2D(goal.x, goal.y, goal.theta)], False)
        return PlatoonPlan([start, Pose2D(3.0, 3.0, math.pi), Pose2D(goal.x, goal.y, goal.theta)],
                           True)


def test_waiting_follower_facing_away_turns():
    """A follower half-way through its departure pivot at the load pose (heading -84 deg of
    -135) is re-planned. From -84 deg no shared path exists and the stand-alone forward path
    crosses the lead's path (YIELD), while the pivot to -135 deg shares the lead's path again.
    Rule: a wait is only for a follower that no departure turn takes off it."""
    facing = -math.pi / 2
    planner = HeadingPlanner(facing, crossing=Pose2D(3.4, 3.9, math.pi))
    commander, world = _setup(planner, leader=(3.6, 4.0, math.pi), follower=(4.4, 3.0, facing))
    want = {'leader': Desired('leader_standby', GoalPose(2.0, 4.0, math.pi), True),
            'follower': Desired('follower_unload', GoalPose(2.0, 3.0, math.pi), True)}
    events = commander.update(want, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['leader'].mode == 'FOLLOW'
    follower = commander.cmds['follower']
    assert ('plan_waiting', 'follower: YIELD_TO_LEAD') not in events
    assert follower.mode == 'FOLLOW' and follower.lead_robot == 'leader', follower
    assert follower.pivot_first and follower.pivot_heading != facing


def _line(x, y, heading, count, step=0.05):
    return tuple(Pose2D(x + i * step * math.cos(heading), y + i * step * math.sin(heading),
                        heading) for i in range(count))


def _turning(robot, poses, heading, goal_key):
    goal = poses[-1]
    return PathCmd(robot=robot, epoch=EPOCH, cmd_id=7, mode='FOLLOW', poses=poses,
                   pivot_first=True, pivot_heading=heading, goal_key=goal_key,
                   goal=GoalPose(goal.x, goal.y, goal.theta))


def test_turn_keeps_clear_of_peer_turn():
    """Leaving the pick together, the leader turns 0 -> 90 deg and the follower, 0.51 m south
    of it, 0 -> -135 deg. Rule: the other rover occupies the rest of its own turn; a follower
    turn that does not keep clear of it is not kept."""
    commander, world = _setup(leader=(4.27, 3.43, 0.0), follower=(4.28, 2.92, 0.0))
    lead = _turning('leader', _line(4.27, 3.43, math.pi / 2, 10), math.pi / 2, 'leader_standby')
    turn = -0.75 * math.pi
    follower = _turning('follower', _line(4.28, 2.92, turn, 10), turn, 'follower_unload')
    commander.cmds.update(leader=lead, follower=follower)
    want = {'leader': Desired('leader_standby', lead.goal, True),
            'follower': Desired('follower_unload', follower.goal, True)}
    commander.check_path('follower')
    commander.update(want, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['leader'] == lead
    assert commander.cmds['follower'] != follower


def _reach(plan):
    front, rear, half = plan.footprint
    return math.hypot(max(front + plan.pivot_offset, rear - plan.pivot_offset), half)


def _follower_turn_kept(south, lead_turns):
    """The follower ``south`` m south of the leader (both heading 0) has a -135 deg
    departure turn to make (a rear corner sweeps through north); is it still clear?"""
    commander, world = _setup(leader=(4.27, 3.43, 0.0), follower=(4.27, 3.43 - south, 0.0))
    turn = -0.75 * math.pi
    cmd = _turning('follower', _line(4.27, 3.43 - south, turn, 10), turn, 'follower_unload')
    commander.cmds['follower'] = cmd
    if lead_turns:                          # 0 -> 90 deg: a rear corner sweeps through south
        commander.cmds['leader'] = _turning(
            'leader', _line(4.27, 3.43, math.pi / 2, 10), math.pi / 2, 'leader_standby')
    return commander._rest_still_clear('follower', cmd, world, NONE_ARRIVED, 0.0)


def test_simultaneous_turns_keep_margins():
    """``plan.clearance`` covers the error of the rover it belongs to: the follower's turn
    keeps its own and the leader's from the rest of the leader's turn."""
    settings = helpers.settings()
    lead, foll = settings.robots['leader'].plan, settings.robots['follower'].plan
    sweeps_touch = _reach(lead) + _reach(foll)
    both = foll.clearance + lead.clearance
    assert not _follower_turn_kept(sweeps_touch + 0.5 * (foll.clearance + both), lead_turns=True)
    assert _follower_turn_kept(sweeps_touch + both + 0.03, lead_turns=True)


def _leader_in_turn(pose, nav=None):
    """The leader at ``pose`` with a 90 deg leading turn commanded at (4.272, 3.43); ``nav`` =
    (state, detail) of its report for that command, None = no report yet."""
    commander, world = _setup(leader=pose, follower=(4.27, 2.88, 0.0))
    cmd = _turning('leader', _line(4.272, 3.43, math.pi / 2, 10), math.pi / 2, 'leader_standby')
    commander.cmds['leader'] = cmd
    if nav is not None:
        world.update_nav('leader', nav[0], nav[1], cmd.epoch, cmd.cmd_id, now=0.1)
    return commander._turn_rest('leader', world.robots['leader'])


def test_shifted_tracked_point_turn_finishes():
    """25 deg into its 90 deg turn the leader's tracked point is 0.03 m from its first path
    pose; read as 'driven off', the rest of its turn would no longer count."""
    assert _leader_in_turn((4.279, 3.460, math.radians(25)), ('TRACKING', 'PIVOT'))
    assert _leader_in_turn((4.279, 3.460, math.radians(25)))
    assert not _leader_in_turn((4.279, 3.460, math.radians(88)), ('TRACKING', 'PIVOT'))


def test_position_step_does_not_end_turn():
    """The leader's camera position moved by its arm steps the pose as the wheels start. Where
    the pose is says nothing; the rover says whether it still pivots (TRACKING PIVOT /
    ALIGNING) or follows its path."""
    stepped = (4.272, 3.73, math.radians(25))           # 0.30 m from the first pose
    assert _leader_in_turn(stepped, ('TRACKING', 'PIVOT'))
    assert _leader_in_turn(stepped, ('TRACKING', 'ALIGNING'))
    assert _leader_in_turn(stepped, ('STOPPED', 'STUCK'))     # stopped in its turn: not made
    assert not _leader_in_turn(stepped, ('TRACKING', ''))
    assert not _leader_in_turn(stepped, ('TRACKING', 'CURVE'))
    assert not _leader_in_turn((4.279, 3.460, math.radians(25)), ('PARKED', ''))


def test_other_command_report_ignored_for_turn():
    commander, world = _setup(leader=(4.279, 3.460, math.radians(25)), follower=(4.27, 2.88, 0.0))
    cmd = _turning('leader', _line(4.272, 3.43, math.pi / 2, 10), math.pi / 2, 'leader_standby')
    commander.cmds['leader'] = cmd
    world.update_nav('leader', 'TRACKING', '', cmd.epoch, cmd.cmd_id - 1, now=0.1)
    assert commander._turn_rest('leader', world.robots['leader'])
    world.update_nav('leader', 'TRACKING', '', cmd.epoch + 1, cmd.cmd_id, now=0.2)
    assert commander._turn_rest('leader', world.robots['leader'])


def test_turn_beside_standing_rover_own_margin():
    """The taught pick and load poses leave 0.10 m between one rover's sweep and the other
    standing: a second margin there would never let the leader leave a pick."""
    settings = helpers.settings()
    lead, foll = settings.robots['leader'].plan, settings.robots['follower'].plan
    touch = lead.footprint[2] + _reach(foll)        # leader at heading 0: half width
    assert _follower_turn_kept(touch + foll.clearance + 0.025, lead_turns=False)
    assert not _follower_turn_kept(touch + 0.5 * foll.clearance, lead_turns=False)


def test_paths_conflict_is_spatial():
    lane_a = [Pose2D(0.05 * i, 0.0, 0.0) for i in range(60)]
    lane_b = [Pose2D(0.05 * i, 1.0, 0.0) for i in range(60)]
    crossing = [Pose2D(1.0, -1.0 + 0.05 * i, 0.0) for i in range(60)]
    assert not paths_conflict(lane_a, lane_b, 0.5)                 # parallel, 1 m apart
    assert paths_conflict(crossing, lane_a, 0.5)
    late = [Pose2D(1.0, -6.0 + 0.05 * i, 0.0) for i in range(140)]
    assert paths_conflict(late, lane_a, 0.5)        # crosses where the lead WILL be, however late
    assert not paths_conflict(late, lane_a[30:], 0.5)              # lead already past x=1.5


def test_no_path_stops_blocks_and_retries():
    planner = NoPathPlanner()
    commander, world = _setup(planner)
    want = {'leader': _want(world, 'pick_gear')}
    events = commander.update(want, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['leader'].mode == 'STOP' and commander.blocked('leader')
    assert ('plan_failed', 'leader -> pick_gear') in events
    calls = planner.calls
    assert commander.update(want, world, NONE_ARRIVED, 0.2) == []
    assert planner.calls == calls
    retry = commander.settings.supervisor.plan_retry_period
    helpers.place(world, 'leader', 2.6, 3.3, 0.0, retry + 0.1)
    commander.update(want, world, NONE_ARRIVED, retry + 0.1)
    assert planner.calls > calls and commander.blocked('leader')
    planner.fail = False
    world.set_manual_obstacles([(5.5, 4.3, 0.1)])                  # obstacle change -> retry now
    helpers.place(world, 'leader', 2.6, 3.3, 0.0, retry + 0.3)
    events = commander.update(want, world, NONE_ARRIVED, retry + 0.3)
    assert ('plan_recovered', 'leader') in events
    assert commander.cmds['leader'].mode == 'FOLLOW' and not commander.blocked('leader')


def test_blocked_plan_probed_while_paused():
    planner = NoPathPlanner()
    commander, world = _setup(planner)
    commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 0.0)
    paused = {'leader': _want(world, 'pick_gear', follow=False)}
    planner.fail = False
    helpers.place(world, 'leader', 2.6, 3.3, 0.0, 5.0)
    events = commander.update(paused, world, NONE_ARRIVED, 5.0)
    assert ('plan_recovered', 'leader') in events and not commander.blocked('leader')
    assert commander.cmds['leader'].mode == 'STOP'


def test_stale_pose_is_never_planned_from():
    commander, world = _setup()
    events = commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 5.0)
    assert commander.cmds['leader'].mode == 'STOP'
    assert ('plan_waiting', 'leader: STALE_POSE') in events
    assert not commander.blocked('leader')
    helpers.place(world, 'leader', 2.6, 3.3, 0.0, 5.2)
    commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 5.2)
    assert commander.cmds['leader'].mode == 'FOLLOW'


def test_unaligned_gets_provisional_then_real_path():
    commander, world = _setup()
    helpers.place(world, 'leader', 2.6, 3.3, 0.0, 0.0, aligned=False)
    want = {'leader': _want(world, 'pick_bolt')}
    commander.update(want, world, NONE_ARRIVED, 0.0)
    first = commander.cmds['leader']
    goal = world.goals.get('pick_bolt')
    assert first.mode == 'FOLLOW' and first.provisional
    assert abs(first.poses[0].theta - math.atan2(goal.y - 3.3, goal.x - 2.6)) < 1e-6
    helpers.place(world, 'leader', 2.8, 3.4, 1.0, 0.4, aligned=True)
    commander.update(want, world, NONE_ARRIVED, 0.4)
    second = commander.cmds['leader']
    assert second.cmd_id == first.cmd_id + 1 and not second.provisional
    assert abs(second.poses[0].theta - 1.0) < 1e-6


@pytest.mark.parametrize('dx, dy', [(0.02, 0.0), (-0.02, 0.01), (0.0, -0.03), (-0.01, -0.02)])
def test_unaligned_at_goal_gets_creep_path(dx, dy):
    """The alignment drive starts at home: the bearing to a goal the rover stands on is the
    direction of the pose noise, so the guess is the goal heading reversed: a loop with
    length (a path of no length is judged in place by the rover)."""
    commander, world = _setup(follower=(4.5, 2.6, 0.0))
    goal = world.goals.get('leader_standby')
    helpers.place(world, 'leader', goal.x + dx, goal.y + dy, 0.0, 0.0, aligned=False)
    commander.update({'leader': _want(world, 'leader_standby')}, world, NONE_ARRIVED, 0.0)
    first = commander.cmds['leader']
    assert first.mode == 'FOLLOW' and first.provisional
    assert planning.path_length(first.poses) > 1.0
    assert abs(abs(math.remainder(first.poses[0].theta - goal.theta, math.tau)) - math.pi) < 1e-6


def test_pathless_guessed_heading_replaced():
    """A provisional heading is a guess (the rover creeps straight ahead whatever the path):
    one without a path is not a failure while another guess has one."""
    asked = []

    class ThirdGuess:
        def plan_path(self, start, goal, *args, **kwargs):
            asked.append(start.theta)
            return None if len(asked) < 3 else [start, Pose2D(goal.x, goal.y, goal.theta)]

    commander, world = _setup(planner=ThirdGuess())
    helpers.place(world, 'leader', 2.6, 3.3, 0.0, 0.0, aligned=False)
    events = commander.update({'leader': _want(world, 'pick_bolt')}, world, NONE_ARRIVED, 0.0)
    first = commander.cmds['leader']
    goal = world.goals.get('pick_bolt')
    bearing = math.atan2(goal.y - 3.3, goal.x - 2.6)
    turns = [round(math.remainder(theta - bearing, math.tau), 3) for theta in asked]
    assert turns == [0.0, 1.047, -1.047]
    assert first.mode == 'FOLLOW' and first.provisional and first.poses[0].theta == asked[2]
    assert not [e for e in events if e[0] == 'plan_failed']


def test_guessed_heading_with_path_is_no_path():
    class Never:
        def plan_path(self, *args, **kwargs):
            return None

    commander, world = _setup(planner=Never())
    helpers.place(world, 'leader', 2.6, 3.3, 0.0, 0.0, aligned=False)
    events = commander.update({'leader': _want(world, 'pick_bolt')}, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['leader'].mode == 'STOP'
    assert ('plan_failed', 'leader -> pick_bolt') in events


def test_rover_at_goal_not_sent_around_loop():
    commander, world = _setup(leader=(4.88, 2.91, 0.03))
    commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['leader'].poses == (Pose2D(4.9, 2.9, 0.0),)
    commander.request_replan('leader', reentry=True)
    commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 0.2)
    assert len(commander.cmds['leader'].poses) > 10


def test_reentry_uses_a_longer_run_in():
    commander, world = _setup(leader=(4.6, 3.1, 0.6))
    want = {'leader': _want(world, 'pick_gear')}
    commander.update(want, world, NONE_ARRIVED, 0.0)
    commander.request_replan('leader', reentry=True)
    commander.update(want, world, NONE_ARRIVED, 0.2)
    run_in = commander.settings.supervisor.reentry_approach_len
    tail = commander.cmds['leader'].poses[-int(run_in / 0.05) + 1:]
    assert len(tail) >= 10 and all(abs(p.y - 2.9) < 0.01 and abs(p.theta) < 0.02 for p in tail)


# ---------- re-entry of a goal missed in heading only ----------
AWAY = (5.0, 4.0, 0.0)      # the follower out of the way: today's re-entry loop has room


def _reentry(commander, world, want, today=False):
    """The command a re-entry (GOAL_MISSED) of ``want`` gets; ``today``: with the turn at
    the goal switched off (the plain re-entry path, or STOP when there is none)."""
    if today:
        commander._turn_at_goal = lambda *args, **kwargs: False
    commander.request_replan('leader', reentry=True)
    commander.update({'leader': want}, world, NONE_ARRIVED, 0.0)
    return commander.cmds['leader']


def _turns_at_goal(cmd, goal):
    return cmd.mode == 'FOLLOW' and cmd.pivot_first and cmd.poses == (goal,)


def test_heading_miss_reentered_by_turn():
    """Home 2.6 cm off (inside goal_pos_tol), 8.5 deg off in heading; the re-entry was a
    2.3 m loop. A turn in place at the goal fixes it."""
    commander, world = _setup(leader=(2.62, 3.28, 0.15), follower=AWAY)
    want = _want(world, 'leader_standby')
    cmd = _reentry(commander, world, want)
    goal = Pose2D(2.6, 3.3, 0.0)
    assert _turns_at_goal(cmd, goal) and cmd.pivot_heading == goal.theta
    assert planning.path_length(cmd.poses) == 0.0 and not cmd.fixture_ahead


def test_turned_away_rover_turns_back():
    """Trip abandoned during the departure turn: the rover has not left its home spot.
    Not a re-entry, the first plan home; a forward path would be a loop."""
    commander, world = _setup(leader=(2.61, 3.29, -1.0), follower=AWAY)
    commander.update({'leader': _want(world, 'leader_standby')}, world, NONE_ARRIVED, 0.0)
    cmd = commander.cmds['leader']
    goal = Pose2D(2.6, 3.3, 0.0)
    assert _turns_at_goal(cmd, goal) and cmd.pivot_heading == goal.theta


def _assert_re_entered_as_today(setup, want_of, leader):
    commander, world = setup(leader)
    cmd = _reentry(commander, world, want_of(world))
    commander, world = setup(leader)
    today = _reentry(commander, world, want_of(world), today=True)
    assert not (cmd.pivot_first and len(cmd.poses) == 1)
    assert (cmd.mode, cmd.poses, cmd.pivot_first, cmd.pivot_heading) == \
        (today.mode, today.poses, today.pivot_first, today.pivot_heading)


def test_position_miss_reentered_by_path():
    _assert_re_entered_as_today(lambda leader: _setup(leader=leader, follower=AWAY),
                                lambda world: _want(world, 'leader_standby'),
                                (2.66, 3.30, 0.15))         # 0.06 m off: outside goal_pos_tol


def test_pick_heading_miss_reentered_by_run_in():
    """A pick (pick_at_goal) or a goal facing a fixture is reached by the straight run-in, never by a turn at the goal."""
    _assert_re_entered_as_today(lambda leader: _setup(leader=leader, follower=AWAY),
                                lambda world: _want(world, 'leader_standby', pick=True),
                                (2.62, 3.28, 0.15))
    _assert_re_entered_as_today(_fixture_setup,
                                lambda world: Desired('pick', GoalPose(4.89, 2.69, 0.0), True),
                                (4.87, 2.70, 0.15))


def test_turn_into_peer_reentered():
    """The follower stands clear of the leader but inside the sweep of its turn (the
    rear-left corner swings toward it)."""
    _assert_re_entered_as_today(lambda leader: _setup(leader=leader, follower=(2.37, 3.64, 0.0)),
                                lambda world: _want(world, 'leader_standby'),
                                (2.62, 3.28, 0.15))


def test_goal_pose_change_replans():
    commander, world = _setup()
    commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 0.0)
    world.goals.set('pick_gear', GoalPose(4.7, 3.0, 0.0))
    commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 0.2)
    cmd = commander.cmds['leader']
    assert cmd.cmd_id == 2 and cmd.poses[-1] == Pose2D(4.7, 3.0, 0.0)


# ---------- departure pivot / fixture goal mark ----------
GEAR_SHELF = Box(5.386, 2.693, 0.4, 0.4)      # layout gear shelf, face at x 5.186


def _fixture_setup(leader, follower=(1.9, 3.8, 0.0)):
    settings = dataclasses.replace(helpers.settings(), fixtures=(GEAR_SHELF,))
    world = WorldModel(pose_timeout=0.5, goals=helpers.goals())
    helpers.place(world, 'leader', *leader, now=0.0)
    helpers.place(world, 'follower', *follower, now=0.0)
    return PathCommander(settings, EPOCH), world


def _assert_no_duplicate_poses(cmd):
    """The poses are a plain path: never two consecutive identical (signalling) poses."""
    assert all(a != b for a, b in zip(cmd.poses, cmd.poses[1:]))


def test_pick_pose_departure_starts_pivot():
    """0.30 m in front of the shelf no forward path leaves; the command starts with an
    in-place turn (pivot_first at poses[0]) and ends with the goal."""
    commander, world = _fixture_setup((4.89, 2.69, 0.0))
    home = GoalPose(2.3, 3.3, math.pi)
    commander.update({'leader': Desired('home', home, True)}, world, NONE_ARRIVED, 0.0)
    cmd = commander.cmds['leader']
    assert cmd.mode == 'FOLLOW'
    assert cmd.pivot_first and (cmd.poses[0].x, cmd.poses[0].y) == (4.89, 2.69)
    assert cmd.poses[0].theta == cmd.pivot_heading != 0.0
    assert cmd.poses[-1] == Pose2D(2.3, 3.3, math.pi)
    _assert_no_duplicate_poses(cmd)


def test_fixture_goal_marked_run_in_exempt():
    """The pick goal faces the shelf: fixture_ahead is set (the rover exempts the final
    run-in from the front stop even before pick_at_goal). Other goals are not."""
    commander, world = _fixture_setup((2.6, 3.3, 0.0))
    pick = GoalPose(4.89, 2.69, 0.0)
    commander.update({'leader': Desired('pick', pick, True)}, world, NONE_ARRIVED, 0.0)
    cmd = commander.cmds['leader']
    assert not cmd.pick_at_goal and cmd.fixture_ahead and not cmd.pivot_first
    assert cmd.poses[-1] == Pose2D(4.89, 2.69, 0.0)
    _assert_no_duplicate_poses(cmd)
    commander.update({'leader': Desired('pick', pick, True, True)}, world, {'leader': True}, 1.0)
    assert commander.cmds['leader'].poses == (Pose2D(4.89, 2.69, 0.0),)
    assert commander.cmds['leader'].fixture_ahead
    commander, world = _fixture_setup((2.6, 3.3, 0.0))
    commander.update({'leader': Desired('x', GoalPose(4.0, 3.3, 0.0), True)}, world,
                     NONE_ARRIVED, 0.0)
    assert not commander.cmds['leader'].fixture_ahead


def test_no_consecutive_identical_poses():
    """Pivot and fixture goals are signalled by message fields, not by repeated poses."""
    for follower_start, goal in (((4.89, 2.69, 0.0), GoalPose(2.3, 3.3, math.pi)),
                                 ((2.6, 3.3, 0.0), GoalPose(4.89, 2.69, 0.0))):
        commander, world = _fixture_setup(follower_start)
        commander.update({'leader': Desired('g', goal, True)}, world, NONE_ARRIVED, 0.0)
        cmd = commander.cmds['leader']
        assert cmd.mode == 'FOLLOW' and cmd.pivot_first != cmd.fixture_ahead
        _assert_no_duplicate_poses(cmd)


def test_flag_resend_keeps_pivot_en_route():
    """The rover judges itself whether it is past the departure point; a re-sent path
    keeps pivot_first/pivot_heading. After arrival the one-pose path has no pivot."""
    commander, world = _fixture_setup((4.89, 2.69, 0.0))
    home = GoalPose(2.3, 3.3, math.pi)
    commander.update({'leader': Desired('home', home, True)}, world, NONE_ARRIVED, 0.0)
    first = commander.cmds['leader']
    assert first.pivot_first
    commander.update({'leader': Desired('home', home, True, True)}, world, NONE_ARRIVED, 1.0)
    resent = commander.cmds['leader']
    assert resent.cmd_id == first.cmd_id + 1 and resent.pick_at_goal
    assert (resent.poses, resent.pivot_first, resent.pivot_heading) == \
        (first.poses, True, first.pivot_heading)
    commander.update({'leader': Desired('home', home, True)}, world, {'leader': True}, 2.0)
    parked = commander.cmds['leader']
    assert parked.poses == (first.poses[-1],) and not parked.pivot_first \
        and parked.pivot_heading == 0.0


def test_flag_resend_keeps_fixture_ahead():
    commander, world = _fixture_setup((2.6, 3.3, 0.0))
    pick = GoalPose(4.89, 2.69, 0.0)
    commander.update({'leader': Desired('pick', pick, True)}, world, NONE_ARRIVED, 0.0)
    first = commander.cmds['leader']
    commander.update({'leader': Desired('pick', pick, True, True)}, world, NONE_ARRIVED, 1.0)
    resent = commander.cmds['leader']
    assert resent.cmd_id == first.cmd_id + 1 and resent.poses == first.poses
    assert first.fixture_ahead and resent.fixture_ahead


# ---------- start-up check: pick goals without a registered fixture ----------
def test_pick_goals_no_fixtures_names_all():
    """The shipped config has ``fixtures: []``: every pick goal is reported (by name)."""
    settings = helpers.settings()
    assert settings.fixtures == ()
    assert pick_goals_without_fixture(settings, helpers.scenario(), helpers.goals()) \
        == ['pick_gear', 'pick_bolt']


def test_pick_goals_skip_fixture_facing():
    settings = dataclasses.replace(helpers.settings(), fixtures=(GEAR_SHELF,))
    assert pick_goals_without_fixture(settings, helpers.scenario(), helpers.goals()) \
        == ['pick_bolt']


# ---------- front-stop zone (the rover's gate refuses to drive with an object in it) ----------
LEADER_ZONE = (0.045, 0.59, 0.19)       # config/mission.yaml leader plan.front_zone


def _with_zone(settings, zone=LEADER_ZONE, runin=0.6):
    robots = {name: dataclasses.replace(cfg, plan=dataclasses.replace(
        cfg.plan, front_zone=zone, front_zone_runin=runin))
        for name, cfg in settings.robots.items()}
    return dataclasses.replace(settings, robots=robots)


def _zone_setup(settings, leader=(2.6, 3.3, 0.0), follower=(1.9, 3.3, 0.0)):
    world = WorldModel(pose_timeout=0.5, goals=helpers.goals())
    helpers.place(world, 'leader', *leader, now=0.0)
    helpers.place(world, 'follower', *follower, now=0.0)
    return PathCommander(settings, EPOCH), world


def _box_appears_in_the_zone_at_the_goal(settings):
    """Leader issued a path to pick_gear (no fixture there), then a small box is mapped
    0.45 m ahead of the goal: inside the zone, clear of the footprint's clearance on every
    pose (the path ends at the goal). Returns (first cmd, cmd after the re-check)."""
    commander, world = _zone_setup(settings)
    want = {'leader': _want(world, 'pick_gear')}
    commander.update(want, world, NONE_ARRIVED, 0.0)
    first = commander.cmds['leader']
    goal = first.poses[-1]
    box = (goal.x + 0.45, goal.y, 0.03)
    params = settings.robots['leader'].plan
    assert planning.min_body_gap(first.poses, [Circle(*box)], params) > params.clearance
    here = first.poses[len(first.poses) // 2]
    helpers.place(world, 'leader', here.x, here.y, here.theta, 0.2)
    world.set_mapped_obstacles([box])
    commander.check_path('leader')
    commander.update(want, world, NONE_ARRIVED, 0.2)
    return first, commander.cmds['leader']


def test_box_in_front_zone_triggers_replan():
    first, after = _box_appears_in_the_zone_at_the_goal(_with_zone(helpers.settings()))
    assert first.mode == 'FOLLOW'
    assert after.cmd_id > first.cmd_id          # re-planned; the goal itself is now refused
    assert after.mode == 'STOP'                 # (the gate would stop 0.5 m short of the box)


def test_without_zone_same_box_keeps_path():
    first, after = _box_appears_in_the_zone_at_the_goal(helpers.settings())
    assert after == first


def test_pick_departure_turns_zone_clear():
    """Pick pose facing the shelf (0.30 m from its face): the shelf is in the zone, so the
    path home starts with a pivot after which no pose has it in the zone; the pick goal
    itself (fixture_ahead) is plannable because its run-in is exempt."""
    settings = _with_zone(dataclasses.replace(helpers.settings(), fixtures=(GEAR_SHELF,)))
    strict = dataclasses.replace(settings.robots['leader'].plan, front_zone_runin=0.0)
    commander, world = _zone_setup(settings, leader=(4.89, 2.69, 0.0), follower=(1.9, 3.8, 0.0))
    home = GoalPose(2.3, 3.3, math.pi)
    commander.update({'leader': Desired('home', home, True)}, world, NONE_ARRIVED, 0.0)
    cmd = commander.cmds['leader']
    assert cmd.mode == 'FOLLOW' and cmd.pivot_first
    assert planning.keeps_clearance(cmd.poses, [GEAR_SHELF], strict)
    commander, world = _zone_setup(settings, follower=(1.9, 3.8, 0.0))
    pick = GoalPose(4.89, 2.69, 0.0)
    commander.update({'leader': Desired('pick', pick, True)}, world, NONE_ARRIVED, 0.0)
    cmd = commander.cmds['leader']
    assert cmd.mode == 'FOLLOW' and cmd.fixture_ahead and cmd.poses[-1] == Pose2D(pick.x, pick.y, pick.theta)
    assert not planning.keeps_clearance(cmd.poses, [GEAR_SHELF], strict)


# ---------- plans computed beside the update (plan_jobs) ----------
class _ManualJob:
    def __init__(self, fn, args):
        self.fn, self.args = fn, args
        self.value, self.finished, self.cancelled = None, False, False

    def done(self):
        return self.finished

    def result(self):
        return self.value

    def cancel(self):
        self.cancelled = not self.finished
        return self.cancelled


class _Manual:
    """Runner whose jobs finish when the test says so."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append(_ManualJob(fn, args))
        return self.jobs[-1]

    def waiting(self):
        return [job for job in self.jobs if not job.finished and not job.cancelled]

    def finish(self):
        for job in self.waiting():
            job.value, job.finished = job.fn(*job.args), True


def _setup_with(runner):
    world = WorldModel(pose_timeout=0.5, goals=helpers.goals())
    helpers.place(world, 'leader', 2.6, 3.3, 0.0, now=0.0)
    helpers.place(world, 'follower', 1.9, 3.3, 0.0, now=0.0)
    return PathCommander(helpers.settings(), EPOCH, runner=runner), world


def test_update_does_not_wait_for_plan():
    runner = _Manual()
    commander, world = _setup_with(runner)
    want = {'leader': _want(world, 'pick_gear')}
    assert commander.update(want, world, NONE_ARRIVED, 0.0) == []
    assert 'leader' not in commander.cmds and len(runner.waiting()) == 1
    commander.update(want, world, NONE_ARRIVED, 0.2)
    assert 'leader' not in commander.cmds and len(runner.jobs) == 1
    runner.finish()
    events = commander.update(want, world, NONE_ARRIVED, 0.4)
    inline, _world = _setup()
    inline.update(want, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['leader'].poses == inline.cmds['leader'].poses
    assert [kind for kind, _detail in events] == ['path_issued']
    assert events[0][1].endswith('planned in 0.4 s)')       # asked at 0.0, there at 0.4


def _tables(commander):
    """Copies of every table (dict / set / list field) of a commander but its job list."""
    return {name: copy.deepcopy(value) for name, value in vars(commander).items()
            if isinstance(value, (dict, set, list)) and name != '_jobs'}


def test_computing_plan_writes_no_state():
    """A plan is computed on another thread, on a shallow copy of the commander that shares
    every table ``_frozen`` does not copy. That is safe only as long as the computation
    writes none of them."""
    commander, world = _setup()
    commander.update(_both(world), world, NONE_ARRIVED, 0.0)
    runner = _Manual()
    commander.runner = runner
    commander.request_replan('leader')
    commander.update(_both(world), world, NONE_ARRIVED, 0.2)
    job = runner.waiting()[0]
    twin = job.fn.__self__
    assert twin is not commander
    before = _tables(commander), _tables(twin)
    runner.finish()
    assert job.value[0] is not None
    assert (_tables(commander), _tables(twin)) == before


def test_stop_during_compute_voids_plan():
    runner = _Manual()
    commander, world = _setup_with(runner)
    commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 0.0)
    stop = {'leader': _want(world, 'pick_gear', follow=False)}
    commander.update(stop, world, NONE_ARRIVED, 0.2)
    assert commander.cmds['leader'].mode == 'STOP'
    assert runner.waiting() == []
    runner.finish()
    commander.update(stop, world, NONE_ARRIVED, 0.4)
    assert commander.cmds['leader'].mode == 'STOP'


def test_plan_for_unwanted_goal_not_issued():
    runner = _Manual()
    commander, world = _setup_with(runner)
    commander.update({'leader': _want(world, 'pick_gear')}, world, NONE_ARRIVED, 0.0)
    first = runner.jobs[0]
    first.value, first.finished = first.fn(*first.args), True
    other = {'leader': _want(world, 'pick_bolt')}
    commander.update(other, world, NONE_ARRIVED, 0.2)
    assert 'leader' not in commander.cmds and len(runner.waiting()) == 1
    runner.finish()
    commander.update(other, world, NONE_ARRIVED, 0.4)
    assert commander.cmds['leader'].goal_key == 'pick_bolt'


def test_follower_plan_asked_after_lead_path():
    """The follower's plan reads the lead's command: computed side by side it would be
    planned against a path the lead is about to drop."""
    runner = _Manual()
    commander, world = _setup_with(runner)
    want = _both(world)
    commander.update(want, world, NONE_ARRIVED, 0.0)
    assert len(runner.jobs) == 1 and runner.jobs[0].args[0] == 'leader'
    runner.finish()
    commander.update(want, world, NONE_ARRIVED, 0.2)
    assert commander.cmds['leader'].mode == 'FOLLOW' and 'follower' not in commander.cmds
    assert len(runner.jobs) == 2 and runner.jobs[1].args[0] == 'follower'
    runner.finish()
    commander.update(want, world, NONE_ARRIVED, 0.4)
    inline, _world = _setup()
    inline.update(want, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['follower'].lead_robot == inline.cmds['follower'].lead_robot
    assert commander.cmds['follower'].poses == inline.cmds['follower'].poses


def test_box_appearing_mid_compute_avoided():
    runner = _Manual()
    commander, world = _setup_with(runner)
    want = {'leader': _want(world, 'pick_gear')}
    commander.update(want, world, NONE_ARRIVED, 0.0)
    job = runner.jobs[0]
    plan, _reason = job.fn(*job.args)
    mid = plan.poses[len(plan.poses) // 2]
    world.set_manual_obstacles([(mid.x, mid.y, 0.05)])
    job.value, job.finished = (plan, _reason), True
    commander.update(want, world, NONE_ARRIVED, 0.2)
    commander.update(want, world, NONE_ARRIVED, 0.4)
    assert len(runner.waiting()) == 1
    runner.finish()
    commander.update(want, world, NONE_ARRIVED, 0.6)
    cfg = helpers.settings().robots['leader']
    assert min(math.hypot(p.x - mid.x, p.y - mid.y) for p in commander.cmds['leader'].poses) \
        > 0.05 + cfg.plan.clearance


def test_worker_plan_equals_inline_plan():
    import time
    runner = plan_jobs.worker()
    commander, world = _setup_with(runner)
    want = _both(world)
    deadline = time.monotonic() + 20.0
    step = 0
    while len(commander.cmds) < 2 and time.monotonic() < deadline:
        commander.update(want, world, NONE_ARRIVED, 0.2 * step)
        helpers.place(world, 'leader', 2.6, 3.3, 0.0, now=0.2 * step)
        helpers.place(world, 'follower', 1.9, 3.3, 0.0, now=0.2 * step)
        step += 1
        time.sleep(0.01)
    runner.shutdown()
    inline, _world = _setup()
    inline.update(want, world, NONE_ARRIVED, 0.0)
    assert {r: c.poses for r, c in commander.cmds.items()} \
        == {r: c.poses for r, c in inline.cmds.items()}


# ---------- turn-in point (platoon.dock_turn) ----------
def _north_along(x, y_from, y_to):
    count = int(round((y_to - y_from) / 0.05))
    return [Pose2D(x, y_from + 0.05 * i, math.pi / 2) for i in range(count + 1)]


def test_turn_in_point_at_lead_path_crossing():
    slot = Pose2D(4.0, 2.0, 0.0)                 # entered heading east, along y = 2.0
    lead = _north_along(3.4, 1.0, 3.0)           # crosses that line 0.6 m behind the slot
    point = turn_in_point(lead, slot, 0.4)
    assert point is not None and abs(point.x - 3.4) < 1e-9 and abs(point.y - 2.0) < 0.026
    assert point in lead                         # a pose of the lead path: it can be ridden to
    assert turn_in_point(lead, slot, 0.7) is None
    assert turn_in_point(_north_along(4.5, 1.0, 3.0), slot, 0.4) is None
    assert turn_in_point(_north_along(3.4, 2.2, 3.0), slot, 0.4) is None


def test_turn_in_point_is_the_last_crossing():
    slot = Pose2D(4.0, 2.0, 0.0)
    back_south = [Pose2D(3.0, 3.0 - 0.05 * i, -math.pi / 2) for i in range(41)]
    lead = _north_along(3.4, 1.0, 3.0) + back_south       # crosses at x = 3.4, then at x = 3.0
    assert abs(turn_in_point(lead, slot, 0.4).x - 3.0) < 1e-9


def test_turn_in_follower_planned_against_at_goal():
    """The lead's plan takes a follower on its way at its goal. The end of a turn-in path
    lies on the lead's own path: taken as the parking spot, the lead's path re-check would
    find the follower parked on the rest of its path."""
    commander, world = _setup()
    slot = world.goals.get('load_gear')
    on_lead_path = (Pose2D(3.0, 3.0, 0.0), Pose2D(3.5, 3.0, 0.0))
    commander.cmds['follower'] = PathCmd(
        robot='follower', epoch=EPOCH, cmd_id=4, mode='FOLLOW', poses=on_lead_path,
        lead_robot='leader', gap=0.6, goal_key='load_gear', goal=slot, via=True)
    assert commander._parking_spot('follower', world, NONE_ARRIVED) == (slot.x, slot.y, slot.theta)
    commander.cmds['follower'] = dataclasses.replace(commander.cmds['follower'], via=False)
    assert commander._parking_spot('follower', world, NONE_ARRIVED) == (3.5, 3.0, 0.0)


def _beside_the_lead(hold_ride=True):
    """Lead on its way to a goal straight ahead; the follower's goal right beside it (0.32 m
    to its right, same heading), inside the lead's front-stop zone along its run-in."""
    commander, world = _setup(leader=(2.0, 3.3, 0.0), follower=(1.3, 3.3, 0.0))
    settings = commander.settings
    leader = settings.robots['leader']
    site_leader = dataclasses.replace(leader, plan=dataclasses.replace(  # site profile
        leader.plan, footprint=(0.09, 0.09, 0.09), clearance=0.03, front_zone=(0.045, 0.59, 0.19)))
    commander.settings = dataclasses.replace(
        settings, robots={**settings.robots, 'leader': site_leader},
        platoon=dataclasses.replace(settings.platoon, hold_ride=hold_ride))
    want = {'leader': Desired('pick', GoalPose(4.0, 3.3, 0.0), True),
            'follower': Desired('load', GoalPose(4.0, 2.98, 0.0), True)}
    return commander, world, want


def test_blocking_follower_rides_to_hold_point():
    """Instead of waiting where it stands until the lead has parked (its goal, parked
    there, would block the lead's run-in), the follower rides behind the lead to a hold
    point on the lead path, at least ``gap`` short of the lead's goal, and stops there."""
    commander, world, want = _beside_the_lead()
    events = commander.update(want, world, NONE_ARRIVED, 0.0)
    lead, follower = commander.cmds['leader'], commander.cmds['follower']
    assert follower.mode == 'FOLLOW' and follower.via and follower.hold
    assert follower.lead_robot == 'leader' and not follower.fixture_ahead
    hold = follower.poses[-1]
    lead_goal = lead.poses[-1]
    assert min(math.hypot(hold.x - p.x, hold.y - p.y) for p in lead.poses) < 1e-6
    assert math.hypot(hold.x - lead_goal.x, hold.y - lead_goal.y) \
        >= commander.settings.platoon.gap - 1e-6
    assert any('to its hold point' in detail for kind, detail in events if kind == 'path_issued')
    # the lead plans around it where it stands (behind it), not at the hold point on its path
    assert commander._parking_spot('follower', world, NONE_ARRIVED) == world.robots['follower'].pose


def test_hold_point_follower_docks_after_lead():
    commander, world, want = _beside_the_lead()
    commander.update(want, world, NONE_ARRIVED, 0.0)
    hold, lead_goal = commander.cmds['follower'].poses[-1], commander.cmds['leader'].poses[-1]
    helpers.place(world, 'leader', lead_goal.x - 0.3, lead_goal.y, 0.0, 1.0)
    helpers.place(world, 'follower', hold.x, hold.y, hold.theta, 1.0)
    commander.request_replan('follower')            # as at the end of a via path
    events = commander.update(want, world, NONE_ARRIVED, 1.0)
    assert commander.cmds['follower'].mode == 'STOP'
    assert ('plan_waiting', 'follower: YIELD_TO_LEAD') in events
    helpers.place(world, 'leader', lead_goal.x, lead_goal.y, 0.0, 9.0)
    helpers.place(world, 'follower', hold.x, hold.y, hold.theta, 9.0)
    commander.update(want, world, {'leader': True, 'follower': False}, 9.0)
    follower = commander.cmds['follower']
    assert follower.mode == 'FOLLOW' and not follower.via and follower.lead_robot == ''
    assert follower.poses[-1] == Pose2D(4.0, 2.98, 0.0)


def test_without_hold_ride_follower_waits():
    commander, world, want = _beside_the_lead(hold_ride=False)
    events = commander.update(want, world, NONE_ARRIVED, 0.0)
    assert commander.cmds['leader'].mode == 'FOLLOW'
    assert commander.cmds['follower'].mode == 'STOP'
    assert ('plan_waiting', 'follower: YIELD_TO_LEAD') in events
