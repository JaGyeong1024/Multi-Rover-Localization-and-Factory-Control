"""Platoon rules (path_commander): the follower's new leg waits while the lead turns in place, and the lead plans wide enough for the follower's body."""
import math

from host_planning.types import Pose2D

import helpers
from host_mission.path_commander import FOLLOW, LEAD_TURNING, PathCmd, PathCommander, Desired

HOME = helpers.goals()


def _commander_with_turning_lead(lead_yaw, cmd_goal_key='load_gear'):
    eng = helpers.engine()
    commander, world = eng.commander, eng.world
    helpers.place(world, 'leader', 4.45, 1.13, lead_yaw, 0.0)
    helpers.place(world, 'follower', 4.42, 0.80, 0.0, 0.0)
    commander.cmds['leader'] = PathCmd(
        robot='leader', epoch=helpers.EPOCH, cmd_id=5, mode=FOLLOW,
        poses=(Pose2D(4.45, 1.13, 2.62), Pose2D(4.0, 1.4, 2.8)), pivot_first=True,
        pivot_heading=2.62, goal_key='leader_standby')
    want = Desired('follower_unload', HOME.get('follower_unload'), follow=True)
    commander.cmds['follower'] = PathCmd(robot='follower', epoch=helpers.EPOCH, cmd_id=4,
                                         mode=FOLLOW, goal_key=cmd_goal_key,
                                         goal=HOME.get(cmd_goal_key))
    commander._desired = {'follower': want}
    return commander, world, want


def test_new_leg_waits_for_lead_turn():
    commander, world, want = _commander_with_turning_lead(0.0)
    plan, reason = commander._compute('follower', want, world, {}, 0.0)
    assert plan is None and reason == LEAD_TURNING
    helpers.place(world, 'leader', 4.45, 1.13, 2.62, 0.1)
    assert not commander._waits_for_lead_turn('follower', want, world)


def test_follower_on_leg_never_held():
    commander, world, want = _commander_with_turning_lead(0.0, cmd_goal_key='follower_unload')
    assert not commander._waits_for_lead_turn('follower', want, world)


def test_platoon_margin_footprint_minus_clearance():
    commander = PathCommander(helpers.settings(), helpers.EPOCH)
    lead, follow = (commander.settings.robots[r].plan for r in ('leader', 'follower'))
    expected = max(f + follow.clearance - (l + lead.clearance)
                   for f, l in zip(follow.footprint, lead.footprint))
    commander._desired = {'follower': Desired('x', None, follow=True)}
    assert math.isclose(commander._platoon_margin('leader'), max(0.0, expected))
    assert commander._platoon_margin('follower') == 0.0
    commander._desired = {'follower': Desired('x', None, follow=False)}
    assert commander._platoon_margin('leader') == 0.0


def test_lead_turn_ends_by_report_not_pose():
    # A pose away from the first path pose says nothing: an unknown rotation centre moves the
    # tracked point ~0.12 m, and a camera position moved by the arm steps the pose 0.14 m.
    # The rover tells whether it still pivots or follows its path.
    commander, world, want = _commander_with_turning_lead(1.0)
    lead = commander.cmds['leader']
    helpers.place(world, 'leader', 4.45 + 0.08, 1.13 + 0.09, 1.0, 0.1)
    assert commander._waits_for_lead_turn('follower', want, world)
    world.update_nav('leader', 'TRACKING', 'PIVOT', lead.epoch, lead.cmd_id, now=0.2)
    helpers.place(world, 'leader', 4.45 - 0.30, 1.13 + 0.05, 1.0, 0.2)   # 0.3 m step, pivoting
    assert commander._waits_for_lead_turn('follower', want, world)
    world.update_nav('leader', 'TRACKING', '', lead.epoch, lead.cmd_id, now=0.4)
    assert not commander._waits_for_lead_turn('follower', want, world)

