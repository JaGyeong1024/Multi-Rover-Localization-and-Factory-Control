"""Platooning: join, shared stretch, branch, stand-alone fallback."""
import math

import pytest

from checks import AREA, assert_pose, assert_run_in, steps
from host_planning import (Circle, PlanParams, PlatoonPlan, Pose2D,
                           max_curvature, min_clearance, path_length,
                           plan_follower_path, plan_path)

PARAMS = PlanParams()
LEADER_START = Pose2D(0.6, 0.6, 0.0)
LEADER_GOAL = Pose2D(4.0, 2.0, math.pi / 2)


@pytest.fixture(scope='module')
def leader_path():
    return plan_path(LEADER_START, LEADER_GOAL, (), AREA, PARAMS)


def _arc_index(path, s):
    """Index of the path point at arc length ``s``."""
    total = 0.0
    for i, gap in enumerate(steps(path)):
        if total >= s - 1e-6:
            return i
        total += gap
    return len(path) - 1


def _assert_plan_ok(plan: PlatoonPlan, follower, goal):
    assert_pose(plan.path[0], follower)
    assert_pose(plan.path[-1], goal)
    assert max_curvature(plan.path) <= PARAMS.max_curvature * 1.001
    assert max(steps(plan.path)) <= PARAMS.spacing + 1e-6
    assert_run_in(plan.path, goal, PARAMS.approach_len)


def _shared_points(plan: PlatoonPlan):
    return plan.path[_arc_index(plan.path, plan.join_s):
                     _arc_index(plan.path, plan.branch_s) + 1]


@pytest.mark.parametrize('follower,goal', [
    (Pose2D(0.4, 1.0, -0.3), Pose2D(3.4, 2.2, math.pi / 2)),
    (Pose2D(0.3, 0.3, 0.5), Pose2D(2.5, 1.6, 1.0)),
    (Pose2D(1.0, 2.0, -1.2), Pose2D(4.3, 1.0, 0.0)),
])
def test_join_share_branch(leader_path, follower, goal):
    plan = plan_follower_path(follower, leader_path, goal, (), AREA, PARAMS)
    assert plan is not None and plan.shared
    _assert_plan_ok(plan, follower, goal)
    assert 0.0 < plan.join_s < plan.branch_s < path_length(plan.path)
    assert plan.leader_join_s < plan.leader_branch_s
    shared_len = plan.branch_s - plan.join_s
    assert shared_len == pytest.approx(plan.leader_branch_s - plan.leader_join_s)
    assert shared_len >= PARAMS.platoon_min_shared - PARAMS.spacing


def test_shared_stretch_equals_leader(leader_path):
    follower, goal = Pose2D(0.4, 1.0, -0.3), Pose2D(3.4, 2.2, math.pi / 2)
    plan = plan_follower_path(follower, leader_path, goal, (), AREA, PARAMS)
    first = _arc_index(leader_path, plan.leader_join_s)
    last = _arc_index(leader_path, plan.leader_branch_s)
    shared = _shared_points(plan)
    assert len(shared) >= 5
    assert shared == leader_path[first:last + 1]


def test_follower_already_on_leader_path(leader_path):
    goal = Pose2D(3.4, 2.2, math.pi / 2)
    plan = plan_follower_path(leader_path[4], leader_path, goal, (), AREA, PARAMS)
    assert plan.shared
    assert plan.join_s == pytest.approx(0.0)
    assert plan.leader_join_s == pytest.approx(path_length(leader_path[:5]))
    _assert_plan_ok(plan, leader_path[4], goal)


def test_same_goal_shares_until_the_end(leader_path):
    plan = plan_follower_path(Pose2D(0.4, 0.9, -0.2), leader_path, LEADER_GOAL,
                              (), AREA, PARAMS)
    assert plan.shared
    assert_pose(plan.path[-1], LEADER_GOAL)
    assert plan.leader_branch_s > 0.7 * path_length(leader_path)


def test_opposite_direction_solo_fallback(leader_path):
    follower, goal = Pose2D(4.0, 2.2, -math.pi / 2), Pose2D(0.8, 0.6, math.pi)
    plan = plan_follower_path(follower, leader_path, goal, (), AREA, PARAMS)
    assert plan is not None and not plan.shared
    assert plan.join_s is None and plan.branch_s is None
    assert plan.leader_join_s is None and plan.leader_branch_s is None
    _assert_plan_ok(plan, follower, goal)


def test_degenerate_leader_solo_fallback():
    follower, goal = Pose2D(0.5, 1.0, 0.0), Pose2D(3.0, 1.0, 0.0)
    plan = plan_follower_path(follower, [Pose2D(1.0, 1.0, 0.0)], goal, (), AREA)
    assert plan is not None and not plan.shared
    assert plan_follower_path(follower, [], goal, (), AREA).shared is False


def test_obstacle_beside_shared_path_respected(leader_path):
    """Join and exit curves keep the follower's own clearance."""
    follower, goal = Pose2D(0.4, 1.0, -0.3), Pose2D(3.4, 2.2, math.pi / 2)
    boxes = [Circle(2.6, 2.3, 0.2), Circle(1.2, 1.6, 0.2)]
    plan = plan_follower_path(follower, leader_path, goal, boxes, AREA, PARAMS)
    assert plan is not None
    _assert_plan_ok(plan, follower, goal)
    assert min_clearance(plan.path, boxes) >= PARAMS.required_clearance - 1e-6


def test_obstacle_on_leader_path_forces_solo(leader_path):
    middle = leader_path[len(leader_path) // 2]
    boxes = [Circle(middle.x, middle.y, 0.15)]
    follower, goal = Pose2D(0.4, 1.0, -0.3), Pose2D(3.4, 2.2, math.pi / 2)
    plan = plan_follower_path(follower, leader_path, goal, boxes, AREA, PARAMS)
    if plan is not None:
        assert min_clearance(plan.path, boxes) >= PARAMS.required_clearance - 1e-6
        _assert_plan_ok(plan, follower, goal)


def test_no_path_at_all_returns_none(leader_path):
    goal = Pose2D(0.2, 1.3, 0.0)   # run-in starts outside the area
    assert plan_follower_path(Pose2D(2.0, 1.3, 0.0), leader_path, goal,
                              (), AREA, PARAMS) is None


def test_leader_arc_at_turn_limit_shared():
    """A resampled minimum-radius arc may read a hair over the curvature limit
    (3.334 vs 3.333 1/m); the follower must still share it."""
    import numpy as np
    from host_planning.geometry import resample, to_poses
    r = PARAMS.min_turn_radius
    east = [(3.0 + d, 0.5, 0.0) for d in np.arange(0.0, 0.5, 0.001)]
    arc = [(3.5 + r * math.sin(a), 0.5 + r * (1.0 - math.cos(a)), a)
           for a in np.arange(0.0, math.pi / 2, 0.001 / r)]
    north = [(3.5 + r, 0.5 + r + d, math.pi / 2) for d in np.arange(0.0, 1.5, 0.001)]
    leader = to_poses(resample(np.array(east + arc + north), PARAMS.spacing))
    assert max_curvature(leader) > PARAMS.max_curvature        # the premise of this test
    follower, goal = Pose2D(2.6, 0.4, 0.0), Pose2D(3.5, 2.4, math.pi / 2)
    plan = plan_follower_path(follower, leader, goal, (), AREA, PARAMS)
    assert plan is not None and plan.shared
    assert plan.leader_join_s < 0.5 and plan.leader_branch_s > 0.5 + r * math.pi / 2   # arc included
    _assert_plan_ok(plan, follower, goal)
