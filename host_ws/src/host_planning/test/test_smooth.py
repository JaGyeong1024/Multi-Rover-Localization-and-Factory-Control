"""Obstacle-free smooth paths: headings, curvature, spacing, bounds, run-in."""
import math

import numpy as np
import pytest

from checks import AREA, assert_path_ok, assert_pose, steps
from host_planning import (Bounds, PlanParams, Pose2D, max_curvature,
                           min_clearance, path_length, plan_path)
from host_planning import dubins
from host_planning.geometry import curvatures, to_array, wrap_angle

PARAMS = PlanParams()

CASES = {
    'straight': (Pose2D(0.5, 1.0, 0.0), Pose2D(3.5, 1.0, 0.0)),
    'lane_change': (Pose2D(0.5, 0.8, 0.0), Pose2D(3.5, 1.8, 0.0)),
    'left_turn': (Pose2D(0.6, 0.6, 0.0), Pose2D(4.0, 2.0, math.pi / 2)),
    'right_turn': (Pose2D(0.6, 2.0, 0.0), Pose2D(4.0, 0.7, -math.pi / 2)),
    'goal_behind': (Pose2D(2.4, 1.3, 0.0), Pose2D(1.2, 1.3, 0.0)),
    'u_turn': (Pose2D(1.0, 0.8, 0.0), Pose2D(1.0, 1.9, math.pi)),
    'opposite_heading': (Pose2D(1.0, 1.3, 0.0), Pose2D(3.5, 1.3, math.pi)),
    'close_misaligned': (Pose2D(2.0, 1.2, 0.0), Pose2D(2.3, 1.4, math.pi / 2)),
    'same_spot_turned': (Pose2D(2.4, 1.3, 0.0), Pose2D(2.4, 1.3, math.pi)),
    'wrap_boundary': (Pose2D(3.5, 1.0, math.pi), Pose2D(1.0, 1.6, -3.0)),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_contract(name):
    start, goal = CASES[name]
    path = plan_path(start, goal, bounds=AREA, params=PARAMS)
    assert path is not None
    assert_path_ok(path, start, goal, PARAMS)


def test_straight_is_straight_and_shortest():
    start, goal = CASES['straight']
    path = plan_path(start, goal, bounds=AREA)
    assert path_length(path) == pytest.approx(3.0, abs=1e-6)
    assert max_curvature(path) < 1e-6


def test_curvature_is_nearly_continuous():
    """Zero curvature at both ends and no large jump between samples."""
    for name in ('lane_change', 'left_turn', 'goal_behind', 'u_turn'):
        path = plan_path(*CASES[name], bounds=AREA, params=PARAMS)
        kappa = curvatures(to_array(path))
        assert abs(kappa[0]) < 0.5 * PARAMS.max_curvature
        assert abs(kappa[-1]) < 1e-6
        ramp_rate = PARAMS.max_curvature / PARAMS.curv_ramp_len
        assert np.abs(np.diff(kappa)).max() <= 1.5 * ramp_rate * PARAMS.spacing


def test_no_reverse_motion():
    """Each step moves along the local heading (forward only)."""
    for start, goal in CASES.values():
        path = plan_path(start, goal, bounds=AREA)
        for a, b in zip(path, path[1:]):
            forward = (b.x - a.x) * math.cos(a.theta) + (b.y - a.y) * math.sin(a.theta)
            assert forward > 0.0


def test_start_on_run_in_drives_straight():
    goal = Pose2D(3.0, 1.0, 0.0)
    path = plan_path(Pose2D(2.8, 1.0, 0.0), goal, bounds=AREA)
    assert path_length(path) == pytest.approx(0.2, abs=1e-6)
    assert_pose(path[-1], goal)


def test_start_behind_goal_drives_straight_in():
    """A rover turned onto the goal heading 0.5 m in front of its goal, 2 cm beside the
    line (pose noise): no connector to the run-in's entry fits in 0.1 m at R 0.30, so
    without the skew tolerance there is no path that is not a loop."""
    goal = Pose2D(3.0, 1.0, 0.0)
    start = Pose2D(2.5, 1.02, 0.0)
    loop = plan_path(start, goal, bounds=AREA)
    assert loop is None or path_length(loop) > 1.0
    direct = PlanParams(direct_skew_tol=0.08)
    path = plan_path(start, goal, bounds=AREA, params=direct)
    assert path_length(path) == pytest.approx(math.hypot(0.5, 0.02), abs=1e-6)
    assert_pose(path[-1], goal)
    wide = plan_path(Pose2D(2.5, 1.06, 0.0), goal, bounds=AREA, params=direct)   # 0.12 rad
    assert wide is None or path_length(wide) > 1.0
    past = plan_path(Pose2D(3.2, 1.0, 0.0), goal, bounds=AREA, params=direct)    # beyond it
    assert past is None or path_length(past) > 1.0


def test_start_equals_goal():
    pose = Pose2D(2.0, 1.0, 0.3)
    path = plan_path(pose, pose, bounds=AREA)
    assert len(path) == 1
    assert_pose(path[0], pose)


def test_spacing_parameter_is_respected():
    params = PlanParams(spacing=0.02)
    path = plan_path(*CASES['left_turn'], bounds=AREA, params=params)
    assert max(steps(path)) <= 0.02 + 1e-6


def test_larger_turn_radius_lowers_curvature():
    params = PlanParams(min_turn_radius=0.6)
    start, goal = CASES['left_turn']
    path = plan_path(start, goal, bounds=AREA, params=params)
    assert_path_ok(path, start, goal, params)


def test_impossible_in_bounds_returns_none():
    """Run-in would start outside the area -> no path."""
    goal = Pose2D(0.2, 1.3, 0.0)   # needs to arrive from x = -0.2
    assert plan_path(Pose2D(2.0, 1.3, 0.0), goal, bounds=AREA) is None


def test_too_narrow_to_turn_returns_none():
    corridor = Bounds(0.0, 4.8, 1.0, 1.4)   # 0.4 m wide, radius 0.3
    assert plan_path(Pose2D(3.0, 1.2, 0.0), Pose2D(1.0, 1.2, math.pi),
                     bounds=corridor) is None


def test_min_approach_len_shortens_run_in():
    start, goal = Pose2D(2.0, 1.3, math.pi), Pose2D(0.65, 1.3, 0.0)
    strict = plan_path(start, goal, bounds=AREA)
    relaxed = plan_path(start, goal, bounds=AREA,
                        params=PlanParams(min_approach_len=0.2))
    assert strict is None and relaxed is not None
    assert_pose(relaxed[-1], goal)


def test_no_bounds_and_no_obstacles():
    start, goal = Pose2D(0.0, 0.0, 0.0), Pose2D(-2.0, 0.5, 1.0)
    path = plan_path(start, goal)
    assert_path_ok(path, start, goal, PARAMS, bounds=None)
    assert min_clearance(path, ()) == float('inf')


def test_dubins_words_reach_goal_and_relax():
    """Closed-form words end on the goal; relaxation keeps the end pose."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        sx, sy, gx, gy = rng.uniform(0.0, 3.0, 4)
        start = Pose2D(sx, sy, rng.uniform(-math.pi, math.pi))
        goal = Pose2D(gx, gy, rng.uniform(-math.pi, math.pi))
        words = dubins.dubins_words(start, goal, 0.3)
        assert words
        for _, segments in words:
            x, y, theta = dubins.end_pose(start, segments, 1.0 / 0.3)
            assert math.hypot(x - goal.x, y - goal.y) < 1e-6
            assert abs(wrap_angle(theta - goal.theta)) < 1e-6
        relaxed = dubins.relax(start, goal, words[0][1], 1.0 / 0.3, 0.1, 0.01)
        if relaxed is not None:
            assert np.abs(curvatures(relaxed)).max() <= 1.0 / 0.3 + 1e-6
            assert math.hypot(relaxed[-1, 0] - goal.x, relaxed[-1, 1] - goal.y) < 1e-9
