"""Shared assertions for the planner tests."""
import math

from host_planning import (Bounds, PlanParams, Pose2D, max_curvature,
                           min_clearance)

AREA = Bounds(0.0, 4.8, 0.0, 2.7)
TOL = 1e-6


def angle_diff(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def steps(path):
    return [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:])]


def assert_pose(actual: Pose2D, expected: Pose2D):
    assert math.hypot(actual.x - expected.x, actual.y - expected.y) < TOL
    assert angle_diff(actual.theta, expected.theta) < TOL


def assert_run_in(path, goal: Pose2D, length: float):
    """The last ``length`` metres (by arc length) lie on the goal-heading line."""
    remaining = 0.0
    for i in range(len(path) - 1, -1, -1):
        pose = path[i]
        if i < len(path) - 1:
            remaining += math.hypot(path[i + 1].x - pose.x, path[i + 1].y - pose.y)
        if remaining > length - 1e-3:
            break
        dx, dy = goal.x - pose.x, goal.y - pose.y
        lateral = -dx * math.sin(goal.theta) + dy * math.cos(goal.theta)
        assert abs(lateral) < 1e-6
        assert angle_diff(pose.theta, goal.theta) < 1e-6
    assert remaining > length - 1e-3


def assert_path_ok(path, start, goal, params: PlanParams, bounds=AREA, obstacles=()):
    """All contract properties of a planned path."""
    assert_pose(path[0], start)
    assert_pose(path[-1], goal)
    assert max_curvature(path) <= params.max_curvature * 1.001
    gaps = steps(path)
    assert max(gaps) <= params.spacing + TOL
    assert min(gaps) >= 0.45 * params.spacing
    if bounds is not None:
        assert all(bounds.xmin - TOL <= p.x <= bounds.xmax + TOL for p in path)
        assert all(bounds.ymin - TOL <= p.y <= bounds.ymax + TOL for p in path)
    assert_run_in(path, goal, params.approach_len)
    assert all(-math.pi < p.theta <= math.pi for p in path)
    if obstacles:
        assert min_clearance(path, obstacles) >= params.required_clearance - TOL
