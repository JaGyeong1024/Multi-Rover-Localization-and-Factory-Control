"""Exact clearance (footprint rectangle vs. oriented boxes) and drop-point goal regions."""
import math

import pytest

from host_planning import Box, Bounds, Circle, PlanParams, Pose2D, min_body_gap, plan_path
from host_planning.region import DropRegion, feasible_goals

# Shelf 0.4 x 0.4 m, the leader's pick goal 0.30 m in front of its face.
BOUNDS = Bounds(1.44, 5.83, 2.17, 4.46)
SHELF = Box(5.386, 2.693, 0.40, 0.40)
PICK = Pose2D(4.89, 2.69, 0.0)
LEADER = PlanParams(footprint=(0.13, 0.23, 0.18), clearance=0.05)
FOLLOWER = PlanParams(footprint=(0.15, 0.15, 0.155), clearance=0.05)


def test_pick_goal_plannable_with_exact_box():
    """Circumscribed circle: the shelf corner bulge eats the 0.17 m body gap -> no path.
    Exact rectangle: the path keeps the clearance to the real faces all the way in."""
    start = Pose2D(1.82, 3.29, 0.0)
    covering = Circle(SHELF.x, SHELF.y, math.hypot(0.20, 0.20))
    assert plan_path(start, PICK, [covering], BOUNDS, PlanParams()) is None
    path = plan_path(start, PICK, [SHELF], BOUNDS, LEADER)
    assert path is not None and path[-1] == PICK
    assert min_body_gap(path, [SHELF], LEADER) >= LEADER.clearance - 1e-9
    gap_at_goal = SHELF.x - SHELF.width / 2 - (PICK.x + LEADER.footprint[0])
    assert min_body_gap([PICK], [SHELF], LEADER) == pytest.approx(gap_at_goal)


def test_load_region_follower_in_bounds():
    """A drop point just outside the bounds has no fixed load pose, but the region does."""
    drop = (PICK.x, PICK.y - 0.55)
    region = DropRegion(drop, PICK.theta, (0.0, 0.0, 0.20, 0.20), 0.03)
    leader_body = Box(PICK.x - 0.05, PICK.y, 0.36, 0.36)
    goals = feasible_goals(region, [SHELF, leader_body], BOUNDS, FOLLOWER)
    assert drop[1] < BOUNDS.ymin and goals
    best = goals[0]
    assert best.theta == PICK.theta                          # parallel to the leader first
    assert BOUNDS.ymin <= best.y <= drop[1] + 0.07 + 1e-9
    assert region.holds(best, 0.0)
    assert min_body_gap([best], [SHELF, leader_body], FOLLOWER) >= FOLLOWER.clearance
    far = DropRegion((PICK.x, BOUNDS.ymin - 0.20), PICK.theta, (0.0, 0.0, 0.20, 0.20), 0.03)
    assert feasible_goals(far, [SHELF, leader_body], BOUNDS, FOLLOWER) == []   # NO_PATH
