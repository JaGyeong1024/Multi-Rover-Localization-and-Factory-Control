"""Departure pivot: swept-footprint clearance and pivot-first paths."""
import math

import pytest

from host_planning import Bounds, Box, PlanParams, Pose2D, plan_departure, plan_path
from host_planning.pivot import pivot_clear

BOUNDS = Bounds(1.44, 5.83, 2.17, 4.46)
LEADER = PlanParams(footprint=(0.13, 0.23, 0.18), clearance=0.05,
                    pivot_step=math.radians(15))
SHELF = Box(5.386, 2.693, 0.4, 0.4)       # gear shelf; its face is at x 5.186
PICK = Pose2D(4.89, 2.69, 0.0)            # 0.30 m in front of the face, facing it


def test_pivot_clearance_rejects_blocked_sweep():
    """A box behind-right is clear at the start but blocks a left turn, not a right one."""
    box = Box(PICK.x - 0.05, PICK.y - 0.40, 0.30, 0.20)   # top edge 0.30 below the centre
    params = LEADER
    assert pivot_clear(PICK, 0.0, [box], BOUNDS, params)          # standing there is fine
    assert not pivot_clear(PICK, math.radians(90), [box], BOUNDS, params)
    assert pivot_clear(PICK, math.radians(-15), [box], BOUNDS, params)


def test_full_turn_at_pick_hits_shelf():
    """Turning 165 deg in front of the shelf swings the rear corners within 0.29 m ahead."""
    assert not pivot_clear(PICK, math.radians(165), [SHELF], BOUNDS, LEADER)
    assert pivot_clear(PICK, math.radians(90), [SHELF], BOUNDS, LEADER)


def test_pick_departure_starts_with_pivot():
    """Forward-only cannot leave 0.30 m in front of the shelf; the path turns in place first."""
    home = Pose2D(1.82, 3.29, math.pi)
    assert plan_path(PICK, home, [SHELF], BOUNDS, LEADER) is None
    path, heading = plan_departure(PICK, home, [SHELF], BOUNDS, LEADER)
    assert path is not None
    assert heading is not None and pivot_clear(PICK, heading - PICK.theta, [SHELF], BOUNDS,
                                               LEADER)
    assert (path[0].x, path[0].y) == (PICK.x, PICK.y) and path[0].theta == heading
    assert path[-1] == home
    assert all(math.hypot(a.x - b.x, a.y - b.y) > 1e-3 for a, b in zip(path, path[1:]))


def test_facing_away_turns_to_goal_heading():
    """Turn onto the goal heading (no multiple of 15 deg) and drive straight in."""
    home = Pose2D(1.82, 3.29, 3.05)
    out = Pose2D(home.x - 0.5 * math.cos(home.theta) + 0.02 * math.sin(home.theta),
                 home.y - 0.5 * math.sin(home.theta) - 0.02 * math.cos(home.theta),
                 home.theta - math.pi)
    params = PlanParams(footprint=LEADER.footprint, clearance=0.05,
                        pivot_step=math.radians(15), pivot_cost=0.5, direct_skew_tol=0.08)
    path, heading = plan_departure(out, home, [], BOUNDS, params)
    assert heading == pytest.approx(home.theta)
    assert path[-1] == home
    length = sum(math.hypot(a.x - b.x, a.y - b.y) for a, b in zip(path, path[1:]))
    assert length == pytest.approx(math.hypot(0.5, 0.02), abs=1e-3)


def test_no_pivot_if_disabled_or_cheap():
    home = Pose2D(1.82, 3.29, math.pi)
    off = PlanParams(footprint=LEADER.footprint, clearance=0.05)          # pivot_step 0
    assert plan_departure(PICK, home, [SHELF], BOUNDS, off) == (None, None)
    free = Pose2D(3.0, 3.3, 0.0)
    ahead = Pose2D(4.0, 3.3, 0.0)
    path, heading = plan_departure(free, ahead, [], BOUNDS, LEADER)
    assert path is not None and heading is None


def test_rotation_centre_ahead_widens_sweep():
    # Centre ahead of the tracked point (pivot_offset < 0): the farthest body point is
    # |offset| further out, not closer.
    from host_planning.pivot import _reach
    ahead = PlanParams(robot_radius=0.2, pivot_offset=-0.063, footprint=None)
    behind = PlanParams(robot_radius=0.2, pivot_offset=0.063, footprint=None)
    assert _reach(ahead) == pytest.approx(0.263)
    assert _reach(ahead) == _reach(behind)
