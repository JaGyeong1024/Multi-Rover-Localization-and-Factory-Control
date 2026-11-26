"""Front-stop zone: no planned pose has an object where the rover's own gate stops."""
import dataclasses
import math

import numpy as np

from host_planning import (Bounds, Box, PlanParams, Pose2D, plan_departure,
                           plan_follower_path, plan_path)
from host_planning.api import keeps_clearance
from host_planning.geometry import obstacle_array, runin_exempt, to_array, zone_clear

BOUNDS = Bounds(1.44, 5.83, 2.17, 4.46)
LEADER = PlanParams(footprint=(0.13, 0.23, 0.18), clearance=0.05,
                    pivot_step=math.radians(15), front_zone=(0.045, 0.59, 0.19),
                    front_zone_runin=0.6)
OFF = dataclasses.replace(LEADER, front_zone=None, front_zone_runin=0.0)
STRICT = dataclasses.replace(LEADER, front_zone_runin=0.0)     # no run-in exemption
FOLLOWER = PlanParams(footprint=(0.15, 0.15, 0.155), clearance=0.05,
                      front_zone=(0.0, 0.33, 0.20))
SHELF = Box(5.386, 2.693, 0.4, 0.4)       # gear shelf; its face is at x 5.186
PICK = Pose2D(4.89, 2.69, 0.0)            # 0.30 m in front of the face, facing it


def _line(x0, x1, y=3.3, step=0.05):
    count = int(round((x1 - x0) / step))
    return [Pose2D(x0 + i * (x1 - x0) / count, y, 0.0) for i in range(count + 1)]


def _zone_clear(path, obstacles, params):
    return zone_clear(to_array(path), obstacle_array(obstacles), params)


def _box_ahead(x_face, y=3.3):
    """0.2 x 0.2 box whose west face is at ``x_face``."""
    return Box(x_face + 0.1, y, 0.2, 0.2)


def test_box_in_zone_invalidates_path():
    path = _line(2.0, 3.0)                       # ends at x 3.0, heading +x
    inside = _box_ahead(3.0 + 0.50)              # 0.455 from the camera: in the zone
    beyond = _box_ahead(3.0 + 0.70)              # 0.655 from the camera: past it
    assert keeps_clearance(path, [inside], OFF)  # the footprint keeps its clearance
    assert not keeps_clearance(path, [inside], STRICT)
    assert keeps_clearance(path, [beyond], STRICT)
    assert keeps_clearance(path, [inside], LEADER)   # goal facing it: exempt run-in
    assert plan_path(Pose2D(2.0, 3.3, 0.0), Pose2D(3.0, 3.3, 0.0), [inside], BOUNDS,
                     STRICT) is None
    assert plan_path(Pose2D(2.0, 3.3, 0.0), Pose2D(3.0, 3.3, 0.0), [inside], BOUNDS,
                     OFF) is not None


def test_arena_bounds_not_zone_objects():
    goal = Pose2D(BOUNDS.xmax - 0.05, 3.3, 0.0)  # nose to the edge of the camera's view
    path = plan_path(Pose2D(4.0, 3.3, 0.0), goal, [], BOUNDS, LEADER)
    assert path is not None and path[-1] == goal


def test_lane_beside_box_stays_valid():
    box = Box(2.5, 3.3 + 0.18 + 0.05 + 0.1 + 0.01, 0.2, 0.2)  # just clear of the body
    assert keeps_clearance(_line(2.0, 3.0), [box], LEADER)


def test_departure_pivots_until_zone_clear():
    """Fixture 0.45 m ahead in the zone, footprint clear: the path starts with a pivot."""
    start = Pose2D(2.0, 3.3, 0.0)
    fixture = Box(2.55, 3.3 + 0.25, 0.2, 0.2)    # x 2.45-2.65, y 3.45-3.65
    goal = Pose2D(3.8, 2.7, 0.0)
    assert not _zone_clear([start], [fixture], LEADER)      # in the zone where it stands
    old, old_heading = plan_departure(start, goal, [fixture], BOUNDS, OFF)
    assert old is not None and not _zone_clear(old, [fixture], LEADER)
    path, heading = plan_departure(start, goal, [fixture], BOUNDS, LEADER)
    assert path is not None and heading is not None and path[0].theta == heading
    assert _zone_clear(path, [fixture], LEADER)              # from the post-pivot pose on
    assert plan_path(start, goal, [fixture], BOUNDS, LEADER) is None   # no forward way out


def test_pick_departure_and_approach_zone():
    """The way home from the pick pose pivots the shelf out of the zone; the pick goal
    stays plannable through the exempt run-in, and is refused without it."""
    assert not _zone_clear([PICK], [SHELF], STRICT)
    home = Pose2D(1.82, 3.29, math.pi)
    path, heading = plan_departure(PICK, home, [SHELF], BOUNDS, LEADER)
    assert path is not None and heading is not None
    assert _zone_clear(path, [SHELF], STRICT)
    start = Pose2D(2.6, 3.3, 0.0)
    assert plan_path(start, PICK, [SHELF], BOUNDS, LEADER) is not None
    assert plan_path(start, PICK, [SHELF], BOUNDS, STRICT) is None


def test_exempt_run_in_is_straight_last_stretch():
    goal = Pose2D(3.0, 3.3, 0.0)
    arr = to_array([Pose2D(2.39, 3.3, 0.0), Pose2D(2.41, 3.3, 0.0), Pose2D(2.8, 3.3, 0.1),
                    Pose2D(2.8, 3.32, 0.0), goal])
    assert list(runin_exempt(arr, (goal.x, goal.y, goal.theta), LEADER)) == \
        [False, True, False, False, True]
    assert not runin_exempt(arr, (goal.x, goal.y, goal.theta), STRICT).any()


def test_follower_zone_ignores_lead_on_shared():
    """Follower goal 0.50 m behind the lead: the lead is inside its zone, ignored on a
    shared plan only, and never for the body clearance."""
    lead_goal = Pose2D(4.0, 3.3, 0.0)
    lead_path = _line(1.8, 4.0)
    lead_box = Box(lead_goal.x - 0.05, lead_goal.y, 0.36, 0.36)   # footprint (0.13, 0.23)
    follower, goal = Pose2D(1.6, 3.3, 0.0), Pose2D(3.5, 3.3, 0.0)
    assert plan_follower_path(follower, lead_path, goal, [lead_box], BOUNDS, FOLLOWER) is None
    plan = plan_follower_path(follower, lead_path, goal, [lead_box], BOUNDS, FOLLOWER,
                              zone_ignore=[lead_box])
    assert plan is not None and plan.shared
    assert not _zone_clear(plan.path, [lead_box], FOLLOWER)    # the lead IS in it
    assert keeps_clearance(plan.path, [lead_box], FOLLOWER, zone_ignore=[lead_box])
    off = dataclasses.replace(FOLLOWER, front_zone=None)
    assert plan_follower_path(follower, lead_path, goal, [lead_box], BOUNDS, off).shared


def test_zone_off_is_the_old_validator():
    """``front_zone`` None: identical results (the default)."""
    rng = np.random.default_rng(7)
    obstacles = [Box(float(x), float(y), 0.3, 0.3) for x, y in
                 zip(rng.uniform(2.0, 5.0, 3), rng.uniform(2.5, 4.0, 3))]
    start, goal = Pose2D(1.7, 2.5, 0.0), Pose2D(5.4, 4.1, 1.0)
    plain = PlanParams(footprint=(0.13, 0.23, 0.18), clearance=0.05)
    with_runin = dataclasses.replace(plain, front_zone_runin=0.6)   # no zone: no effect
    assert plan_path(start, goal, obstacles, BOUNDS, plain) == \
        plan_path(start, goal, obstacles, BOUNDS, with_runin)
