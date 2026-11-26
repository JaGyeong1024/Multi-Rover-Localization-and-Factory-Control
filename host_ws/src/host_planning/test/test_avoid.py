"""Obstacle avoidance: clearance, contract, no-path cases, grid search."""
import math
import time

import numpy as np

from checks import AREA, assert_path_ok
from host_planning import (Circle, PlanParams, Pose2D, min_clearance,
                           path_length, plan_path)
from host_planning import avoid
from host_planning.geometry import circle_shapes

PARAMS = PlanParams()


def test_single_box_on_the_straight_line():
    start, goal = Pose2D(0.5, 1.35, 0.0), Pose2D(4.3, 1.35, 0.0)
    boxes = [Circle(2.4, 1.35, 0.25)]
    path = plan_path(start, goal, boxes, AREA, PARAMS)
    assert path is not None
    assert_path_ok(path, start, goal, PARAMS, obstacles=boxes)
    assert path_length(path) > 3.8          # had to leave the straight line
    assert path_length(path) < 1.5 * 3.8    # but without a large detour


def test_obstacle_off_route_changes_nothing():
    start, goal = Pose2D(0.5, 0.6, 0.0), Pose2D(4.3, 0.6, 0.0)
    free = plan_path(start, goal, (), AREA, PARAMS)
    cluttered = plan_path(start, goal, [Circle(2.4, 2.2, 0.2)], AREA, PARAMS)
    assert cluttered == free


def test_slalom_between_boxes():
    start, goal = Pose2D(0.4, 1.35, 0.0), Pose2D(4.4, 1.35, 0.0)
    boxes = [Circle(1.5, 1.0, 0.25), Circle(2.4, 1.8, 0.25), Circle(3.3, 1.0, 0.25)]
    path = plan_path(start, goal, boxes, AREA, PARAMS)
    assert path is not None
    assert_path_ok(path, start, goal, PARAMS, obstacles=boxes)


def test_box_as_several_circles():
    """A 0.9 x 0.3 m box approximated by three circles."""
    start, goal = Pose2D(0.5, 1.2, 0.0), Pose2D(4.3, 1.5, 0.0)
    boxes = [Circle(2.4, y, 0.21) for y in (0.9, 1.2, 1.5)]
    path = plan_path(start, goal, boxes, AREA, PARAMS)
    assert path is not None
    assert_path_ok(path, start, goal, PARAMS, obstacles=boxes)


def test_clearance_parameter_is_honoured():
    start, goal = Pose2D(0.5, 1.35, 0.0), Pose2D(4.3, 1.35, 0.0)
    boxes = [Circle(2.4, 1.35, 0.25)]
    wide = PlanParams(clearance=0.20)
    path = plan_path(start, goal, boxes, AREA, wide)
    assert min_clearance(path, boxes) >= wide.robot_radius + 0.20 - 1e-6


def test_wall_of_boxes_returns_none():
    boxes = [Circle(2.4, y, 0.3) for y in np.arange(0.0, 2.71, 0.45)]
    assert plan_path(Pose2D(0.5, 1.35, 0.0), Pose2D(4.3, 1.35, 0.0),
                     boxes, AREA, PARAMS) is None


def test_goal_inside_obstacle_returns_none():
    boxes = [Circle(4.3, 1.35, 0.3)]
    assert plan_path(Pose2D(0.5, 1.35, 0.0), Pose2D(4.3, 1.35, 0.0),
                     boxes, AREA, PARAMS) is None


def test_start_inside_inflation_returns_none():
    """Head-on: every forward path gets closer than the start already is."""
    boxes = [Circle(0.7, 1.35, 0.2)]
    assert plan_path(Pose2D(0.5, 1.35, 0.0), Pose2D(4.3, 1.35, 0.0),
                     boxes, AREA, PARAMS) is None


def test_escape_start_next_to_an_obstacle():
    """A rover stopped closer than the required clearance (e.g. after a front stop)
    may leave: below the clearance only until it first reaches it, never closer
    than the start. Other obstacles get the full clearance."""
    box = Circle(2.0, 1.55, 0.2)
    start, goal = Pose2D(2.0, 1.35, 0.0), Pose2D(4.3, 1.35, 0.0)
    start_gap = min_clearance([start], [box])
    assert start_gap < PARAMS.required_clearance
    path = plan_path(start, goal, [box], AREA, PARAMS)
    assert path is not None and path[0] == start and path[-1] == goal
    gaps = np.array([min_clearance([p], [box]) for p in path])
    assert gaps.min() >= start_gap - 1e-9
    first = int(np.argmax(gaps >= PARAMS.required_clearance))
    assert first > 0 and gaps[first:].min() >= PARAMS.required_clearance - 1e-9
    # the exemption is the start's own obstacle only: a second box near the way out
    second = Circle(2.6, 1.0, 0.1)
    path = plan_path(start, goal, [box, second], AREA, PARAMS)
    assert path is not None
    assert min_clearance(path, [second]) >= PARAMS.required_clearance - 1e-9


def test_without_bounds():
    start, goal = Pose2D(0.0, 0.0, 0.0), Pose2D(3.0, 0.0, 0.0)
    boxes = [Circle(1.5, 0.0, 0.3)]
    path = plan_path(start, goal, boxes)
    assert path is not None
    assert_path_ok(path, start, goal, PARAMS, bounds=None, obstacles=boxes)


def test_astar_finds_shortest_grid_route():
    blocked = np.zeros((10, 10), dtype=bool)
    blocked[5, 0:9] = True
    cells = avoid.astar(blocked, (0, 0), (9, 0))
    assert cells[0] == (0, 0) and cells[-1] == (9, 0)
    assert (5, 9) in cells
    assert not any(blocked[c] for c in cells)
    blocked[5, 9] = True
    assert avoid.astar(blocked, (0, 0), (9, 0)) is None


def test_simplify_keeps_line_of_sight():
    circles = circle_shapes([(1.0, 0.0, 0.3)])
    points = [(0.0, 0.0), (0.5, 0.6), (1.0, 0.7), (1.5, 0.6), (2.0, 0.0)]
    kept = avoid.simplify(points, circles, 0.2)
    assert kept[0] == points[0] and kept[-1] == points[-1]
    assert 2 < len(kept) < len(points)


def test_planning_time_four_boxes():
    start, goal = Pose2D(0.4, 0.5, 0.0), Pose2D(4.4, 2.2, 0.0)
    boxes = [Circle(1.3, 0.8, 0.25), Circle(2.2, 1.7, 0.25),
             Circle(3.1, 0.9, 0.25), Circle(3.7, 1.5, 0.2)]
    began = time.perf_counter()
    path = plan_path(start, goal, boxes, AREA, PARAMS)
    elapsed = time.perf_counter() - began
    assert path is not None
    assert_path_ok(path, start, goal, PARAMS, obstacles=boxes)
    assert elapsed < 0.5
    assert math.isfinite(min_clearance(path, boxes))
