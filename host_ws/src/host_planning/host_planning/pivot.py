"""Leading pivot: an in-place rotation at the start of a path (departure only).

For a rover parked close to a shelf or wall with no forward path out. The rotation turns the
short way (``|turn| <= max_pivot < pi``) about ``pivot_offset`` behind the tracked point; a
larger turn onto the goal heading needs the sweep clear both ways. Footprints are checked
every ``pivot_sweep_step``, padded by the largest body-point move between samples.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from .geometry import (body_shapes, clear_of, include, inside, obstacle_array, shape_gaps,
                       wrap_angle)
from .types import Bounds, PlanParams, Pose2D

T = TypeVar('T')
_SAME_POS = 1e-3     # m: two poses at the same position


def candidate_turns(theta: float, params: PlanParams,
                    goal_heading: Optional[float] = None) -> List[float]:
    """Signed turns to the candidate headings, smallest first. Headings are world-fixed
    multiples of ``pivot_step`` (a re-plan mid-pivot finds the same target), plus ``goal_heading``."""
    if params.pivot_step <= 0.0:
        return []
    count = int(round(2.0 * math.pi / params.pivot_step))
    turns = {round(float(wrap_angle(k * params.pivot_step - theta)), 9) for k in range(count)}
    turns = {t for t in turns if abs(t) <= params.max_pivot}
    if goal_heading is not None:    # also beyond max_pivot: search_departure checks both ways
        turns.add(round(float(wrap_angle(goal_heading - theta)), 9))
    turns = [t for t in turns if abs(t) > _SAME_POS]
    return sorted(turns, key=lambda t: (abs(t), -t))


def swept_poses(start: Pose2D, turn: float, params: PlanParams) -> np.ndarray:
    steps = max(1, int(math.ceil(abs(turn) / params.pivot_sweep_step)))
    yaw = start.theta + np.linspace(0.0, turn, steps + 1)
    off = params.pivot_offset
    cx, cy = start.x - off * math.cos(start.theta), start.y - off * math.sin(start.theta)
    return np.column_stack([cx + off * np.cos(yaw), cy + off * np.sin(yaw), yaw])


def _reach(params: PlanParams) -> float:
    if params.footprint is None:
        return params.robot_radius + abs(params.pivot_offset)
    front, rear, half = params.footprint
    off = params.pivot_offset
    return math.hypot(max(front + off, rear - off), half)


def pivot_clear(start: Pose2D, turn: float, obstacles: Sequence[object],
                bounds: Optional[Bounds], params: PlanParams) -> bool:
    """True when the footprint swept by turning ``turn`` (rad, + = CCW) keeps ``clearance``
    (escape rule as for paths) and the tracked point stays in ``bounds``. The front-stop zone
    is not checked (exempt while pivoting)."""
    arr = swept_poses(start, turn, params)
    if not inside(arr, include(bounds, (start.x, start.y))):
        return False
    obs = obstacle_array(obstacles)
    if len(obs) == 0:
        return True
    step = abs(turn) / (len(arr) - 1)
    shapes = body_shapes(arr, params)
    shapes[:, 5] += 2.0 * _reach(params) * math.sin(step / 4.0)
    return clear_of(shape_gaps(shapes, obs), params.clearance - 1e-9, escape=True)


def search_departure(start: Pose2D, goal_xy: Tuple[float, float],
                     plan_from: Callable[[Pose2D], Optional[T]],
                     length_of: Callable[[T], float],
                     obstacles: Sequence[object], bounds: Optional[Bounds],
                     params: PlanParams,
                     forward: Optional[T] = None,
                     goal_heading: Optional[float] = None
                     ) -> Tuple[Optional[T], Optional[float]]:
    """Best of the forward plan (``plan_from(start)``) and pivot-first plans: (plan, pivot
    heading). A pivot plan costs length + ``pivot_cost`` per rad and wins only when cheaper."""
    best, best_heading = forward, None
    best_cost = math.inf if forward is None else length_of(forward)
    floor = math.hypot(goal_xy[0] - start.x, goal_xy[1] - start.y)
    for turn in candidate_turns(start.theta, params, goal_heading):
        if floor + params.pivot_cost * abs(turn) >= best_cost:
            break   # turns only grow from here
        if not pivot_clear(start, turn, obstacles, bounds, params):
            continue
        other_way = turn - math.copysign(2.0 * math.pi, turn)
        if abs(turn) > params.max_pivot \
                and not pivot_clear(start, other_way, obstacles, bounds, params):
            continue    # near half a turn the rover picks the direction
        heading = float(wrap_angle(start.theta + turn))
        plan = plan_from(Pose2D(start.x, start.y, heading))
        if plan is None:
            continue
        cost = length_of(plan) + params.pivot_cost * abs(turn)
        if cost < best_cost:
            best, best_heading, best_cost = plan, heading, cost
    return best, best_heading
