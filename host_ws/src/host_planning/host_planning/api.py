"""Public entry points of the planning library: forward-only paths of Pose2D in ``uwb_global``
at most ``PlanParams.spacing`` apart, ending at the goal pose."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from . import avoid, pivot, platoon
from .geometry import (arc_lengths, body_gaps, body_shapes, clear_of, clearance, curvatures,
                       obstacle_array, shape_gaps, to_array, to_poses, zone_clear, zone_mask)
from .types import Bounds, Box, Circle, PlanParams, PlatoonPlan, Pose2D


def plan_path(start: Pose2D, goal: Pose2D, obstacles: Sequence[Circle] = (),
              bounds: Optional[Bounds] = None,
              params: Optional[PlanParams] = None) -> Optional[List[Pose2D]]:
    """Smooth forward path, or None; keeps both headings, curvature <= 1/min_turn_radius,
    ``bounds``, ``clearance``, and ends with a straight run-in of ``approach_len``."""
    arr = avoid.plan_around(start, goal, obstacles, bounds, params)
    return None if arr is None else to_poses(arr)


def plan_departure(start: Pose2D, goal: Pose2D, obstacles: Sequence[Circle] = (),
                   bounds: Optional[Bounds] = None,
                   params: Optional[PlanParams] = None
                   ) -> Tuple[Optional[List[Pose2D]], Optional[float]]:
    """:func:`plan_path`, or one starting with an in-place rotation when that is the only or
    cheaper way (``pivot_step`` > 0). Returns (path, pivot heading); heading None = plain path."""
    params = params or PlanParams()

    def plan_from(pose):
        return plan_path(pose, goal, obstacles, bounds, params)

    forward = plan_from(start)
    path, heading = pivot.search_departure(start, (goal.x, goal.y), plan_from, path_length,
                                           obstacles, bounds, params, forward, goal.theta)
    return path, heading


def plan_follower_path(follower_pose: Pose2D, leader_path: Sequence[Pose2D],
                       follower_goal: Pose2D, obstacles: Sequence[Circle] = (),
                       bounds: Optional[Bounds] = None,
                       params: Optional[PlanParams] = None,
                       zone_ignore: Sequence[Union[Circle, Box]] = ()
                       ) -> Optional[PlatoonPlan]:
    """Follower path that joins, shares and leaves the leader path (``shared`` False =
    stand-alone); None when no path exists. ``zone_ignore``: ignored by the front-stop zone."""
    return platoon.plan_platoon(follower_pose, leader_path, follower_goal,
                                obstacles, bounds, params, zone_ignore)


def keeps_clearance(path: Sequence[Pose2D], obstacles: Sequence[Union[Circle, Box]] = (),
                    params: Optional[PlanParams] = None,
                    zone_ignore: Sequence[Union[Circle, Box]] = ()) -> bool:
    """True when ``path`` passes the obstacle test :func:`plan_path` applies."""
    params = params or PlanParams()
    obs = obstacle_array(obstacles)
    if len(obs) == 0 or len(path) == 0:
        return True
    arr = to_array(path)
    if not clear_of(body_gaps(arr, obs, params), params.clearance - 1e-9, escape=True):
        return False
    return zone_clear(arr, obs, params, goal=arr[-1], seen=zone_mask(obstacles, zone_ignore))


def path_length(path: Sequence[Pose2D]) -> float:
    if len(path) < 2:
        return 0.0
    return float(arc_lengths(to_array(path))[-1])


def max_curvature(path: Sequence[Pose2D]) -> float:
    kappa = curvatures(to_array(path))
    return float(np.abs(kappa).max()) if len(kappa) else 0.0


def min_clearance(path: Sequence[Pose2D], obstacles: Sequence[Circle]) -> float:
    """Smallest distance from a path point to an obstacle surface (m); ``inf`` if none."""
    return clearance(to_array(path), obstacle_array(obstacles))


def min_body_gap(path: Sequence[Pose2D], obstacles: Sequence[Union[Circle, Box]],
                 params: Optional[PlanParams] = None) -> float:
    """Smallest footprint gap to any obstacle along the path (m, negative = overlap)."""
    params = params or PlanParams()
    obs = obstacle_array(obstacles)
    if len(obs) == 0 or len(path) == 0:
        return float('inf')
    return float(shape_gaps(body_shapes(to_array(path), params), obs).min())
