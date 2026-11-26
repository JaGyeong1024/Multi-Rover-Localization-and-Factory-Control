"""Obstacle-free smooth path from a start pose to a goal pose (forward only).

The path is ``connector(start -> approach pose) + straight run-in``; the approach
pose sits ``approach_len`` behind the goal on the goal heading.

The connector is one of:

1. Quintic Bezier with collinear end control triples (preferred): zero curvature at
   both ends, so it joins the run-in without a curvature step. Several tangent
   lengths are tried; the valid candidate with the lowest ``length + curv_weight *
   max|curvature|`` wins. Candidates whose curvature exceeds ``1 / min_turn_radius``
   are rejected.

2. Curvature-relaxed Dubins (fallback: goal behind the start, close poses, tight
   walls), with curvature steps replaced by linear ramps (see :mod:`dubins`); plain
   Dubins if the relaxation does not converge.

Reversing is never used. ``None`` means no candidate satisfies curvature, bounds and
the caller's validator.
"""
from __future__ import annotations

import functools
import math
import struct
from typing import List, Optional, Tuple

import numpy as np

from . import dubins
from .geometry import (Validator, arc_lengths, obstacle_array, curvatures, join,
                       make_validator, resample, shrink, straight, wrap_angle)
from .types import Bounds, PlanParams, Pose2D

_BINOMIAL_5 = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
_BINOMIAL_4 = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
_SAME_POINT = 1e-6


def _row(pose: Pose2D) -> np.ndarray:
    return np.array([[pose.x, pose.y, pose.theta]])


def _bernstein(t: np.ndarray, binomial: np.ndarray) -> np.ndarray:
    degree = len(binomial) - 1
    k = np.arange(degree + 1)
    return binomial * t[:, None] ** k * (1.0 - t[:, None]) ** (degree - k)


def _control_points(start: Pose2D, goal: Pose2D, tangent: float) -> np.ndarray:
    u0 = np.array([math.cos(start.theta), math.sin(start.theta)])
    u1 = np.array([math.cos(goal.theta), math.sin(goal.theta)])
    p0, p5 = np.array([start.x, start.y]), np.array([goal.x, goal.y])
    return np.array([p0, p0 + tangent * u0, p0 + 2.0 * tangent * u0,
                     p5 - 2.0 * tangent * u1, p5 - tangent * u1, p5])


@functools.lru_cache(maxsize=4096)
def _bases(intervals: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read-only quintic and quartic Bernstein matrices at ``intervals`` + 1 uniform
    parameters; they depend on the sample count only, so they are cached per count."""
    t = np.linspace(0.0, 1.0, intervals + 1)
    bases = (_bernstein(t, _BINOMIAL_5), _bernstein(t, _BINOMIAL_4))
    for basis in bases:
        basis.setflags(write=False)
    return bases


def _bezier(start: Pose2D, goal: Pose2D, tangent: float, step: float) -> np.ndarray:
    ctrl = _control_points(start, goal, tangent)
    polygon = float(np.hypot(*np.diff(ctrl, axis=0).T).sum())
    quintic, quartic = _bases(max(8, int(math.ceil(polygon / step))))
    xy = quintic @ ctrl
    deriv = quartic @ (5.0 * np.diff(ctrl, axis=0))
    theta = np.arctan2(deriv[:, 1], deriv[:, 0])
    theta[0], theta[-1] = start.theta, goal.theta
    return np.column_stack([xy, theta])


def _cost(arr: np.ndarray, params: PlanParams) -> float:
    kappa = curvatures(arr)
    peak = float(np.abs(kappa).max()) if len(kappa) else 0.0
    return float(arc_lengths(arr)[-1]) + params.curv_weight * peak


def _bits(pose: Pose2D) -> bytes:
    return struct.pack('3d', pose.x, pose.y, pose.theta)


def memoized(fn):
    """Cache ``fn(start, goal, params, valid, max_len)`` in ``valid.memo``.

    Poses are keyed by bit pattern (0.0 and -0.0 differ in the copied heading).
    Cached arrays are read-only.
    """
    @functools.wraps(fn)
    def inner(start, goal, params, valid, max_len):
        memo = getattr(valid, 'memo', None)
        if memo is None:
            return fn(start, goal, params, valid, max_len)
        key = (fn.__name__, _bits(start), _bits(goal), params, max_len)
        if key not in memo:
            out = fn(start, goal, params, valid, max_len)
            if out is not None:
                out.setflags(write=False)
            memo[key] = out
        return memo[key]
    return inner


@memoized
def _best_bezier(start, goal, params, valid, max_len) -> Optional[np.ndarray]:
    chord = math.hypot(goal.x - start.x, goal.y - start.y)
    best, best_cost = None, float('inf')
    for scale in params.tangent_scales:
        arr = _bezier(start, goal, scale * chord, params.dense_step)
        if not _accept(arr, valid, max_len):
            continue
        cost = _cost(arr, params)
        if cost < best_cost:
            best, best_cost = arr, cost
    return best


def _accept(arr: np.ndarray, valid: Validator, max_len: Optional[float]) -> bool:
    if max_len is not None and arc_lengths(arr)[-1] > max_len:
        return False
    return valid(arr)


def _dubins_candidates(start, goal, params) -> List[tuple]:
    """(length, segments, radius) over all radius scales, shortest first."""
    found = []
    for scale in params.dubins_radius_scales:
        radius = params.min_turn_radius * scale
        for length, segments in dubins.dubins_words(start, goal, radius):
            found.append((length, segments, radius))
    return sorted(found, key=lambda item: item[0])


@memoized
def _first_dubins(start, goal, params, valid, max_len) -> Optional[np.ndarray]:
    step = params.dense_step
    for _, segments, radius in _dubins_candidates(start, goal, params):
        kappa = 1.0 / radius
        relaxed = dubins.relax(start, goal, segments, kappa, params.curv_ramp_len,
                               step, params.relax_iterations, params.relax_tol)
        if relaxed is not None and _accept(relaxed, valid, max_len):
            return relaxed
        plain = dubins.integrate(start, segments, kappa, 0.0,
                                 dubins.sample_counts(segments, step))
        plain[-1] = (goal.x, goal.y, goal.theta)
        if _accept(plain, valid, max_len):
            return plain
    return None


def connect(start: Pose2D, goal: Pose2D, params: PlanParams, valid: Validator,
            allow_dubins: bool = True,
            max_len: Optional[float] = None) -> Optional[np.ndarray]:
    """Dense smooth connector between two poses (no run-in), or None.

    ``valid`` decides curvature / bounds / collision; ``max_len`` rejects detours.
    """
    chord = math.hypot(goal.x - start.x, goal.y - start.y)
    aligned = abs(wrap_angle(goal.theta - start.theta)) < params.heading_tol
    if chord < _SAME_POINT and aligned:
        return _row(goal)
    best = _best_bezier(start, goal, params, valid, max_len) \
        if chord >= _SAME_POINT else None
    if best is not None or not allow_dubins:
        return best
    return _first_dubins(start, goal, params, valid, max_len)


def approach_pose(goal: Pose2D, length: float) -> Pose2D:
    return Pose2D(goal.x - length * math.cos(goal.theta),
                  goal.y - length * math.sin(goal.theta), goal.theta)


def approach_lengths(params: PlanParams) -> List[float]:
    """Run-in lengths to try, longest (nominal) first."""
    floor = params.min_approach_len
    if floor is None or floor >= params.approach_len:
        return [params.approach_len]
    return list(np.linspace(params.approach_len, floor,
                            params.approach_shrink_steps))


def _on_run_in(start: Pose2D, goal: Pose2D, params: PlanParams) -> bool:
    """True when the start already sits on the run-in line, facing the goal.

    With ``direct_skew_tol`` > 0 also a start behind the goal whose straight line to the
    goal is within that angle of the goal heading (else it would get a loop).
    """
    dx, dy = goal.x - start.x, goal.y - start.y
    along = dx * math.cos(goal.theta) + dy * math.sin(goal.theta)
    lateral = -dx * math.sin(goal.theta) + dy * math.cos(goal.theta)
    if abs(wrap_angle(goal.theta - start.theta)) >= params.heading_tol or along < -_SAME_POINT:
        return False
    if abs(lateral) < params.on_line_tol and along <= params.approach_len:
        return True
    return along > _SAME_POINT and abs(math.atan2(lateral, along)) <= params.direct_skew_tol


def _direct(start: Pose2D, goal: Pose2D, params: PlanParams) -> np.ndarray:
    line = straight((start.x, start.y), (goal.x, goal.y), goal.theta,
                    params.dense_step)
    line[0, 2] = start.theta
    return line


def dense_path(start: Pose2D, goal: Pose2D, params: PlanParams, valid: Validator,
               allow_dubins: bool = True,
               max_len: Optional[float] = None) -> Optional[np.ndarray]:
    """Dense connector + straight run-in to the goal, or None."""
    if _on_run_in(start, goal, params):
        line = _direct(start, goal, params)
        return line if valid(line) else None
    for length in approach_lengths(params):
        entry = approach_pose(goal, length)
        run_in = straight((entry.x, entry.y), (goal.x, goal.y), goal.theta,
                          params.dense_step)
        if not valid(run_in):
            continue
        link = connect(start, entry, params, valid, allow_dubins, max_len)
        if link is not None:
            return join([link, run_in[1:]])
    return None


def smooth_path(start: Pose2D, goal: Pose2D, bounds: Optional[Bounds] = None,
                params: Optional[PlanParams] = None) -> Optional[np.ndarray]:
    """Smooth obstacle-free path as an (N, 3) array at ``params.spacing``, or None."""
    params = params or PlanParams()
    valid = make_validator(obstacle_array(()),
                           shrink(bounds, params.bounds_margin), params)
    dense = dense_path(start, goal, params, valid)
    return None if dense is None else resample(dense, params.spacing)
