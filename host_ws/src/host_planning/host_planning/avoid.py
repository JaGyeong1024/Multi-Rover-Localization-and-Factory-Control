"""Path around obstacles (circles, oriented boxes): grid A* -> simplify -> smooth
-> re-check.

Pipeline: (1) the obstacle-free smooth path if its footprint keeps the clearance;
otherwise (2) 8-connected A* on a grid inflated by ``lateral_radius + clearance +
buffer``, (3) line-of-sight shortcut into via points headed along the neighbour
chord, (4) smooth connectors stitched through the via poses (:func:`smooth.connect`),
(5) every connector validated against the un-buffered clearance; on failure the buffer
grows by ``inflate_step`` (up to ``max_retries``). The grid search first runs with
virtual end-guard circles (:func:`end_guards`) so the route respects the start and
goal headings, then without them.

Escape start: a start closer than the required clearance is accepted; the path may
leave that obstacle but never gets closer than the start is (:func:`geometry.clear_of`),
and the grid search starts from the nearest free cell within ``snap_radius``.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence

import numpy as np

from . import smooth
from .geometry import (Validator, bounding_radius, circle_shapes, join,
                       make_validator, obstacle_array, point_shapes, resample,
                       shape_gaps, shrink, straight)
from .types import Bounds, Circle, PlanParams, Pose2D

_NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
               (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0))]


class Grid:
    """Occupancy grid over the bounds; nodes are cell centres."""

    def __init__(self, bounds: Bounds, res: float):
        self.x0, self.y0, self.res = bounds.xmin, bounds.ymin, res
        self.nx = max(1, int(math.ceil((bounds.xmax - bounds.xmin) / res)))
        self.ny = max(1, int(math.ceil((bounds.ymax - bounds.ymin) / res)))
        self.xs = self.x0 + (np.arange(self.nx) + 0.5) * res
        self.ys = self.y0 + (np.arange(self.ny) + 0.5) * res

    def blocked(self, shapes: np.ndarray, inflate: float) -> np.ndarray:
        if len(shapes) == 0:
            return np.zeros((self.nx, self.ny), dtype=bool)
        gx, gy = np.meshgrid(self.xs, self.ys, indexing='ij')
        cells = point_shapes(np.stack([gx.ravel(), gy.ravel()], axis=1))
        gap = shape_gaps(cells, shapes, near=inflate).min(axis=1)
        return (gap < inflate).reshape(self.nx, self.ny)

    def nearest_free(self, xy: Sequence[float], blocked: np.ndarray,
                     max_dist: float) -> Optional[tuple]:
        dist = np.hypot(self.xs[:, None] - xy[0], self.ys[None, :] - xy[1])
        dist = np.where(blocked, np.inf, dist)
        cell = np.unravel_index(int(np.argmin(dist)), dist.shape)
        return (int(cell[0]), int(cell[1])) if dist[cell] <= max_dist else None

    def centre(self, cell: tuple) -> tuple:
        return (float(self.xs[cell[0]]), float(self.ys[cell[1]]))


_DIAGONAL_EXTRA = math.sqrt(2.0) - 1.0


def _octile(a: tuple, b: tuple) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + _DIAGONAL_EXTRA * min(dx, dy)


def _backtrack(parent: dict, cell: tuple) -> List[tuple]:
    cells = [cell]
    while cell in parent:
        cell = parent[cell]
        cells.append(cell)
    return cells[::-1]


def astar(blocked: np.ndarray, start: tuple, goal: tuple) -> Optional[List[tuple]]:
    """8-connected A* over free cells; returns the cell path or None."""
    # Hot loop: flat lists and an inlined octile heuristic. A cell is the flat index
    # (x + 1) * w + (y + 1) of a grid with a blocked one-cell border (no range checks).
    nx, ny = blocked.shape
    w = ny + 2
    free = [False] * ((nx + 2) * w)
    for x, row in enumerate((~blocked).tolist()):
        free[(x + 1) * w + 1:(x + 1) * w + 1 + ny] = row
    moves = [(dx * w + dy, dx, dy, step) for dx, dy, step in _NEIGHBOURS]
    gx, gy = goal[0] + 1, goal[1] + 1
    first, last = (start[0] + 1) * w + start[1] + 1, gx * w + gy
    best = [math.inf] * len(free)
    best[first] = 0.0
    parent: dict = {}
    queue = [(_octile(start, goal), 0.0, first)]
    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cell == last:
            return [((c // w) - 1, (c % w) - 1) for c in _backtrack(parent, cell)]
        if cost > best[cell]:
            continue
        cx, cy = divmod(cell, w)
        for offset, dx, dy, step in moves:
            nxt = cell + offset
            if not free[nxt]:
                continue
            new_cost = cost + step
            if new_cost < best[nxt]:
                best[nxt], parent[nxt] = new_cost, cell
                ax, ay = abs(cx + dx - gx), abs(cy + dy - gy)
                rest = ax + _DIAGONAL_EXTRA * ay if ax >= ay else ay + _DIAGONAL_EXTRA * ax
                heapq.heappush(queue, (new_cost + rest, new_cost, nxt))
    return None


def _free_segments(a, ends, shapes: np.ndarray, inflate: float) -> np.ndarray:
    """For each end point b: True when segment a-b stays outside every inflated obstacle."""
    if len(shapes) == 0:
        return np.ones(len(ends), dtype=bool)
    a, b = np.asarray(a, dtype=float), np.asarray(ends, dtype=float)
    mid, ab = (a + b) / 2.0, b - a
    zero = np.zeros(len(b))
    segments = np.column_stack([mid, np.arctan2(ab[:, 1], ab[:, 0]),
                                np.hypot(ab[:, 0], ab[:, 1]) / 2.0, zero, zero])
    return shape_gaps(segments, shapes, near=inflate).min(axis=1) >= inflate


def simplify(points: List[tuple], circles: np.ndarray, inflate: float) -> List[tuple]:
    """Greedy line-of-sight shortcut of a polyline: from each kept point jump to the
    farthest point it sees (all candidates tested in one batch)."""
    kept, i = [points[0]], 0
    while i < len(points) - 1:
        ahead = points[i + 2:]
        seen = np.flatnonzero(_free_segments(points[i], ahead, circles, inflate)) \
            if ahead else ()
        j = i + 2 + int(seen[-1]) if len(seen) else i + 1
        kept.append(points[j])
        i = j
    return kept


def grid_route(start_xy, goal_xy, circles: np.ndarray, bounds: Bounds,
               inflate: float, params: PlanParams) -> Optional[List[tuple]]:
    """Simplified collision-free polyline from start to goal, or None."""
    grid = Grid(bounds, params.grid_res)
    blocked = grid.blocked(circles, inflate)
    first = grid.nearest_free(start_xy, blocked, params.snap_radius)
    last = grid.nearest_free(goal_xy, blocked, params.snap_radius)
    if first is None or last is None:
        return None
    cells = astar(blocked, first, last)
    if cells is None:
        return None
    points = [tuple(start_xy)] + [grid.centre(c) for c in cells] + [tuple(goal_xy)]
    return simplify(points, circles, inflate)


def via_poses(points: List[tuple], start: Pose2D, end: Pose2D) -> List[Pose2D]:
    poses = [start]
    for prev, here, nxt in zip(points, points[1:], points[2:]):
        heading = math.atan2(nxt[1] - prev[1], nxt[0] - prev[0])
        poses.append(Pose2D(here[0], here[1], heading))
    return poses + [end]


def stitch(poses: List[Pose2D], params: PlanParams, valid: Validator,
           allow_dubins: bool) -> Optional[np.ndarray]:
    pieces, i = [], 0
    while i < len(poses) - 1:
        for j in range(len(poses) - 1, i, -1):
            link = smooth.connect(poses[i], poses[j], params, valid, allow_dubins)
            if link is not None:
                break
        else:
            return None
        pieces.append(link if not pieces else link[1:])
        i = j
    return join(pieces)


def _auto_bounds(start: Pose2D, goal: Pose2D, circles: np.ndarray, pad: float) -> Bounds:
    xs = np.concatenate(([start.x, goal.x], circles[:, 0]))
    ys = np.concatenate(([start.y, goal.y], circles[:, 1]))
    reach = pad + (float(bounding_radius(circles).max()) if len(circles) else 0.0)
    return Bounds(xs.min() - reach, xs.max() + reach, ys.min() - reach, ys.max() + reach)


def end_guards(start: Pose2D, entry: Pose2D, inflate: float,
               params: PlanParams) -> np.ndarray:
    """Virtual circles that make the grid search leave the start forwards and reach the
    run-in entry from behind.

    The two minimum-radius circles tangent to a pose (slightly shrunk) plus one behind
    the start / ahead of the entry are blocked for the grid search only; radii are
    pre-compensated for the inflation.
    """
    reach = params.min_turn_radius
    radius = params.guard_scale * reach - inflate
    rows = []
    for pose, sign in ((start, -1.0), (entry, 1.0)):
        ux, uy = math.cos(pose.theta), math.sin(pose.theta)
        for ox, oy in ((-uy, ux), (uy, -ux), (sign * ux, sign * uy)):
            rows.append((pose.x + reach * ox, pose.y + reach * oy, radius))
    return circle_shapes(rows)


def _via_route(start, entry, run_in, circles, area, params, valid,
               guarded: bool) -> Optional[np.ndarray]:
    for attempt in range(params.max_retries + 1):
        inflate = (params.lateral_radius + params.clearance + params.plan_buffer
                   + attempt * params.inflate_step)
        blockers = circles if not guarded else np.vstack(
            [circles, end_guards(start, entry, inflate, params)])
        points = grid_route((start.x, start.y), (entry.x, entry.y), blockers,
                            area, inflate, params)
        if points is None:
            break
        poses = via_poses(points, start, entry)
        for allow_dubins in (False, True):
            body = stitch(poses, params, valid, allow_dubins)
            if body is not None:
                return join([body, run_in[1:]])
    return None


def plan_around(start: Pose2D, goal: Pose2D, obstacles: Sequence[Circle] = (),
                bounds: Optional[Bounds] = None,
                params: Optional[PlanParams] = None) -> Optional[np.ndarray]:
    """Smooth collision-free path as an (N, 3) array, or None.

    Same guarantees as :func:`smooth.smooth_path` plus ``clearance`` between the
    footprint and every obstacle and, with ``params.front_zone``, no obstacle in the
    front-stop zone except on the exempt run-in (:func:`geometry.zone_clear`).
    """
    params = params or PlanParams()
    circles = obstacle_array(obstacles)
    limits = shrink(bounds, params.bounds_margin)
    valid = make_validator(circles, limits, params, start=(start.x, start.y),
                           goal=(goal.x, goal.y, goal.theta))
    crowded = len(circles) > 0
    dense = smooth.dense_path(start, goal, params, valid, allow_dubins=not crowded)
    if dense is None and crowded:
        dense = _detour(start, goal, circles, limits, params, valid)
    if dense is None and crowded:   # last resort: a direct Dubins manoeuvre
        dense = smooth.dense_path(start, goal, params, valid)
    return None if dense is None else resample(dense, params.spacing)


def _detour(start, goal, circles, limits, params, valid) -> Optional[np.ndarray]:
    area = limits or _auto_bounds(start, goal, circles, params.auto_bounds_pad)
    for length in smooth.approach_lengths(params):
        entry = smooth.approach_pose(goal, length)
        run_in = straight((entry.x, entry.y), (goal.x, goal.y), goal.theta,
                          params.dense_step)
        if not valid(run_in):
            continue
        for guarded in (True, False):
            dense = _via_route(start, entry, run_in, circles, area, params,
                               valid, guarded)
            if dense is not None:
                return dense
    return None
