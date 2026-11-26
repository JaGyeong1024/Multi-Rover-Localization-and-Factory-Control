"""Small numpy helpers shared by the planners.

Internally a path is an ``(N, 3)`` float array of ``x, y, theta`` rows; the
public API converts to and from lists of :class:`Pose2D`.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from .types import Bounds, Box, PlanParams, Pose2D

_EPS = 1e-9

Validator = Callable[[np.ndarray], bool]


def wrap_angle(angle):
    """Wrap an angle (scalar or array) into (-pi, pi]."""
    return -((np.pi - angle) % (2.0 * np.pi) - np.pi)


def to_array(path: Iterable[Pose2D]) -> np.ndarray:
    rows = [(p.x, p.y, p.theta) for p in path]
    return np.array(rows, dtype=float).reshape(-1, 3)


def to_poses(arr: np.ndarray) -> List[Pose2D]:
    theta = wrap_angle(arr[:, 2])
    return [Pose2D(float(x), float(y), float(t))
            for x, y, t in zip(arr[:, 0], arr[:, 1], theta)]


def _diff(column: np.ndarray) -> np.ndarray:
    return column[1:] - column[:-1]


def arc_lengths(arr: np.ndarray) -> np.ndarray:
    chords = np.hypot(_diff(arr[:, 0]), _diff(arr[:, 1]))
    return np.concatenate(([0.0], np.cumsum(chords)))


def curvatures(arr: np.ndarray) -> np.ndarray:
    """Signed curvature of every segment (exact for circular arcs)."""
    if len(arr) < 2:
        return np.zeros(0)
    chords = np.hypot(_diff(arr[:, 0]), _diff(arr[:, 1]))
    dtheta = wrap_angle(_diff(arr[:, 2]))
    keep = chords > _EPS
    return 2.0 * np.sin(dtheta[keep] / 2.0) / chords[keep]


def resample(arr: np.ndarray, spacing: float) -> np.ndarray:
    """Resample to uniform spacing <= ``spacing``, keeping both end poses."""
    s = arc_lengths(arr)
    keep = np.concatenate(([True], np.diff(s) > _EPS))
    arr, s = arr[keep], s[keep]
    if len(arr) < 2:
        return arr[-1:].copy()
    count = max(1, int(np.ceil(s[-1] / spacing - 1e-6)))
    query = np.linspace(0.0, s[-1], count + 1)
    theta = np.unwrap(arr[:, 2])
    out = np.column_stack([np.interp(query, s, arr[:, 0]),
                           np.interp(query, s, arr[:, 1]),
                           wrap_angle(np.interp(query, s, theta))])
    out[0], out[-1] = arr[0], arr[-1]
    return out


def straight(start_xy: Sequence[float], end_xy: Sequence[float],
             theta: float, step: float) -> np.ndarray:
    length = float(np.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))
    count = max(1, int(np.ceil(length / step)))
    t = np.linspace(0.0, 1.0, count + 1)
    return np.column_stack([start_xy[0] + t * (end_xy[0] - start_xy[0]),
                            start_xy[1] + t * (end_xy[1] - start_xy[1]),
                            np.full(count + 1, theta)])


def join(pieces: Sequence[np.ndarray]) -> np.ndarray:
    return np.vstack([p for p in pieces if len(p)])


def shrink(bounds: Optional[Bounds], margin: float) -> Optional[Bounds]:
    if bounds is None:
        return None
    return Bounds(bounds.xmin + margin, bounds.xmax - margin,
                  bounds.ymin + margin, bounds.ymax - margin)


def include(bounds: Optional[Bounds], xy: Optional[Sequence[float]]) -> Optional[Bounds]:
    """Bounds grown just enough to contain ``xy`` (a start slightly outside)."""
    if bounds is None or xy is None:
        return bounds
    return Bounds(min(bounds.xmin, xy[0]), max(bounds.xmax, xy[0]),
                  min(bounds.ymin, xy[1]), max(bounds.ymax, xy[1]))


def inside(arr: np.ndarray, bounds: Optional[Bounds]) -> bool:
    if bounds is None:
        return True
    x, y = arr[:, 0], arr[:, 1]
    return bool(x.min() >= bounds.xmin - _EPS and x.max() <= bounds.xmax + _EPS
                and y.min() >= bounds.ymin - _EPS
                and y.max() <= bounds.ymax + _EPS)


# ---------------------------------------------------------------- shapes
# Every obstacle and body is a padded oriented rectangle, one row
# ``cx, cy, yaw, half_len, half_wid, pad``. A circle is a zero-size rectangle with
# pad = radius, a disc body likewise.
SHAPE_COLS = 6


def obstacle_array(obstacles: Iterable[object]) -> np.ndarray:
    """Obstacles (:class:`Circle` / :class:`Box`) as an (M, 6) shape array."""
    rows = []
    for o in obstacles:
        if isinstance(o, Box):
            rows.append((o.x, o.y, o.yaw, o.width / 2.0, o.height / 2.0, o.pad))
        else:
            rows.append((o.x, o.y, 0.0, 0.0, 0.0, o.r))
    return np.array(rows, dtype=float).reshape(-1, SHAPE_COLS)


def point_shapes(points: np.ndarray) -> np.ndarray:
    out = np.zeros((len(points), SHAPE_COLS))
    out[:, :2] = points[:, :2]
    return out


def circle_shapes(rows: Iterable[Sequence[float]]) -> np.ndarray:
    rows = np.array(list(rows), dtype=float).reshape(-1, 3)
    out = point_shapes(rows)
    out[:, 5] = rows[:, 2]
    return out


def body_shapes(arr: np.ndarray, params: PlanParams) -> np.ndarray:
    if params.footprint is None:
        out = point_shapes(arr)
        out[:, 5] = params.robot_radius
        return out
    front, rear, half_wid = params.footprint
    shift = (front - rear) / 2.0
    out = np.empty((len(arr), SHAPE_COLS))
    out[:, 0] = arr[:, 0] + shift * np.cos(arr[:, 2])
    out[:, 1] = arr[:, 1] + shift * np.sin(arr[:, 2])
    out[:, 2] = arr[:, 2]
    out[:, 3] = (front + rear) / 2.0
    out[:, 4] = half_wid
    out[:, 5] = 0.0
    return out


def bounding_radius(shapes: np.ndarray) -> np.ndarray:
    return np.hypot(shapes[:, 3], shapes[:, 4]) + shapes[:, 5]


# Corner signs along / across the heading, counter-clockwise; and each corner's successor.
_CORNER_U = np.array([1.0, -1.0, -1.0, 1.0])
_CORNER_V = np.array([1.0, 1.0, -1.0, -1.0])
_NEXT_CORNER = np.array([1, 2, 3, 0])

# The rectangle tests below batch corners, axes and both directions into single numpy
# calls to cut per-call overhead.


def _corners(s: np.ndarray, c: np.ndarray, sn: np.ndarray):
    along = _CORNER_U * s[:, 3:4]       # (K, 4): +-half_len
    across = _CORNER_V * s[:, 4:5]      # (K, 4): +-half_wid
    x = s[:, 0:1] + along * c[:, None] + across * (-sn)[:, None]
    y = s[:, 1:2] + along * sn[:, None] + across * c[:, None]
    return x, y


def _vertex_edge(px: np.ndarray, py: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    abx = (qx[:, _NEXT_CORNER] - qx)[:, None, :]
    aby = (qy[:, _NEXT_CORNER] - qy)[:, None, :]
    apx = px[:, :, None] - qx[:, None, :]
    apy = py[:, :, None] - qy[:, None, :]
    denom = np.maximum(abx * abx + aby * aby, _EPS)
    t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    return np.hypot(apx - t * abx, apy - t * aby).min(axis=(1, 2))


def _trig(s: np.ndarray):
    turned = s[:, 2] + np.pi / 2.0
    return np.cos(s[:, 2]), np.sin(s[:, 2]), np.cos(turned), np.sin(turned)


def _rect_distance(a: np.ndarray, b: np.ndarray, trig_a=None, trig_b=None) -> np.ndarray:
    """(K,) signed distance between paired rectangles (pads not included).

    Separated: Euclidean corner-to-edge distance. Overlapping: minus the smallest
    overlap along the four edge normals (separating-axis test).
    """
    ca, sa, ca90, sa90 = _trig(a) if trig_a is None else trig_a
    cb, sb, cb90, sb90 = _trig(b) if trig_b is None else trig_b
    ax, ay = _corners(a, ca, sa)
    bx, by = _corners(b, cb, sb)
    k = len(a)
    # Both directions in one call: corners of a to edges of b, then b to a.
    both = _vertex_edge(np.concatenate([ax, bx]), np.concatenate([ay, by]),
                        np.concatenate([bx, ax]), np.concatenate([by, ay]))
    dist = np.minimum(both[:k], both[k:])
    # The four edge normals as the columns of (K, 4).
    axis_x = np.array([ca, ca90, cb, cb90]).T
    axis_y = np.array([sa, sa90, sb, sb90]).T
    reach = _half_extent(a, ca, sa, axis_x, axis_y) + _half_extent(b, cb, sb, axis_x, axis_y)
    dx, dy = b[:, 0:1] - a[:, 0:1], b[:, 1:2] - a[:, 1:2]
    sep = np.abs(dx * axis_x + dy * axis_y) - reach
    overlap = (sep < 0.0).all(axis=1)
    depth = (-sep).min(axis=1)
    return np.where(overlap, -depth, dist)


def _half_extent(s, c, sn, axis_x, axis_y) -> np.ndarray:
    c, sn = c[:, None], sn[:, None]
    along = np.abs(axis_x * c + axis_y * sn)
    across = np.abs(-axis_x * sn + axis_y * c)
    return s[:, 3:4] * along + s[:, 4:5] * across


def shape_gaps(a: np.ndarray, b: np.ndarray, near: float = np.inf) -> np.ndarray:
    """(N, M) free gap between every shape of ``a`` and of ``b`` (negative = overlap).

    Exact below ``near``; pairs whose bounding circles are ``near`` apart keep that
    lower bound (broad phase).
    """
    delta = a[:, None, :2] - b[None, :, :2]
    gap = np.hypot(delta[..., 0], delta[..., 1]) \
        - bounding_radius(a)[:, None] - bounding_radius(b)[None, :]
    a_sized = (a[:, 3] + a[:, 4]) > 0.0
    b_sized = (b[:, 3] + b[:, 4]) > 0.0
    i, j = np.nonzero((a_sized[:, None] | b_sized[None, :]) & (gap < near))
    if len(i):
        both = a_sized[i] & b_sized[j]
        exact = np.empty(len(i))
        if both.any():
            # cos / sin per shape, gathered per pair
            ii, jj = i[both], j[both]
            trig_a = [t[ii] for t in _trig(a)]
            trig_b = [t[jj] for t in _trig(b)]
            exact[both] = _rect_distance(a[ii], b[jj], trig_a, trig_b)
        one = ~both
        if one.any():          # a point against a rectangle (either way round)
            ai, bj = a[i[one]], b[j[one]]
            flip = ~a_sized[i[one]]
            rect = np.where(flip[:, None], bj, ai)
            point = np.where(flip[:, None], ai, bj)
            exact[one] = _point_rect_distance(point[:, :2], rect)
        gap[i, j] = exact - a[i, 5] - b[j, 5]
    return gap


def _point_rect_distance(points: np.ndarray, rect: np.ndarray) -> np.ndarray:
    c, sn = np.cos(rect[:, 2]), np.sin(rect[:, 2])
    dx, dy = points[:, 0] - rect[:, 0], points[:, 1] - rect[:, 1]
    ex = np.abs(c * dx + sn * dy) - rect[:, 3]
    ey = np.abs(-sn * dx + c * dy) - rect[:, 4]
    outside = np.hypot(np.maximum(ex, 0.0), np.maximum(ey, 0.0))
    return np.where((ex < 0.0) & (ey < 0.0), np.maximum(ex, ey), outside)


def clearance(points: np.ndarray, obstacles: np.ndarray) -> float:
    """Smallest distance from any point to any obstacle surface (inf if none)."""
    if len(obstacles) == 0 or len(points) == 0:
        return float('inf')
    return float(shape_gaps(point_shapes(points), obstacles).min())


def clear_of(dist: np.ndarray, need: float, escape: bool = False) -> bool:
    """True when every gap of an (N poses, M obstacles) matrix keeps ``need``.

    ``escape`` (row 0 is the real start): an obstacle already closer than ``need`` may
    stay so until the path first reaches ``need`` from it, never closer than the start.
    """
    if dist.size == 0:
        return True
    floor = np.full(dist.shape, need)
    if escape:
        for j in np.flatnonzero(dist[0] < need):
            reached = dist[:, j] >= need
            first = int(np.argmax(reached)) if reached.any() else len(dist)
            floor[:first, j] = dist[0, j]
    return bool((dist >= floor - 1e-9).all())


def body_gaps(arr: np.ndarray, obstacles: np.ndarray, params: PlanParams) -> np.ndarray:
    return shape_gaps(body_shapes(arr, params), obstacles, near=params.clearance + 1e-6)


def zone_shapes(arr: np.ndarray, params: PlanParams) -> np.ndarray:
    start, length, half_wid = params.front_zone
    centre = start + length / 2.0
    out = np.empty((len(arr), SHAPE_COLS))
    out[:, 0] = arr[:, 0] + centre * np.cos(arr[:, 2])
    out[:, 1] = arr[:, 1] + centre * np.sin(arr[:, 2])
    out[:, 2] = arr[:, 2]
    out[:, 3] = length / 2.0
    out[:, 4] = half_wid
    out[:, 5] = 0.0
    return out


def runin_exempt(arr: np.ndarray, goal: Optional[Sequence[float]],
                 params: PlanParams) -> np.ndarray:
    """(N,) mask of poses on the goal line and heading, at most ``front_zone_runin``
    before ``goal`` (x, y, theta): exempt from the front stop."""
    if goal is None or params.front_zone_runin <= 0.0:
        return np.zeros(len(arr), dtype=bool)
    c, s = np.cos(goal[2]), np.sin(goal[2])
    dx, dy = arr[:, 0] - goal[0], arr[:, 1] - goal[1]
    before = -(c * dx + s * dy)
    lateral = np.abs(-s * dx + c * dy)
    turn = np.abs(wrap_angle(arr[:, 2] - goal[2]))
    return ((before >= -_EPS) & (before <= params.front_zone_runin + _EPS)
            & (lateral <= params.on_line_tol) & (turn <= params.heading_tol))


def zone_clear(arr: np.ndarray, obstacles: np.ndarray, params: PlanParams,
               goal: Optional[Sequence[float]] = None,
               seen: Optional[np.ndarray] = None) -> bool:
    """True when no obstacle overlaps the front-stop zone at any pose of ``arr`` (except
    the run-in to ``goal``, :func:`runin_exempt`). ``seen`` (M,) masks the obstacles
    that count (None = all). No escape rule."""
    if params.front_zone is None or len(arr) == 0:
        return True
    if seen is not None:
        obstacles = obstacles[seen]
    if len(obstacles) == 0:
        return True
    rows = ~runin_exempt(arr, goal, params)
    if not rows.any():
        return True
    gaps = shape_gaps(zone_shapes(arr[rows], params), obstacles, near=1e-6)
    return bool((gaps >= -1e-9).all())


def zone_mask(obstacles: Sequence[object], ignore: Sequence[object]) -> Optional[np.ndarray]:
    """(M,) mask of ``obstacles`` that count for the front-stop zone: all but those in
    ``ignore`` (None when nothing is ignored)."""
    if not ignore:
        return None
    return np.array([o not in ignore for o in obstacles], dtype=bool)


def _same_xy(row: np.ndarray, xy: Sequence[float]) -> bool:
    for a, b in ((float(row[0]), float(xy[0])), (float(row[1]), float(xy[1]))):
        if not (a == b or abs(a - b) <= 1e-9):
            return False
    return True


ISSUED_CURVATURE_SLACK = 1e-3  # relative; resampling to ``spacing`` can read a
                                # minimum-radius arc slightly tighter (~0.03 %)


def make_validator(obstacles: np.ndarray, bounds: Optional[Bounds],
                   params: PlanParams,
                   start: Optional[Sequence[float]] = None,
                   curvature_slack: float = 0.0,
                   goal: Optional[Sequence[float]] = None,
                   seen: Optional[np.ndarray] = None) -> Validator:
    """Predicate: path stays in bounds, footprint keeps ``clearance``, front-stop zone
    free of every obstacle in ``seen``, curvature under the limit.

    ``bounds`` must already be shrunk by the caller. ``start`` (x, y) enables the
    escape-start rule (:func:`clear_of`) and widens ``bounds`` to it. ``goal``
    (x, y, theta) places the run-in exempt from the zone. ``curvature_slack``
    (relative) is for judging an already issued path.
    """
    bounds = include(bounds, start)
    limit = params.max_curvature * (1.0 + 1e-6 + curvature_slack) + 1e-9
    need = params.clearance - 1e-9

    def valid(arr: np.ndarray) -> bool:
        # Cheap tests (bounds, curvature) first; the obstacle tests cost far more.
        if not inside(arr, bounds):
            return False
        kappa = curvatures(arr)
        if len(kappa) and not np.abs(kappa).max() <= limit:
            return False
        if len(obstacles) and len(arr):
            escape = start is not None and _same_xy(arr[0], start)
            if not clear_of(body_gaps(arr, obstacles, params), need, escape):
                return False
            if not zone_clear(arr, obstacles, params, goal, seen):
                return False
        return True

    # Cache for smooth.memoized; lives as long as this validator.
    valid.memo = {}
    return valid
