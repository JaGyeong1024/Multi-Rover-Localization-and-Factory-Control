"""Arc-length parameterised polyline path."""
import math
from typing import NamedTuple, Optional

import numpy as np

_MIN_SEGMENT = 1e-6  # numerical guard: shorter segments are merged


class Projection(NamedTuple):
    """Result of projecting a point on the path."""

    s: float        # arc length of the foot point
    lateral: float  # signed cross-track error, + = point is left of the path
    dist: float     # distance to the foot point (>= |lateral| past the ends)


class Segments(NamedTuple):
    """A stretch of the path as straight pieces, in order (all arrays of the same length)."""

    start: np.ndarray   # (m, 2) first point of each piece
    unit: np.ndarray    # (m, 2) direction of travel
    length: np.ndarray  # (m,) [m]
    s: np.ndarray       # (m,) arc length at each start


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def _dedupe(pts: np.ndarray) -> np.ndarray:
    keep = [0]
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - pts[keep[-1]])) > _MIN_SEGMENT:
            keep.append(i)
    return pts[keep]


def _vertex_curvature(yaw: np.ndarray, seg_len: np.ndarray) -> np.ndarray:
    kappa = np.zeros(len(seg_len) + 1)
    if len(seg_len) < 2:
        return kappa
    dyaw = np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw)))
    kappa[1:-1] = dyaw / (0.5 * (seg_len[:-1] + seg_len[1:]))
    return kappa


class Path:
    """Polyline with cumulative arc length, tangent and curvature lookup."""

    def __init__(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            raise ValueError('path needs at least one point')
        self._pts = _dedupe(pts)
        seg = np.diff(self._pts, axis=0)
        self._seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self._unit = seg / self._seg_len[:, None] if len(seg) else np.zeros((0, 2))
        self._s = np.concatenate(([0.0], np.cumsum(self._seg_len)))
        self._yaw = np.arctan2(seg[:, 1], seg[:, 0])
        self._kappa = _vertex_curvature(self._yaw, self._seg_len)

    @property
    def length(self) -> float:
        return float(self._s[-1])

    @property
    def end(self) -> tuple:
        return float(self._pts[-1, 0]), float(self._pts[-1, 1])

    def _segment_index(self, s: float) -> int:
        idx = int(np.searchsorted(self._s, s, side='right')) - 1
        return min(max(idx, 0), len(self._seg_len) - 1)

    def point(self, s: float, extrapolate: bool = False) -> tuple:
        """Point at arc length s. Past the ends: clamp, or extend along the end tangent."""
        if len(self._seg_len) == 0:
            return self.end
        if not extrapolate:
            s = min(max(s, 0.0), self.length)
        i = self._segment_index(s)
        p = self._pts[i] + (s - self._s[i]) * self._unit[i]
        return float(p[0]), float(p[1])

    def tangent(self, s: float) -> float:
        """Direction of travel (yaw) at arc length s."""
        if len(self._seg_len) == 0:
            return 0.0
        return float(self._yaw[self._segment_index(s)])

    def curvature(self, s: float) -> float:
        """Signed curvature at arc length s (+ = turning left)."""
        return float(np.interp(s, self._s, self._kappa))

    def max_abs_curvature(self, s0: float, s1: float) -> float:
        """Largest |curvature| on [s0, s1]."""
        inside = np.abs(self._kappa[(self._s >= s0) & (self._s <= s1)])
        ends = max(abs(self.curvature(s0)), abs(self.curvature(s1)))
        return max(ends, float(inside.max()) if inside.size else 0.0)

    def segments(self, s_min: float, s_max: float) -> Segments:
        """The path between ``s_min`` and ``s_max`` (clamped) as straight pieces, the first
        and last cut to the range."""
        lo = min(max(s_min, 0.0), self.length)
        hi = min(max(s_max, lo), self.length)
        i0, i1 = self._segment_index(lo), self._segment_index(hi)
        seg_s = self._s[i0:i1 + 1]
        begin = np.maximum(seg_s, lo)
        length = np.minimum(seg_s + self._seg_len[i0:i1 + 1], hi) - begin
        unit = self._unit[i0:i1 + 1]
        start = self._pts[i0:i1 + 1] + (begin - seg_s)[:, None] * unit
        keep = length > _MIN_SEGMENT
        return Segments(start[keep], unit[keep], length[keep], begin[keep])

    def straight_after(self, s: float, kappa_eps: float) -> bool:
        """No heading change at any vertex beyond arc length s (the rest is one line)."""
        return bool(np.all(np.abs(self._kappa[self._s > s]) <= kappa_eps))

    def project(self, x: float, y: float, s_min: Optional[float] = None,
                s_max: Optional[float] = None, tie_margin: float = 0.0) -> Projection:
        """Closest path point within [s_min, s_max] (whole path by default)."""
        p = np.array([x, y], dtype=float)
        if len(self._seg_len) == 0:
            return Projection(0.0, 0.0, float(np.hypot(*(p - self._pts[0]))))
        lo = 0.0 if s_min is None else min(max(s_min, 0.0), self.length)
        hi = self.length if s_max is None else min(max(s_max, lo), self.length)
        i0, i1 = self._segment_index(lo), self._segment_index(hi)
        start = self._pts[i0:i1 + 1]
        unit = self._unit[i0:i1 + 1]
        along = np.einsum('ij,ij->i', p - start, unit)
        along = np.clip(along, 0.0, self._seg_len[i0:i1 + 1])
        s_foot = np.clip(self._s[i0:i1 + 1] + along, lo, hi)
        foot = start + (s_foot - self._s[i0:i1 + 1])[:, None] * unit
        delta = p - foot
        dist = np.hypot(delta[:, 0], delta[:, 1])
        j = int(np.argmax(dist <= dist.min() + tie_margin))
        lateral = float(unit[j, 0] * delta[j, 1] - unit[j, 1] * delta[j, 0])
        return Projection(float(s_foot[j]), lateral, float(dist[j]))


class PathProgress:
    """Monotonic projection: progress along the path never moves backwards."""

    def __init__(self, path: Path, window: float, tie_margin: float = 0.0):
        self._path = path
        self._window = window
        self._tie_margin = tie_margin
        self._s: Optional[float] = None

    @property
    def s(self) -> float:
        return 0.0 if self._s is None else self._s

    def update(self, x: float, y: float) -> Projection:
        if self._s is None:
            proj = self._path.project(x, y, tie_margin=self._tie_margin)
        else:
            proj = self._path.project(x, y, self._s, self._s + self._window)
        self._s = proj.s
        return proj
