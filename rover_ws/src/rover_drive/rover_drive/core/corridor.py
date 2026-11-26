"""Front stop zone along the remaining path (gate option ``front_path``; pure numpy, no ROS)."""
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .path import Path, PathProgress, Projection, wrap_angle

# Rover farther than this from the path [m] is not following it.
XTRACK_TOL = 0.10


def _direction(path: Path, s: float, fallback: float) -> float:
    return path.tangent(s) if path.length > 0.0 else fallback


class PathCorridor:
    """Stop zone of one path command, with the rover's progress on it."""

    def __init__(self, poses: Sequence[tuple], half_width: float, front_offset: float,
                 window: float, tie_margin: float):
        xy = np.array([(p[0], p[1]) for p in poses], dtype=float)
        base = Path(xy)
        goal_yaw = float(poses[-1][2])
        start_yaw = _direction(base, 0.0, goal_yaw)
        end_yaw = _direction(base, base.length, goal_yaw)
        before = xy[0] - XTRACK_TOL * np.array([math.cos(start_yaw), math.sin(start_yaw)])
        after = xy[-1] + front_offset * np.array([math.cos(end_yaw), math.sin(end_yaw)])
        self._path = Path(np.vstack((before, xy, after)))
        self._end_yaw = end_yaw
        self._half_width = half_width
        self._progress = PathProgress(self._path, window, tie_margin)
        self._proj: Optional[Projection] = None

    def update(self, x: float, y: float) -> None:
        """New own position: advance the progress (monotonic, as the tracker's)."""
        first = self._proj is None
        self._proj = self._progress.update(x, y)
        if first:
            self._proj = self._progress.update(x, y)

    def follows(self, yaw: float, heading_tol: float) -> bool:
        """On the path: within ``XTRACK_TOL`` of the progress point, heading within ``heading_tol``."""
        proj = self._proj
        if proj is None or proj.dist > XTRACK_TOL:
            return False
        tangent = _direction(self._path, proj.s, self._end_yaw)
        return abs(wrap_angle(yaw - tangent)) <= heading_tol

    @property
    def progress(self) -> float:
        """The rover's arc length on the (extended) path."""
        return self._progress.s

    def nearest(self, pose: tuple, bx, by, reach: float) -> Optional[Tuple[float, int]]:
        """(distance along the path, index) of the first body-frame point reached inside
        the corridor within ``reach``; None = none."""
        along = self.first_cover(pose, bx, by, reach) - self._progress.s
        if not along.size:
            return None
        i = int(np.argmin(along))
        return (float(along[i]), i) if math.isfinite(along[i]) else None

    def first_cover(self, pose: tuple, bx, by, reach: float) -> np.ndarray:
        """Per body-frame point: arc length where the corridor first covers it (inf = not
        inside, or beyond ``reach``). Depends on the global position only."""
        bx, by = np.asarray(bx, dtype=float), np.asarray(by, dtype=float)
        cover = np.full(bx.shape, np.inf)
        s0 = self._progress.s
        seg = self._path.segments(s0, s0 + reach)
        if not len(seg.length):
            return cover
        near = np.flatnonzero(bx * bx + by * by <= self._bound(pose, s0, seg) ** 2)
        if near.size:
            c, s = math.cos(pose[2]), math.sin(pose[2])
            cover[near] = self._cover(pose[0] + c * bx[near] - s * by[near],
                                      pose[1] + s * bx[near] + c * by[near], seg)
        return cover

    def _bound(self, pose: tuple, s0: float, seg) -> float:
        px, py = self._path.point(s0)
        length = seg.s[-1] + seg.length[-1] - s0
        return length + self._half_width + math.hypot(pose[0] - px, pose[1] - py)

    def _cover(self, gx: np.ndarray, gy: np.ndarray, seg) -> np.ndarray:
        ux, uy, length = seg.unit[:, 0], seg.unit[:, 1], seg.length
        rel_x = gx[:, None] - seg.start[:, 0]
        rel_y = gy[:, None] - seg.start[:, 1]
        t = rel_x * ux + rel_y * uy
        lateral = rel_y * ux - rel_x * uy
        hw2 = self._half_width ** 2
        rect = (t > 0.0) & (t <= length) & (lateral * lateral <= hw2)
        cover = np.where(rect, seg.s + t, np.inf).min(axis=1)
        if len(length) > 1:
            wedge = (t[:, :-1] > length[:-1]) & (t[:, 1:] < 0.0) \
                & (rel_x[:, 1:] ** 2 + rel_y[:, 1:] ** 2 <= hw2)
            cover = np.minimum(cover, np.where(wedge, seg.s[1:], np.inf).min(axis=1))
        return cover
