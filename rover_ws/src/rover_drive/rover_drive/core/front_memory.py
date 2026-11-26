"""Memory of near obstacle points for the front stop (pure numpy, no ROS)."""
import math
from typing import Optional, Tuple

import numpy as np

# Grid that de-duplicates the remembered points [m] (finer than depth noise carries no info).
CELL_M = 0.02
# A beam clears a point only when it measures this much beyond it [m]. Kept small: a larger
# margin keeps the nearest noise draws of every earlier scan and the union drifts toward
# the rover.
CLEAR_MARGIN_M = CELL_M


def measured_beams(ranges, range_min: float) -> np.ndarray:
    """Mask of the beams that are measurements: finite, > 0 and at least ``range_min``."""
    r = np.asarray(ranges, dtype=float)
    return np.isfinite(r) & (r > 0.0) & (r >= range_min)


class FrontMemory:
    """Remembered near points, body frame (x forward, y left), at most one per grid cell."""

    def __init__(self, near_m: float):
        self._near = near_m
        self._pts = np.zeros((0, 2))

    def __len__(self) -> int:
        return len(self._pts)

    def points(self) -> np.ndarray:
        """Remembered points, body frame (n, 2)."""
        return self._pts.copy()

    def move(self, dx: float, dy: float, dyaw: float) -> None:
        """Re-express the points after the rover moved (dx, dy) in its previous body frame
        and turned by ``dyaw``."""
        if not len(self._pts) or (dx == 0.0 and dy == 0.0 and dyaw == 0.0):
            return
        c, s = math.cos(dyaw), math.sin(dyaw)
        px, py = self._pts[:, 0] - dx, self._pts[:, 1] - dy
        self._pts = np.column_stack((c * px + s * py, -s * px + c * py))

    def update(self, ranges, angle_min: float, angle_increment: float,
               range_min: float) -> None:
        """One scan: forget what it shows free or what is behind, then add its near points."""
        r = np.asarray(ranges, dtype=float)
        measured = measured_beams(r, range_min)
        if len(self._pts):
            self._pts = self._pts[~self._shown_free(r, measured, angle_min, angle_increment,
                                                    range_min)]
        angle = angle_min + angle_increment * np.arange(len(r))
        x, y = r * np.cos(angle), r * np.sin(angle)
        near = measured & (x > 0.0) & (x < self._near)
        if near.any():
            pts = np.vstack((np.column_stack((x[near], y[near])), self._pts))
            _, first = np.unique(np.floor(pts / CELL_M).astype(np.int64), axis=0,
                                 return_index=True)
            self._pts = pts[np.sort(first)]      # one per cell, the newest measurement wins

    def _shown_free(self, r: np.ndarray, measured: np.ndarray, angle_min: float,
                    angle_increment: float, range_min: float) -> np.ndarray:
        bx, by = self._pts[:, 0], self._pts[:, 1]
        behind = bx <= 0.0
        dist = np.hypot(bx, by)
        beam = np.rint((np.arctan2(by, bx) - angle_min) / angle_increment).astype(np.int64)
        visible = ~behind & (beam >= 0) & (beam < len(r)) & (dist >= range_min)
        idx = np.clip(beam, 0, max(len(r) - 1, 0))
        beyond = visible & measured[idx] & (r[idx] > dist + CLEAR_MARGIN_M)
        return behind | beyond

    def nearest_ahead(self, half_width: float,
                      half_angle: float) -> Optional[Tuple[float, float]]:
        """Nearest remembered point ahead in the corridor |y| <= ``half_width``."""
        if not len(self._pts):
            return None
        bx, by = self._pts[:, 0], self._pts[:, 1]
        ahead = bx > 0.0
        if half_width > 0.0:
            ahead &= np.abs(by) <= half_width
        else:
            ahead &= np.abs(np.arctan2(by, bx)) <= half_angle
        if not ahead.any():
            return None
        i = int(np.argmin(np.where(ahead, bx, np.inf)))
        return float(bx[i]), float(by[i])
