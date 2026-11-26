"""Square up to a fixture face with the depth scan (opt-in ``face_align``)."""
import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np

from .config import DriveConfig
from .depth_front import DepthScan
from .front_memory import measured_beams

MAX_RMS_M = 0.02          # residual rms of a flat face; above it the points are not one face
# Widest lateral gap between neighbouring points of one face [m] (~5 beams at 0.9 m).
MAX_GAP_M = 0.06
# Outlier rejection: points farther than this many rms from the first line, but never
# nearer than the floor (depth noise ~1 cm).
_REJECT_RMS = 2.5
_REJECT_FLOOR_M = 0.01
# A gyro gap longer than this [s] is not integrated, and a gyro silent this long is lost.
GYRO_TIMEOUT_S = 0.5

# FaceFit.reason
FIT_OK = ''
FEW_POINTS = 'FEW_POINTS'
BAD_FIT = 'BAD_FIT'
WEAK_FIT = 'WEAK_FIT'
TOO_SKEWED = 'TOO_SKEWED'


class FaceFit(NamedTuple):
    """One measurement of the face ahead."""

    reason: str        # FIT_OK, or why this is no measurement
    phi: float         # rad, rover yaw from the face normal (CCW +); the rover turns by -phi
    dist: float        # m, face ahead of the camera on its axis (y = 0)
    rms: float         # m, residual of the points used
    points: int        # points used (after rejection), or found in the window

    @property
    def ok(self) -> bool:
        return self.reason == FIT_OK

    def describe(self) -> str:
        if self.reason == FEW_POINTS:
            return f'{self.points} points in the window'
        face = f'face {self.dist:.2f} m' if math.isfinite(self.dist) else 'face not in view'
        return (f'phi {math.degrees(self.phi):+.1f} deg, {face}, '
                f'rms {self.rms * 1000.0:.1f} mm, {self.points} points')


def face_points(scan: DepthScan, cfg: DriveConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Measured scan points (x, y) inside the range and window."""
    r = np.asarray(scan.ranges, dtype=float)
    angle = scan.angle_min + scan.angle_increment * np.arange(len(r))
    measured = measured_beams(r, scan.range_min)
    x, y = r * np.cos(angle), r * np.sin(angle)
    near, far = cfg.face_align_range
    keep = measured & (x >= near) & (x <= far) & (np.abs(y) <= cfg.face_align_window)
    return x[keep], y[keep]


def _line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    design = np.column_stack((np.ones_like(y), y))
    (a, b), *_ = np.linalg.lstsq(design, x, rcond=None)
    return float(a), float(b), x - a - b * y


def _rms(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(residuals * residuals)))


def fit_face(scan: DepthScan, cfg: DriveConfig) -> FaceFit:
    """The face ahead from one depth scan."""
    x, y = face_points(scan, cfg)
    if len(x) < cfg.face_align_min_points:
        return FaceFit(FEW_POINTS, math.nan, math.nan, math.nan, len(x))
    _, _, residuals = _line(x, y)
    keep = np.abs(residuals) <= max(_REJECT_RMS * _rms(residuals), _REJECT_FLOOR_M)
    x, y = x[keep], y[keep]
    if len(x) < cfg.face_align_min_points:
        return FaceFit(FEW_POINTS, math.nan, math.nan, math.nan, len(x))
    a, b, residuals = _line(x, y)
    rms = _rms(residuals)
    fit = FaceFit(FIT_OK, math.atan(b), a, rms, len(x))
    spread = float(np.sum((y - y.mean()) ** 2))
    # standard error of the slope as an angle
    phi_error = rms / math.sqrt(spread) / (1.0 + b * b) if spread > 0.0 else math.inf
    if rms > MAX_RMS_M or np.max(np.diff(np.sort(y))) > MAX_GAP_M:
        return fit._replace(reason=BAD_FIT)
    if phi_error > cfg.face_align_tol_rad:
        return fit._replace(reason=WEAK_FIT)
    if abs(fit.phi) > cfg.face_align_max_rad:
        return fit._replace(reason=TOO_SKEWED)
    return fit


class GyroYaw:
    """Yaw integrated from the gyro z rate (differences only). A gap longer than
    ``GYRO_TIMEOUT_S`` (or a stamp going back) is not integrated and counts in ``gaps``."""

    def __init__(self):
        self.yaw = 0.0
        self.gaps = 0
        self._last: Optional[Tuple[float, float]] = None   # (stamp, rate)
        self._heard: Optional[float] = None                # receive time of the last sample

    def add(self, rate: float, stamp: float, now: float) -> None:
        if not (math.isfinite(rate) and math.isfinite(stamp)):
            return
        if self._last is not None:
            dt = stamp - self._last[0]
            if 0.0 < dt <= GYRO_TIMEOUT_S:
                self.yaw += 0.5 * (rate + self._last[1]) * dt
            else:
                self.gaps += 1
        self._last, self._heard = (stamp, rate), now

    def fresh(self, now: float) -> bool:
        return self._heard is not None and now - self._heard <= GYRO_TIMEOUT_S


class RememberedFace(NamedTuple):
    """A good run-in fit with the gyro yaw then (the face may be too near to see at the
    goal; phi now = phi then + gyro yaw change)."""

    fit: FaceFit
    yaw: float         # GyroYaw.yaw at the fit
    gaps: int          # GyroYaw.gaps at the fit
    stamp: float       # receive time of the scan

    def carried(self, gyro: GyroYaw, now: float, max_age: float) -> Optional[FaceFit]:
        """The fit moved to the current yaw; None if too old or the gyro lost track."""
        if now - self.stamp > max_age or gyro.gaps != self.gaps:
            return None
        return self.fit._replace(phi=self.fit.phi + gyro.yaw - self.yaw, dist=math.nan)


@dataclass
class AlignRun:
    """Progress of one squaring-up at a goal."""

    start: float                    # when the position arrival was latched
    measure_after: float            # only a scan received after this measures the face
    turns: int = 0
    target: Optional[float] = None  # gyro yaw being turned to; None = measuring
    turn_from: float = 0.0          # gyro yaw where the current turn started
    origin: float = 0.0             # gyro yaw at the start of the run
