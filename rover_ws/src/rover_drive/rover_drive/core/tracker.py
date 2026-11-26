"""Arc-length pure pursuit."""
import math
from typing import NamedTuple

from .config import DriveConfig
from .path import Path, wrap_angle

_STRAIGHT_KAPPA = 1e-3  # [1/m] below this the rest of the path counts as straight


class Steering(NamedTuple):
    """Pure pursuit result for one control step."""

    curvature: float      # commanded curvature [1/m], + = left
    heading_error: float  # lookahead bearing minus its bearing from the path pose at s [rad]
    lookahead: float      # Ld used [m]


def lookahead_distance(cfg: DriveConfig, speed: float) -> float:
    """Speed-proportional lookahead with lower and upper bounds."""
    return min(max(cfg.lookahead_time_s * abs(speed), cfg.lookahead_min), cfg.lookahead_max)


def rotation_centre(cfg: DriveConfig, x: float, y: float, yaw: float) -> tuple:
    """Rotation centre from the tracked point (which sits ref_offset_x ahead of it)."""
    return (x - cfg.ref_offset_x * math.cos(yaw), y - cfg.ref_offset_x * math.sin(yaw))


def _to_body(cx: float, cy: float, yaw: float, px: float, py: float) -> tuple:
    dx, dy = px - cx, py - cy
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _arc_curvature(cfg: DriveConfig, alpha: float, chord: float) -> float:
    if abs(alpha) > math.pi / 2.0:
        return math.copysign(cfg.max_curvature, alpha)
    kappa = 2.0 * math.sin(alpha) / chord
    return min(max(kappa, -cfg.max_curvature), cfg.max_curvature)


def steer(cfg: DriveConfig, path: Path, s: float, x: float, y: float, yaw: float,
          speed: float) -> Steering:
    """Curvature that drives the rotation centre toward the path point Ld ahead of it."""
    ld = lookahead_distance(cfg, speed)
    # Ld is measured from the rotation centre, which sits ref_offset_x behind the tracked
    # point s, so the chord stays Ld whatever the offset.
    target = path.point(s + ld - cfg.ref_offset_x, extrapolate=True)  # run-in past the goal
    cx, cy = rotation_centre(cfg, x, y, yaw)
    bx, by = _to_body(cx, cy, yaw, *target)
    chord = max(math.hypot(bx, by), cfg.lookahead_min)
    alpha = math.atan2(by, bx)
    kappa = _arc_curvature(cfg, alpha, chord) + _runin_heading_term(cfg, path, s, yaw)
    kappa = min(max(kappa, -cfg.max_curvature), cfg.max_curvature)
    return Steering(kappa, wrap_angle(alpha - _path_bearing(cfg, path, s, target)), ld)


def _path_bearing(cfg: DriveConfig, path: Path, s: float, target: tuple) -> float:
    tangent = path.tangent(s)
    px, py = path.point(s)
    cx, cy = rotation_centre(cfg, px, py, tangent)
    bx, by = _to_body(cx, cy, tangent, *target)
    return math.atan2(by, bx)


def _runin_heading_term(cfg: DriveConfig, path: Path, s: float, yaw: float) -> float:
    if cfg.runin_heading_gain <= 0.0 or path.max_abs_curvature(s, path.length) > _STRAIGHT_KAPPA:
        return 0.0
    return -cfg.runin_heading_gain * math.sin(wrap_angle(yaw - path.tangent(path.length)))


class CrossTrackIntegral:
    """Integral of the signed cross-track error over the distance driven (|v| dt)."""

    def __init__(self, cfg: DriveConfig):
        self._gain = cfg.xtrack_integral_gain
        self._limit = cfg.xtrack_integral_limit
        self._integral = 0.0   # [m^2]

    def reset(self) -> None:
        self._integral = 0.0

    def update(self, lateral: float, speed: float, dt: float) -> None:
        """Add one step; ``lateral`` > 0 = tracked point left of the path."""
        if self._gain <= 0.0:
            return
        bound = self._limit / self._gain
        self._integral = min(max(self._integral + lateral * abs(speed) * dt, -bound), bound)

    @property
    def curvature(self) -> float:
        """Curvature to add to the pure pursuit curvature (+ = left)."""
        return -self._gain * self._integral
