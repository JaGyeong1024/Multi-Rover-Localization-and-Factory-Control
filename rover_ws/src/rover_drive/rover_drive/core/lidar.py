"""LaserScan -> nearest obstacle in front of the rover (pure numpy, no ROS)."""
import math
from typing import NamedTuple, Optional, Sequence

import numpy as np

from .config import GateConfig

_TWO_PI = 2.0 * math.pi


class FrontObstacle(NamedTuple):
    """Nearest scan point in front of the rover, in the sensor/body frame."""

    dist: float       # forward distance [m]; inf = nothing ahead; NaN = blind (too few beams)
    x: float          # the blocking point (NaN when there is none)
    y: float
    valid_beams: int  # measurements inside the forward sector


class ScanPoints(NamedTuple):
    """Scan beams as body-frame points; ``measured`` marks the usable ones."""

    x: np.ndarray
    y: np.ndarray
    angle: np.ndarray
    measured: np.ndarray


class LeadGap(NamedTuple):
    """Free gap to the lead rover seen in one scan."""

    gap: float   # my front edge to the lead's nearest point [m]; NaN = not measured
    points: int  # scan points attributed to the lead rover


NO_LEAD_GAP = LeadGap(math.nan, 0)


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def in_sectors(angle: np.ndarray, sectors: Sequence[float]) -> np.ndarray:
    """Mask of angles inside any [from, to] pair, counter-clockwise from ``from``
    (pairs may cross +-pi; from == to is empty)."""
    mask = np.zeros(angle.shape, dtype=bool)
    for start, end in zip(sectors[0::2], sectors[1::2]):
        width = (end - start) % _TWO_PI
        mask |= ((angle - start) % _TWO_PI) < width
    return mask


def scan_points(ranges, angle_min: float, angle_increment: float, range_min: float,
                cfg: GateConfig) -> ScanPoints:
    """LaserScan -> body-frame points (non-measurements and ignored sectors masked)."""
    r = np.asarray(ranges, dtype=float)
    angle = _wrap(angle_min + angle_increment * np.arange(len(r)) + cfg.lidar_yaw_offset)
    measured = np.isfinite(r) & (r >= max(cfg.lidar_min_range, range_min)) & (r > 0.0)
    measured &= ~in_sectors(angle, cfg.lidar_ignore_sectors)
    r = np.where(measured, r, 0.0)
    return ScanPoints(r * np.cos(angle), r * np.sin(angle), angle, measured)


def lead_ahead(lead_xy: tuple, cfg: GateConfig) -> bool:
    """True when the lead (body frame) is ahead: positive x and bearing inside the front
    sector. A lead beside or behind me has no gap along my heading (a "gap" to it would be
    bogus) and is an ordinary obstacle instead."""
    x, y = lead_xy
    return x > 0.0 and abs(math.atan2(y, x)) <= cfg.lidar_front_half_angle


def _near_lead(x: np.ndarray, y: np.ndarray, lead_xy: Optional[tuple],
               radius: float) -> np.ndarray:
    if lead_xy is None:
        return np.zeros(x.shape, dtype=bool)
    return np.hypot(x - lead_xy[0], y - lead_xy[1]) <= radius


def obstacle_points(pts: ScanPoints, cfg: GateConfig, lead_xy: Optional[tuple]) -> np.ndarray:
    """Mask of the measured points not within ``lidar_lead_radius`` of the lead (body frame)."""
    return pts.measured & ~_near_lead(pts.x, pts.y, lead_xy, cfg.lidar_lead_radius)


def front_obstacle(pts: ScanPoints, cfg: GateConfig,
                   lead_xy: Optional[tuple] = None) -> FrontObstacle:
    """Nearest measured point inside the forward sector (and corridor), lead left out."""
    x, y = pts.x, pts.y
    sector = obstacle_points(pts, cfg, lead_xy)
    sector &= np.abs(pts.angle) <= cfg.lidar_front_half_angle
    count = int(sector.sum())
    if count < cfg.lidar_min_valid_beams:
        return FrontObstacle(math.nan, math.nan, math.nan, count)
    ahead = sector & (x > 0.0)
    if cfg.lidar_half_width > 0.0:
        ahead &= np.abs(y) <= cfg.lidar_half_width
    if not ahead.any():
        return FrontObstacle(math.inf, math.nan, math.nan, count)
    i = int(np.argmin(np.where(ahead, x, np.inf)))
    return FrontObstacle(float(x[i]), float(x[i]), float(y[i]), count)


def lead_gap(pts: ScanPoints, cfg: GateConfig, lead_xy: Optional[tuple]) -> LeadGap:
    """Free gap to the lead along my heading: nearest lead point x - lidar_to_front."""
    if lead_xy is None:
        return NO_LEAD_GAP
    lead = pts.measured & (pts.x > 0.0)
    lead &= _near_lead(pts.x, pts.y, lead_xy, cfg.lidar_lead_radius)
    count = int(lead.sum())
    if count < cfg.lidar_lead_min_points:
        return LeadGap(math.nan, count)
    return LeadGap(float(pts.x[lead].min()) - cfg.lidar_to_front, count)


def scan_front(ranges, angle_min: float, angle_increment: float, range_min: float,
               cfg: GateConfig, lead_xy: Optional[tuple] = None) -> FrontObstacle:
    """``front_obstacle`` straight from LaserScan contents."""
    return front_obstacle(scan_points(ranges, angle_min, angle_increment, range_min, cfg),
                          cfg, lead_xy)
