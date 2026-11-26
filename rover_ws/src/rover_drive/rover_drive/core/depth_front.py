"""Depth image -> distance to the nearest thing ahead (pure numpy, no ROS, no cv_bridge)."""
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import DepthFrontConfig

ENC_16UC1 = '16UC1'      # uint16, millimetres, 0 = no measurement
ENC_32FC1 = '32FC1'      # float32, metres, NaN / 0 = no measurement
_MM_PER_M = 1000.0
_FORMATS = {ENC_16UC1: ('u2', 1.0 / _MM_PER_M), ENC_32FC1: ('f4', 1.0)}


def decode_depth(data, height: int, width: int, encoding: str, step: int = 0,
                 is_bigendian: bool = False) -> np.ndarray:
    """``sensor_msgs/Image`` fields -> (height, width) float array in metres."""
    if encoding not in _FORMATS:
        raise ValueError(f'unsupported depth encoding {encoding!r}')
    kind, scale = _FORMATS[encoding]
    dtype = np.dtype(('>' if is_bigendian else '<') + kind)
    row_bytes = step or width * dtype.itemsize
    rows = np.frombuffer(data, dtype=np.uint8, count=height * row_bytes)
    rows = rows.reshape(height, row_bytes)[:, :width * dtype.itemsize]
    pixels = np.ascontiguousarray(rows).view(dtype).reshape(height, width)
    return pixels.astype(np.float32) * scale


def _span(lo: float, hi: float, size: int) -> slice:
    start, stop = int(lo * size), int(math.ceil(hi * size))
    return slice(start, max(stop, start + 1))


def _band(depth: np.ndarray, cfg: DepthFrontConfig) -> np.ndarray:
    rows = _span(cfg.depth_row_min, cfg.depth_row_max, depth.shape[0])
    return depth[rows, _span(cfg.depth_col_min, cfg.depth_col_max, depth.shape[1])]


def _valid(pixels: np.ndarray, cfg: DepthFrontConfig) -> np.ndarray:
    return np.isfinite(pixels) & (pixels >= cfg.depth_valid_min_m) \
        & (pixels <= cfg.depth_valid_max_m)


def band_slopes(col_min: float, col_max: float, hfov: float) -> Tuple[float, float]:
    """Lateral offset per metre of depth (y / Z, y left) at the band's right and left edge
    (ideal centred pinhole)."""
    scale = 2.0 * math.tan(hfov / 2.0)
    return (0.5 - col_max) * scale, (0.5 - col_min) * scale


def front_distance(depth_m: np.ndarray, cfg: DepthFrontConfig) -> float:
    """Low-percentile depth of the band [m], or NaN if too little of it is valid."""
    band = _band(depth_m, cfg)
    valid = band[_valid(band, cfg)]
    if band.size == 0 or valid.size < cfg.depth_min_valid_ratio * band.size or valid.size == 0:
        return math.nan
    return float(np.percentile(valid, cfg.depth_percentile))


# ---------------------------------------------------------------- depth scan
@dataclass(frozen=True)
class Intrinsics:
    """Horizontal pinhole intrinsics of the depth image [px]."""

    fx: float
    cx: float
    width: int


def intrinsics_from_fov(width: int, hfov: float) -> Intrinsics:
    """Ideal pinhole with the given horizontal field of view [rad], centred."""
    return Intrinsics(fx=(width / 2.0) / math.tan(hfov / 2.0), cx=(width - 1) / 2.0,
                      width=int(width))


def intrinsics_from_camera_info(k: Sequence[float], info_width: int,
                                image_width: int) -> Optional[Intrinsics]:
    """``CameraInfo.k`` (row-major 3x3) -> intrinsics for an image ``image_width`` wide
    (rescaled if the camera_info is for another resolution); None if uncalibrated."""
    fx, cx = float(k[0]), float(k[2])
    if fx <= 0.0 or info_width <= 0:
        return None
    scale = image_width / float(info_width)
    return Intrinsics(fx=fx * scale, cx=(cx + 0.5) * scale - 0.5, width=int(image_width))


@dataclass(frozen=True)
class DepthScan:
    """A planar scan in the camera frame (x forward, y left), LaserScan layout."""

    ranges: np.ndarray          # (beams,) metres; 0.0 = no return
    angle_min: float            # bearing of beam 0 (the rightmost one) [rad]
    angle_increment: float
    range_min: float
    range_max: float

    @property
    def angle_max(self) -> float:
        return self.angle_min + self.angle_increment * (len(self.ranges) - 1)


def depth_scan(depth_m: np.ndarray, intr: Intrinsics, cfg: DepthFrontConfig) -> DepthScan:
    """Depth image [m] -> planar scan from rows ``depth_scan_row_min/max``."""
    rows = depth_m[_span(cfg.depth_scan_row_min, cfg.depth_scan_row_max, depth_m.shape[0])]
    width = rows.shape[1]
    tan = (np.arange(width) - intr.cx) / intr.fx
    secant = np.sqrt(1.0 + tan * tan)
    right = -math.atan((width - 0.5 - intr.cx) / intr.fx)
    left = math.atan((intr.cx + 0.5) / intr.fx)
    beams = max(1, int(cfg.depth_scan_beams))
    increment = (left - right) / beams
    beam_of_column = np.clip(((-np.arctan(tan) - right) / increment).astype(int), 0, beams - 1)
    ranges_px = rows * secant[None, :]
    valid = _valid(rows, cfg)
    ranges = np.zeros(beams)
    for beam in range(beams):
        cols = beam_of_column == beam
        ok = valid[:, cols]
        if ok.size and ok.sum() >= max(1, cfg.depth_scan_min_valid_ratio * ok.size):
            ranges[beam] = np.percentile(ranges_px[:, cols][ok], cfg.depth_percentile)
    return DepthScan(ranges=ranges, angle_min=right + increment / 2.0,
                     angle_increment=increment, range_min=cfg.depth_valid_min_m,
                     range_max=cfg.depth_valid_max_m * float(secant.max()))
