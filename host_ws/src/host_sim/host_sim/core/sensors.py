"""Sensor models: host localization output and a 2D LiDAR scan."""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from host_sim.core.faults import FaultState
from host_sim.core.geometry import (Circle, ray_circle, ray_circle_span, ray_segment_matrix,
                                    ray_segments)
from host_sim.core.util import wrap_angle

ALIGN_IMMEDIATE = 'immediate'
ALIGN_AFTER_DISTANCE = 'after_distance'
_TIME_EPS = 1e-9


@dataclass
class LocalizationParams:
    """Host localization output model (m, rad, s)."""

    rate_hz: float = 30.0
    delay: float = 0.08
    pos_noise_std: float = 0.02           # per axis, quality GOOD
    degraded_pos_noise_std: float = 0.08  # per axis, DEGRADED / LOST
    heading_noise_std: float = 0.02
    heading_bias: float = 0.0
    # slowly varying error: Gauss-Markov stationary std / correlation time (0 = off)
    pos_bias_std: float = 0.0
    pos_bias_tau: float = 10.0
    heading_drift_std: float = 0.0
    heading_drift_tau: float = 10.0
    align_mode: str = ALIGN_IMMEDIATE     # or 'after_distance'
    align_distance: float = 0.3           # straight driving until aligned
    align_max_omega: float = 0.2          # rad/s; faster turning restarts the count
    align_min_speed: float = 0.02         # m/s
    quality: str = 'GOOD'


@dataclass
class PoseMeasurement:
    """One localization sample."""

    stamp: float            # s, time the pose was true
    x: float
    y: float
    theta: float            # 0 and meaningless while heading_aligned is false
    heading_aligned: bool
    quality: str


class LocalizationSensor:
    """Delayed, noisy pose of the rover's tracked point at a fixed rate."""

    def __init__(self, params=None, rng=None):
        self.params = params or LocalizationParams()
        if self.params.align_mode not in (ALIGN_IMMEDIATE, ALIGN_AFTER_DISTANCE):
            raise ValueError(f'unknown align_mode {self.params.align_mode!r}')
        self._rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        """Forget history (after teleporting the rover)."""
        self._buffer = deque()
        self._next_due = None
        self._last_t = None
        self._straight_run = 0.0
        self._aligned = self.params.align_mode == ALIGN_IMMEDIATE
        self._pos_bias = GaussMarkov(self.params.pos_bias_std, self.params.pos_bias_tau,
                                     2, self._rng)
        self._heading_drift = GaussMarkov(self.params.heading_drift_std,
                                          self.params.heading_drift_tau, 1, self._rng)

    def status(self, fault=None):
        fault = fault or FaultState()
        aligned = self._aligned and not fault.hold_unaligned
        return aligned, fault.quality or self.params.quality

    def poll(self, t, robot, fault=None, others=()):
        """Feed the true state at t; a PoseMeasurement when due. ``others`` unused."""
        fault = fault or FaultState()
        self._track_alignment(t, robot)
        self._buffer.append((t,) + tuple(robot.tracked_point()))
        if not self._is_due(t) or fault.pose_blackout:
            return None
        delayed = self._delayed_sample(t - self.params.delay)
        if delayed is None:
            return None
        return self._measure(delayed, fault)

    def _track_alignment(self, t, robot):
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        if self._aligned:
            return
        p = self.params
        speed = robot.v * robot.traction
        if abs(robot.w * robot.traction) > p.align_max_omega:
            self._straight_run = 0.0
        elif speed > p.align_min_speed:
            self._straight_run += speed * dt
        self._aligned = self._straight_run >= p.align_distance

    def _is_due(self, t):
        period = 1.0 / self.params.rate_hz
        if self._next_due is None:
            self._next_due = t
        if t < self._next_due - _TIME_EPS:
            return False
        self._next_due += period
        if self._next_due < t:  # fell behind: resynchronise
            self._next_due = t + period
        return True

    def _delayed_sample(self, target_t):
        buf = self._buffer
        while len(buf) > 1 and buf[1][0] <= target_t + _TIME_EPS:
            buf.popleft()
        return buf[0] if buf[0][0] <= target_t + _TIME_EPS else None

    def _measure(self, sample, fault):
        stamp, x, y, theta = sample
        aligned, quality = self.status(fault)
        p = self.params
        std = p.pos_noise_std if quality == 'GOOD' else p.degraded_pos_noise_std
        noise = self._rng.normal(0.0, std, 2) if std > 0.0 else (0.0, 0.0)
        bias = self._pos_bias.step(1.0 / p.rate_hz)
        jump = fault.jump_once or (0.0, 0.0)
        fault.jump_once = None
        x += noise[0] + bias[0] + fault.jump_offset[0] + jump[0]
        y += noise[1] + bias[1] + fault.jump_offset[1] + jump[1]
        return PoseMeasurement(stamp, x, y, self._heading(theta, aligned), aligned, quality)

    def _heading(self, theta, aligned):
        if not aligned:
            return 0.0
        p = self.params
        noise = self._rng.normal(0.0, p.heading_noise_std) if p.heading_noise_std > 0.0 else 0.0
        drift = self._heading_drift.step(1.0 / p.rate_hz)[0]
        return wrap_angle(theta + p.heading_bias + drift + noise)


class GaussMarkov:
    """First-order Gauss-Markov error: stationary ``std``, correlation time ``tau``; std 0 = off."""

    def __init__(self, std, tau, size, rng):
        self._std, self._tau, self._rng = std, tau, rng
        self._value = np.zeros(size)
        self._started = False

    def step(self, dt):
        if self._std <= 0.0:
            return self._value
        if not self._started:          # start in the stationary distribution
            self._started = True
            self._value = self._rng.normal(0.0, self._std, self._value.shape)
            return self._value
        keep = math.exp(-dt / self._tau) if self._tau > 0.0 else 0.0
        kick = self._std * math.sqrt(1.0 - keep * keep)
        self._value = keep * self._value + self._rng.normal(0.0, kick, self._value.shape)
        return self._value


@dataclass
class LidarParams:
    """2D LiDAR model; with ``fov`` < 2 pi the beams span [-fov/2, fov/2] (depth camera as scan)."""

    rate_hz: float = 5.0
    num_beams: int = 360
    range_min: float = 0.12
    range_max: float = 3.5
    noise_std: float = 0.01               # m
    mount_x: float = 0.0                  # m, ahead of the rotation centre
    mount_yaw: float = 0.0                # rad, relative to the body
    # occluded intervals [start_deg, end_deg), CCW from sensor +x; may wrap 360
    blind_sectors: List[List[float]] = field(default_factory=list)
    invalid_value: float = 0.0            # blind or under-range beams
    fov: float = 2.0 * math.pi
    out_of_range_value: float = math.inf  # beyond range_max
    # depth-camera band: height h at distance d is seen only if h >= sensor_height - d*tan(lower)
    sensor_height: Optional[float] = None  # m above floor; None = everything is seen
    band_lower_angle: float = 0.0          # rad below the optical axis


class Lidar:
    """Ray-casting scanner. Beam i points at angle_min + i * angle_increment."""

    def __init__(self, params=None, rng=None):
        self.params = params or LidarParams()
        self._rng = rng or np.random.default_rng()
        fov, beams = self.params.fov, self.params.num_beams
        if fov >= 2.0 * math.pi - _TIME_EPS:
            self.angle_min, self.angle_increment = 0.0, 2.0 * math.pi / beams
        else:
            self.angle_increment = fov / beams
            self.angle_min = -fov / 2.0 + self.angle_increment / 2.0
        self.angles = self.angle_min + self.angle_increment * np.arange(self.params.num_beams)
        self._blind = self._blind_mask()
        self._next_due = None

    def is_due(self, t):
        if self._next_due is None:
            self._next_due = t
        if t < self._next_due - _TIME_EPS:
            return False
        self._next_due = max(self._next_due + 1.0 / self.params.rate_hz, t)
        return True

    def sensor_pose(self, robot):
        p = self.params
        return (robot.x + p.mount_x * math.cos(robot.theta),
                robot.y + p.mount_x * math.sin(robot.theta), robot.theta + p.mount_yaw)

    def scan(self, sensor_pose, circles, segments, low=()):
        """Ranges (num_beams,) against circles and (M, 2, 2) segments. ``low``: (obstacle, top)
        pairs for obstacles with a height, seen per the vertical band rule."""
        p = self.params
        origin = np.array(sensor_pose[:2], dtype=float)
        world_angles = self.angles + sensor_pose[2]
        dirs = np.stack([np.cos(world_angles), np.sin(world_angles)], axis=1)
        ranges = ray_segments(origin, dirs, segments)
        for circle in circles:
            ranges = np.minimum(ranges, ray_circle(origin, dirs, circle))
        for obstacle, top in low:
            ranges = np.minimum(ranges, self._low_hit(origin, dirs, obstacle, top))
        if p.noise_std > 0.0:
            ranges = ranges + self._rng.normal(0.0, p.noise_std, len(ranges))
        ranges = np.where(ranges > p.range_max, p.out_of_range_value, ranges)
        invalid = self._blind | (ranges < p.range_min)
        return np.where(invalid, p.invalid_value, ranges)

    def _low_hit(self, origin, dirs, obstacle, top):
        """Per-ray distance at which an obstacle of height ``top`` is seen (inf = not seen); the
        band reaches the top at d* = (sensor_height - top) / tan(band_lower_angle)."""
        if isinstance(obstacle, Circle):
            entry, leave = ray_circle_span(origin, dirs, obstacle)
            return self._seen(entry, leave, top)
        if isinstance(obstacle, np.ndarray):                 # thin lines: each on its own
            hits = ray_segment_matrix(origin, dirs, obstacle)
            return self._seen(hits, hits, top).min(axis=1, initial=np.inf)
        hits = ray_segment_matrix(origin, dirs, obstacle.segments())
        entry = hits.min(axis=1)
        leave = np.where(np.isfinite(hits), hits, -np.inf).max(axis=1)
        return self._seen(entry, leave, top)

    def _seen(self, entry, leave, top):
        p = self.params
        if p.sensor_height is None or top >= p.sensor_height:
            return entry
        slope = math.tan(p.band_lower_angle)
        if slope <= 0.0:
            return np.full_like(entry, np.inf)
        reach = (p.sensor_height - top) / slope
        return np.where(entry >= reach, entry, np.where(reach <= leave, reach, np.inf))

    def _blind_mask(self):
        mask = np.zeros(self.params.num_beams, dtype=bool)
        degrees = np.degrees(self.angles) % 360.0
        for start, end in self.params.blind_sectors:
            start, end = start % 360.0, end % 360.0
            if start <= end:
                mask |= (degrees >= start) & (degrees < end)
            else:
                mask |= (degrees >= start) | (degrees < end)
        return mask
