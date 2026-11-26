"""Raw localization inputs: ceiling camera, UWB tag, rover IMU (feed FusedLocalization).

Each samples the true state at its own rate; a measurement carries its stamp and ``deliver_at``."""
import math
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

from host_sim.core.sensors import GaussMarkov
from host_sim.core.util import wrap_angle

_TIME_EPS = 1e-9


class Ticker:
    """Fixed-rate schedule starting at the first query."""

    def __init__(self, rate_hz):
        self.period = 1.0 / rate_hz
        self._next = None

    def due(self, t):
        """True once per period (a coarser caller gets one tick per call)."""
        if self._next is None:
            self._next = t
        if t < self._next - _TIME_EPS:
            return False
        self._next += self.period
        if self._next < t:
            self._next = t + self.period
        return True

    def ticks_until(self, t):
        """Tick times up to t not returned yet."""
        if self._next is None:
            self._next = t
        ticks = []
        while self._next <= t + _TIME_EPS:
            ticks.append(self._next)
            self._next += self.period
        return ticks


@dataclass
class CameraParams:
    """Ceiling camera (YOLO-OBB) detection of a rover's tracked point."""

    rate_hz: float = 21.0
    latency: float = 0.055                    # s, frame to pose
    jitter_std: float = 0.002                 # m per axis, white
    bias_std: float = 0.0                     # m per axis, Gauss-Markov (0 = off)
    bias_tau: float = 10.0                    # s
    axis_noise_std: float = math.radians(2.4)  # rad, box axis scatter
    axis_period: float = math.pi / 2.0        # rad, box axis ambiguity (square box)
    occlusion_dist: float = 0.40              # m, another rover closer hides it
    roi: Optional[List[float]] = None         # [x_min, x_max, y_min, y_max]; None = arena


class CameraFix(NamedTuple):
    stamp: float               # s, time of the frame
    deliver_at: float          # s
    x: float
    y: float
    axis: Optional[float]      # rad, box axis in [-pi/2, pi/2)


class CeilingCamera:
    """Tracked-point detection at a fixed rate, delivered ``latency`` later; none outside the
    ROI, when occluded, or when lost. ``box_axis`` adds the OBB axis."""

    def __init__(self, params, roi, box_axis, rng):
        self.params = params
        self._roi = params.roi if params.roi is not None else roi
        self._box_axis = box_axis
        self._rng = rng
        self._ticker = Ticker(params.rate_hz)
        self._bias = GaussMarkov(params.bias_std, params.bias_tau, 2, rng)

    def capture(self, t, point, others=(), lost=False):
        if not self._ticker.due(t):
            return None
        p = self.params
        bias = self._bias.step(self._ticker.period)
        if lost or not self._in_roi(point) or self._occluded(point, others):
            return None
        noise = self._rng.normal(0.0, p.jitter_std, 2) if p.jitter_std > 0.0 else (0.0, 0.0)
        axis = self._axis(point[2]) if self._box_axis else None
        return CameraFix(t, t + p.latency, point[0] + noise[0] + bias[0],
                         point[1] + noise[1] + bias[1], axis)

    def _in_roi(self, point):
        x_min, x_max, y_min, y_max = self._roi
        return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max

    def _occluded(self, point, others):
        limit = self.params.occlusion_dist
        return any(math.hypot(ox - point[0], oy - point[1]) < limit for ox, oy in others)

    def _axis(self, heading):
        """One of the box's equivalent axes (square box), wrapped to [-pi/2, pi/2)."""
        p = self.params
        noise = self._rng.normal(0.0, p.axis_noise_std) if p.axis_noise_std > 0.0 else 0.0
        side = self._rng.integers(int(round(math.pi / p.axis_period)))
        axis = heading + noise + side * p.axis_period
        return (axis + math.pi / 2.0) % math.pi - math.pi / 2.0


@dataclass
class UwbParams:
    """UWB tag fix (firmware trilateration)."""

    enabled: bool = True
    rate_hz: float = 10.0
    noise_std: float = 0.04                   # m per axis, white
    offset_x: float = 0.0                     # m, constant offset of the UWB frame
    offset_y: float = 0.0
    bias_std: float = 0.0                     # m per axis, Gauss-Markov (0 = off)
    bias_tau: float = 10.0                    # s
    qf: float = 80.0                          # firmware quality factor


class UwbFix(NamedTuple):
    stamp: float
    deliver_at: float
    x: float
    y: float
    qf: float


class UwbTag:
    """Position fix of the tracked point in the UWB frame (offset + bias + white noise)."""

    def __init__(self, params, rng):
        self.params = params
        self._rng = rng
        self._ticker = Ticker(params.rate_hz)
        self._bias = GaussMarkov(params.bias_std, params.bias_tau, 2, rng)

    def capture(self, t, point):
        p = self.params
        if not p.enabled or not self._ticker.due(t):
            return None
        bias = self._bias.step(self._ticker.period)
        noise = self._rng.normal(0.0, p.noise_std, 2) if p.noise_std > 0.0 else (0.0, 0.0)
        return UwbFix(t, t, point[0] + p.offset_x + bias[0] + noise[0],
                      point[1] + p.offset_y + bias[1] + noise[1], p.qf)


@dataclass
class ImuParams:
    """Rover IMU (orientation yaw + gyro z)."""

    rate_hz: float = 20.0
    yaw_drift_std: float = 0.0                # rad/sqrt(s), random walk (0 = off)
    yaw_rate_scale: float = 1.0               # integrated yaw / true change
    gyro_bias: float = 0.0                    # rad/s
    gyro_noise_std: float = 0.0               # rad/s, white


class ImuSample(NamedTuple):
    stamp: float
    deliver_at: float
    yaw: float                 # rad, arbitrary zero (heading alignment finds it)
    wz: float                  # rad/s


class Imu:
    """Yaw with a random zero, optional drift and scale error."""

    def __init__(self, params, rng):
        self.params = params
        self._rng = rng
        self._ticker = Ticker(params.rate_hz)
        self.zero = float(rng.uniform(-math.pi, math.pi))   # rad, the IMU's own yaw zero
        self._yaw = None          # integrated yaw (unwrapped, without the zero)
        self._last_theta = None
        self._drift = 0.0

    def capture(self, t, theta, yaw_rate):
        if not self._ticker.due(t):
            return None
        p = self.params
        self._integrate(theta)
        noise = self._rng.normal(0.0, p.gyro_noise_std) if p.gyro_noise_std > 0.0 else 0.0
        wz = p.yaw_rate_scale * yaw_rate + p.gyro_bias + noise
        return ImuSample(t, t, wrap_angle(self.zero + self._yaw + self._drift), wz)

    def _integrate(self, theta):
        p = self.params
        if self._yaw is None:
            self._yaw = theta
        else:
            self._yaw += p.yaw_rate_scale * wrap_angle(theta - self._last_theta)
            if p.yaw_drift_std > 0.0:
                step = p.yaw_drift_std * math.sqrt(self._ticker.period)
                self._drift += self._rng.normal(0.0, step)
        self._last_theta = theta
