"""Single-rover planar kinematics with lag, acceleration limits and skid-steer slip.

Input: Twist (+v forward, +omega CCW). State (x, y, theta): body origin in the global frame;
the rotation centre is ``icr_offset_x`` ahead of it, the tracked point ``ref_offset_x`` further."""
import math
from dataclasses import dataclass
from typing import Optional

from host_sim.core.util import wrap_angle

_STRAIGHT_EPS = 1e-9
_QUANTIZE_MODES = ('round', 'truncate')
_TRUNCATE_EPS = 1e-6  # in quanta; keeps exact multiples from falling one step short


@dataclass
class RobotParams:
    """Base model parameters; defaults describe an ideal (slip-free) rover."""

    max_linear_speed: float = 0.22      # m/s, command clamp
    max_angular_speed: float = 2.84     # rad/s
    linear_time_constant: float = 0.15  # s, first-order lag (0 = instant)
    angular_time_constant: float = 0.10
    max_linear_accel: float = 1.0       # m/s^2
    max_angular_accel: float = 6.0      # rad/s^2
    cmd_timeout: Optional[float] = 0.5  # s without a command -> stop (None/<= 0: never)
    forward_efficiency: float = 1.0     # actual / commanded forward speed
    tread_ratio: float = 1.0            # actual / commanded yaw rate
    icr_offset_x: float = 0.0           # m, rotation centre ahead of the body origin
    linear_quantum: float = 0.0         # m/s command resolution (0 = off)
    linear_quantize: str = 'round'      # or 'truncate'
    omega_quantum: float = 0.0          # rad/s command resolution (0 = off)
    omega_quantize: str = 'round'       # or 'truncate'
    omega_deadband: float = 0.0         # rad/s
    hw_angular_sign: float = 1.0        # -1 = hardware turns opposite to the command
    ref_offset_x: float = 0.0           # m, tracked point ahead of the rotation centre
    radius: float = 0.15                # m, collision footprint circle


class Robot:
    """One rover: integrates the commanded Twist into a planar pose."""

    def __init__(self, params=None, x=0.0, y=0.0, theta=0.0):
        self.params = params or RobotParams()
        self.t = 0.0
        self.traction = 1.0  # 0 = wheels spin but the body does not move (stuck)
        self.reset_pose(x, y, theta)

    def reset_pose(self, x, y, theta):
        """Teleport to a pose and zero velocities and odometer."""
        self.x, self.y, self.theta = float(x), float(y), wrap_angle(theta)
        self.v = 0.0  # body forward speed the wheels currently produce
        self.w = 0.0  # body yaw rate the wheels currently produce
        self.odometer = 0.0
        self._cmd = (0.0, 0.0)
        self._cmd_time = None

    def set_command(self, v, w):
        self._cmd = (float(v), float(w))
        self._cmd_time = self.t

    @property
    def lateral_velocity(self):
        return -self.w * self.params.icr_offset_x

    def tracked_point(self):
        off = self.params.icr_offset_x + self.params.ref_offset_x
        return (self.x + off * math.cos(self.theta),
                self.y + off * math.sin(self.theta), self.theta)

    def step(self, dt):
        target_v, target_w = self._target_rates()
        p = self.params
        self.v = _lag_step(self.v, target_v, dt, p.linear_time_constant, p.max_linear_accel)
        self.w = _lag_step(self.w, target_w, dt, p.angular_time_constant, p.max_angular_accel)
        self._integrate(dt)
        self.t += dt

    def _active_command(self):
        if self._cmd_time is None or self._command_timed_out():
            return 0.0, 0.0
        return self._cmd

    def _command_timed_out(self):
        timeout = self.params.cmd_timeout
        if timeout is None or timeout <= 0.0:
            return False  # base keeps executing the last command forever
        return self.t - self._cmd_time > timeout

    def _target_rates(self):
        p = self.params
        cmd_v, cmd_w = self._active_command()
        cmd_v = _quantise(_clamp(cmd_v, p.max_linear_speed), p.linear_quantum, p.linear_quantize)
        cmd_w = _quantise(_clamp(cmd_w, p.max_angular_speed), p.omega_quantum, p.omega_quantize)
        if abs(cmd_w) < p.omega_deadband:
            cmd_w = 0.0
        return p.forward_efficiency * cmd_v, p.hw_angular_sign * p.tread_ratio * cmd_w

    def _integrate(self, dt):
        vx = self.v * self.traction
        vy = self.lateral_velocity * self.traction
        w = self.w * self.traction
        dx, dy = _arc_displacement(vx, vy, w, dt)
        c, s = math.cos(self.theta), math.sin(self.theta)
        self.x += c * dx - s * dy
        self.y += s * dx + c * dy
        self.theta = wrap_angle(self.theta + w * dt)
        self.odometer += math.hypot(dx, dy)


def _clamp(value, limit):
    return max(-limit, min(limit, value))


def _quantise(value, quantum, mode='round'):
    if quantum <= 0.0:
        return value
    if mode not in _QUANTIZE_MODES:
        raise ValueError(f'quantize mode must be one of {_QUANTIZE_MODES}, got {mode!r}')
    steps = value / quantum
    if mode == 'truncate':
        return math.copysign(math.floor(abs(steps) + _TRUNCATE_EPS), steps) * quantum
    return round(steps) * quantum


def _lag_step(current, target, dt, time_constant, max_accel):
    if time_constant <= 0.0:
        delta = target - current
    else:
        delta = (target - current) * (1.0 - math.exp(-dt / time_constant))
    return current + _clamp(delta, max_accel * dt)


def _arc_displacement(vx, vy, w, dt):
    angle = w * dt
    if abs(angle) < _STRAIGHT_EPS:
        return vx * dt, vy * dt
    sin_term = math.sin(angle) / w
    cos_term = (1.0 - math.cos(angle)) / w
    return sin_term * vx - cos_term * vy, cos_term * vx + sin_term * vy
