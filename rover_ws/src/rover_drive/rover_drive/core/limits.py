"""Speed limiters and the velocity ramp."""
import math
from typing import Iterable, NamedTuple, Optional

from .config import DriveConfig

NO_LIMIT = math.inf


class Limit(NamedTuple):
    speed: float
    reason: str


def strictest(limits: Iterable[Limit]) -> Limit:
    """The lowest speed bound (first one wins ties)."""
    return min(limits, key=lambda lim: lim.speed)


def curvature_limit(cfg: DriveConfig, kappa_ahead: float) -> Limit:
    """Bound lateral acceleration and angular speed on the upcoming curve."""
    k = abs(kappa_ahead)
    if k <= 0.0:
        return Limit(NO_LIMIT, 'CURVE')
    return Limit(min(math.sqrt(cfg.max_lateral_accel / k), cfg.max_angular_speed / k), 'CURVE')


def heading_limit(cfg: DriveConfig, heading_error: float, v_max: float) -> Limit:
    """Slow down for a large heading error (start-up): crawl and allow a tight arc."""
    err = abs(heading_error)
    if err <= cfg.heading_free_rad:
        return Limit(NO_LIMIT, 'HEADING')
    span = max(cfg.heading_slow_rad - cfg.heading_free_rad, 1e-6)
    ratio = min((err - cfg.heading_free_rad) / span, 1.0)
    return Limit(v_max + ratio * (min(cfg.turn_speed, v_max) - v_max), 'HEADING')


def goal_limit(cfg: DriveConfig, remaining: float) -> Limit:
    """Decelerate toward the goal; exactly 0 once the remaining arc length is used up."""
    if remaining <= cfg.goal_stop_margin:
        return Limit(0.0, 'GOAL')
    profile = math.sqrt(2.0 * cfg.goal_decel * (remaining - cfg.goal_stop_margin))
    return Limit(max(profile, cfg.approach_speed), 'GOAL')


def arrival_limit(cfg: DriveConfig, remaining: float) -> Limit:
    """Speed whose lookahead still fits ``runin_lookaheads`` times into the path left,
    so the rover can arrive on the line and at the goal heading. Floor: shortest lookahead."""
    if cfg.runin_lookaheads <= 0.0 or cfg.lookahead_time_s <= 0.0:
        return Limit(NO_LIMIT, 'GOAL')
    lookahead = max(remaining / cfg.runin_lookaheads, cfg.lookahead_min)
    return Limit(lookahead / cfg.lookahead_time_s, 'GOAL')


def quality_limit(cfg: DriveConfig, quality: str) -> Limit:
    """Localization quality: DEGRADED slows, LOST (or anything unknown) stops."""
    if quality == 'GOOD':
        return Limit(NO_LIMIT, 'QUALITY_LOW')
    if quality == 'DEGRADED':
        return Limit(cfg.degraded_speed, 'QUALITY_LOW')
    return Limit(0.0, 'QUALITY_LOW')


def stopping_speed(dist: float, base: float, decel: float, reaction_s: float) -> float:
    """Highest speed that can still stop ``base`` short of an object ``dist`` ahead."""
    free = dist - base
    if free <= 0.0:
        return 0.0
    return decel * (-reaction_s + math.sqrt(reaction_s ** 2 + 2.0 * free / decel))


def lidar_limit(cfg: DriveConfig, front_dist: Optional[float]) -> Limit:
    """LiDAR proximity (interface only: the safety node owns the scan processing)."""
    if front_dist is None:
        return Limit(NO_LIMIT, 'LIDAR_SLOW')
    speed = stopping_speed(front_dist, cfg.lidar_stop_base, cfg.lidar_brake_decel,
                           cfg.lidar_reaction_s)
    return Limit(speed, 'LIDAR_SLOW' if speed > 0.0 else 'LIDAR_STOP')


class GapLimiter:
    """Keep the gap to the rover ahead: speed = ``gap_gain * e`` with e = gap - wanted gap."""

    def __init__(self, cfg: DriveConfig):
        self._cfg = cfg
        self._waiting = False

    def reset(self) -> None:
        self._waiting = False

    def limit(self, gap_error: float, reason: str = 'GAP') -> Limit:
        """``reason`` names the gap source (GAP = path gap, GAP_LIDAR = measured free gap)."""
        cfg = self._cfg
        if self._waiting:
            if gap_error < cfg.gap_resume_margin:
                return Limit(0.0, reason)
            self._waiting = False
        speed = cfg.gap_gain * gap_error
        if speed <= 0.0:
            self._waiting = True
            return Limit(0.0, reason)
        return Limit(speed, reason)


QUANTIZE_ROUND = 'round'
QUANTIZE_TRUNCATE = 'truncate'


class OmegaQuantizer:
    """Send omega as whole steps of the base's resolution, keeping the mean."""

    def __init__(self, quantum: float, mode: str = QUANTIZE_ROUND, margin: float = 0.0,
                 omega_max: float = math.inf):
        if mode not in (QUANTIZE_ROUND, QUANTIZE_TRUNCATE):
            raise ValueError(f'omega_quantize must be round or truncate, got {mode!r}')
        self._q = quantum
        self._margin = margin if mode == QUANTIZE_TRUNCATE else 0.0
        self._k_max = math.inf
        if quantum > 0.0 and math.isfinite(omega_max):  # the sent value must stay inside it
            self._k_max = math.floor(omega_max / quantum - self._margin)
        self._residual = 0.0

    def reset(self) -> None:
        self._residual = 0.0

    def apply(self, omega: float) -> float:
        if self._q <= 0.0:
            return omega
        want = omega + self._residual
        # A step is sent only once a whole quantum is owed (toward zero, not nearest):
        # half a quantum of hysteresis, so +1/-1 steps do not alternate on a straight.
        k = min(max(int(want / self._q), -self._k_max), self._k_max)
        # Bounded carry: a saturated step must not wind the error up.
        self._residual = min(max(want - k * self._q, -self._q), self._q)
        if k == 0:
            return 0.0
        return math.copysign((abs(k) + self._margin) * self._q, k)


class VelocityRamp:
    """Acceleration, deceleration, jerk (on speeding up) and angular accel limits."""

    def __init__(self, cfg: DriveConfig):
        self._cfg = cfg
        self.v = 0.0
        self.w = 0.0
        self._accel = 0.0

    def reset(self) -> None:
        self.v = self.w = self._accel = 0.0

    def _ramp_linear(self, target: float, dt: float) -> float:
        cfg = self._cfg
        if target <= self.v:  # braking is never delayed by the jerk limit
            self._accel = 0.0
            return max(target, self.v - cfg.max_linear_decel * dt)
        if cfg.max_linear_jerk > 0.0:
            self._accel = min(self._accel + cfg.max_linear_jerk * dt, cfg.max_linear_accel)
        else:
            self._accel = cfg.max_linear_accel
        return min(target, self.v + self._accel * dt)

    def _ramp_angular(self, target: float, dt: float) -> float:
        step = self._cfg.max_angular_accel * dt
        return self.w + min(max(target - self.w, -step), step)

    def step(self, v_target: float, curvature: float, dt: float) -> tuple:
        """Advance one control period toward (v_target, curvature)."""
        w_max = self._cfg.max_angular_speed
        self.v = self._ramp_linear(max(v_target, 0.0), dt)
        self.w = self._ramp_angular(min(max(curvature * self.v, -w_max), w_max), dt)
        return self.v, self.w
