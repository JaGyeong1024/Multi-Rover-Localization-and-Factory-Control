"""Monitors: pose jump, off-path, stuck and arrival judgement."""
import math
from statistics import median
from typing import List, NamedTuple, Optional

from .config import DriveConfig
from .path import wrap_angle


class PoseJumpFilter:
    """Discard samples farther from the last accepted one than
    ``jump_speed_factor * v_max * dt + jump_slack_m``; rejected samples that agree for
    ``jump_reacquire_s`` are adopted. Never smoothed."""

    def __init__(self, cfg: DriveConfig):
        self._cfg = cfg
        self._last: Optional[tuple] = None       # (x, y, t) last accepted
        self._candidate: Optional[tuple] = None  # (x, y, t) last rejected
        self._candidate_since = 0.0

    @property
    def rejecting(self) -> bool:
        return self._candidate is not None

    def _reachable(self, ref: tuple, x: float, y: float, now: float) -> bool:
        cfg = self._cfg
        allowed = cfg.jump_speed_factor * cfg.max_linear_speed * (now - ref[2]) + cfg.jump_slack_m
        return math.hypot(x - ref[0], y - ref[1]) <= allowed

    def _accept(self, x: float, y: float, now: float) -> bool:
        self._last = (x, y, now)
        self._candidate = None
        return True

    def accept(self, x: float, y: float, now: float) -> bool:
        """True if the sample may be used."""
        if self._last is None or self._reachable(self._last, x, y, now):
            return self._accept(x, y, now)
        if self._candidate is None or not self._reachable(self._candidate, x, y, now):
            self._candidate_since = now
        self._candidate = (x, y, now)
        if now - self._candidate_since >= self._cfg.jump_reacquire_s:
            return self._accept(x, y, now)
        return False


def is_off_path(cfg: DriveConfig, dist_to_path: float) -> bool:
    """Cross-track limit exceeded (also true when far beyond either path end)."""
    return dist_to_path > cfg.off_path_max


class StuckMonitor:
    """Commands go out but the rover does not move: each time ``stuck_cmd_dist`` of travel
    has been commanded, the tracked point must have moved at least ``stuck_min_ratio`` of it."""

    def __init__(self, cfg: DriveConfig):
        self._cfg = cfg
        self._origin: Optional[tuple] = None
        self._commanded = 0.0

    def reset(self) -> None:
        self._origin = None
        self._commanded = 0.0

    def update(self, v_cmd: float, x: float, y: float, dt: float) -> bool:
        """Feed the effective commanded speed and the position. True = STUCK."""
        if self._origin is None:
            self._origin = (x, y)
        self._commanded += abs(v_cmd) * dt
        if self._commanded < self._cfg.stuck_cmd_dist:
            return False
        moved = math.hypot(x - self._origin[0], y - self._origin[1])
        stuck = moved < self._cfg.stuck_min_ratio * self._commanded
        self.reset()
        return stuck


class PivotMonitor:
    """Leading pivot: the heading must follow the commanded rotation."""

    def __init__(self, cfg: DriveConfig):
        self._cfg = cfg
        self.reset()

    def reset(self) -> None:
        """A new pivot."""
        self._direction = 0.0     # sign of the initial error (0 = not started)
        self._budget = 0.0
        self._commanded = 0.0
        self._progress = 0.0
        self._last_yaw: Optional[float] = None

    def pause(self) -> None:
        """Heading not valid (not aligned): the next yaw starts a new difference."""
        self._last_yaw = None

    def update(self, error: float, yaw: float, w_cmd: float, dt: float) -> bool:
        """One step: current heading error, measured yaw, rate commanded over the last dt."""
        cfg = self._cfg
        ratio = cfg.pivot_min_turn_ratio
        if not self._direction:
            self._direction = math.copysign(1.0, error)
            self._budget = abs(error) / ratio + cfg.pivot_slack_rad
        if self._last_yaw is not None:
            self._progress += self._direction * wrap_angle(yaw - self._last_yaw)
        self._last_yaw = yaw
        self._commanded += abs(w_cmd) * dt
        if self._commanded > self._budget:
            return True
        return self._commanded >= cfg.pivot_check_rad and self._progress < ratio * self._commanded


class Arrival(NamedTuple):
    ok: bool
    pos_error: float
    yaw_error: float  # NaN when the heading was not checked


def judge_arrival(pose: tuple, goal: tuple, pos_tol: float, yaw_tol: float,
                  heading_valid: bool) -> Arrival:
    """Arrival only inside the tolerances; ``yaw_tol == 0`` = heading not checked, and a
    required heading that cannot be verified (not aligned) is a miss."""
    pos_error = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
    if yaw_tol <= 0.0:
        return Arrival(pos_error <= pos_tol, pos_error, math.nan)
    if not heading_valid:
        return Arrival(False, pos_error, math.nan)
    yaw_error = abs(wrap_angle(pose[2] - goal[2]))
    return Arrival(pos_error <= pos_tol and yaw_error <= yaw_tol, pos_error, yaw_error)


class ArrivalWindow:
    """Pose samples received while the rover stands still at the end of its path."""

    def __init__(self):
        self._since: Optional[float] = None
        self._samples: List[tuple] = []

    @property
    def is_open(self) -> bool:
        return self._since is not None

    def open(self, pose: tuple, now: float) -> None:
        """Start collecting (no-op while already open)."""
        if self._since is None:
            self._since, self._samples = now, [pose]

    def close(self) -> None:
        self._since, self._samples = None, []

    def add(self, pose: tuple) -> None:
        if self._since is not None:
            self._samples.append(pose)

    def elapsed(self, now: float) -> float:
        return 0.0 if self._since is None else now - self._since

    def estimate(self) -> tuple:
        """Median position and median heading of the samples (the window is open)."""
        ref_yaw = self._samples[0][2]
        yaw_offsets = [wrap_angle(p[2] - ref_yaw) for p in self._samples]
        return (median(p[0] for p in self._samples), median(p[1] for p in self._samples),
                wrap_angle(ref_yaw + median(yaw_offsets)))
