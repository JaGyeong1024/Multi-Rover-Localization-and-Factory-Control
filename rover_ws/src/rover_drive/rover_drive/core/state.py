"""Rover drive state, command acceptance and the heading alignment manoeuvre."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

IDLE = 'IDLE'
TRACKING = 'TRACKING'
HOLDING = 'HOLDING'
PARKED = 'PARKED'
STOPPED = 'STOPPED'
ERROR = 'ERROR'

MODE_FOLLOW = 'FOLLOW'

# STOPPED reasons that only a new command (host re-plan) clears.
LATCHED_STOPS = ('OFF_PATH', 'STUCK', 'GOAL_MISSED')

# Command classification.
NEW = 'NEW'
KEEPALIVE = 'KEEPALIVE'
REJECT = 'REJECT'

# Alignment manoeuvre status.
ALIGNED = 'ALIGNED'
ALIGNING = 'ALIGNING'
ALIGN_FAILED = 'ALIGN_FAILED'


@dataclass(frozen=True)
class PathCmd:
    """ROS-free copy of ``host_mission_interfaces/PathCommand``."""

    epoch: int
    cmd_id: int
    mode: str = MODE_FOLLOW
    poses: Tuple[Tuple[float, float, float], ...] = ()  # (x, y, theta); last = goal
    speed_cap: float = 0.0
    goal_pos_tol: float = 0.0
    goal_yaw_tol: float = 0.0
    pick_at_goal: bool = False
    lead_robot: str = ''
    gap: float = 0.0
    pivot_first: bool = False       # rotate in place at poses[0] to pivot_heading first
    pivot_heading: float = 0.0      # rad
    fixture_ahead: bool = False     # goal faces a fixture: final run-in exempt from front stop


def classify_command(current: Optional[PathCmd], incoming: PathCmd) -> str:
    """Acceptance rule: larger epoch wins; within an epoch cmd_id must not go back."""
    if current is None or incoming.epoch > current.epoch:
        return NEW
    if incoming.epoch < current.epoch or incoming.cmd_id < current.cmd_id:
        return REJECT
    return KEEPALIVE if incoming.cmd_id == current.cmd_id else NEW


def command_is_finite(cmd: PathCmd) -> bool:
    """Every number a FOLLOW command is driven with is finite (NaN slips through comparisons)."""
    used = [cmd.speed_cap, cmd.goal_pos_tol, cmd.goal_yaw_tol]
    used += [c for pose in cmd.poses for c in pose]
    if cmd.pivot_first:
        used.append(cmd.pivot_heading)
    if cmd.lead_robot:
        used.append(cmd.gap)
    return all(math.isfinite(v) for v in used)


class DriveState:
    """Current state + detail, with the latching rules."""

    def __init__(self):
        self.state = IDLE
        self.detail = ''

    @property
    def latched(self) -> bool:
        if self.state in (HOLDING, PARKED, ERROR):
            return True
        return self.state == STOPPED and self.detail in LATCHED_STOPS

    def _set(self, state: str, detail: str) -> None:
        if not self.latched:
            self.state, self.detail = state, detail

    def reset(self) -> None:
        """A new command was accepted: clear any latch."""
        self.state, self.detail = IDLE, ''

    def track(self, detail: str = '') -> None:
        self._set(TRACKING, detail)

    def stop(self, detail: str) -> None:
        self._set(STOPPED, detail)

    def fail(self, detail: str) -> None:
        self._set(ERROR, detail)

    def arrive(self, pick_at_goal: bool, detail: str = '') -> None:
        """Inside the goal tolerances: HOLDING only at a pick point."""
        self._set(HOLDING if pick_at_goal else PARKED, detail)


class AlignManeuver:
    """Heading not aligned: creep straight until the host can observe the heading
    (derived from motion). Still unaligned after ``max_dist`` from where the unaligned
    period began: ERROR ``ALIGN_FAILED``."""

    def __init__(self, max_dist: float):
        self._max_dist = max_dist
        self._origin: Optional[tuple] = None

    def reset(self) -> None:
        self._origin = None

    def update(self, aligned: bool, x: float, y: float) -> str:
        if aligned:
            self._origin = None
            return ALIGNED
        if self._origin is None:
            self._origin = (x, y)
        travelled = math.hypot(x - self._origin[0], y - self._origin[1])
        return ALIGN_FAILED if travelled >= self._max_dist else ALIGNING
