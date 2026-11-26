"""Fault injection: per-rover fault flags and an injector with optional durations."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from host_sim.core.geometry import Circle

QUALITIES = ('GOOD', 'DEGRADED', 'LOST')


@dataclass
class FaultState:
    """Active faults of one rover. Read by the world, base model and sensors."""

    pose_blackout: bool = False                       # localization stops publishing
    jump_offset: Tuple[float, float] = (0.0, 0.0)     # sustained position offset
    jump_once: Optional[Tuple[float, float]] = None   # offset for the next sample only
    quality: Optional[str] = None                     # forced localization quality
    hold_unaligned: bool = False                      # heading_aligned forced false
    cmd_blocked: bool = False                         # base stops receiving commands
    stuck: bool = False                               # wheels spin, body does not move
    camera_lost: bool = False                         # ceiling camera does not see the rover


class FaultInjector:
    """Applies faults to a world; a fault with a duration reverts by itself."""

    def __init__(self, world):
        self._world = world
        self._timers = []  # (expiry time, revert callable)

    def update(self, t):
        """Revert faults whose duration has elapsed."""
        expired = [timer for timer in self._timers if timer[0] <= t]
        self._timers = [timer for timer in self._timers if timer[0] > t]
        for _, revert in expired:
            revert()

    def pose_blackout(self, robot, enabled=True, duration=None):
        self._set(robot, 'pose_blackout', enabled, False, duration)

    def pose_jump(self, robot, dx, dy, sustained=False, duration=None):
        """Shift the reported position: one sample, or sustained until cleared."""
        if sustained:
            self._set(robot, 'jump_offset', (float(dx), float(dy)), (0.0, 0.0), duration)
        else:
            self._world.faults[robot].jump_once = (float(dx), float(dy))

    def set_quality(self, robot, quality='DEGRADED', duration=None):
        """Force the localization quality (None restores the configured one)."""
        if quality is not None and quality not in QUALITIES:
            raise ValueError(f'quality must be one of {QUALITIES}')
        self._set(robot, 'quality', quality, None, duration)

    def hold_unaligned(self, robot, enabled=True, duration=None):
        self._set(robot, 'hold_unaligned', enabled, False, duration)

    def block_commands(self, robot, enabled=True, duration=None):
        self._set(robot, 'cmd_blocked', enabled, False, duration)

    def stuck(self, robot, enabled=True, duration=None):
        self._set(robot, 'stuck', enabled, False, duration)

    def camera_lost(self, robot, enabled=True, duration=None):
        """The ceiling camera stops detecting the rover (localization mode fused only)."""
        self._set(robot, 'camera_lost', enabled, False, duration)

    def obstacle_ahead(self, robot, distance, radius, duration=None):
        """Drop a circular obstacle `distance` ahead of the rover's tracked point."""
        x, y, theta = self._world.robots[robot].tracked_point()
        shape = Circle(x + distance * math.cos(theta), y + distance * math.sin(theta), radius)
        name = self._world.add_obstacle(shape)
        if duration is not None:
            self._schedule(duration, lambda: self._world.remove_obstacle(name))
        return name

    def clear(self, robot=None):
        """Remove all faults of one rover (or of every rover) and pending timers."""
        names = [robot] if robot else list(self._world.faults)
        for name in names:
            self._world.faults[name].__init__()
        self._timers = []

    def _set(self, robot, field, value, default, duration):
        state = self._world.faults[robot]
        setattr(state, field, value)
        if duration is not None:
            self._schedule(duration, lambda: setattr(state, field, default))

    def _schedule(self, duration, revert):
        self._timers.append((self._world.t + float(duration), revert))
