"""World: arena, obstacles, rovers with their sensors, and collision monitoring."""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from host_sim.core.faults import FaultInjector, FaultState
from host_sim.core.fused import MODE_FUSED, MODE_IDEAL, FusedLocalization, FusedParams
from host_sim.core.geometry import (Circle, Rect, box_segments,
                                    point_shape_distance)
from host_sim.core.robot import Robot, RobotParams
from host_sim.core.sensors import (Lidar, LidarParams, LocalizationParams,
                                   LocalizationSensor)
from host_sim.core.util import build_params

KIND_OBSTACLE = 'robot_obstacle'
KIND_ROBOT = 'robot_robot'
KIND_ARENA = 'out_of_arena'
_CONFIG_KEYS = {'seed', 'arena', 'robots', 'obstacles', 'node'}
_ROBOT_KEYS = {'start', 'base', 'localization', 'lidar', 'depth_scan'}


@dataclass
class Arena:
    """Axis-aligned working area (default: ceiling-camera visible range)."""

    x_min: float = 1.24
    x_max: float = 6.03
    y_min: float = 1.97
    y_max: float = 4.66
    walls_enabled: bool = True  # whether the boundary shows up in LiDAR scans
    # Boundary: None = a wall (taller than any sensor), a number = a pipe of that height
    # (m), 0 = nothing.
    border_top: Optional[float] = None
    # A room wall this far outside the boundary (m); None = none.
    room_wall_offset: Optional[float] = None

    def contains(self, x, y):
        """Return True when the point lies inside the area."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def segments(self):
        """Boundary edges as a (4, 2, 2) array."""
        return box_segments(self.x_min, self.x_max, self.y_min, self.y_max)

    def room_segments(self):
        """Room wall edges as a (4, 2, 2) array (requires room_wall_offset)."""
        d = self.room_wall_offset
        return box_segments(self.x_min - d, self.x_max + d, self.y_min - d, self.y_max + d)


@dataclass
class CollisionEvent:
    """A contact that started at time t."""

    t: float
    kind: str    # robot_obstacle | robot_robot | out_of_arena
    robot: str
    other: str   # obstacle name, other rover name, or 'arena'

    def describe(self):
        """One-line text for logs and the /sim/collision topic."""
        return f'{self.kind} {self.robot} {self.other} t={self.t:.2f}'


@dataclass
class SafetyRecord:
    """Closest approaches and collisions seen so far (footprint edge to edge)."""

    min_obstacle_clearance: float = math.inf
    min_robot_gap: float = math.inf
    collisions: List[CollisionEvent] = field(default_factory=list)

    @property
    def min_distance(self):
        """Smallest clearance to anything (obstacle or other rover)."""
        return min(self.min_obstacle_clearance, self.min_robot_gap)


class World:
    """Container that steps the rovers and answers sensor queries."""

    def __init__(self, arena=None, seed=None):
        self.arena = arena or Arena()
        self.t = 0.0
        self.robots = {}
        self.sensors = {}
        self.lidars = {}
        self.depth_scans = {}       # optional second scanner per rover (camera-like)
        self.faults = {}
        self.obstacles = []
        self.record = SafetyRecord()
        self.current_min_distance = math.inf
        self.injector = FaultInjector(self)
        self._rng = np.random.default_rng(seed)
        self._contacts = set()
        self._obstacle_count = 0

    @classmethod
    def from_config(cls, config, seed=None):
        """Build a world from a plain dict (see config/sim_default.yaml)."""
        _check_keys('config', config, _CONFIG_KEYS)
        seed = config.get('seed') if seed is None else seed
        world = cls(build_params(Arena, config.get('arena')), seed)
        for name, entry in (config.get('robots') or {}).items():
            _check_keys(f'robots.{name}', entry, _ROBOT_KEYS)
            robot = Robot(build_params(RobotParams, entry.get('base')),
                          *entry.get('start', (0.0, 0.0, 0.0)))
            lidar, depth = entry.get('lidar', {}), entry.get('depth_scan')
            world.add_robot(name, robot,
                            localization_params(entry.get('localization')),
                            False if lidar is None else build_params(LidarParams, lidar),
                            build_params(LidarParams, depth) if depth else None)
        for spec in config.get('obstacles') or []:
            world.add_obstacle(shape_from_dict(spec))
        return world

    def add_robot(self, name, robot, localization=None, lidar=None, depth_scan=None):
        """Register a rover with its localization and LiDAR models.

        ``localization``: LocalizationParams (ideal, default) or FusedParams (sensor
        models + the real fusion front-end). ``lidar``: LidarParams, None = default,
        False = none. ``depth_scan`` (LidarParams) adds a second, camera-like scanner.
        """
        robot.t = self.t
        self.robots[name] = robot
        self.sensors[name] = self._localization(name, localization)
        lidar_rng = self._child_rng()      # drawn either way so other sensors keep their noise
        if lidar is not False:
            self.lidars[name] = Lidar(lidar, lidar_rng)
        if depth_scan is not None:
            self.depth_scans[name] = Lidar(depth_scan, self._child_rng())
        self.faults[name] = FaultState()

    def add_obstacle(self, shape):
        """Add a Circle or Rect; returns its (possibly generated) name."""
        if not shape.name:
            self._obstacle_count += 1
            shape.name = f'obs{self._obstacle_count}'
        self.obstacles.append(shape)
        return shape.name

    def remove_obstacle(self, name):
        """Remove an obstacle by name; returns whether it existed."""
        before = len(self.obstacles)
        self.obstacles = [o for o in self.obstacles if o.name != name]
        return len(self.obstacles) < before

    def clear_obstacles(self):
        """Remove every obstacle."""
        self.obstacles = []

    def set_command(self, name, v, w):
        """Deliver a base Twist to a rover (dropped while its command path is cut)."""
        if not self.faults[name].cmd_blocked:
            self.robots[name].set_command(v, w)

    def reset_pose(self, name, x, y, theta):
        """Teleport a rover and restart its localization history."""
        self.robots[name].reset_pose(x, y, theta)
        self.sensors[name].reset()

    def reset_record(self):
        """Forget recorded minimum distances and collisions."""
        self.record = SafetyRecord()
        self._contacts = set()

    def step(self, dt):
        """Advance all rovers by dt; returns the collisions that just started."""
        self.t += dt
        self.injector.update(self.t)
        for name, robot in self.robots.items():
            robot.traction = 0.0 if self.faults[name].stuck else 1.0
            robot.step(dt)
            robot.t = self.t
        return self._monitor()

    def poll_pose(self, name):
        """Localization sample for a rover if one is due now, else None."""
        return self.sensors[name].poll(self.t, self.robots[name], self.faults[name],
                                       self._other_points(name))

    def pose_status(self, name):
        """Return (heading_aligned, quality) as currently reported for a rover."""
        return self.sensors[name].status(self.faults[name])

    def lidar_due(self, name):
        """Return True once per scan period of a rover's LiDAR (False without one)."""
        return name in self.lidars and self.lidars[name].is_due(self.t)

    def scan(self, name):
        """LiDAR ranges for a rover against obstacles, other rovers and walls."""
        return self._cast(name, self.lidars[name])

    def depth_scan_due(self, name):
        """Return True once per period of a rover's depth scanner (False without one)."""
        return name in self.depth_scans and self.depth_scans[name].is_due(self.t)

    def depth_scan(self, name):
        """Ranges of a rover's depth (camera-like) scanner."""
        return self._cast(name, self.depth_scans[name])

    def _cast(self, name, lidar):
        tall = [o for o in self.obstacles if o.top is None]
        low = [(o, o.top) for o in self.obstacles if o.top is not None and o.top > 0.0]
        circles = [o for o in tall if isinstance(o, Circle)]
        circles += [Circle(r.x, r.y, r.params.radius)
                    for other, r in self.robots.items() if other != name]
        segments = [o.segments() for o in tall if isinstance(o, Rect)]
        arena = self.arena
        if arena.walls_enabled and arena.border_top is None:
            segments.append(arena.segments())
        elif arena.walls_enabled and arena.border_top > 0.0:
            low.append((arena.segments(), arena.border_top))
        if arena.room_wall_offset is not None:
            segments.append(arena.room_segments())
        stacked = np.concatenate(segments) if segments else np.zeros((0, 2, 2))
        return lidar.scan(lidar.sensor_pose(self.robots[name]), circles, stacked, low)

    def _localization(self, name, params):
        if isinstance(params, FusedParams):
            arena = self.arena
            roi = (arena.x_min, arena.x_max, arena.y_min, arena.y_max)
            return FusedLocalization(name, params, roi, self._child_rng())
        return LocalizationSensor(params, self._child_rng())

    def _other_points(self, name):
        """True tracked points (x, y) of the other rovers."""
        return [robot.tracked_point()[:2] for other, robot in self.robots.items()
                if other != name]

    def _child_rng(self):
        return np.random.default_rng(self._rng.integers(0, 2 ** 32))

    def _monitor(self):
        """Update the safety record; return newly started contacts."""
        contacts = set()
        obstacle_min = self._check_obstacles(contacts)
        robot_min = self._check_robots(contacts)
        self._check_arena(contacts)
        record = self.record
        record.min_obstacle_clearance = min(record.min_obstacle_clearance, obstacle_min)
        record.min_robot_gap = min(record.min_robot_gap, robot_min)
        self.current_min_distance = min(obstacle_min, robot_min)
        started = [CollisionEvent(self.t, *key) for key in sorted(contacts - self._contacts)]
        record.collisions.extend(started)
        self._contacts = contacts
        return started

    def _check_obstacles(self, contacts):
        closest = math.inf
        for name, robot in self.robots.items():
            for shape in self.obstacles:
                gap = point_shape_distance((robot.x, robot.y), shape) - robot.params.radius
                closest = min(closest, gap)
                if gap <= 0.0:
                    contacts.add((KIND_OBSTACLE, name, shape.name))
        return closest

    def _check_robots(self, contacts):
        closest = math.inf
        names = sorted(self.robots)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                a, b = self.robots[first], self.robots[second]
                gap = math.hypot(a.x - b.x, a.y - b.y) - a.params.radius - b.params.radius
                closest = min(closest, gap)
                if gap <= 0.0:
                    contacts.add((KIND_ROBOT, first, second))
        return closest

    def _check_arena(self, contacts):
        for name, robot in self.robots.items():
            if not self.arena.contains(robot.x, robot.y):
                contacts.add((KIND_ARENA, name, 'arena'))


def localization_params(values):
    """Params of a rover's ``localization`` section: mode ``ideal`` (default) uses the
    section's own keys, ``fused`` uses its ``fused`` subsection."""
    values = dict(values or {})
    mode = values.pop('mode', MODE_IDEAL)
    fused = values.pop('fused', None)
    ideal = build_params(LocalizationParams, values)
    if mode == MODE_IDEAL:
        return ideal
    if mode == MODE_FUSED:
        return FusedParams.from_dict(fused)
    raise ValueError(f'localization.mode must be {MODE_IDEAL!r} or {MODE_FUSED!r}, got {mode!r}')


def shape_from_dict(spec):
    """Build an obstacle from {'type': 'circle'|'rect', ...}."""
    spec = dict(spec)
    kind = spec.pop('type', None)
    if kind == 'circle':
        return build_params(Circle, spec)
    if kind == 'rect':
        return build_params(Rect, spec)
    raise ValueError(f"obstacle type must be 'circle' or 'rect', got {kind!r}")


def _check_keys(label, mapping, allowed):
    unknown = sorted(set(mapping or {}) - allowed)
    if unknown:
        raise ValueError(f'{label}: unknown keys {unknown}')
