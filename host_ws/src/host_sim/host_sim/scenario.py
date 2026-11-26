"""ROS-free closed-loop scenario runner.

A controller callback per rover returns the base Twist (v, w) every control_period (None
sends nothing, so the base stops after its command timeout):

    result = run_scenario(world, {'leader': controller}, duration=20.0,
                          goals={'leader': Goal(4.0, 3.3)})
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from host_sim.core.geometry import point_polyline_distance
from host_sim.core.sensors import PoseMeasurement
from host_sim.core.util import wrap_angle
from host_sim.core.world import CollisionEvent


@dataclass
class Goal:
    """Target pose of the tracked point."""

    x: float
    y: float
    theta: Optional[float] = None   # None = heading not checked
    pos_tol: float = 0.05           # m
    yaw_tol: float = 0.1            # rad
    stop_speed: float = 0.02        # m/s, must be slower than this to count


@dataclass
class Observation:
    """What a controller sees at one control tick."""

    t: float
    robot: str
    pose: Optional[PoseMeasurement]             # latest own sample (None before first)
    pose_is_new: bool                           # a sample arrived since the last tick
    poses: Dict[str, Optional[PoseMeasurement]]  # latest sample of every rover
    scan: Optional[np.ndarray]                  # latest LiDAR ranges (None before first)
    scan_is_new: bool
    scan_angles: Optional[np.ndarray]           # rad, sensor frame; None without LiDAR
    # depth (camera-like) scanner, same layout as the LiDAR
    depth_scan: Optional[np.ndarray] = None
    depth_scan_is_new: bool = False
    depth_scan_angles: Optional[np.ndarray] = None


@dataclass
class RobotResult:
    """Outcome for one rover (errors on the true tracked point)."""

    arrived: bool = False
    arrival_time: Optional[float] = None
    final_pos_error: float = math.nan
    final_yaw_error: float = math.nan
    max_cross_track: float = math.nan
    path_length: float = 0.0
    trace: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # t, x, y, theta


@dataclass
class ScenarioResult:
    """Outcome of a run."""

    robots: Dict[str, RobotResult]
    duration: float
    min_distance: float             # footprint clearance to obstacles and rovers
    min_obstacle_clearance: float
    min_robot_gap: float
    collisions: List[CollisionEvent]

    @property
    def collided(self):

        return bool(self.collisions)


def run_scenario(world, controllers, duration, goals=None, reference_paths=None,
                 events=None, dt=0.01, control_period=0.05, stop_on_arrival=True,
                 settle_time=0.5):
    """Closed-loop run. controllers: {robot: fn(Observation) -> (v, w) | None}; goals: arrival =
    inside tolerances and slow for settle_time; reference_paths: (N, 2) polylines for cross-track
    error; events: [(time, fn(world))] e.g. fault injection."""
    runner = _Runner(world, controllers, goals or {}, reference_paths or {},
                     events or [], control_period, settle_time)
    steps = int(round(duration / dt))
    for _ in range(steps):
        runner.step(dt)
        if stop_on_arrival and runner.all_arrived():
            break
    return runner.result()


class _Runner:


    def __init__(self, world, controllers, goals, paths, events, control_period, settle_time):
        self.world = world
        self.controllers = controllers
        self.goals = goals
        self.paths = {name: np.asarray(p, dtype=float) for name, p in paths.items()}
        self.events = sorted(events, key=lambda item: item[0])
        self.control_period = control_period
        self.settle_time = settle_time
        self.start_t = world.t
        self.next_control = world.t
        self.latest_pose = {name: None for name in world.robots}
        self.latest_scan = {name: None for name in world.robots}
        self.latest_depth = {name: None for name in world.robots}
        self.fresh_pose = set()
        self.fresh_scan = set()
        self.fresh_depth = set()
        self.hold_start = {name: None for name in goals}
        self.max_cross = {name: 0.0 for name in self.paths}
        self.traces = {name: [] for name in world.robots}

    def step(self, dt):

        self._fire_events()
        self._sense()
        if self.world.t >= self.next_control - 1e-9:
            self._control()
            self.next_control += self.control_period
        self.world.step(dt)
        self._track()

    def all_arrived(self):

        return bool(self.goals) and all(self._arrived(name) for name in self.goals)

    def result(self):

        record = self.world.record
        robots = {name: self._robot_result(name) for name in self.world.robots}
        return ScenarioResult(robots, self.world.t - self.start_t, record.min_distance,
                              record.min_obstacle_clearance, record.min_robot_gap,
                              list(record.collisions))

    def _fire_events(self):
        while self.events and self.events[0][0] <= self.world.t - self.start_t + 1e-9:
            self.events.pop(0)[1](self.world)

    def _sense(self):
        for name in self.world.robots:
            sample = self.world.poll_pose(name)
            if sample is not None:
                self.latest_pose[name] = sample
                self.fresh_pose.add(name)
            if name in self.controllers and self.world.lidar_due(name):
                self.latest_scan[name] = self.world.scan(name)
                self.fresh_scan.add(name)
            if name in self.controllers and self.world.depth_scan_due(name):
                self.latest_depth[name] = self.world.depth_scan(name)
                self.fresh_depth.add(name)

    def _control(self):
        for name, controller in self.controllers.items():
            command = controller(self._observation(name))
            if command is not None:
                self.world.set_command(name, command[0], command[1])
        self.fresh_pose.clear()
        self.fresh_scan.clear()
        self.fresh_depth.clear()

    def _observation(self, name):
        lidar, depth = self.world.lidars.get(name), self.world.depth_scans.get(name)
        return Observation(
            t=self.world.t, robot=name, pose=self.latest_pose[name],
            pose_is_new=name in self.fresh_pose, poses=dict(self.latest_pose),
            scan=self.latest_scan[name], scan_is_new=name in self.fresh_scan,
            scan_angles=None if lidar is None else lidar.angles,
            depth_scan=self.latest_depth[name], depth_scan_is_new=name in self.fresh_depth,
            depth_scan_angles=None if depth is None else depth.angles)

    def _track(self):
        for name, robot in self.world.robots.items():
            x, y, theta = robot.tracked_point()
            self.traces[name].append((self.world.t, x, y, theta))
            if name in self.paths:
                error = point_polyline_distance((x, y), self.paths[name])
                self.max_cross[name] = max(self.max_cross[name], error)
            if name in self.goals:
                self._update_hold(name)

    def _goal_errors(self, name):
        goal = self.goals[name]
        x, y, theta = self.world.robots[name].tracked_point()
        pos_error = math.hypot(x - goal.x, y - goal.y)
        yaw_error = 0.0 if goal.theta is None else abs(wrap_angle(theta - goal.theta))
        return pos_error, yaw_error

    def _update_hold(self, name):
        goal = self.goals[name]
        pos_error, yaw_error = self._goal_errors(name)
        inside = (pos_error <= goal.pos_tol and yaw_error <= goal.yaw_tol
                  and abs(self.world.robots[name].v) <= goal.stop_speed)
        if not inside:
            self.hold_start[name] = None
        elif self.hold_start[name] is None:
            self.hold_start[name] = self.world.t

    def _arrived(self, name):
        start = self.hold_start[name]
        return start is not None and self.world.t - start >= self.settle_time - 1e-9

    def _robot_result(self, name):
        result = RobotResult(path_length=self.world.robots[name].odometer,
                             trace=np.array(self.traces[name]).reshape(-1, 4))
        if name in self.paths:
            result.max_cross_track = self.max_cross[name]
        if name in self.goals:
            result.final_pos_error, result.final_yaw_error = self._goal_errors(name)
            result.arrived = self._arrived(name)
            if result.arrived:
                result.arrival_time = self.hold_start[name] - self.start_t
        return result
