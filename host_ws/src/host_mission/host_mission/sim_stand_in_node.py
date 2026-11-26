#!/usr/bin/env python3
"""sim_stand_in_node: simulation aid for the node-mode lap, beside host_sim.

Stands in, on a simulator domain only, for the pieces that are hardware or a person on the
real system (as the in-process layout runner does, host_mission/scenarios/layout_runner.py):

  pick&place   /leader/nav_feedback HOLDING -> /leader/work_state PICKING, after
               ``pick_duration`` s -> DONE + /leader/pick_place_done true (both latched)
  front stop   ``front_source`` depth (default): /leader/depth_scan (simulated depth camera,
               0.0 = no depth) -> /leader/front_distance (Float32, NaN = unknown), reduced
               like depth_front_node (rover_drive test/host_loop.py depth_band_distance;
               band params = leader.yaml). ``front_source`` scan: nearest return of
               /leader/scan within +-``front_half_angle``. The scenario leader has no
               LiDAR, so without the depth source its gate stays STOPPED on STALE_SCAN.
  unload       ``auto_unload``: /mission/state UNLOADING -> one /unload/done (false = press
               it on the dashboard)
  dispatch     ``dispatch`` = part id: one /task/add "fetch:<id>" once the mission is IDLE
               and both rovers report nav feedback ('' = use the dashboard)

Never run it beside the real rovers: it fakes a finished pick."""
import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Empty, Float32, String

from host_mission_interfaces.msg import NavFeedback

_DEPTH = 10
_TICK_S = 0.1


def _latched():
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=1)


def front_distance(ranges, angle_min, angle_increment, half_angle):
    ranges = np.asarray(ranges, dtype=float)
    angles = angle_min + angle_increment * np.arange(ranges.size)
    wrapped = (angles + math.pi) % (2.0 * math.pi) - math.pi
    beams = ranges[np.abs(wrapped) <= half_angle]
    valid = beams[(beams > 0.0) & np.isfinite(beams)]
    return float(valid.min()) if valid.size else math.nan


# depth_front_node band of rover_drive config/leader.yaml (DepthFrontConfig).
DEPTH_BAND = {'depth_hfov': 1.019, 'depth_col_min': 0.30, 'depth_col_max': 0.70,
              'depth_valid_min_m': 0.30, 'depth_valid_max_m': 6.0, 'depth_percentile': 5.0,
              'depth_min_valid_ratio': 0.30}


def depth_band_distance(ranges, angle_min, angle_increment, band):
    """Depth-camera scan -> depth_front_node's front distance (NaN = unknown)."""
    ranges = np.asarray(ranges, dtype=float)
    angles = angle_min + angle_increment * np.arange(ranges.size)
    col = 0.5 - np.tan(angles) / (2.0 * math.tan(band['depth_hfov'] / 2.0))
    inside = (col >= band['depth_col_min']) & (col <= band['depth_col_max'])
    depth = ranges[inside] * np.cos(angles[inside])
    valid = depth[np.isfinite(depth) & (depth >= band['depth_valid_min_m'])
                  & (depth <= band['depth_valid_max_m'])]
    if valid.size == 0 or valid.size < band['depth_min_valid_ratio'] * inside.sum():
        return math.nan
    return float(np.percentile(valid, band['depth_percentile']))


class PickerStandIn:
    """Leader pick coordinator stand-in: HOLDING -> PICKING -> DONE."""

    def __init__(self, duration, repeat_delay=2.0):
        self.duration, self.repeat_delay = duration, repeat_delay
        self._started = None
        self._ready_at = 0.0

    def step(self, nav_state, now):
        if self._started is None:
            if nav_state == 'HOLDING' and now >= self._ready_at:
                self._started = now
                return [('PICKING', None)]
            return []
        if now - self._started >= self.duration:
            self._started = None
            self._ready_at = now + self.repeat_delay
            return [('DONE', True)]
        return []


class EdgeTrigger:

    def __init__(self, target):
        self.target, self._previous = target, None

    def __call__(self, state):
        fire = state == self.target and self._previous != self.target
        self._previous = state
        return fire


class SimStandInNode(Node):
    def __init__(self):
        super().__init__('sim_stand_in')
        declare = self.declare_parameter
        self.picker_name = declare('picker', 'leader').value
        self.picker = PickerStandIn(float(declare('pick_duration', 5.0).value))
        self.half_angle = float(declare('front_half_angle', 0.15).value)
        self.front_source = str(declare('front_source', 'depth').value)
        if self.front_source not in ('depth', 'scan'):
            raise ValueError(f"front_source must be 'depth' or 'scan', got {self.front_source!r}")
        self.depth_band = {k: float(declare(k, v).value) for k, v in DEPTH_BAND.items()}
        self.auto_unload = bool(declare('auto_unload', True).value)
        self.dispatch_part = str(declare('dispatch', '').value).strip()
        self.dispatch_delay = float(declare('dispatch_delay', 3.0).value)
        self.robots = list(declare('robots', ['leader', 'follower']).value)
        self._nav = {}
        self._mission_state = ''
        self._idle_since = None
        self._unload_edge = EdgeTrigger('UNLOADING')
        p = self.picker_name
        self.work_pub = self.create_publisher(String, f'/{p}/work_state', _latched())
        self.done_pub = self.create_publisher(Bool, f'/{p}/pick_place_done', _latched())
        self.front_pub = self.create_publisher(
            Float32, declare('front_distance_topic', f'/{p}/front_distance').value, _DEPTH)
        self.unload_pub = self.create_publisher(Empty, '/unload/done', _DEPTH)
        self.task_pub = self.create_publisher(String, '/task/add', _DEPTH)
        for robot in self.robots:
            self.create_subscription(NavFeedback, f'/{robot}/nav_feedback',
                                     lambda m, r=robot: self._nav.__setitem__(r, m.state),
                                     _DEPTH)
        default_scan = f'/{p}/depth_scan' if self.front_source == 'depth' else f'/{p}/scan'
        self.create_subscription(LaserScan, declare('front_scan_topic', default_scan).value,
                                 self._on_scan, qos_profile_sensor_data)
        self.create_subscription(String, '/mission/state', self._on_state, _DEPTH)
        self.work_pub.publish(String(data='NAVIGATING'))
        self.create_timer(_TICK_S, self._tick)
        self.get_logger().info(
            f'sim stand-ins: picker={p} ({self.picker.duration} s), front={self.front_source}, auto_unload='
            f'{self.auto_unload}, dispatch={self.dispatch_part or "(dashboard)"}')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_scan(self, msg):
        if self.front_source == 'depth':
            value = depth_band_distance(msg.ranges, msg.angle_min, msg.angle_increment,
                                        self.depth_band)
        else:
            value = front_distance(msg.ranges, msg.angle_min, msg.angle_increment,
                                   self.half_angle)
        self.front_pub.publish(Float32(data=value))

    def _on_state(self, msg):
        self._mission_state = msg.data
        if self.auto_unload and self._unload_edge(msg.data):
            self.get_logger().info('UNLOADING -> /unload/done (auto_unload)')
            self.unload_pub.publish(Empty())

    def _tick(self):
        now = self._now()
        for work_state, done in self.picker.step(self._nav.get(self.picker_name, ''), now):
            self.work_pub.publish(String(data=work_state))
            if done is not None:
                self.done_pub.publish(Bool(data=done))
            self.get_logger().info(f'pick stand-in: {work_state}')
        self._maybe_dispatch(now)

    def _maybe_dispatch(self, now):
        ready = (self.dispatch_part and self._mission_state == 'IDLE'
                 and all(r in self._nav for r in self.robots))
        if not ready:
            self._idle_since = None
            return
        if self._idle_since is None:
            self._idle_since = now
        if now - self._idle_since >= self.dispatch_delay:
            self.get_logger().info(f'dispatch fetch:{self.dispatch_part}')
            self.task_pub.publish(String(data=f'fetch:{self.dispatch_part}'))
            self.dispatch_part = ''


def main():
    rclpy.init()
    node = SimStandInNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
