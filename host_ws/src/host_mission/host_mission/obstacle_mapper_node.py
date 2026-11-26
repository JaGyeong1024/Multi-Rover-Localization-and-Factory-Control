#!/usr/bin/env python3
"""obstacle_mapper_node: range scans of every rover -> static obstacle rectangles for the
planner.

Thin ROS shell around ``ObstacleMapper`` (see that module for the method). Inputs are
stamped with the host's own monotonic receive time; the arena is the ``bounds`` of the
mission settings file.

Scan sources (parameter ``sources``, one block of parameters per name, see
config/obstacle_mapper.yaml): scan topic, the rover carrying the sensor, mount x/y/yaw
relative to that rover's tracked point, min / max use range, max turn rate. The host is
the single place the sensor mounts are configured.

Subscribes
  <source>.scan_topic                 LaserScan     sensor-data QoS, e.g. /follower/scan,
                                                    /leader/depth_scan
  /{robot}/global/position            PoseStamped   uwb_global, every rover in ``robots``
  /{robot}/heading_aligned            Bool          latched
  /{robot}/localization_quality       String        latched; GOOD | DEGRADED | LOST
  /mission/state                      String        latched; with ``freeze_while_mission``
                                                    scans are integrated only while IDLE
                                                    (obstacle_mapper.integrates_scans)
Publishes
  <output_topic> (/mission/mapped_obstacles)  String  latched JSON, only when the set
      changed: [{"x","y","r","width","height","yaw","pad"}, ...] = rectangle (x, y) of
      width (along yaw) x height grown by pad, the exact planner shape; "r" = covering
      circle radius for display-only consumers (dashboard)"""
import dataclasses
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from host_mission.obstacle_mapper import (DEFAULT_HEADING_ACCURACY, DEFAULT_ROBOT_RADII,
                                          MapperParams, ObstacleMapper, Region, ScanData,
                                          ScanSource, integrates_scans)
from host_mission.path_coordinator_node import load_node_settings

_DEPTH = 10
_DEFAULT_SOURCE = ScanSource()


def _latched(depth):
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def shape_json(shape):
    """JSON dict of an (x, y, pad, width, height, yaw) shape; "r" covers it with one circle
    (display)."""
    x, y, pad, width, height, yaw = shape
    cover = math.hypot(width, height) / 2.0 + pad
    return {'x': round(x, 3), 'y': round(y, 3), 'r': round(cover, 3),
            'width': round(width, 3), 'height': round(height, 3), 'yaw': round(yaw, 4),
            'pad': round(pad, 3)}


def parse_zones(text):
    return [Region(*map(float, zone)) for zone in json.loads(text or '[]')]


class ObstacleMapperNode(Node):
    def __init__(self):
        super().__init__('obstacle_mapper_node')
        declare = self.declare_parameter
        output_topic = declare('output_topic', '/mission/mapped_obstacles').value
        self._freeze = declare('freeze_while_mission', False).value
        self._mission_state = None
        sources, topics = self._sources()
        radii = self._robot_radii()
        heading = self._heading_accuracy({s.robot for s in sources})
        settings = self._settings()
        bounds = settings.bounds
        arena = Region(bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax)
        self.mapper = ObstacleMapper(self._mapper_params(), arena,
                                     parse_zones(declare('ignore_zones', '[]').value),
                                     sources, radii, settings.fixtures, heading)
        self.pub = self.create_publisher(String, output_topic, _latched(1))
        for robot in sorted(self.mapper.robots):
            self._subscribe_robot(robot)
        for source, topic in zip(sources, topics):
            self.create_subscription(LaserScan, topic,
                                     lambda msg, name=source.name: self._on_scan(name, msg),
                                     qos_profile_sensor_data)
        self.create_subscription(String, '/mission/state', self._on_mission_state, _latched(10))
        self._last_reason = {}
        self._publish()
        layer = next(iter(self.mapper.layers.values()))
        self.get_logger().info(
            f'obstacle mapper ready: sources='
            f'{", ".join(f"{s.name}<-{t} on {s.robot}" for s, t in zip(sources, topics))}, '
            f'excluded rovers={radii}, region={self.mapper.region}, '
            f'grid={layer.nx}x{layer.ny}, out={output_topic}')

    def _mapper_params(self):
        values = {f.name: self.declare_parameter(f.name, f.default).value
                  for f in dataclasses.fields(MapperParams)}
        return MapperParams(**values)

    def _sources(self):
        """ScanSource per name in ``sources`` (+ its scan topic) from '<name>.<field>'."""
        names = self.declare_parameter('sources', [_DEFAULT_SOURCE.name]).value
        sources, topics = [], []
        for name in names:
            values = {f.name: self.declare_parameter(f'{name}.{f.name}', f.default).value
                      for f in dataclasses.fields(ScanSource) if f.name != 'name'}
            sources.append(ScanSource(name=name, **values))
            topics.append(self.declare_parameter(f'{name}.scan_topic',
                                                 f'/{values["robot"]}/scan').value)
        return sources, topics

    def _robot_radii(self):
        """Exclusion radius of every rover in ``robots`` ('robot_radius.<robot>')."""
        robots = self.declare_parameter('robots', sorted(DEFAULT_ROBOT_RADII)).value
        return {robot: self.declare_parameter(f'robot_radius.{robot}',
                                              DEFAULT_ROBOT_RADII.get(robot, 0.35)).value
                for robot in robots if robot}

    def _heading_accuracy(self, robots):
        """Heading accuracy (rad) per rover carrying a source ('heading_accuracy.<robot>')."""
        return {robot: self.declare_parameter(f'heading_accuracy.{robot}',
                                              DEFAULT_HEADING_ACCURACY.get(robot, 0.14)).value
                for robot in sorted(robots)}

    def _settings(self):
        name, settings = load_node_settings(self)
        if settings.bounds is None:
            raise ValueError(f'{name}: "bounds" is required by the obstacle mapper')
        return settings

    def _subscribe_robot(self, robot):
        sub, mapper = self.create_subscription, self.mapper
        sub(PoseStamped, f'/{robot}/global/position',
            lambda m: self._on_pose(robot, m), _DEPTH)
        sub(Bool, f'/{robot}/heading_aligned',
            lambda m: mapper.update_status(robot, aligned=m.data), _latched(1))
        sub(String, f'/{robot}/localization_quality',
            lambda m: mapper.update_status(robot, quality=m.data), _latched(1))

    def _on_pose(self, robot, msg):
        p = msg.pose.position
        self.mapper.update_pose(robot, p.x, p.y, _yaw(msg.pose.orientation), time.monotonic())

    def _on_mission_state(self, msg):
        self._mission_state = msg.data

    def _on_scan(self, source, msg):
        if not integrates_scans(self._mission_state, self._freeze):
            self._log_reason(source, f'map frozen while the mission is {self._mission_state}')
            return
        scan = ScanData(np.asarray(msg.ranges, dtype=float), msg.angle_min,
                        msg.angle_increment, msg.range_min, msg.range_max)
        verdict = self.mapper.process_scan(source, scan, time.monotonic())
        self._log_reason(source, verdict.reason, verdict.used)
        if verdict.changed:
            self._publish()

    def _log_reason(self, source, reason, used=False):
        if reason != self._last_reason.get(source):
            self.get_logger().info(f'{source}: scans {"used" if used else "skipped"}'
                                   f'{": " + reason if reason else ""}')
            self._last_reason[source] = reason

    def _publish(self):
        shapes = self.mapper.shapes()
        self.pub.publish(String(data=json.dumps([shape_json(s) for s in shapes])))
        self.get_logger().info(f'mapped obstacles: {len(shapes)} rectangles')


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
