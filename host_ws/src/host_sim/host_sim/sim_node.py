#!/usr/bin/env python3
"""sim_node: ROS 2 front end of the 2D kinematic rover simulator.

Stands in for "rover base + host localization" so the real rover drive code
(tracker, safety node) and the real host mission / planning code run unchanged.

Subscribes (per rover, topic from parameter `<robot>_cmd_topic`):
  /{robot}/cmd_vel_out   geometry_msgs/Twist   base input = safety node output
  /sim/command           std_msgs/String       JSON, see host_sim.core.commands
Publishes:
  /{robot}/global/position        PoseStamped (uwb_global, tracked point, noisy, delayed)
  /{robot}/heading_aligned        Bool   (transient_local)
  /{robot}/localization_quality   String (transient_local) GOOD | DEGRADED | LOST
  /{robot}/scan                   LaserScan (frame base_scan, sensor-data QoS)
  /{robot}/depth_scan             LaserScan (frame camera_depth_frame), only for a rover
                                  with a `depth_scan` section (camera-like scanner)
  /sim/ground_truth/{robot}       PoseStamped (true rotation-centre pose)
  /sim/min_distance               Float32 (current footprint clearance, m)
  /sim/collision                  String (one message per contact that starts)
  /sim/command_result             String (ok/error text for each /sim/command)
"""
import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String

from host_sim.config import load_config
from host_sim.core.commands import apply_command
from host_sim.core.world import World

_NODE_DEFAULTS = {
    'step_hz': 100.0, 'max_step': 0.05, 'ground_truth_hz': 30.0,
    'min_distance_hz': 10.0, 'status_hz': 1.0,
    'frame_id': 'uwb_global', 'scan_frame_id': 'base_scan',
    'depth_scan_frame_id': 'camera_depth_frame',
}
_QUEUE_DEPTH = 10


def _latched_qos():
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=1)


class _RobotIo:
    """Publishers and last-published status of one rover."""

    def __init__(self, node, name, cmd_topic, depth_scan=False):
        self.pose = node.create_publisher(PoseStamped, f'/{name}/global/position', _QUEUE_DEPTH)
        self.depth_scan = node.create_publisher(
            LaserScan, f'/{name}/depth_scan', qos_profile_sensor_data) if depth_scan else None
        self.aligned = node.create_publisher(Bool, f'/{name}/heading_aligned', _latched_qos())
        self.quality = node.create_publisher(
            String, f'/{name}/localization_quality', _latched_qos())
        self.scan = node.create_publisher(LaserScan, f'/{name}/scan', qos_profile_sensor_data)
        self.truth = node.create_publisher(
            PoseStamped, f'/sim/ground_truth/{name}', _QUEUE_DEPTH)
        self.cmd_sub = node.create_subscription(
            Twist, cmd_topic, lambda msg: node.on_cmd(name, msg), _QUEUE_DEPTH)
        self.last_status = None


class SimNode(Node):
    """Steps the world in real time and bridges it to ROS topics."""

    def __init__(self):
        super().__init__('host_sim')
        config = load_config(self.declare_parameter('config_file', '').value)
        self._cfg = {**_NODE_DEFAULTS, **(config.get('node') or {})}
        self.world = World.from_config(config)
        self._io = {name: self._make_io(name) for name in self.world.robots}
        self._min_pub = self.create_publisher(Float32, '/sim/min_distance', _QUEUE_DEPTH)
        self._collision_pub = self.create_publisher(String, '/sim/collision', _QUEUE_DEPTH)
        self._result_pub = self.create_publisher(String, '/sim/command_result', _QUEUE_DEPTH)
        self.create_subscription(String, '/sim/command', self._on_command, _QUEUE_DEPTH)
        self._last_tick = None
        self.create_timer(1.0 / self._cfg['step_hz'], self._tick)
        self.create_timer(1.0 / self._cfg['ground_truth_hz'], self._publish_truth)
        self.create_timer(1.0 / self._cfg['min_distance_hz'], self._publish_min_distance)
        self.create_timer(1.0 / self._cfg['status_hz'], self._republish_status)
        self.get_logger().info(f'host_sim up: robots={sorted(self.world.robots)}')

    def _make_io(self, name):
        topic = self.declare_parameter(f'{name}_cmd_topic', f'/{name}/cmd_vel_out').value
        self.get_logger().info(f'{name}: base input <- {topic}')
        return _RobotIo(self, name, topic, depth_scan=name in self.world.depth_scans)

    def on_cmd(self, name, msg):
        """Base Twist input of one rover."""
        self.world.set_command(name, msg.linear.x, msg.angular.z)

    def _on_command(self, msg):
        try:
            text = 'ok: ' + apply_command(self.world, json.loads(msg.data))
        except (ValueError, KeyError, TypeError) as error:
            text = f'error: {error}'
            self.get_logger().warn(f'/sim/command rejected: {error}')
        self._result_pub.publish(String(data=text))

    def _tick(self):
        now = self.get_clock().now()
        elapsed = 0.0 if self._last_tick is None else (now - self._last_tick).nanoseconds * 1e-9
        self._last_tick = now
        for event in self.world.step(min(elapsed, self._cfg['max_step'])):
            self._collision_pub.publish(String(data=event.describe()))
            self.get_logger().warn(f'collision: {event.describe()}')
        stamp = now.to_msg()
        for name, io in self._io.items():
            self._publish_localization(name, io, stamp)
            if self.world.lidar_due(name):
                io.scan.publish(self._scan_msg(
                    self.world.lidars[name], self.world.scan(name), stamp,
                    self._cfg['scan_frame_id']))
            if io.depth_scan is not None and self.world.depth_scan_due(name):
                io.depth_scan.publish(self._scan_msg(
                    self.world.depth_scans[name], self.world.depth_scan(name), stamp,
                    self._cfg['depth_scan_frame_id']))

    def _publish_localization(self, name, io, stamp):
        sample = self.world.poll_pose(name)
        if sample is not None:
            io.pose.publish(self._pose_msg(stamp, sample.x, sample.y, sample.theta))
        self._publish_status(name, io, force=False)

    def _publish_status(self, name, io, force):
        status = self.world.pose_status(name)
        if force or status != io.last_status:
            io.last_status = status
            io.aligned.publish(Bool(data=bool(status[0])))
            io.quality.publish(String(data=status[1]))

    def _republish_status(self):
        for name, io in self._io.items():
            self._publish_status(name, io, force=True)

    def _publish_truth(self):
        stamp = self.get_clock().now().to_msg()
        for name, io in self._io.items():
            robot = self.world.robots[name]
            io.truth.publish(self._pose_msg(stamp, robot.x, robot.y, robot.theta))

    def _publish_min_distance(self):
        self._min_pub.publish(Float32(data=float(self.world.current_min_distance)))

    def _pose_msg(self, stamp, x, y, theta):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._cfg['frame_id']
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.orientation.w = math.cos(theta / 2.0)
        return msg

    @staticmethod
    def _scan_msg(lidar, ranges, stamp, frame_id):
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.angle_min = float(lidar.angle_min)
        msg.angle_max = float(lidar.angles[-1])
        msg.angle_increment = float(lidar.angle_increment)
        msg.scan_time = 1.0 / lidar.params.rate_hz
        msg.range_min = float(lidar.params.range_min)
        msg.range_max = float(lidar.params.range_max)
        msg.ranges = ranges.astype('float32').tolist()
        return msg


def main():
    rclpy.init()
    node = SimNode()
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
