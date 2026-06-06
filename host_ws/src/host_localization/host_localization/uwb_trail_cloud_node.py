#!/usr/bin/env python3
"""Accumulate raw UWB poses into a PointCloud2 trail for RViz."""

from __future__ import annotations

import struct
from collections import deque

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def _rgb_float(r: int, g: int, b: int) -> float:
    packed = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack('f', struct.pack('I', packed))[0]


class UWBTrailCloudNode(Node):
    def __init__(self):
        super().__init__('uwb_trail_cloud_node')

        self.declare_parameter('input_topic', '/uwb/leader_pose')
        self.declare_parameter('output_topic', '/analysis/uwb_leader_raw_trail')
        self.declare_parameter('frame_id', 'uwb_global')
        self.declare_parameter('publish_rate', 2.0)
        # 0 means keep every point for the whole run.
        self.declare_parameter('max_points', 0)
        self.declare_parameter('color_rgb', [255, 70, 70])
        self.declare_parameter('z_mode', 'raw')  # raw | fixed
        self.declare_parameter('fixed_z', 0.02)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.z_mode = str(self.get_parameter('z_mode').value).lower()
        self.fixed_z = float(self.get_parameter('fixed_z').value)
        if self.z_mode not in ('raw', 'fixed'):
            raise ValueError('z_mode must be "raw" or "fixed"')

        max_points = int(self.get_parameter('max_points').value)
        self.points = deque(maxlen=max_points if max_points > 0 else None)
        rgb = [int(v) for v in self.get_parameter('color_rgb').value]
        self.rgb = _rgb_float(rgb[0], rgb[1], rgb[2])

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pub = self.create_publisher(PointCloud2, output_topic, qos)
        self.sub = self.create_subscription(PoseStamped, input_topic, self._on_pose, 100)
        self.timer = self.create_timer(
            1.0 / float(self.get_parameter('publish_rate').value), self._publish)

        self.get_logger().info(
            f'Accumulating raw UWB trail from {input_topic} -> {output_topic}, '
            f'max_points={max_points or "unlimited"}, z_mode={self.z_mode}')

    def _on_pose(self, msg: PoseStamped):
        p = msg.pose.position
        z = self.fixed_z if self.z_mode == 'fixed' else float(p.z)
        self.points.append((float(p.x), float(p.y), z, self.rgb))

    def _publish(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.pub.publish(point_cloud2.create_cloud(header, fields, list(self.points)))


def main(args=None):
    rclpy.init(args=args)
    node = UWBTrailCloudNode()
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
