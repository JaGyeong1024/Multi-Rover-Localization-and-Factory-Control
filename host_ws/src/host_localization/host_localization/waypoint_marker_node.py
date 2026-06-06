#!/usr/bin/env python3
"""waypoint_marker_node — interactive ground-truth labeller for UWB bags.

Type a waypoint id (1..N) and Enter at the moment you have settled on a
grid cell. The node publishes that id on /test/active_waypoint (Int32) at
`rate_hz` for `duration` seconds, then idles until the next input. Record
this topic alongside the UWB topics; in post-processing, samples whose
timestamp lies inside an active_waypoint=N publish window are labelled
GT = waypoint N.

Tip: stay still for longer than `duration` (e.g. 5-6 s of physical hold
for a 3 s label window) so footstep wobble isn't captured in the labelled
segment.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


class WaypointMarkerNode(Node):
    def __init__(self):
        super().__init__('waypoint_marker_node')
        self.declare_parameter('topic', '/test/active_waypoint')
        self.declare_parameter('duration', 3.0)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('min_id', 1)
        self.declare_parameter('max_id', 36)

        topic = self.get_parameter('topic').value
        self.duration = float(self.get_parameter('duration').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.min_id = int(self.get_parameter('min_id').value)
        self.max_id = int(self.get_parameter('max_id').value)

        self.pub = self.create_publisher(Int32, topic, 10)
        self.get_logger().info(
            f'Ready. Topic={topic}, window={self.duration:.1f}s @ '
            f'{self.rate_hz:.1f} Hz, range=[{self.min_id},{self.max_id}].')

    def mark(self, n):
        msg = Int32()
        msg.data = n
        period = 1.0 / self.rate_hz
        end = time.monotonic() + self.duration
        count = 0
        while time.monotonic() < end and rclpy.ok():
            self.pub.publish(msg)
            count += 1
            time.sleep(period)
        self.get_logger().info(f'waypoint {n}: published {count} msgs')


def main():
    rclpy.init()
    node = WaypointMarkerNode()
    prompt = (f'waypoint # ({node.min_id}..{node.max_id}, '
              "'q' to quit): ")
    try:
        while rclpy.ok():
            try:
                s = input(prompt).strip()
            except EOFError:
                break
            if not s:
                continue
            if s.lower() in ('q', 'quit', 'exit'):
                break
            try:
                n = int(s)
            except ValueError:
                print(f'  not an integer: {s!r}', file=sys.stderr)
                continue
            if not (node.min_id <= n <= node.max_id):
                print(f'  out of range [{node.min_id},{node.max_id}]',
                      file=sys.stderr)
                continue
            node.mark(n)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
