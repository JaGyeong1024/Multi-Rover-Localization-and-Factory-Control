#!/usr/bin/env python3
"""mock_pickplace_node — TEST AID (stands in for teammate A's pick&place).

Subscribes the host's PickPlaceCommand and replays a PickPlaceStatus sequence
APPROACH -> PICKING -> PLACING -> DONE over `duration` seconds, echoing cmd_id. Lets
the mock demo exercise the host PICK_PLACE phase + 관제 sub-state without the real arm.
The real picker (leader, teammate A) replaces this with its manipulator pipeline.
"""
import rclpy
from rclpy.node import Node

from host_mission_interfaces.msg import PickPlaceCommand, PickPlaceStatus


class MockPickPlace(Node):
    def __init__(self):
        super().__init__('mock_pickplace')
        self.robot = self.declare_parameter('robot', 'leader').value
        self.dur = float(self.declare_parameter('duration', 3.0).value)
        self.create_subscription(PickPlaceCommand, f'/{self.robot}/pickplace_cmd',
                                 self._on_cmd, 10)
        self.pub = self.create_publisher(PickPlaceStatus, f'/{self.robot}/pickplace_status', 10)
        self.cmd_id = 0
        self.part = ''
        self.t0 = None
        self.create_timer(0.1, self._tick)
        self.get_logger().info(f'mock_pickplace[{self.robot}] ready (duration {self.dur}s)')

    def _on_cmd(self, msg):
        if msg.robot != self.robot or msg.cmd_id == self.cmd_id:
            return
        self.cmd_id = msg.cmd_id
        self.part = msg.part_id
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        if self.t0 is None:
            return
        e = self.get_clock().now().nanoseconds * 1e-9 - self.t0
        if e < self.dur * 0.33:
            state = 'APPROACH'
        elif e < self.dur * 0.66:
            state = 'PICKING'
        elif e < self.dur:
            state = 'PLACING'
        else:
            state = 'DONE'
        m = PickPlaceStatus()
        m.header.stamp = self.get_clock().now().to_msg()
        m.robot = self.robot
        m.cmd_id = self.cmd_id
        m.part_id = self.part
        m.state = state
        self.pub.publish(m)


def main():
    rclpy.init()
    node = MockPickPlace()
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
