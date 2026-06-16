#!/usr/bin/env python3
"""mock_pickplace_node — TEST AID (stands in for 택주's pick&place).

New contract: subscribes the host's /pnp/start (Empty) and replays the '작업중' flag
/pnp/working (Bool) — true on start, false after `duration` seconds. The host completes
its LOADING phase on the working-flag FALLING edge (success/fail not modeled). Lets the
mock demo exercise the LOADING phase without the real arm; 택주 replaces this with the
manipulator pipeline.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty


class MockPnP(Node):
    def __init__(self):
        super().__init__('mock_pnp')
        self.dur = float(self.declare_parameter('duration', 3.0).value)
        self.working = False
        self.t0 = None
        self.create_subscription(Empty, '/pnp/start', self._on_start, 10)
        self.pub = self.create_publisher(Bool, '/pnp/working', 10)
        self.create_timer(0.1, self._tick)
        self.get_logger().info(f'mock_pnp ready (duration {self.dur}s)')

    def _set(self, value):
        self.working = value
        self.pub.publish(Bool(data=value))

    def _on_start(self, _msg):
        self.t0 = self.get_clock().now().nanoseconds * 1e-9
        self._set(True)
        self.get_logger().info('pnp start -> working=true')

    def _tick(self):
        if self.t0 is None:
            return
        if self.working and self.get_clock().now().nanoseconds * 1e-9 - self.t0 >= self.dur:
            self._set(False)
            self.t0 = None
            self.get_logger().info('pnp done -> working=false')


def main():
    rclpy.init()
    node = MockPnP()
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
