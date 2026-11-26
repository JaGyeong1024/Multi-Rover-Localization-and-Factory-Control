#!/usr/bin/env python3
"""cmd_vel_mux: picks one base velocity source by priority.

    align > pnp (only while /pnp/working is true) > nav

A source counts only if its last message is newer than input_timeout_sec.
With no fresh source the mux publishes a zero Twist.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class CmdVelMux(Node):
    def __init__(self):
        super().__init__('cmd_vel_mux')
        g = lambda k, v: self.declare_parameter(k, v).value
        self.out_topic = g('output_topic', '/cmd_vel')
        self.align_topic = g('align_topic', '/cmd_vel_align')
        self.pnp_topic = g('pnp_topic', '/cmd_vel_pnp')
        self.nav_topic = g('nav_topic', '/leader/cmd_vel')
        self.working_topic = g('pnp_working_topic', '/pnp/working')
        self.timeout = float(g('input_timeout_sec', 0.5))
        rate = float(g('publish_rate', 50.0))

        self.last = {'align': None, 'pnp': None, 'nav': None}  # key -> (t, Twist)
        self.pnp_working = False
        self._active = '<init>'

        self.pub = self.create_publisher(Twist, self.out_topic, 10)
        self.create_subscription(Twist, self.align_topic, lambda m: self._set('align', m), 10)
        self.create_subscription(Twist, self.pnp_topic, lambda m: self._set('pnp', m), 10)
        self.create_subscription(Twist, self.nav_topic, lambda m: self._set('nav', m), 10)
        self.create_subscription(Bool, self.working_topic,
                                 lambda m: setattr(self, 'pnp_working', bool(m.data)), 10)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'cmd_vel_mux: align[{self.align_topic}] > '
            f'pnp[{self.pnp_topic}@{self.working_topic}] > nav[{self.nav_topic}] -> {self.out_topic}')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _set(self, key, msg):
        self.last[key] = (self._now(), msg)

    def _fresh(self, key):
        e = self.last[key]
        return e is not None and (self._now() - e[0]) <= self.timeout

    def _tick(self):
        if self._fresh('align'):
            src = 'align'
        elif self.pnp_working and self._fresh('pnp'):
            src = 'pnp'
        elif self._fresh('nav'):
            src = 'nav'
        else:
            src = None
        if src != self._active:
            self.get_logger().info(f'cmd_vel_mux active -> {src or "STOP(0)"}')
            self._active = src
        self.pub.publish(self.last[src][1] if src else Twist())


def main():
    rclpy.init()
    node = CmdVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.ok() and rclpy.shutdown()


if __name__ == '__main__':
    main()
