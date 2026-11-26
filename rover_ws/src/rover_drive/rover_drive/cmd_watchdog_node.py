"""cmd_watchdog: zero the base when the safety gate goes silent."""
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Empty

from .core.config import WatchdogConfig
from .core.watchdog import CmdWatchdog
from .ros_util import declare_config, now_s, run_node


class CmdWatchdogNode(Node):

    def __init__(self):
        super().__init__('cmd_watchdog')
        self._cfg = declare_config(self, WatchdogConfig)
        self._watchdog = CmdWatchdog(self._cfg.watchdog_timeout_s)
        self._holding = None
        alive_topic = self._cfg.alive_topic or f'/{self._cfg.robot}/gate_alive'
        self._pub = self.create_publisher(Twist, self._cfg.output_topic, 10)
        self.create_subscription(Empty, alive_topic, self._on_alive, 10)
        self.create_timer(1.0 / self._cfg.watchdog_rate_hz, self._on_timer)
        self.get_logger().info(
            f'cmd_watchdog[{self._cfg.robot}] {alive_topic} silent for '
            f'{self._cfg.watchdog_timeout_s} s -> zero on {self._cfg.output_topic}')

    def _on_alive(self, _msg: Empty) -> None:
        self._watchdog.on_alive(now_s())

    def _on_timer(self) -> None:
        holding = self._watchdog.must_stop(now_s())
        if holding != self._holding:
            self._holding = holding
            self.get_logger().warn('gate silent: holding the base at zero' if holding
                                   else 'gate alive: released')
        if holding:
            self.publish_stop()

    def publish_stop(self) -> None:
        self._pub.publish(Twist())


def main() -> None:
    run_node(CmdWatchdogNode)


if __name__ == '__main__':
    main()
