#!/usr/bin/env python3
"""check_imu_sync — one-shot clock-sync sanity check for the bridged rover IMU.

Measures each topic's receive delay = host_clock_now - msg.header.stamp, which is
(rover↔host clock skew) + (bridge/WiFi latency). With chrony sub-ms sync the delay
is just WiFi latency (single-digit ms); a large delay means clock skew, so the
IMU and host UWB timestamps would misalign in a co-recorded bag.

Run automatically ~4 s after rover_bridge.launch.py starts (one-shot), or:
    ros2 run host_localization check_imu_sync
Exits after sampling (or timeout). Prints PASS / WARN(skew) / NO DATA per topic;
does NOT block the bridge.
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class CheckImuSync(Node):
    def __init__(self):
        super().__init__('check_imu_sync')
        self.declare_parameter('topics', ['/leader/imu', '/follower/imu'])
        self.declare_parameter('samples', 15)
        self.declare_parameter('timeout', 6.0)
        # WiFi IMU latency is a few ms; clock skew shows as 10s-100s of ms.
        self.declare_parameter('warn_ms', 30.0)

        self.topics = list(self.get_parameter('topics').value)
        self.samples = int(self.get_parameter('samples').value)
        self.timeout = float(self.get_parameter('timeout').value)
        self.warn_ms = float(self.get_parameter('warn_ms').value)
        self.delays = {t: [] for t in self.topics}
        # IMU는 best_effort(sensor_data)로 발행 — 구독자도 맞춰야 수신됨
        # (reliable로 두면 조용히 0개 수신 → 잘못된 NO DATA).
        for t in self.topics:
            self.create_subscription(
                Imu, t, lambda m, top=t: self._cb(top, m), qos_profile_sensor_data)

    def _cb(self, topic, msg):
        buf = self.delays[topic]
        if len(buf) < self.samples:
            now = self.get_clock().now().nanoseconds
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            buf.append((now - stamp) / 1e6)  # ms

    def done(self):
        return all(len(b) >= self.samples for b in self.delays.values())

    def report(self):
        self.get_logger().debug('── clock-sync (IMU delay = skew + latency) ──')
        worst_ok = True
        for t in self.topics:
            b = self.delays[t]
            if not b:
                self.get_logger().warn(
                    f'{t}: NO DATA — 로버 bringup 꺼졌거나 토픽 미발행?')
                worst_ok = False
                continue
            mean = sum(b) / len(b)
            if mean < self.warn_ms:
                self.get_logger().debug(f'{t}: PASS  ~{mean:.1f}ms (동기 양호, n={len(b)})')
            else:
                self.get_logger().warn(
                    f'{t}: ⚠️ {mean:.1f}ms — 클럭 skew 의심 '
                    f'(chronyc tracking 확인; co-record 타임스탬프 어긋남)')
                worst_ok = False
        self.get_logger().info(
            '결과: ' + ('전부 OK ✅' if worst_ok else '주의 필요 ⚠️ (위 참조)'))


def main(args=None):
    rclpy.init(args=args)
    node = CheckImuSync()
    deadline = time.time() + node.timeout
    while time.time() < deadline and not node.done():
        rclpy.spin_once(node, timeout_sec=0.2)
    node.report()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
