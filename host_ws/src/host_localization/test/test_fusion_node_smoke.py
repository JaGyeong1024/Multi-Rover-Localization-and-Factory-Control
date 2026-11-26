"""fusion_node wiring check: the real node as a process, fed over real topics.

The estimator and the output gating are tested without ROS (test_fusion_filter.py,
test_frontend.py). This test only guards what those cannot see: topic names, QoS
compatibility and the node's glue around RobotFusion.
"""
import math
import subprocess
import sys
import time

import pytest

rclpy = pytest.importorskip('rclpy')
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Imu  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
SPEED = 0.20        # [m/s] straight drive along +x, above the course gate
RATE = 20.0         # [Hz] camera and IMU
DRIVE_S = 6.0       # 1.2 m: enough course samples to anchor the heading
SILENCE_S = 2.5     # longer than pose_timeout and bev_timeout (1 s each)


class _Probe:
    def __init__(self, node):
        self.poses, self.aligned, self.quality = [], [], []
        node.create_subscription(PoseStamped, '/follower/pose', self.poses.append, qos_profile_sensor_data)
        node.create_subscription(Bool, '/follower/heading_aligned',
                                 lambda m: self.aligned.append(m.data), LATCHED)
        node.create_subscription(String, '/follower/localization_quality',
                                 lambda m: self.quality.append(m.data), LATCHED)


def _spin_for(node, seconds, tick=None):
    end = time.monotonic() + seconds
    next_tick = time.monotonic()
    while time.monotonic() < end:
        if tick is not None and time.monotonic() >= next_tick:
            tick()
            next_tick += 1.0 / RATE
        rclpy.spin_once(node, timeout_sec=0.005)


@pytest.fixture
def fusion_process():
    proc = subprocess.Popen([sys.executable, '-m', 'host_localization.fusion.fusion_node'])
    yield proc
    proc.terminate()
    proc.wait(timeout=5)


def test_node_publishes_pose_align_quality(fusion_process):
    rclpy.init()
    node = rclpy.create_node('fusion_smoke_probe')
    try:
        probe = _Probe(node)
        bev_pub = node.create_publisher(PoseStamped, '/perception/follower/bev_pose', qos_profile_sensor_data)
        imu_pub = node.create_publisher(Imu, '/follower/imu', qos_profile_sensor_data)
        start = time.monotonic()

        def drive():
            stamp = node.get_clock().now().to_msg()
            bev = PoseStamped()
            bev.header.stamp = stamp
            bev.pose.position.x = 1.0 + SPEED * (time.monotonic() - start)
            bev.pose.position.y = 2.0
            bev.pose.orientation.w = 1.0            # identity = no box axis
            bev_pub.publish(bev)
            imu = Imu()
            imu.header.stamp = stamp
            imu.orientation.z, imu.orientation.w = math.sin(0.6), math.cos(0.6)   # arbitrary IMU zero
            imu_pub.publish(imu)

        _spin_for(node, 2.0)                        # discovery
        _spin_for(node, DRIVE_S, drive)
        assert fusion_process.poll() is None, 'fusion_node exited'
        assert len(probe.poses) > DRIVE_S * RATE, 'pose is published on every measurement'
        assert probe.aligned and probe.aligned[0] is False and probe.aligned[-1] is True
        assert probe.quality[-1] == 'GOOD'
        last = probe.poses[-1].pose
        heading = 2.0 * math.atan2(last.orientation.z, last.orientation.w)
        assert abs(heading) < math.radians(10.0), 'heading follows the driven course, not the IMU zero'
        assert abs(last.position.y - 2.0) < 0.05

        n_before = len(probe.poses)

        def imu_only():
            imu = Imu()
            imu.header.stamp = node.get_clock().now().to_msg()
            imu.orientation.z, imu.orientation.w = math.sin(0.6), math.cos(0.6)
            imu_pub.publish(imu)

        _spin_for(node, SILENCE_S, imu_only)
        n_at_timeout = len(probe.poses)
        _spin_for(node, 1.0, imu_only)
        assert len(probe.poses) == n_at_timeout, 'no pose without a position source (IMU only)'
        assert n_at_timeout - n_before < 1.5 * RATE, 'pose stops about pose_timeout after the last fix'
        assert probe.quality[-1] == 'LOST'
    finally:
        node.destroy_node()
        rclpy.shutdown()
