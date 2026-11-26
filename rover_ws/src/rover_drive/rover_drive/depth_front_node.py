"""depth_front: nearest distance ahead + a planar scan from the front depth camera (leader)."""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Float32

from .core.config import DepthFrontConfig
from .core.depth_front import (decode_depth, depth_scan, front_distance,
                               intrinsics_from_camera_info, intrinsics_from_fov)
from .ros_util import declare_config, now_s

_WARN_PERIOD_S = 5.0


class _RateLimit:
    """True at most once per 1 / rate_hz seconds."""

    def __init__(self, rate_hz: float):
        self._period = 1.0 / rate_hz
        self._last = None

    def due(self, now: float) -> bool:
        if self._last is not None and now - self._last < self._period:
            return False
        self._last = now
        return True


class DepthFrontNode(Node):

    def __init__(self):
        super().__init__('depth_front')
        cfg = self._cfg = declare_config(self, DepthFrontConfig)
        out_topic = cfg.front_distance_topic or f'/{cfg.robot}/front_distance'
        self._pub = self.create_publisher(Float32, out_topic, 10)
        self._front_due = _RateLimit(cfg.depth_rate_hz)
        self._scan_pub = None
        self._camera_info = None              # (k, width) of the latest calibrated info
        if cfg.depth_scan_enabled:
            scan_topic = cfg.depth_scan_topic or f'/{cfg.robot}/depth_scan'
            self._scan_pub = self.create_publisher(LaserScan, scan_topic,
                                                   qos_profile_sensor_data)
            self._scan_due = _RateLimit(cfg.depth_scan_rate_hz)
            if cfg.depth_camera_info_topic:
                self.create_subscription(CameraInfo, cfg.depth_camera_info_topic,
                                         self._on_camera_info, qos_profile_sensor_data)
            self.get_logger().info(f'depth_front: scan -> {scan_topic} '
                                   f'(frame {cfg.depth_scan_frame})')
        self.create_subscription(Image, cfg.depth_topic, self._on_image,
                                 qos_profile_sensor_data)
        self.get_logger().info(f'depth_front: {cfg.depth_topic} -> {out_topic} '
                               f'depth_scan_enabled={cfg.depth_scan_enabled}')

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if msg.k[0] > 0.0 and msg.width > 0:
            if self._camera_info is None:
                self.get_logger().info(f'depth_front: intrinsics from camera_info '
                                       f'(fx {msg.k[0]:.1f}, cx {msg.k[2]:.1f})')
            self._camera_info = (tuple(msg.k), int(msg.width))

    def _on_image(self, msg: Image) -> None:
        now = now_s()
        front = self._front_due.due(now)
        scan = self._scan_pub is not None and self._scan_due.due(now)
        if not (front or scan):
            return
        depth = self._decode(msg)
        if front:
            dist = math.nan if depth is None else front_distance(depth, self._cfg)
            self._pub.publish(Float32(data=dist))
        if scan and depth is not None:
            self._scan_pub.publish(self._scan_msg(depth, msg.header))

    def _decode(self, msg: Image):
        try:
            return decode_depth(msg.data, msg.height, msg.width, msg.encoding, msg.step,
                                bool(msg.is_bigendian))
        except ValueError as exc:  # wrong encoding or truncated data: unknown, not a crash
            self.get_logger().warn(f'depth image not usable: {exc}',
                                   throttle_duration_sec=_WARN_PERIOD_S)
            return None

    def _intrinsics(self, width: int):
        if self._camera_info is not None:
            intr = intrinsics_from_camera_info(*self._camera_info, width)
            if intr is not None:
                return intr
        return intrinsics_from_fov(width, self._cfg.depth_hfov)

    def _scan_msg(self, depth, header) -> LaserScan:
        scan = depth_scan(depth, self._intrinsics(depth.shape[1]), self._cfg)
        msg = LaserScan()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self._cfg.depth_scan_frame
        msg.angle_min = float(scan.angle_min)
        msg.angle_max = float(scan.angle_max)
        msg.angle_increment = float(scan.angle_increment)
        msg.scan_time = 1.0 / self._cfg.depth_scan_rate_hz
        msg.range_min = float(scan.range_min)
        msg.range_max = float(scan.range_max)
        msg.ranges = scan.ranges.astype('float32').tolist()
        return msg


def main() -> None:
    rclpy.init()
    node = DepthFrontNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
