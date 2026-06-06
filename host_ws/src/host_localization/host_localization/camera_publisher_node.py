#!/usr/bin/env python3
"""camera_publisher_node — global/ceiling BRIO publisher with the FROZEN v4l2 pipeline.

Intrinsic phase: applies the GEOMETRIC layer only (1080p MJPG + zoom 1x + autofocus
off + fixed focus) — the settings that fix the intrinsic matrix (K, D). Exposure /
gain / white-balance are left on AUTO because they do not affect K,D. This is the
node handed to whoever computes the intrinsics. The FINAL runtime node additionally
freezes the photometric layer (camera_tuner values) and publishes camera_info(K,D).

Publishes COMPRESSED only (raw 1080p BGR8 @ 30fps is ~1.5 Gbps — too heavy to ship
to a teammate). Consumers subscribe with the compressed transport, e.g. ROS
camera_calibration:
    ros2 run camera_calibration cameracalibrator --size 9x6 --square 0.025 \\
        --ros-args -r image:=/global_cam/image_raw -p image_transport:=compressed
(`image:=` uses the BASE name; image_transport finds `<base>/compressed`.)

The v4l2 freeze controls are parameters (defaults = camera_io.DEFAULT_CONTROLS).
Replace them with camera_tuner.py's output BEFORE calibration so calibration,
distortion, YOLO, and runtime share one pixel pipeline (config/global_cam_params.yaml).

frame_id is a placeholder ('global_cam') — intrinsic calibration doesn't use TF.
The FINAL runtime node additionally publishes camera_info with the returned K, D.
"""
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from host_localization.camera_io import open_camera, GEOMETRIC_CONTROLS


class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')

        self.declare_parameter('device', '/dev/global_cam')
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('fps', 30)
        self.declare_parameter('compressed_topic', '/global_cam/image_raw/compressed')
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('frame_id', 'global_cam')

        device = self.get_parameter('device').value
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        fps = int(self.get_parameter('fps').value)
        compressed_topic = self.get_parameter('compressed_topic').value
        self.quality = int(self.get_parameter('jpeg_quality').value)
        self.frame_id = self.get_parameter('frame_id').value

        # Layer 1 (GEOMETRIC) ONLY — fixes zoom/focus so the intrinsic K,D transfers
        # to runtime. Photometric (exposure/gain/WB) is left on AUTO; it does NOT
        # affect K,D. The FINAL runtime node adds the photometric freeze + camera_info.
        controls = {}
        for key, default in GEOMETRIC_CONTROLS.items():
            self.declare_parameter(key, default)
            controls[key] = int(self.get_parameter(key).value)
        controls['auto_exposure'] = 3   # auto exposure for intrinsic capture

        self.cap = open_camera(device, width, height, fps, controls=controls, verbose=False)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f'global cam {device} {w}x{h} MJPG @ {fps}, geometric fixed '
            f'(zoom/focus), exposure AUTO -> {compressed_topic} (JPEG q={self.quality}, compressed-only)')

        self.pub = self.create_publisher(CompressedImage, compressed_topic, 5)
        self.create_timer(1.0 / max(1, fps), self._tick)

    def _tick(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('frame grab failed', throttle_duration_sec=2.0)
            return
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self.pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
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
