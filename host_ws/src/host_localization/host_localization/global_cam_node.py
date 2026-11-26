#!/usr/bin/env python3
"""Ceiling camera node: one frozen-control capture, published at several stages.

    image_raw/compressed   raw capture (JPEG)                  [always]
    camera_info            K, D                                [if intrinsic]
    image_rect             undistort(K, D), uncompressed       [if intrinsic]
    bev                    undistort+H fused remap, uncompressed [if remap]
    processed/compressed   bev cropped to ROI, JPEG            [if remap]

Only raw and processed are compressed. Consumers (localization, YOLO) use `processed`.
Undistortion uses the full 14-coeff rational D via cv2 (see build_combined_remap).
"""
import os
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from host_localization.camera_io import DEFAULT_CONTROLS, ReconnectingCapture, open_camera


def _sensor_qos(depth):
    """best_effort QoS for image/camera streams."""
    return QoSProfile(depth=depth, reliability=ReliabilityPolicy.BEST_EFFORT)


def _distortion_model(n_coeffs):
    """Approximate ROS distortion_model label (the full D is still in camera_info.d)."""
    return 'plumb_bob' if n_coeffs <= 5 else 'rational_polynomial'


class GlobalCamNode(Node):
    def __init__(self):
        super().__init__('global_cam_node')

        self.declare_parameter('device', '/dev/global_cam')
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('fps', 30)
        self.declare_parameter('compressed_topic', '/global_cam/image_raw/compressed')
        self.declare_parameter('camera_info_topic', '/global_cam/camera_info')
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('frame_id', 'global_cam')
        self.declare_parameter('intrinsic_path',
                               os.path.expanduser('~/charuco_intrinsics.npz'))
        # ── processing pipeline ──────────────────────────────────────────────
        self.declare_parameter('remap_path',
                               os.path.expanduser('~/.factory_calibration/remap.npz'))
        self.declare_parameter('calibration_path',
                               os.path.expanduser('~/.factory_calibration/calibration.yaml'))
        self.declare_parameter('rect_topic', '/global_cam/image_rect')
        self.declare_parameter('bev_topic', '/global_cam/bev')
        self.declare_parameter('processed_topic', '/global_cam/processed/compressed')
        self.declare_parameter('publish_rect', True)
        self.declare_parameter('publish_bev', True)
        self.declare_parameter('publish_processed', True)
        # processed-crop ROI [m, uwb_global]; fallback for global_cam_params.yaml (perception reads it too)
        self.declare_parameter('roi_x', [1.63, 5.73])
        self.declare_parameter('roi_y', [2.10, 4.64])

        device = self.get_parameter('device').value
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        fps = int(self.get_parameter('fps').value)
        self.quality = int(self.get_parameter('jpeg_quality').value)
        self.frame_id = self.get_parameter('frame_id').value
        intrinsic_path = os.path.expanduser(self.get_parameter('intrinsic_path').value)

        # FULL freeze (geometric + photometric). Defaults from camera_io SoT.
        controls = {}
        for key, default in DEFAULT_CONTROLS.items():
            self.declare_parameter(key, default)
            controls[key] = int(self.get_parameter(key).value)

        self.K = None
        self.D = None
        self.camera_info = self._load_camera_info(intrinsic_path, width, height)
        self.bridge = CvBridge()

        # Stage maps. rect = undistort(K,D); bev = fused undistort+H (remap.npz).
        self.rect_map = self._build_rect_map(width, height)
        self.bev_map, self.roi = self._load_bev()

        # one open path for the first open and every reopen (same resolution/fps/controls)
        self.cap = ReconnectingCapture(
            lambda: open_camera(device, width, height, fps, controls=controls, verbose=False),
            time.monotonic, on_event=self._on_cam_event)

        self.pub_img = self.create_publisher(
            CompressedImage, self.get_parameter('compressed_topic').value, _sensor_qos(5))
        self.pub_info = self.create_publisher(
            CameraInfo, self.get_parameter('camera_info_topic').value, _sensor_qos(5))
        self.pub_rect = self._maybe_pub('publish_rect', self.rect_map is not None,
                                        Image, 'rect_topic')
        self.pub_bev = self._maybe_pub('publish_bev', self.bev_map is not None,
                                       Image, 'bev_topic')
        self.pub_processed = self._maybe_pub(
            'publish_processed', self.bev_map is not None,
            CompressedImage, 'processed_topic', depth=5, reliable=True)  # web_video_server compatible

        stages = ['raw'] + (['info'] if self.camera_info else []) \
            + (['rect'] if self.pub_rect else []) + (['bev'] if self.pub_bev else []) \
            + (['processed'] if self.pub_processed else [])
        self.get_logger().info(
            f'global cam {device} {width}x{height} MJPG @ {fps}, FULL freeze; '
            f'stages: {", ".join(stages)}')

        self.create_timer(1.0 / max(1, fps), self._tick)

    def _on_cam_event(self, kind, msg):
        if kind == 'recovered':
            self.get_logger().info(msg)
        else:
            self.get_logger().warn(msg, throttle_duration_sec=10.0)

    def _maybe_pub(self, flag_param, available, msg_type, topic_param, depth=2, reliable=False):
        if not (self.get_parameter(flag_param).value and available):
            return None
        # reliable=True so web_video_server can receive; best_effort subscribers still work.
        qos = (QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE)
               if reliable else _sensor_qos(depth))
        return self.create_publisher(msg_type, self.get_parameter(topic_param).value, qos)

    def _load_camera_info(self, path, width, height):
        if not os.path.isfile(path):
            self.get_logger().warn(f'intrinsic 없음: {path} — camera_info/rect 미발행')
            return None
        data = np.load(path)
        self.K = np.asarray(data['K'], dtype=float).reshape(3, 3)
        self.D = np.asarray(data['D'], dtype=float).reshape(1, -1)

        info = CameraInfo()
        info.width = int(data['image_width']) if 'image_width' in data.files else width
        info.height = int(data['image_height']) if 'image_height' in data.files else height
        info.distortion_model = _distortion_model(self.D.size)
        info.d = self.D.flatten().tolist()
        info.k = self.K.flatten().tolist()
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [self.K[0, 0], 0.0, self.K[0, 2], 0.0,
                  0.0, self.K[1, 1], self.K[1, 2], 0.0,
                  0.0, 0.0, 1.0, 0.0]
        if self.D.size not in (5, 8):
            # not an error: the pipeline uses the full D
            self.get_logger().debug(
                f'D has {self.D.size} coeffs (non-standard for ROS); full vector '
                f'kept in camera_info.d, label "{info.distortion_model}".')
        self.get_logger().debug(f'camera_info from {path} '
                                f'({info.width}x{info.height}, D={self.D.size})')
        return info

    def _build_rect_map(self, width, height):
        """Undistort map from the full D (cv2 handles the 14-coeff rational model)."""
        if self.K is None or not self.get_parameter('publish_rect').value:
            return None
        m1, m2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.K, (width, height), cv2.CV_32FC1)
        return (m1, m2)

    def _load_bev(self):
        """remap.npz (fused undistort+H) + calibration.yaml -> bev maps and the
        world-ROI -> BEV-pixel crop for `processed`. Returns (None, None) if absent."""
        remap_path = os.path.expanduser(self.get_parameter('remap_path').value)
        calib_path = os.path.expanduser(self.get_parameter('calibration_path').value)
        if not (os.path.isfile(remap_path) and os.path.isfile(calib_path)):
            self.get_logger().warn('remap.npz/calibration.yaml 없음 — bev/processed 미발행')
            return None, None
        d = np.load(remap_path)
        bev_map = (d['map1'], d['map2'])
        c = yaml.safe_load(open(calib_path))
        ppm, ox, oy = c['pixels_per_meter'], c['real_origin_x'], c['real_origin_y']
        bw, bh = c['bev_width'], c['bev_height']
        rx = self.get_parameter('roi_x').value
        ry = self.get_parameter('roi_y').value
        # world -> BEV px (y inverted). clamp to canvas.
        x0, x1 = sorted(int(round((x - ox) * ppm)) for x in rx)
        y0, y1 = sorted(int(round(bh - (y - oy) * ppm)) for y in ry)
        roi = (max(0, y0), min(bh, y1), max(0, x0), min(bw, x1))
        self.get_logger().debug(
            f'bev {bw}x{bh}; processed ROI px rows[{roi[0]}:{roi[1]}] cols[{roi[2]}:{roi[3]}]')
        return bev_map, roi

    def _tick(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('frame grab failed', throttle_duration_sec=2.0)
            return
        stamp = self.get_clock().now().to_msg()

        self._pub_compressed(self.pub_img, frame, stamp)
        if self.camera_info is not None:
            self.camera_info.header.stamp = stamp
            self.camera_info.header.frame_id = self.frame_id
            self.pub_info.publish(self.camera_info)

        if self.pub_rect is not None:
            rect = cv2.remap(frame, self.rect_map[0], self.rect_map[1], cv2.INTER_LINEAR)
            self._pub_image(self.pub_rect, rect, stamp)

        bev = None
        if self.bev_map is not None and (self.pub_bev or self.pub_processed):
            bev = cv2.remap(frame, self.bev_map[0], self.bev_map[1], cv2.INTER_LINEAR)
        if self.pub_bev is not None and bev is not None:
            self._pub_image(self.pub_bev, bev, stamp)
        if self.pub_processed is not None and bev is not None:
            y0, y1, x0, x1 = self.roi
            self._pub_compressed(self.pub_processed, bev[y0:y1, x0:x1], stamp)

    def _pub_image(self, pub, img, stamp):
        msg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        pub.publish(msg)

    def _pub_compressed(self, pub, img, stamp):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        # inside the try: opening the camera takes seconds, so Ctrl+C usually lands here
        node = GlobalCamNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
