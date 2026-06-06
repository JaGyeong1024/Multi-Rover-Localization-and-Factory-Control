#!/usr/bin/env python3
"""
ceiling_tracker_node.py
remap 기반 BEV + 색상/YOLO 탐지 + 로버 좌표 퍼블리시
"""

import os
import math
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    from sensor_msgs.msg import Image
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

# ── 공유 경로 ──────────────────────────────────────────────────────────────
DEFAULT_CALIBRATION_DIR  = os.path.join(str(Path.home()), '.factory_calibration')
DEFAULT_CALIBRATION_PATH = os.path.join(DEFAULT_CALIBRATION_DIR, 'calibration.yaml')
DEFAULT_REMAP_PATH        = os.path.join(DEFAULT_CALIBRATION_DIR, 'remap.npz')


class CeilingTrackerNode(Node if HAS_ROS else object):

    def __init__(self):
        if HAS_ROS:
            super().__init__('ceiling_tracker_node')

        # ── 파라미터 ──────────────────────────────────────────────────────
        if HAS_ROS:
            self.declare_parameter('camera_device', '/dev/video2')
            self.declare_parameter('camera_width', 1920)
            self.declare_parameter('camera_height', 1080)
            self.declare_parameter('camera_fps', 30)
            self.declare_parameter('calibration_path', '')
            self.declare_parameter('remap_path', '')
            self.declare_parameter('detection_method', 'color')
            self.declare_parameter('model_path', '')
            self.declare_parameter('yolo_conf_threshold', 0.7)
            self.declare_parameter('publish_rate', 15)
            self.declare_parameter('origin_offset_x', 0.0)
            self.declare_parameter('origin_offset_y', 0.0)
            self.declare_parameter('origin_offset_theta', 0.0)
            self.declare_parameter('color_leader_hsv_low', [35, 100, 100])
            self.declare_parameter('color_leader_hsv_high', [85, 255, 255])
            self.declare_parameter('color_follower_a_hsv_low', [10, 100, 100])
            self.declare_parameter('color_follower_a_hsv_high', [25, 255, 255])
            self.declare_parameter('color_follower_b_hsv_low', [130, 100, 100])
            self.declare_parameter('color_follower_b_hsv_high', [160, 255, 255])
            self.declare_parameter('color_min_area', 200)
            self.declare_parameter('color_max_area', 10000)
            self.declare_parameter('heading_red_hsv_low1', [0, 120, 100])
            self.declare_parameter('heading_red_hsv_high1', [10, 255, 255])
            self.declare_parameter('heading_red_hsv_low2', [160, 120, 100])
            self.declare_parameter('heading_red_hsv_high2', [180, 255, 255])
            self.declare_parameter('heading_blue_hsv_low', [100, 120, 100])
            self.declare_parameter('heading_blue_hsv_high', [130, 255, 255])

            def p(name):
                return self.get_parameter(name).value

            self.camera_device = p('camera_device')
            camera_index = int(self.camera_device.replace('/dev/video', '')) \
                if self.camera_device.startswith('/dev/video') else 2
            self.camera_width   = p('camera_width')
            self.camera_height  = p('camera_height')
            self.camera_fps     = p('camera_fps')
            self.calibration_path = p('calibration_path')
            self.remap_path       = p('remap_path')
            self.detection_method = p('detection_method')
            self.model_path       = p('model_path')
            self.yolo_conf        = p('yolo_conf_threshold')
            self.publish_rate     = p('publish_rate')
            self.origin_offset_x  = p('origin_offset_x')
            self.origin_offset_y  = p('origin_offset_y')
            self.origin_offset_theta = p('origin_offset_theta')

            self.hsv_ranges = {
                'leader':     (np.array(p('color_leader_hsv_low')),
                               np.array(p('color_leader_hsv_high'))),
                'follower_a': (np.array(p('color_follower_a_hsv_low')),
                               np.array(p('color_follower_a_hsv_high'))),
                'follower_b': (np.array(p('color_follower_b_hsv_low')),
                               np.array(p('color_follower_b_hsv_high'))),
            }
            self.min_area = p('color_min_area')
            self.max_area = p('color_max_area')
            self.red_low1  = np.array(p('heading_red_hsv_low1'))
            self.red_high1 = np.array(p('heading_red_hsv_high1'))
            self.red_low2  = np.array(p('heading_red_hsv_low2'))
            self.red_high2 = np.array(p('heading_red_hsv_high2'))
            self.blue_low  = np.array(p('heading_blue_hsv_low'))
            self.blue_high = np.array(p('heading_blue_hsv_high'))
        else:
            # standalone 실행용 기본값
            camera_index = 2
            self.camera_width  = 1920
            self.camera_height = 1080
            self.camera_fps    = 30
            self.calibration_path = ''
            self.remap_path       = ''
            self.detection_method = 'color'
            self.model_path       = ''
            self.yolo_conf        = 0.7
            self.publish_rate     = 15
            self.origin_offset_x  = 0.0
            self.origin_offset_y  = 0.0
            self.origin_offset_theta = 0.0
            self.hsv_ranges = {
                'leader':     (np.array([35, 100, 100]), np.array([85, 255, 255])),
                'follower_a': (np.array([10, 100, 100]), np.array([25, 255, 255])),
                'follower_b': (np.array([130, 100, 100]), np.array([160, 255, 255])),
            }
            self.min_area = 200
            self.max_area = 10000
            self.red_low1  = np.array([0, 120, 100])
            self.red_high1 = np.array([10, 255, 255])
            self.red_low2  = np.array([160, 120, 100])
            self.red_high2 = np.array([180, 255, 255])
            self.blue_low  = np.array([100, 120, 100])
            self.blue_high = np.array([130, 255, 255])

        # ── 캘리브레이션 로드 ─────────────────────────────────────────────
        self.map1 = None
        self.map2 = None
        self.bev_w = self.camera_width
        self.bev_h = self.camera_height
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.real_origin_x = 0.0
        self.real_origin_y = 0.0
        self.ppm = 1.0
        self._load_calibration()

        # ── 카메라 ────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            msg = f"카메라 {camera_index} 열기 실패"
            if HAS_ROS:
                self.get_logger().error(msg)
            else:
                print(f"[오류] {msg}")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_ZOOM, 100)

        # ── YOLO (선택) ───────────────────────────────────────────────────
        self.yolo_model = None
        if self.detection_method == 'yolo' and self.model_path:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.model_path)
                msg = f"YOLO 모델 로드: {self.model_path}"
            except Exception as e:
                msg = f"YOLO 로드 실패: {e} → color 모드로 전환"
                self.detection_method = 'color'
            if HAS_ROS:
                self.get_logger().info(msg)
            else:
                print(msg)

        # ── ROS 퍼블리셔 / 구독자 ─────────────────────────────────────────
        if HAS_ROS:
            # 토픽 분리: legacy BEV는 /{robot}/pose/ceiling (fusion_node의 /{robot}/pose와 충돌 방지).
            self.pose_publishers = {
                name: self.create_publisher(PoseStamped, f'/{name}/pose/ceiling', 10)
                for name in ['leader', 'follower_a', 'follower_b']
            }
            self.debug_pub  = self.create_publisher(Image, '/ceiling_tracker/debug_image', 10)
            self.status_pub = self.create_publisher(String, '/ceiling_tracker/status', 10)
            self.create_timer(1.0 / self.publish_rate, self.timer_callback)
            self.create_timer(1.0, self._publish_status)
            self.get_logger().info(
                f"CeilingTrackerNode 시작 | detection={self.detection_method} | "
                f"cal={'로드됨' if self.map1 is not None else '없음'}")

        self._tracking_count = 0

    # ── 캘리브레이션 로드 ──────────────────────────────────────────────────

    def _load_calibration(self):
        cal_path  = self.calibration_path  or DEFAULT_CALIBRATION_PATH
        remap_path = self.remap_path        or DEFAULT_REMAP_PATH

        if not os.path.isfile(cal_path) or not os.path.isfile(remap_path):
            msg = '캘리브레이션 파일 미발견 — 보정 없이 원본 사용'
            if HAS_ROS:
                self.get_logger().warn(msg)
            else:
                print(f"[경고] {msg}")
            return

        with open(cal_path) as f:
            data = yaml.safe_load(f)

        self.bev_w         = int(data['bev_width'])
        self.bev_h         = int(data['bev_height'])
        self.scale_x       = float(data['scale_x'])
        self.scale_y       = float(data['scale_y'])
        self.ppm           = float(data['pixels_per_meter'])
        self.real_origin_x = float(data['real_origin_x'])
        self.real_origin_y = float(data['real_origin_y'])

        npz = np.load(remap_path)
        self.map1 = npz['map1']
        self.map2 = npz['map2']
        msg = f"캘리브레이션 로드 완료 (BEV {self.bev_w}×{self.bev_h}, ppm={self.ppm})"
        if HAS_ROS:
            self.get_logger().info(msg)
        else:
            print(msg)

    # ── BEV 변환 ──────────────────────────────────────────────────────────

    def _frame_to_bev(self, frame):
        if self.map1 is not None:
            return cv2.remap(frame, self.map1, self.map2,
                             cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        return frame

    def _bev_pixel_to_meter(self, px, py):
        x_m = px * self.scale_x + self.real_origin_x
        y_m = (self.bev_h - py) * self.scale_y + self.real_origin_y
        x_m -= self.origin_offset_x
        y_m -= self.origin_offset_y
        return x_m, y_m

    # ── 색상 탐지 ─────────────────────────────────────────────────────────

    def _detect_color(self, bev):
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        detections = {}
        for name, (low, high) in self.hsv_ranges.items():
            mask = cv2.inRange(hsv, low, high)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            best = None
            best_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if self.min_area <= area <= self.max_area and area > best_area:
                    best_area = area
                    best = cnt
            if best is not None:
                M = cv2.moments(best)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    x1, y1, w, h = cv2.boundingRect(best)
                    detections[name] = {
                        'center': (cx, cy),
                        'bbox': (x1, y1, x1 + w, y1 + h),
                        'area': best_area,
                    }
        return detections

    # ── YOLO 탐지 ─────────────────────────────────────────────────────────

    def _detect_yolo(self, bev):
        if self.yolo_model is None:
            return {}
        results = self.yolo_model(bev, conf=self.yolo_conf, verbose=False)
        detections = {}
        name_map = {0: 'leader', 1: 'follower_a', 2: 'follower_b'}
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in name_map:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections[name_map[cls]] = {
                    'center': (cx, cy),
                    'bbox': (x1, y1, x2, y2),
                }
        return detections

    # ── 헤딩 추정 ─────────────────────────────────────────────────────────

    def _estimate_heading(self, bev, bbox=None):
        if bbox is None:
            return 0.0
        x1, y1, x2, y2 = [int(v) for v in bbox]
        margin = 20
        roi = bev[max(0, y1 - margin):min(bev.shape[0], y2 + margin),
                  max(0, x1 - margin):min(bev.shape[1], x2 + margin)]
        if roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.red_low1, self.red_high1),
            cv2.inRange(hsv, self.red_low2, self.red_high2))
        blue_mask = cv2.inRange(hsv, self.blue_low, self.blue_high)

        def centroid(mask):
            M = cv2.moments(mask)
            if M['m00'] < 10:
                return None
            return (M['m10'] / M['m00'], M['m01'] / M['m00'])

        red_c  = centroid(red_mask)
        blue_c = centroid(blue_mask)
        if red_c is None or blue_c is None:
            return 0.0

        theta = math.atan2(-(red_c[1] - blue_c[1]), red_c[0] - blue_c[0])
        return theta - self.origin_offset_theta

    # ── 메인 루프 ─────────────────────────────────────────────────────────

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        bev = self._frame_to_bev(frame)

        if self.detection_method == 'yolo':
            detections = self._detect_yolo(bev)
        else:
            detections = self._detect_color(bev)

        self._tracking_count = len(detections)
        debug = bev.copy()

        for rover_name in ['leader', 'follower_a', 'follower_b']:
            if rover_name not in detections:
                continue
            d = detections[rover_name]
            cx_px, cy_px = d['center']
            x_m, y_m = self._bev_pixel_to_meter(cx_px, cy_px)
            theta = self._estimate_heading(bev, d.get('bbox'))

            if HAS_ROS:
                self.pose_publishers[rover_name].publish(
                    self._create_pose_stamped(x_m, y_m, theta))

            # 디버그 오버레이
            cv2.circle(debug, (int(cx_px), int(cy_px)), 8, (0, 255, 0), -1)
            arrow_len = 30
            ex = int(cx_px + arrow_len * math.cos(theta))
            ey = int(cy_px - arrow_len * math.sin(theta))
            cv2.arrowedLine(debug, (int(cx_px), int(cy_px)), (ex, ey),
                            (0, 0, 255), 2)
            cv2.putText(debug, f"{rover_name} ({x_m:.2f},{y_m:.2f})",
                        (int(cx_px) + 10, int(cy_px) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if HAS_ROS:
            try:
                from cv_bridge import CvBridge
                bridge = CvBridge()
                self.debug_pub.publish(bridge.cv2_to_imgmsg(debug, 'bgr8'))
            except Exception:
                pass

        cv2.imshow('Ceiling Tracker', debug)
        cv2.waitKey(1)

    def _publish_status(self):
        if not HAS_ROS:
            return
        msg = String()
        msg.data = f"tracking:{self._tracking_count}"
        self.status_pub.publish(msg)

    def _create_pose_stamped(self, x_m, y_m, theta):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = x_m
        msg.pose.position.y = y_m
        msg.pose.position.z = 0.0
        # theta → quaternion (z축 회전)
        msg.pose.orientation.w = math.cos(theta / 2)
        msg.pose.orientation.z = math.sin(theta / 2)
        return msg

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        if HAS_ROS:
            super().destroy_node()


def main(args=None):
    if not HAS_ROS:
        # ROS 없는 환경에서도 standalone으로 실행 가능
        print("[INFO] ROS 없음 — standalone 모드로 실행")
        node = CeilingTrackerNode()
        try:
            while True:
                node.timer_callback()
        except KeyboardInterrupt:
            pass
        finally:
            node.cap.release()
            cv2.destroyAllWindows()
        return

    rclpy.init(args=args)
    node = CeilingTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
