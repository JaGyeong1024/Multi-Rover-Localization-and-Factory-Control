#!/usr/bin/env python3
"""perception_node — 천장 BEV YOLO 검출 → /perception/{robot}/bev_pose (역할: 이미지 → 검출).

런타임 파이프라인의 Perception 역할. `/global_cam/processed/compressed`(BEV)에 YOLO-OBB 추론,
검출 centroid를 uwb_global 미터로 투영해 **로버별** pose 발행. 융합·마커·검증·UWB/IMU는
별도(fusion_node) — 이 노드는 *검출만*. (yolo_bev_verify_node에서 검출부만 추출.)

class→robot: 기본 = v2 2클래스 가중치. ⚠ Roboflow가 클래스를 **알파벳순**으로 매기므로
  실제 인덱스는 **0=follower, 1=leader** (직관과 반대!). 기본 `class_map="0:follower,1:leader"`.
  다른 가중치로 교체 시 그 data.yaml 의 names 순서에 맞춰 class_map 만 바꾸면 됨(코드 불변).

환경: `.venv_yolo` python + ROS source (rclpy=PYTHONPATH, ultralytics=venv). CompressedImage는
  cv2.imdecode로 처리(cv_bridge 회피). 헤딩은 fusion에서 IMU 사용 → bev_pose는 *위치 전용*
  (orientation=identity).
출력: /perception/{robot}/bev_pose (PoseStamped, uwb_global) + /perception/{robot}/bbox
  (PolygonStamped, uwb_global — OBB 4코너를 px2world로 점별 정합. ⚠ y-flip은 *코너 변환엔 무관*
   [px2world가 점별 처리]; OBB *각도*→월드헤딩만 flip 필요한데 헤딩=IMU라 각도 미사용 → 코너만 발행).
"""
import os
import math
import yaml
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, PolygonStamped, Point32
from ultralytics import YOLO

WEIGHTS = "~/capstone/data/models/yolo11n-obb_2class_v3.pt"   # v3: 0=follower, 1=leader (299장/2세션)
CALIB = "~/.factory_calibration/calibration.yaml"


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        p = lambda n, d: self.declare_parameter(n, d).value
        weights = os.path.expanduser(p("weights", WEIGHTS))
        calib = os.path.expanduser(p("calib", CALIB))
        self.conf = float(p("conf", 0.5))
        self.imgsz = int(p("imgsz", 960))
        img_topic = p("image_topic", "/global_cam/processed/compressed")
        roi_x = list(p("roi_x", [1.64, 5.72]))
        roi_y = list(p("roi_y", [2.11, 4.63]))
        cmap_s = p("class_map", "0:follower,1:leader")   # v2 가중치 클래스 순서(알파벳순)
        self.class_map = {int(k): v for k, v in (kv.split(":") for kv in cmap_s.split(","))}

        # processed 픽셀 <-> uwb_global 미터 어파인 (calibration.yaml + ROI 좌상단)
        c = yaml.safe_load(open(calib))
        self.ppm = c["pixels_per_meter"]
        self.ox = c["real_origin_x"]
        self.oy = c["real_origin_y"]
        self.bh = c["bev_height"]
        self.x0 = round((roi_x[0] - self.ox) * self.ppm)
        self.y0 = round(self.bh - (roi_y[1] - self.oy) * self.ppm)

        self.model = YOLO(weights)
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=5)
        # bev_pose=내부 고빈도 → BEST_EFFORT(fusion 구독과 일치). bbox=viz(eval_overlay)용 → default 유지.
        self.pubs = {name: self.create_publisher(PoseStamped, f"/perception/{name}/bev_pose", be)
                     for name in self.class_map.values()}
        self.bbox_pubs = {name: self.create_publisher(PolygonStamped, f"/perception/{name}/bbox", 10)
                          for name in self.class_map.values()}   # OBB 4코너(uwb_global, 좌표정합 완료)
        self.create_subscription(CompressedImage, img_topic, self.on_image, be)
        self.n = 0
        self.get_logger().info(f"perception: weights={os.path.basename(weights)} conf={self.conf} "
                               f"classes={self.class_map} -> /perception/<robot>/{{bev_pose,bbox}}")

    def px2world(self, u, v):
        return (float(self.ox + (self.x0 + u) / self.ppm),
                float(self.oy + (self.bh - (self.y0 + v)) / self.ppm))

    def on_image(self, msg):
        if not rclpy.ok():
            return
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, verbose=False, device=0)[0]
        if r.obb is None or len(r.obb) == 0:
            return
        cls = r.obb.cls.cpu().numpy().astype(int)
        conf = r.obb.conf.cpu().numpy()
        xywhr = r.obb.xywhr.cpu().numpy()
        best = {}                                    # 로버별 최고 conf 검출
        for i in range(len(cls)):
            robot = self.class_map.get(int(cls[i]))
            if robot is None:
                continue
            if robot not in best or conf[i] > best[robot][1]:
                best[robot] = (i, conf[i])
        for robot, (i, _) in best.items():
            cx, cy, w, h, rot = (float(v) for v in xywhr[i])
            x, y = self.px2world(cx, cy)
            m = PoseStamped()
            m.header.stamp = msg.header.stamp        # BEV 프레임 시각 보존(시간동기)
            m.header.frame_id = "uwb_global"
            m.pose.position.x, m.pose.position.y = x, y
            m.pose.orientation.w = 1.0               # 위치 전용 (헤딩=IMU @ fusion)
            self.pubs[robot].publish(m)
            # bbox: OBB 4코너 → px2world 점별 정합 → PolygonStamped(uwb_global)
            corners = cv2.boxPoints(((cx, cy), (w, h), math.degrees(rot)))
            poly = PolygonStamped()
            poly.header = m.header
            poly.polygon.points = [Point32(x=wx, y=wy, z=0.0)
                                   for wx, wy in (self.px2world(float(u), float(v)) for u, v in corners)]
            self.bbox_pubs[robot].publish(poly)
        self.n += 1
        if self.n % 60 == 0:
            self.get_logger().debug(f"frames={self.n}, detected={list(best)}")


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
