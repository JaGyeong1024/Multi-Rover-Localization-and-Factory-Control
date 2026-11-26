#!/usr/bin/env python3
"""Ceiling BEV YOLO-OBB detection -> /perception/{robot}/bev_pose (and /bbox, uwb_global corners).

Runs in `.venv_yolo` with ROS sourced; detection only (fusion lives in fusion_node).
class_map maps class index -> robot and must match the weights' data.yaml order.
bev_pose.orientation = OBB long axis (ambiguous by 180 deg), not the heading;
publish_box_axis:=false gives identity. The ROI origin is read from global_cam_params.yaml
(roi_x/roi_y parameters override). top_height_<robot> [m] enables parallax correction.
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
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PolygonStamped, Point32
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
from host_localization.ceiling_parallax import camera_centre_from_calibration, correct_height_parallax

# repo root: two levels above host_ws/perception/
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.path.join(_REPO, "data/models/yolo11n-obb_2class_v3.pt")   # 0=follower, 1=leader
CALIB = "~/.factory_calibration/calibration.yaml"
INTRINSICS = "~/.factory_calibration/charuco_intrinsics.npz"
CAM_NODE = "global_cam_node"          # yaml section that owns the processed-crop ROI


def default_cam_params():
    return os.path.join(get_package_share_directory("host_localization"),
                        "config", "global_cam_params.yaml")


def load_roi(cam_params):
    """ROI (world metres) as the camera node crops it."""
    with open(os.path.expanduser(cam_params)) as f:
        prm = yaml.safe_load(f)[CAM_NODE]["ros__parameters"]
    return list(prm["roi_x"]), list(prm["roi_y"])


def box_axis_yaw(w, h, rot):
    """OBB long-axis yaw in uwb_global [rad], modulo pi in [-pi/2, pi/2); image y is down (sign flip)."""
    a = -(rot if w >= h else rot + math.pi / 2.0)
    a = (a + math.pi / 2.0) % math.pi - math.pi / 2.0
    return a if a != 0.0 else 1e-9      # exact identity means "no angle"


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        p = lambda n, d: self.declare_parameter(n, d).value
        weights = os.path.expanduser(p("weights", WEIGHTS))
        calib = os.path.expanduser(p("calib", CALIB))
        self.conf = float(p("conf", 0.5))
        self.imgsz = int(p("imgsz", 960))
        img_topic = p("image_topic", "/global_cam/processed/compressed")
        roi_x, roi_y = load_roi(p("cam_params", default_cam_params()))
        roi_x = list(p("roi_x", roi_x))            # override only
        roi_y = list(p("roi_y", roi_y))
        self.publish_axis = bool(p("publish_box_axis", True))
        cmap_s = p("class_map", "0:follower,1:leader")   # default weights: alphabetical class order
        self.class_map = {int(k): v for k, v in (kv.split(":") for kv in cmap_s.split(","))}

        # processed pixel <-> uwb_global metre affine (calibration + ROI top-left)
        c = yaml.safe_load(open(calib))
        self.ppm = c["pixels_per_meter"]
        self.ox = c["real_origin_x"]
        self.oy = c["real_origin_y"]
        self.bh = c["bev_height"]
        self.x0 = round((roi_x[0] - self.ox) * self.ppm)
        self.y0 = round(self.bh - (roi_y[1] - self.oy) * self.ppm)

        # Top-plate height [m] for parallax correction; 0.0 = off.
        self.top_height = {name: float(p(f"top_height_{name}", 0.0)) for name in self.class_map.values()}
        self.cam = self._load_camera_centre(calib, os.path.expanduser(p("intrinsics", INTRINSICS)))

        self.model = YOLO(weights)
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=5)
        # bev_pose is BEST_EFFORT like the fusion subscriber; bbox is visualisation
        self.pubs = {name: self.create_publisher(PoseStamped, f"/perception/{name}/bev_pose", be)
                     for name in self.class_map.values()}
        self.bbox_pubs = {name: self.create_publisher(PolygonStamped, f"/perception/{name}/bbox", 10)
                          for name in self.class_map.values()}   # OBB 4 corners (uwb_global)
        # bev_all: one PoseArray per class per frame (also empty), sorted by confidence,
        # position.z = confidence; lets fusion_node associate class-flipped boxes. Off by default.
        self.publish_all = bool(p("publish_all_boxes", False))
        self.all_pubs = {name: self.create_publisher(PoseArray, f"/perception/{name}/bev_all", be)
                         for name in self.class_map.values()} if self.publish_all else {}
        self.create_subscription(CompressedImage, img_topic, self.on_image, be)
        self.n = 0
        self.get_logger().info(f"perception: roi_x={roi_x} roi_y={roi_y} crop origin px=({self.x0},{self.y0})")
        self.get_logger().info(f"perception: weights={os.path.basename(weights)} conf={self.conf} "
                               f"classes={self.class_map} -> /perception/<robot>/{{bev_pose,bbox}}")
        self.get_logger().info(f"top_height[m]={self.top_height}")

    def _load_camera_centre(self, calib, intrinsics):
        """Camera centre, or None when unusable and no robot needs it."""
        needed = any(h != 0.0 for h in self.top_height.values())
        try:
            cam = camera_centre_from_calibration(calib, intrinsics)
        except (OSError, KeyError, ValueError) as e:
            if needed:
                raise RuntimeError(f"top_height set but camera centre unavailable: {e}") from e
            self.get_logger().warning(f"camera centre unavailable (parallax correction off): {e}")
            return None
        self.get_logger().info(
            f"camera centre: nadir=({cam.nadir_x:.3f}, {cam.nadir_y:.3f}) H={cam.height:.3f} m "
            f"tilt={cam.tilt_deg:.1f} deg (R residuals: norm {cam.norm_mismatch:.4f}, ortho {cam.ortho_residual:.4f})")
        return cam

    def floor_xy(self, robot, u, v):
        """BEV pixel on the rover top -> floor position under it (uwb_global)."""
        x, y = self.px2world(u, v)
        h = self.top_height[robot]
        if h == 0.0:
            return x, y
        return correct_height_parallax(x, y, h, self.cam.nadir_x, self.cam.nadir_y, self.cam.height)

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
        if self.publish_all:
            self._publish_all(msg.header.stamp, r)
        if r.obb is None or len(r.obb) == 0:
            return
        cls = r.obb.cls.cpu().numpy().astype(int)
        conf = r.obb.conf.cpu().numpy()
        xywhr = r.obb.xywhr.cpu().numpy()
        best = {}                                    # best-confidence detection per rover
        for i in range(len(cls)):
            robot = self.class_map.get(int(cls[i]))
            if robot is None:
                continue
            if robot not in best or conf[i] > best[robot][1]:
                best[robot] = (i, conf[i])
        for robot, (i, _) in best.items():
            cx, cy, w, h, rot = (float(v) for v in xywhr[i])
            x, y = self.floor_xy(robot, cx, cy)
            m = PoseStamped()
            m.header.stamp = msg.header.stamp
            m.header.frame_id = "uwb_global"
            m.pose.position.x, m.pose.position.y = x, y
            m.pose.orientation.w = 1.0               # identity = no box angle
            if self.publish_axis:
                yaw = box_axis_yaw(w, h, rot)
                m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            self.pubs[robot].publish(m)
            corners = cv2.boxPoints(((cx, cy), (w, h), math.degrees(rot)))
            poly = PolygonStamped()
            poly.header = m.header
            poly.polygon.points = [Point32(x=wx, y=wy, z=0.0)
                                   for wx, wy in (self.floor_xy(robot, float(u), float(v)) for u, v in corners)]
            self.bbox_pubs[robot].publish(poly)
        self.n += 1
        if self.n % 60 == 0:
            self.get_logger().debug(f"frames={self.n}, detected={list(best)}")

    def _publish_all(self, stamp, r):
        """/perception/<class>/bev_all: every detection of this frame."""
        dets = {name: [] for name in self.class_map.values()}
        if r.obb is not None and len(r.obb) > 0:
            cls = r.obb.cls.cpu().numpy().astype(int)
            conf = r.obb.conf.cpu().numpy()
            xywhr = r.obb.xywhr.cpu().numpy()
            for i in range(len(cls)):
                robot = self.class_map.get(int(cls[i]))
                if robot is not None:
                    dets[robot].append((float(conf[i]), i))
        for robot, found in dets.items():
            arr = PoseArray()
            arr.header.stamp = stamp
            arr.header.frame_id = "uwb_global"
            for c, i in sorted(found, reverse=True):
                cx, cy, w, h, rot = (float(v) for v in xywhr[i])
                pose = Pose()
                pose.position.x, pose.position.y = self.floor_xy(robot, cx, cy)
                pose.position.z = c                     # confidence
                pose.orientation.w = 1.0                # identity = no box angle
                if self.publish_axis:
                    yaw = box_axis_yaw(w, h, rot)
                    pose.orientation.z, pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
                arr.poses.append(pose)
            self.all_pubs[robot].publish(arr)


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
