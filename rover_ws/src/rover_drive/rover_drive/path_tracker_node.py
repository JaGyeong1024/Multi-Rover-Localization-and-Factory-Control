"""path_tracker: follow ``/{robot}/path_cmd`` and report ``/{robot}/nav_feedback``."""
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from host_mission_interfaces.msg import NavFeedback, PathCommand, SafetyState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32, String

from .core import state as st
from .core.config import DriveConfig
from .core.controller import DriveController, DriveOutput
from .core.depth_front import DepthScan
from .ros_util import (command_qos, declare_config, latched_qos, now_s, reliable_qos, run_node,
                       to_path_cmd, yaw_from_quaternion)

_NO_NODE = -1  # NavFeedback.current_node: the rover has no graph any more


class PathTrackerNode(Node):

    def __init__(self):
        super().__init__('path_tracker')
        self._cfg = declare_config(self, DriveConfig)
        self._robot = self._cfg.robot
        self._ctrl = DriveController(self._cfg)
        self._lead_subs = {}
        self._last_reported = None
        self._last_feedback_time = 0.0
        self._create_io()
        self.create_timer(1.0 / self._cfg.control_rate_hz, self._on_timer)
        self.get_logger().info(
            f'path_tracker[{self._robot}] v_max={self._cfg.max_linear_speed} '
            f'pivot_omega={self._cfg.pivot_omega} '
            f'xtrack_integral_gain={self._cfg.xtrack_integral_gain} '
            f'runin_heading_gain={self._cfg.runin_heading_gain} '
            f'runin_lookaheads={self._cfg.runin_lookaheads} '
            f'gap_bridge={self._cfg.gap_bridge} '
            f'face_align={self._cfg.face_align} '
            f'(tol {self._cfg.face_align_tol_rad} rad) '
            f'-> {self._cmd_pub.topic_name}')

    def _create_io(self) -> None:
        ns = f'/{self._robot}'
        cfg = self._cfg
        self._cmd_pub = self.create_publisher(Twist, cfg.cmd_vel_topic or f'{ns}/cmd_vel', 10)
        self._fb_pub = self.create_publisher(NavFeedback, f'{ns}/nav_feedback', reliable_qos())
        self._approach_pub = self.create_publisher(
            Bool, cfg.approach_topic or f'{ns}/approach_active', reliable_qos())
        self.create_subscription(PathCommand, f'{ns}/path_cmd', self._on_cmd, command_qos())
        self.create_subscription(PoseStamped, f'{ns}/global/position', self._on_pose,
                                 reliable_qos())
        self.create_subscription(Bool, f'{ns}/heading_aligned', self._on_aligned, latched_qos())
        self.create_subscription(String, f'{ns}/localization_quality', self._on_quality,
                                 latched_qos())
        self.create_subscription(SafetyState, cfg.safety_state_topic or f'{ns}/safety_state',
                                 self._on_gate, reliable_qos())
        self.create_subscription(Float32, cfg.lead_gap_topic or f'{ns}/lead_gap',
                                 self._on_lead_gap, qos_profile_sensor_data)
        if cfg.face_align:
            self.create_subscription(Imu, cfg.imu_topic, self._on_imu, qos_profile_sensor_data)
            self.create_subscription(LaserScan, cfg.depth_scan_topic or f'{ns}/depth_scan',
                                     self._on_depth_scan, qos_profile_sensor_data)

    # ---------------------------------------------------------- callbacks
    def _on_cmd(self, msg: PathCommand) -> None:
        if msg.robot and msg.robot != self._robot:
            return
        cmd = to_path_cmd(msg)
        kind = self._ctrl.on_command(cmd, now_s())
        if kind != st.NEW:
            return
        self._ensure_lead_subscription(cmd.lead_robot)
        self.get_logger().info(
            f'path_cmd accepted: epoch={cmd.epoch} cmd_id={cmd.cmd_id} mode={cmd.mode} '
            f'poses={len(cmd.poses)} pick={cmd.pick_at_goal} lead="{cmd.lead_robot}"')

    def _ensure_lead_subscription(self, lead: str) -> None:
        if not lead or lead in self._lead_subs:
            return
        self._lead_subs[lead] = self.create_subscription(
            PoseStamped, f'/{lead}/global/position', self._on_lead_pose, reliable_qos())

    def _on_pose(self, msg: PoseStamped) -> None:
        p = msg.pose
        self._ctrl.on_pose(p.position.x, p.position.y, yaw_from_quaternion(p.orientation), now_s())

    def _on_lead_pose(self, msg: PoseStamped) -> None:
        self._ctrl.on_lead_pose(msg.pose.position.x, msg.pose.position.y, now_s())

    def _on_lead_gap(self, msg: Float32) -> None:
        self._ctrl.on_lead_gap(float(msg.data), now_s())

    def _on_imu(self, msg: Imu) -> None:
        now = now_s()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._ctrl.on_gyro(float(msg.angular_velocity.z), stamp or now, now)

    def _on_depth_scan(self, msg: LaserScan) -> None:
        scan = DepthScan(ranges=np.asarray(msg.ranges, dtype=float),
                         angle_min=msg.angle_min, angle_increment=msg.angle_increment,
                         range_min=msg.range_min, range_max=msg.range_max)
        self._ctrl.on_depth_scan(scan, now_s())

    def _on_aligned(self, msg: Bool) -> None:
        self._ctrl.set_heading_aligned(bool(msg.data))

    def _on_quality(self, msg: String) -> None:
        self._ctrl.set_quality(str(msg.data).strip().upper())

    def _on_gate(self, msg: SafetyState) -> None:
        self._ctrl.set_gate(msg.state, list(msg.reasons), float(msg.speed_scale), now_s())

    # ------------------------------------------------------------- output
    def _on_timer(self) -> None:
        now = now_s()
        out = self._ctrl.step(now)
        self._publish_twist(out.v, out.w)
        for note in self._ctrl.take_notes():
            self.get_logger().info(note)
        if self._feedback_due(out, now):
            self._publish_feedback(out, now)

    def _feedback_due(self, out: DriveOutput, now: float) -> bool:
        if self._report_key(out) != self._last_reported:
            return True
        return now - self._last_feedback_time >= 1.0 / self._cfg.feedback_rate_hz

    @staticmethod
    def _report_key(out: DriveOutput) -> tuple:
        detail = '' if out.state == st.TRACKING else out.detail
        return out.state, out.epoch, out.cmd_id, detail, out.approach_active

    def _publish_feedback(self, out: DriveOutput, now: float) -> None:
        key = self._report_key(out)
        if key[:4] != (self._last_reported or ())[:4]:
            self.get_logger().info(f'state {out.state} {out.detail} (cmd_id={out.cmd_id})')
        self._last_reported, self._last_feedback_time = key, now
        self._approach_pub.publish(Bool(data=bool(out.approach_active)))
        self._fb_pub.publish(self._feedback_msg(out))

    def _feedback_msg(self, out: DriveOutput) -> NavFeedback:
        msg = NavFeedback()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'uwb_global'
        msg.robot = self._robot
        msg.epoch = out.epoch
        msg.cmd_id = out.cmd_id
        msg.state = out.state
        msg.current_node = _NO_NODE
        msg.progress = float(out.progress)
        msg.detail = out.detail
        return msg

    def _publish_twist(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self._cmd_pub.publish(msg)

    def publish_stop(self) -> None:
        self._publish_twist(0.0, 0.0)


def main() -> None:
    run_node(PathTrackerNode)


if __name__ == '__main__':
    main()
