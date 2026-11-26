"""safety_gate: the last check between any velocity source and the base."""
from geometry_msgs.msg import PoseStamped, Twist
from host_mission_interfaces.msg import NavFeedback, PathCommand, SafetyState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Empty, Float32

from .core.config import GateConfig
from .core.safety import GateOutput, SafetyGate
from .ros_util import (command_qos, declare_config, latched_qos, now_s, reliable_qos, run_node,
                       to_path_cmd, yaw_from_quaternion)


class SafetyGateNode(Node):

    def __init__(self):
        super().__init__('safety_gate')
        self._cfg = declare_config(self, GateConfig)
        self._robot = self._cfg.robot
        self._gate = SafetyGate(self._cfg)
        self._last_reported = None
        self._last_state_time = 0.0
        self._last_alive_time = 0.0
        self._lead_subs = {}
        self._create_io()
        self._create_front_io()
        self.create_timer(1.0 / self._cfg.gate_rate_hz, self._on_timer)
        self.get_logger().info(
            f'safety_gate[{self._robot}] {self._input_topic} -> {self._cfg.output_topic} '
            f'angular_sign={self._cfg.angular_sign} lidar={self._cfg.lidar_enabled} '
            f'v_max={self._cfg.max_linear_speed} front_memory={self._cfg.front_memory} '
            f'front_path={self._cfg.front_path}')

    def _create_io(self) -> None:
        ns = f'/{self._robot}'
        cfg = self._cfg
        self._input_topic = cfg.input_topic or f'{ns}/cmd_vel'
        self._out_pub = self.create_publisher(Twist, cfg.output_topic, 10)
        self._state_pub = self.create_publisher(
            SafetyState, cfg.safety_state_topic or f'{ns}/safety_state', reliable_qos())
        self.create_subscription(Twist, self._input_topic, self._on_input, 10)
        self.create_subscription(Bool, cfg.estop_topic, self._on_estop, latched_qos())
        self.create_subscription(PoseStamped, f'{ns}/global/position', self._on_pose,
                                 reliable_qos())
        self.create_subscription(Bool, f'{ns}/heading_aligned', self._on_aligned, latched_qos())
        self.create_subscription(PathCommand, f'{ns}/path_cmd', self._on_cmd, command_qos())
        self.create_subscription(NavFeedback, f'{ns}/nav_feedback', self._on_tracker,
                                 reliable_qos())
        self.create_subscription(Bool, cfg.approach_topic or f'{ns}/approach_active',
                                 self._on_approach, reliable_qos())
        self._alive_pub = self.create_publisher(Empty, cfg.alive_topic or f'{ns}/gate_alive', 10)

    def _create_front_io(self) -> None:
        cfg = self._cfg
        if cfg.front_distance_topic:
            self.create_subscription(Float32, cfg.front_distance_topic, self._on_front, 10)
            if cfg.front_memory:
                self.create_subscription(
                    LaserScan, cfg.depth_scan_topic or f'/{self._robot}/depth_scan',
                    self._on_depth_scan, qos_profile_sensor_data)
        elif cfg.lidar_enabled:
            self.create_subscription(LaserScan, cfg.scan_topic, self._on_scan,
                                     qos_profile_sensor_data)
            self._gap_pub = self.create_publisher(
                Float32, cfg.lead_gap_topic or f'/{self._robot}/lead_gap',
                qos_profile_sensor_data)

    # ---------------------------------------------------------- callbacks
    def _on_input(self, msg: Twist) -> None:
        self._gate.on_input(msg.linear.x, msg.angular.z, now_s())

    def _on_estop(self, msg: Bool) -> None:
        self._gate.set_estop(bool(msg.data))

    def _on_pose(self, msg: PoseStamped) -> None:
        p = msg.pose
        self._gate.on_pose(p.position.x, p.position.y, yaw_from_quaternion(p.orientation), now_s())

    def _on_aligned(self, msg: Bool) -> None:
        self._gate.set_aligned(bool(msg.data))

    def _on_cmd(self, msg: PathCommand) -> None:
        if msg.robot and msg.robot != self._robot:
            return
        lead = str(msg.lead_robot).strip()
        self._gate.on_cmd(now_s(), lead, to_path_cmd(msg) if self._cfg.front_path else None)
        if lead and lead not in self._lead_subs:
            self._lead_subs[lead] = self.create_subscription(
                PoseStamped, f'/{lead}/global/position',
                lambda m, name=lead: self._on_lead_pose(m, name), reliable_qos())

    def _on_lead_pose(self, msg: PoseStamped, name: str) -> None:
        if name == self._gate.lead_name:
            self._gate.on_lead_pose(msg.pose.position.x, msg.pose.position.y, now_s())

    def _on_approach(self, msg: Bool) -> None:
        self._gate.on_approach(bool(msg.data), now_s())

    def _on_front(self, msg: Float32) -> None:
        self._gate.on_front_distance(float(msg.data), now_s())

    def _on_depth_scan(self, msg: LaserScan) -> None:
        self._gate.on_depth_scan(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min,
                                 now_s())

    def _on_tracker(self, msg: NavFeedback) -> None:
        self._gate.on_tracker(msg.state, msg.detail, now_s())

    def _on_scan(self, msg: LaserScan) -> None:
        self._gate.on_scan(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min,
                           now_s())
        if self._gate.lead_name:
            self._gap_pub.publish(Float32(data=float(self._gate.lead_gap)))

    # ------------------------------------------------------------- output
    def _on_timer(self) -> None:
        now = now_s()
        out = self._gate.step(now)
        self._publish_twist(out.v, out.w)
        self._publish_alive(now)
        key = (out.state, out.reasons)
        changed = key != self._last_reported
        if changed or now - self._last_state_time >= 1.0 / self._cfg.state_rate_hz:
            self._publish_state(out, now, changed)

    def _publish_alive(self, now: float) -> None:
        slack = 0.5 / self._cfg.gate_rate_hz  # timer ticks are not exact multiples
        if now - self._last_alive_time >= 1.0 / self._cfg.alive_rate_hz - slack:
            self._last_alive_time = now
            self._alive_pub.publish(Empty())

    def _publish_state(self, out: GateOutput, now: float, changed: bool) -> None:
        if changed:
            self.get_logger().info(f'gate {out.state} {list(out.reasons)}')
        self._last_reported, self._last_state_time = (out.state, out.reasons), now
        msg = SafetyState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'uwb_global'
        msg.robot = self._robot
        msg.state = out.state
        msg.reasons = list(out.reasons)
        msg.speed_scale = float(out.speed_scale)
        msg.blocked_x = float(out.blocked_x)
        msg.blocked_y = float(out.blocked_y)
        self._state_pub.publish(msg)

    def _publish_twist(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self._out_pub.publish(msg)

    def publish_stop(self) -> None:
        self._publish_twist(0.0, 0.0)


def main() -> None:
    run_node(SafetyGateNode)


if __name__ == '__main__':
    main()
