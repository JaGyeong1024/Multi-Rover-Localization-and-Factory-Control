"""fusion_node: per-robot sensor fusion (UWB + IMU + ceiling camera) -> /{robot}/pose.

One EKFSE2 per robot (se2nhc.py, tracker.py). Inputs: /uwb/{robot}_pose_corrected (+ _qf),
/perception/{robot}/bev_pose, /{robot}/imu, optionally /{robot}/joint_states.
Outputs, published from the measurement callbacks: /{robot}/pose (stops after pose_timeout
without UWB/BEV), /{robot}/velocity (noisy, not for control), latched /{robot}/heading_aligned
and /{robot}/localization_quality (rule in quality.py).
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from geometry_msgs.msg import PoseArray, PoseStamped, TwistStamped
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32, String

from . import quality
from .frame_assoc import Box, process_frame
from .frontend import Gating, RobotFusion
from .tracker import KW, OBB_HEADING, WHEEL_RADIUS, WHEEL_STILL, ZERO_MOTION

# Latched QoS for discrete state: late joiners get the current value.
ALIGNED_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
QUALITY_CHECK_PERIOD = 0.1     # [s] how often the quality rule is evaluated


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def is_identity(q):
    """Exact identity means perception has no box angle."""
    return q.w == 1.0 and q.z == 0.0


class _Robot:
    """ROS-side state of one robot."""

    def __init__(self, name, kw, use_obb, gating, wheel_still=False, zero_motion=False, wheel_radius=None):
        self.name = name
        self.fusion = RobotFusion(kw, use_obb=use_obb, gating=gating, wheel_still=wheel_still,
                                  zero_motion=zero_motion, wheel_radius=wheel_radius)
        self.wheel_joints = ('wheel_left_joint', 'wheel_right_joint')   # names in /joint_states
        self.pub = None
        self.vel_pub = None
        self.aligned_pub = None
        self.quality_pub = None
        self.aligned_last = None         # last published heading_aligned (publish on change)
        self.quality_last = None
        self.quality_pub_t = 0.0


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        # q_vel overrides the per-robot KW value for all robots.
        q_vel = self.declare_parameter('q_vel', 0.02).value
        self.frame = self.declare_parameter('frame_id', 'uwb_global').value
        robots = self.declare_parameter('robots', ['leader', 'follower']).value
        self.exit_on_dup = bool(self.declare_parameter('exit_on_duplicate', False).value)  # exit if another publisher exists
        # COG is suppressed this long [s] after alignment (0 = off): the rover then reverses.
        cog_suppress_s = float(self.declare_parameter('cog_suppress_after_align', 5.0).value)
        bev_timeout = float(self.declare_parameter('quality.bev_timeout', quality.BEV_TIMEOUT).value)
        uwb_timeout = float(self.declare_parameter('quality.uwb_timeout', quality.UWB_TIMEOUT).value)
        # quality while only UWB is fresh: LOST or DEGRADED
        uwb_only_quality = str(self.declare_parameter('quality.uwb_only', quality.LOST).value).upper()
        self.quality_period = float(self.declare_parameter('quality.republish_period', 1.0).value)
        # No UWB and no BEV for this long -> stop /{robot}/pose (IMU-only is not a live position).
        pose_timeout = float(self.declare_parameter('pose_timeout', quality.POSE_TIMEOUT).value)
        self.gating = Gating(pose_timeout=pose_timeout, uwb_timeout=uwb_timeout, bev_timeout=bev_timeout,
                             uwb_only_quality=uwb_only_quality, cog_suppress_s=cog_suppress_s,
                             quality_mode=self._quality_mode(), limits=self._estimator_limits())

        # Per-frame association by position (frame_assoc.py), off by default; needs perception
        # publish_all_boxes:=true and replaces bev_pose for that robot.
        self.frame_robots = [n for n in robots
                             if bool(self.declare_parameter(f'{n}.bev_frames', False).value)]
        # A box of another class is used only this close [m] to the estimate.
        self.cross_max_dist = float(self.declare_parameter('cross_class_max_dist', 0.15).value)
        self._frames = {}                   # image stamp -> {class: PoseArray} being assembled
        self.robots = {}
        for name in robots:
            self.robots[name] = self._make_robot(name, float(q_vel))
        # While the leader's arm works, rovers with standing wheels hold position (tracker._frozen).
        arm_t = self.declare_parameter('arm_work_state', '/leader/work_state').value
        self.arm_busy_states = set(self.declare_parameter('arm_busy_states', ['PICKING']).value)
        self.create_subscription(String, arm_t, self._on_arm_state,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        if self.frame_robots:
            self._frame_classes = list(robots)
            for cls in self._frame_classes:
                topic = self.declare_parameter(f'{cls}.bev_all', f'/perception/{cls}/bev_all').value
                self.create_subscription(PoseArray, topic, lambda m, C=cls: self._on_bev_all(m, C),
                                         qos_profile_sensor_data)
            self.get_logger().info(f'per-frame camera association for {self.frame_robots} '
                                   f'(cross-class max {self.cross_max_dist:.2f} m)')

        # Single-instance guard (a second publisher mixes poses), checked once after discovery.
        self._guard_timer = self.create_timer(2.0, self._instance_guard)
        self.create_timer(QUALITY_CHECK_PERIOD, self._quality_tick)

    def _quality_mode(self):
        """quality.mode; camera_age_hold needs {robot}.zero_motion."""
        mode = str(self.declare_parameter('quality.mode', quality.CAMERA_AGE).value).lower()
        if mode not in quality.MODES:
            raise ValueError(f'quality.mode={mode!r}: one of {quality.MODES}')
        return mode

    def _estimator_limits(self):
        d = quality.EstimatorLimits()
        return quality.EstimatorLimits(**{
            field: type(getattr(d, field))(self.declare_parameter(f'quality.{field}', getattr(d, field)).value)
            for field in d._fields})

    def _make_robot(self, name, q_vel):
        kw = dict(KW.get(name, KW['follower'])); kw['q_vel'] = q_vel
        use_obb = bool(self.declare_parameter(f'{name}.obb_heading', OBB_HEADING.get(name, False)).value)
        # Still wheels freeze the heading offset; no joint_states = never still.
        wheel_still = bool(self.declare_parameter(f'{name}.wheel_still', WHEEL_STILL.get(name, False)).value)
        # Still wheels hold the position (ZUPT); camera or far UWB overrules them.
        zero_motion = bool(self.declare_parameter(f'{name}.zero_motion', ZERO_MOTION.get(name, False)).value)
        radius = float(self.declare_parameter(f'{name}.wheel_radius', WHEEL_RADIUS.get(name, 0.0)).value)
        r = _Robot(name, kw, use_obb, self.gating, wheel_still, zero_motion, radius if radius > 0.0 else None)
        r.wheel_joints = tuple(str(self.declare_parameter(f'{name}.wheel_joint_{side}',
                                                          f'wheel_{side}_joint').value)
                               for side in ('left', 'right'))
        # High-rate topics are best-effort.
        r.pub = self.create_publisher(PoseStamped, f'/{name}/pose', qos_profile_sensor_data)
        r.vel_pub = self.create_publisher(TwistStamped, f'/{name}/velocity', qos_profile_sensor_data)
        r.aligned_pub = self.create_publisher(Bool, f'/{name}/heading_aligned', ALIGNED_QOS)
        r.aligned_pub.publish(Bool(data=False))   # starts unaligned (latched)
        r.aligned_last = False
        r.quality_pub = self.create_publisher(String, f'/{name}/localization_quality', ALIGNED_QOS)
        self._subscribe(name, r)
        self.get_logger().info(f'{name}: EKFSE2 q_vel={kw["q_vel"]} obb_heading={use_obb} '
                               f'wheel_still={wheel_still} zero_motion={zero_motion} '
                               f'quality={self.gating.quality_mode} -> /{name}/pose')
        return r

    def _subscribe(self, name, r):
        uwb_t = self.declare_parameter(f'{name}.uwb', f'/uwb/{name}_pose_corrected').value
        qf_t = self.declare_parameter(f'{name}.qf', f'/uwb/{name}_qf').value
        bev_t = self.declare_parameter(f'{name}.bev', f'/perception/{name}/bev_pose').value
        imu_t = self.declare_parameter(f'{name}.imu', f'/{name}/imu').value
        q = qos_profile_sensor_data
        self.create_subscription(PoseStamped, uwb_t, lambda m, R=r: self._on_uwb(m, R), q)
        self.create_subscription(Float32, qf_t, lambda m, R=r: self._on_qf(m, R), q)
        if name not in self.frame_robots:          # else: boxes come per frame (_on_bev_all)
            self.create_subscription(PoseStamped, bev_t, lambda m, R=r: self._on_bev(m, R), q)
        self.create_subscription(Imu, imu_t, lambda m, R=r: self._on_imu(m, R), q)
        trk = r.fusion.trk
        if trk.wheel_still or trk.zero_motion or self.gating.quality_mode == quality.ESTIMATOR:
            js_t = self.declare_parameter(f'{name}.joint_states', f'/{name}/joint_states').value
            self.create_subscription(JointState, js_t, lambda m, R=r: self._on_joint_states(m, R), q)

    def _instance_guard(self):
        self._guard_timer.cancel()    # one-shot
        dups = [f'/{n}/pose×{c}' for n in self.robots
                if (c := self.count_publishers(f'/{n}/pose')) > 1]
        if not dups:
            return
        self.get_logger().error(
            f'⚠️ 중복 publisher 감지 {dups} — 다른 fusion_node/추정기가 같은 토픽 발행 중. '
            f'헤딩/위치가 섞여 튄다. 이전 인스턴스 정리 후 단독 기동할 것.')
        if self.exit_on_dup:
            self.get_logger().fatal('exit_on_duplicate=true → 종료.')
            raise SystemExit(1)

    def _meas_t(self, msg):
        """Header stamp, capped at host time (a future stamp would freeze the filter clock)."""
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return min(stamp, self.get_clock().now().nanoseconds * 1e-9)

    def _on_arm_state(self, msg):
        busy = msg.data in self.arm_busy_states
        for r in self.robots.values():
            if r.fusion.trk.arm_busy != busy:
                self.get_logger().info(f'{r.name}: arm {msg.data} -> position '
                                       f'{"held while the wheels stand" if busy else "free"}')
            r.fusion.trk.arm_busy = busy

    def _on_qf(self, msg, R):
        R.fusion.on_qf(msg.data)

    def _on_uwb(self, msg, R):
        R.fusion.on_uwb(self._meas_t(msg), time.monotonic(), msg.pose.position.x, msg.pose.position.y)
        self._emit(R)

    def _on_bev(self, msg, R):
        q = msg.pose.orientation
        obb_yaw = None if is_identity(q) else quat_to_yaw(q)      # box long axis (mod pi), optional
        R.fusion.on_bev(self._meas_t(msg), time.monotonic(), msg.pose.position.x, msg.pose.position.y, obb_yaw)
        self._emit(R)

    def _on_bev_all(self, msg, cls):
        """Collect one frame (a PoseArray per class, same stamp); process it when complete or a newer one starts."""
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._frames.setdefault(key, {})[cls] = msg
        for old in sorted(k for k in self._frames if k < key):
            self._process_frame(self._frames.pop(old))
        if len(self._frames.get(key, ())) == len(self._frame_classes):
            self._process_frame(self._frames.pop(key))

    def _process_frame(self, by_class):
        boxes = []
        for cls, msg in by_class.items():
            for p in msg.poses:                  # position.z = detection confidence
                q = p.orientation
                boxes.append(Box(cls, p.position.z, p.position.x, p.position.y,
                                 None if is_identity(q) else quat_to_yaw(q)))
        msg = next(iter(by_class.values()))
        fusions = {n: R.fusion for n, R in self.robots.items()}
        fed = process_frame(fusions, self.frame_robots, boxes, self._meas_t(msg),
                            time.monotonic(), self.cross_max_dist)
        for name in fed:
            self._emit(self.robots[name])

    def _on_imu(self, msg, R):
        a, w = msg.linear_acceleration, msg.angular_velocity
        R.fusion.on_imu(self._meas_t(msg), quat_to_yaw(msg.orientation), w.z,
                        (a.x, a.y, a.z), math.hypot(w.x, w.y))
        self._emit(R)                         # heading follows the IMU, so publish on IMU too

    def _on_joint_states(self, msg, R):
        """Wheels only tell "standing still"; no pose output."""
        try:
            left = msg.position[msg.name.index(R.wheel_joints[0])]
            right = msg.position[msg.name.index(R.wheel_joints[1])]
        except (ValueError, IndexError):
            return
        R.fusion.on_wheels(self._meas_t(msg), left, right)

    def _emit(self, R):
        """Publish pose and velocity after a measurement."""
        now = self.get_clock().now()
        sample = R.fusion.output(time.monotonic(), now.nanoseconds * 1e-9)
        if sample is None:
            return
        R.pub.publish(self._pose_msg(sample, now))
        self._publish_aligned(R)
        R.vel_pub.publish(self._velocity_msg(sample, now))

    def _pose_msg(self, sample, now):
        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame
        msg.pose.position.x, msg.pose.position.y = sample.x, sample.y
        msg.pose.orientation.w = 1.0          # heading not anchored yet: identity
        if sample.heading is not None:        # anchored: publish EKF heading
            _, _, msg.pose.orientation.z, msg.pose.orientation.w = yaw_to_quat(sample.heading)
        return msg

    def _publish_aligned(self, R):
        """Publish heading_aligned (COG anchored) on change."""
        aligned = R.fusion.aligned
        if aligned == R.aligned_last:
            return
        R.aligned_pub.publish(Bool(data=bool(aligned)))
        R.aligned_last = aligned
        suppressed = aligned and self.gating.cog_suppress_s > 0.0
        self.get_logger().info(f'{R.name}: heading_aligned={aligned}'
                               + (f' (COG {self.gating.cog_suppress_s:.0f}s 억제)' if suppressed else ''))

    def _velocity_msg(self, sample, now):
        """World velocity vx, vy and gyro yaw rate (auxiliary, noisy)."""
        tw = TwistStamped()
        tw.header.stamp = now.to_msg(); tw.header.frame_id = self.frame
        tw.twist.linear.x, tw.twist.linear.y = sample.vx, sample.vy
        tw.twist.angular.z = float(sample.wz)
        return tw

    def _quality_tick(self):
        """Publish /{robot}/localization_quality on change and every republish_period."""
        now = time.monotonic()
        for R in self.robots.values():
            q = R.fusion.quality(now)
            changed = q != R.quality_last
            if not changed and now - R.quality_pub_t < self.quality_period:
                continue
            R.quality_pub.publish(String(data=q))
            R.quality_pub_t = now
            if changed:
                self.get_logger().info(f'{R.name}: localization_quality {R.quality_last} -> {q}')
                R.quality_last = q


def main():
    rclpy.init()
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
