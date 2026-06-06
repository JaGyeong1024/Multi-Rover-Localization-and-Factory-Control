"""fusion_node — 호스트 런타임 위치 융합 (역할: 센서 → /{robot}/pose).

**내부 = EKFSE2** (SE(2)+NHC, UWB+IMU+BEV 융합). 벤치마크(docs/fusion_benchmark_*)에서
**EKF 채택**(2026-06-04): BEV 지배 레짐 + worst-case·단순·실시간 우위.
위치·**헤딩 모두 EKFSE2 단일 필터**(2026-06-06 구조 통합). 헤딩 = COG(진행방향)를 h_off 측정으로
EKF에 통합(`update_cog`, INS course-aiding) → NHC 단독의 180° 모호 해소, **별도 HeadingFilter 제거**.
손 게이트(턴·부트스트랩·슬루)는 측정 R=f(속도)+χ²에서 emergent. (이전 PositionKF는 position_kf.py 보존.)
docs/host_fusion_design.md.

front-end 상수는 벤치마크 결정값(`analysis/bench_config.py` SSOT). full-fusion 레짐 = UWB 절대-R(BEV 지배).
입력(로버별): `/uwb/{robot}_pose`(+`_qf`), `/perception/{robot}/bev_pose`(YOLO), `/{robot}/imu`(자이로 yaw).
출력(이벤트구동·저지연 — 측정 콜백서 즉시 발행): `/{robot}/pose`(PoseStamped) + `/{robot}/velocity`
  (TwistStamped, 월드속도 vx·vy + gyro ω). ⚠ 속도는 noisy(위치미분)·제어백본 아님(갭=위치 + leader
  명령속도 rover 방송 권장). **출력 *주기*는 global_position_node가 통제**(이 노드는 신선값만 흘림). CLAUDE.md §2.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from .se2nhc import EKFSE2, VX, VY

COG_MIN_MOVE = 0.15      # COG 측정용 최소 위치변위[m] — 클수록 진행방향 노이즈↓(skid). 이동마다 EKF update_cog

# 벤치마크 결정 front-end 상수(per-robot, §1.3 분기). full-fusion 레짐: uwb_sigma0=절대-R(BEV가 정밀원 지배),
# r_bev=(3cm)² 실측 YOLO잡음, q_vel=0.02(엔코더+NIS 튜닝), q_hoff(leader IMU 드리프트→큼). bench_config SSOT.
KW = {
    'leader':   dict(q_vel=0.02, q_hoff=5e-3, r_nhc=0.10 ** 2, uwb_sigma0=0.12, r_bev=0.03 ** 2, use_wheel=False),
    'follower': dict(q_vel=0.02, q_hoff=2e-4, r_nhc=0.045 ** 2, uwb_sigma0=0.20, r_bev=0.03 ** 2, use_wheel=False),
}


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class _Robot:
    def __init__(self, kw):
        self.est = EKFSE2(**kw)     # SE(2)+NHC EKF — UWB+IMU+BEV 융합 + COG헤딩(update_cog)
        self.qf = 70.0
        self.last_xy = None         # COG 기준 위치(진행방향 산출용)
        self.last_wz = 0.0          # 최근 gyro_z[rad/s] — COG 턴 게이트 입력
        self.last_t = None
        self.pub = None
        self.vel_pub = None

    def advance(self, t):
        """측정 시각까지 EKF predict(측정구동) — IMU가 고레이트라 위치측정 공백에도 coast."""
        if self.last_t is not None and t > self.last_t:
            self.est.predict(t - self.last_t)
        self.last_t = t


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        # 튜닝 노브(제어테스트): q_vel 전역 override, 나머진 per-robot KW. 기본=벤치 결정값.
        q_vel = self.declare_parameter('q_vel', 0.02).value
        self.frame = self.declare_parameter('frame_id', 'uwb_global').value
        robots = self.declare_parameter('robots', ['leader', 'follower']).value
        self.exit_on_dup = bool(self.declare_parameter('exit_on_duplicate', False).value)  # 중복 인스턴스 시 종료

        self.robots = {}
        for name in robots:
            kw = dict(KW.get(name, KW['follower'])); kw['q_vel'] = float(q_vel)
            r = _Robot(kw)
            # 내부 고빈도 = BEST_EFFORT(sensor_data): 최신값 우선·재전송 블로킹 없음. 외부(/global/*)는 그대로.
            r.pub = self.create_publisher(PoseStamped, f'/{name}/pose', qos_profile_sensor_data)
            r.vel_pub = self.create_publisher(TwistStamped, f'/{name}/velocity', qos_profile_sensor_data)  # → global_position 릴레이
            uwb_default = f'/uwb/{name}_pose_corrected'   # 둘 다 보정 입력(follower도 corrected — uwb_correction_node가 발행)
            uwb_t = self.declare_parameter(f'{name}.uwb', uwb_default).value
            qf_t = self.declare_parameter(f'{name}.qf', f'/uwb/{name}_qf').value
            bev_t = self.declare_parameter(f'{name}.bev', f'/perception/{name}/bev_pose').value
            imu_t = self.declare_parameter(f'{name}.imu', f'/{name}/imu').value
            # 구독도 BEST_EFFORT(sensor_data): 고빈도 최신값 우선. (best_effort 구독자는 reliable 발행자와도 호환
            #  → uwb_reader 등 외부 발행자 미변경 OK.)
            self.create_subscription(PoseStamped, uwb_t, lambda m, R=r: self._on_uwb(m, R), qos_profile_sensor_data)
            self.create_subscription(Float32, qf_t, lambda m, R=r: self._on_qf(m, R), qos_profile_sensor_data)
            self.create_subscription(PoseStamped, bev_t, lambda m, R=r: self._on_bev(m, R), qos_profile_sensor_data)
            self.create_subscription(Imu, imu_t, lambda m, R=r: self._on_imu(m, R), qos_profile_sensor_data)
            self.robots[name] = r
            self.get_logger().info(f'{name}: EKFSE2 q_vel={kw["q_vel"]} uwb={uwb_t} bev={bev_t} imu={imu_t} -> /{name}/pose')

        # 발행=이벤트구동(_emit, 측정 콜백서 즉시). 출력 *주기*는 global_position_node가 통제(저지연·이중타이머 회피).
        # 단일-인스턴스 가드: 같은 토픽을 다른 fusion_node/추정기가 발행 중이면 헤딩·위치가 섞여 튄다
        # ("두 헤딩 번갈아"의 구조적 원인). 디스커버리 안정 후 1회 점검.
        self._guard_timer = self.create_timer(2.0, self._instance_guard)

    def _instance_guard(self):
        self._guard_timer.cancel()                       # one-shot
        dups = [f'/{n}/pose×{c}' for n in self.robots
                if (c := self.count_publishers(f'/{n}/pose')) > 1]
        if not dups:
            return
        self.get_logger().error(
            f'⚠️ 중복 publisher 감지 {dups} — 다른 fusion_node/추정기가 같은 토픽 발행 중. '
            f'헤딩/위치가 섞여 튄다. 이전 인스턴스 정리(pkill -f fusion_node) 후 단독 기동할 것.')
        if self.exit_on_dup:
            self.get_logger().fatal('exit_on_duplicate=true → 종료.')
            raise SystemExit(1)

    @staticmethod
    def _stamp_t(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_qf(self, msg, R):
        R.qf = msg.data

    def _on_uwb(self, msg, R):
        R.advance(self._stamp_t(msg))
        R.est.update_uwb(msg.pose.position.x, msg.pose.position.y, R.qf)   # qf 적응 R은 EKF 내부
        self._emit(R)

    def _on_bev(self, msg, R):
        R.advance(self._stamp_t(msg))
        R.est.update_bev(msg.pose.position.x, msg.pose.position.y)         # YOLO centroid (정밀원, r_bev 내부)
        self._emit(R)

    def _on_imu(self, msg, R):
        y = quat_to_yaw(msg.orientation)
        R.advance(self._stamp_t(msg))
        R.est.set_imu_yaw(y)                  # NHC·헤딩용 IMU yaw + predict 구동
        R.last_wz = msg.angular_velocity.z    # COG 턴 게이트 입력(회전 중 COG skip)
        self._emit(R)                         # 헤딩이 IMU 추종하도록 imu에도 발행(회전 중 신선)

    def _emit(self, R):
        """측정 갱신 직후 *즉시* /{r}/pose·/{r}/velocity 발행(이벤트구동·저지연).
        출력 *주기*는 global_position_node가 통제(이 노드는 신선한 값만 흘림)."""
        if not R.est.initialized:
            return
        now = self.get_clock().now()
        x, y, _ = R.est.pose
        # COG(진행방향) → EKF h_off 측정: 위치가 충분히 이동했을 때 진행방향을 헤딩에 통합.
        # 정지·회전 게이트와 슬루는 update_cog 내부(R=f(속도)+χ²)에서 처리.
        if R.last_xy is None:
            R.last_xy = (x, y)
        elif math.hypot(x - R.last_xy[0], y - R.last_xy[1]) >= COG_MIN_MOVE:
            cog = math.atan2(y - R.last_xy[1], x - R.last_xy[0])
            R.est.update_cog(cog, R.est.speed, R.last_wz)
            R.last_xy = (x, y)
        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame
        msg.pose.position.x, msg.pose.position.y = x, y
        if R.est.heading_valid:           # COG로 h_off 앵커됨 → EKF 헤딩 출력
            qx, qy, qz, qw = yaw_to_quat(R.est.heading())
            msg.pose.orientation.x = qx; msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz; msg.pose.orientation.w = qw
        else:                             # 미수렴(첫 전진 전): identity(틀린 헤딩 미발행)
            msg.pose.orientation.w = 1.0
        R.pub.publish(msg)

        # 속도(보조) — EKF 월드속도 vx,vy + gyro yaw-rate. ⚠ noisy(위치미분), 제어백본 X.
        tw = TwistStamped()
        tw.header.stamp = now.to_msg(); tw.header.frame_id = self.frame
        tw.twist.linear.x = float(R.est.x[VX]); tw.twist.linear.y = float(R.est.x[VY])
        tw.twist.angular.z = float(R.last_wz)
        R.vel_pub.publish(tw)


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
