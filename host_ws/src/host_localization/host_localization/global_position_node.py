"""global_position_node — 호스트 최종 글로벌 인터페이스 발행 (로버 제어 입력).

안정 출력 계약: `/{robot}/global/position`(PoseStamped: x·y(z=0)+heading(quat)) +
`/{robot}/global/velocity`(TwistStamped: 월드속도+ω). frame=uwb_global. 내부 추정기와 *분리* —
입력 토픽만 파라미터로 바꾸면 인터페이스 불변(지금=fusion_node, 향후=다른 추정기/tracking). docs/host_fusion_design.md.

**출력 *주기*를 이 노드(계약 소유자)가 통제**(`rate`, 기본 30Hz). 추정기(fusion)는 *이벤트구동*으로
신선한 `/{robot}/pose`·`/velocity`만 흘리고, 여기서 최신값을 잡아 고정 주기로 재발행 →
이중-타이머 지연 회피 + 출력 주기/QoS를 한 곳에서 관리. 소스 stamp 보존(데이터 시각 추적).

호스트 domain 30 발행 → 로버는 host→rover domain_bridge(30→25/73)로 수신.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped


class GlobalPositionNode(Node):
    def __init__(self):
        super().__init__('global_position_node')
        rate = self.declare_parameter('rate', 30.0).value           # 출력 계약 주기(여기서 통제)
        self.frame = self.declare_parameter('frame_id', 'uwb_global').value
        robots = self.declare_parameter('robots', ['leader', 'follower']).value

        self.robots = {}
        for name in robots:
            pose_in = self.declare_parameter(f'{name}.pose_in', f'/{name}/pose').value
            vel_in = self.declare_parameter(f'{name}.vel_in', f'/{name}/velocity').value
            st = {
                'pose': None, 'vel': None,
                # 외부(/global/*) 발행은 default(RELIABLE) 유지 — 브릿지/로버 QoS 정합은 별도(외부 미변경).
                'pose_pub': self.create_publisher(PoseStamped, f'/{name}/global/position', 10),
                'vel_pub': self.create_publisher(TwistStamped, f'/{name}/global/velocity', 10),
            }
            # 내부 입력(fusion 발행)은 BEST_EFFORT로 일치(고빈도 최신값 우선).
            self.create_subscription(PoseStamped, pose_in,
                                     lambda m, s=st: s.__setitem__('pose', m), qos_profile_sensor_data)
            self.create_subscription(TwistStamped, vel_in,
                                     lambda m, s=st: s.__setitem__('vel', m), qos_profile_sensor_data)
            self.robots[name] = st
            self.get_logger().info(
                f'{name}: {pose_in} + {vel_in} -> /{name}/global/{{position,velocity}} @ {rate}Hz')

        self.create_timer(1.0 / float(rate), self._on_timer)

    def _on_timer(self):
        """최신 추정값을 고정 주기로 재발행. 소스 stamp 보존(데이터 시각). 미수신은 skip."""
        for st in self.robots.values():
            m = st['pose']
            if m is not None:
                out = PoseStamped()
                out.header = m.header                          # 소스 stamp 보존
                out.pose.position.x = m.pose.position.x
                out.pose.position.y = m.pose.position.y
                out.pose.position.z = 0.0                      # 2D
                out.pose.orientation = m.pose.orientation      # heading 보존
                st['pose_pub'].publish(out)
            v = st['vel']
            if v is not None:
                out = TwistStamped()
                out.header = v.header
                out.twist = v.twist
                st['vel_pub'].publish(out)


def main():
    rclpy.init()
    node = GlobalPositionNode()
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
