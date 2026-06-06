#!/usr/bin/env python3
"""localization_node — 호스트 최종 위치 추정 발행 (/leader·follower/pose).

여러 모달리티(UWB corrected, 향후 천장 BEV pose, IMU heading)를 융합해 각 로버의
최종 글로벌 pose를 단일 토픽으로 내보내는 '추정 계층'의 소유자.

현 단계(카메라 검출·BEV/UWB fusion 미구현, DEF-004): 가용 입력은 UWB뿐이라 corrected
UWB pose를 **그대로 통과**시켜 발행한다 — leader는 affine 보정된 /uwb/leader_pose_corrected,
follower는 보정 모델이 없어 raw /uwb/follower_pose.

⚠️ 시간 필터링(EMA/이동평균 등) 절대 추가 금지(헌법 #2): UWB raw 측정 보존이 상위 KF의
전제다. 여기는 모달리티 '선택/융합' 자리이지 평활 자리가 아니다. 카메라 검출이 생기면 BEV
pose를 별도 토픽으로 받아 이 노드에서 선택/융합하고, /leader·follower/pose는 계속 이 노드가
**단일 발행자**로 유지한다(ceiling_tracker가 /{name}/pose를 직접 쏘는 옛 경로와의 정합은
DEF-004 — bringup엔 ceiling_tracker가 없어 현재 충돌 없음).
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class LocalizationNode(Node):
    def __init__(self):
        super().__init__('localization_node')
        # (입력, 출력) 쌍. 토픽 재배치는 파라미터로. leader=corrected, follower=raw
        # (follower는 affine 보정 모델 미존재 — 비대칭이 의도된 것).
        pairs = [
            (self.declare_parameter('leader_in', '/uwb/leader_pose_corrected').value,
             self.declare_parameter('leader_out', '/leader/pose').value),
            (self.declare_parameter('follower_in', '/uwb/follower_pose').value,
             self.declare_parameter('follower_out', '/follower/pose').value),
        ]
        for src, dst in pairs:
            pub = self.create_publisher(PoseStamped, dst, 10)
            # 들어온 메시지를 헤더(stamp·frame_id) 그대로 재발행 = 순수 통과(필터 X).
            # 기본인자로 pub 캡처 — 루프 변수 late-binding 방지.
            self.create_subscription(
                PoseStamped, src, lambda msg, p=pub: p.publish(msg), 10)
            self.get_logger().info(f'relay {src} -> {dst}')


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()
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
