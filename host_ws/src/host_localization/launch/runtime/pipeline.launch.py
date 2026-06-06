"""호스트 런타임 위치추정 파이프라인 — colcon 전체 (역할별 노드 한 번에).

  소스: uwb(/uwb/*) · global_cam(/global_cam/*) · rover_bridge(/{robot}/imu·joint_states)
  Fusion: fusion_node (PositionKF + IMU헤딩) → /{robot}/pose
  Output: global_position (relay → /{robot}/global/position → host→rover 브릿지)

⚠️ Perception(YOLO, /perception/{robot}/bev_pose 발행)은 `.venv_yolo`(ultralytics)라 colcon과
   인터프리터가 달라 **별도 프로세스** — 이 런치에 안 들어감. 토픽으로만 연결(run_perception 스크립트).
   YOLO 없을 땐 fusion이 UWB-only로 동작(파이프라인 정상).

Run:  ros2 launch host_localization pipeline.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

_SHARE = get_package_share_directory('host_localization')


def _inc(name, **args):
    # 각 include를 scoped GroupAction으로 격리 (bringup.launch.py의 _include와 동일 패턴).
    # IncludeLaunchDescription은 자체 설정 scope를 안 만들어, 형제 런치가 같은 이름의
    # launch arg('params_file')를 선언하면 먼저 include된 uwb의 값(uwb_config.yaml)이
    # global_cam으로 누수돼 global_cam_node가 uwb_config를 읽고(자기 섹션 없어) intrinsic·
    # ROI가 default로 떨어진다(camera_info 미발행). scoped=True가 각 launch config를 그룹
    # 안에 가둬 누수를 막는다.
    src = PythonLaunchDescriptionSource(os.path.join(_SHARE, 'launch', name))
    inc = IncludeLaunchDescription(src, launch_arguments=args.items() if args else None)
    return GroupAction(scoped=True, actions=[inc])


def generate_launch_description():
    return LaunchDescription([
        _inc('uwb.launch.py'),
        _inc('global_cam_runtime.launch.py'),
        _inc('rover_bridge.launch.py'),
        Node(
            package='host_localization', executable='fusion_node',
            name='fusion_node', output='screen', emulate_tty=True,
        ),
        # Output relay 입력 = 융합 결과 /{robot}/pose (UWB passthrough 아님)
        _inc('global_position.launch.py',
             leader_in='/leader/pose', follower_in='/follower/pose'),
    ])
