"""bag 기반 대시보드 개발용 — 장치 없이 *추정 처리 노드만* (fusion + global_position).

`ros2 bag play <bag>`가 UWB/IMU/joint_states/camera를 도메인 30에 흘리면 이 런치가
fusion → `/{robot}/pose` → global_position → `/{robot}/global/position` 을 생성한다.
장치 읽는 노드(uwb_reader·global_cam 캡처·rover_bridge)는 **안 띄운다**(bag와 충돌 회피).
perception(YOLO)은 `.venv_yolo`라 별도 — `run_dashboard_bag.sh`가 옵션 기동. 없으면 UWB+IMU-only.

⚠ test2 bag 주의: follower는 raw `/uwb/follower_pose`만 녹화됨(corrected 토픽 없음) →
   `follower_uwb` 기본을 raw로(leader는 corrected가 bag에 있음).

Run:  ros2 launch host_localization bag_estimation.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('leader_uwb', default_value='/uwb/leader_pose_corrected'),
        DeclareLaunchArgument('follower_uwb', default_value='/uwb/follower_pose'),
        Node(
            package='host_localization', executable='fusion_node', name='fusion_node',
            output='screen', emulate_tty=True,
            parameters=[{
                'leader.uwb': LaunchConfiguration('leader_uwb'),
                'follower.uwb': LaunchConfiguration('follower_uwb'),
            }],
        ),
        Node(
            package='host_localization', executable='global_position_node',
            name='global_position_node', output='screen', emulate_tty=True,
        ),
    ])
