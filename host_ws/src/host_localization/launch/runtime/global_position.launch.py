"""Host -> rover 글로벌 위치 출력 인터페이스 (domain 30 -> 25 / 73).

global_position_node가 호스트 domain 30에 `/{robot}/global/position`(geometry_msgs/PointStamped,
x·y) 발행 + domain_bridge 2인스턴스가 host(30) -> leader(25)/follower(73)로 전달 = 로버 제어 입력.

입력(내부)은 현재 **UWB**(리더=corrected, 팔로워=raw) — 아직 EKF 융합 전 단계.
융합(fusion_node) 완성 시 `leader_in:=/leader/pose follower_in:=/follower/pose`로 override만 하면
됨 (인터페이스 토픽·타입 불변). docs/host_fusion_design.md.

Run:    ros2 launch host_localization global_position.launch.py
Verify: ROS_DOMAIN_ID=73 ros2 topic echo /follower/global/position --once   # 팔로워 도착
        ROS_DOMAIN_ID=25 ros2 topic echo /leader/global/position --once     # 리더 도착
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('host_localization')
    return LaunchDescription([
        # fusion(EKF) 출력을 글로벌 계약으로. 입력 토픽만 override 하면 추정기 교체 가능.
        DeclareLaunchArgument('leader_in', default_value='/leader/pose'),
        DeclareLaunchArgument('follower_in', default_value='/follower/pose'),
        DeclareLaunchArgument('leader_vel', default_value='/leader/velocity'),
        DeclareLaunchArgument('follower_vel', default_value='/follower/velocity'),
        Node(
            package='host_localization',
            executable='global_position_node',
            name='global_position_node',
            output='screen',
            emulate_tty=True,
            parameters=[{                          # 출력 주기는 노드 rate 파라미터(기본 30Hz)가 통제
                'leader.pose_in': LaunchConfiguration('leader_in'),
                'follower.pose_in': LaunchConfiguration('follower_in'),
                'leader.vel_in': LaunchConfiguration('leader_vel'),
                'follower.vel_in': LaunchConfiguration('follower_vel'),
            }],
        ),
        # domain_bridge: config 경로를 positional로; rclcpp가 launch가 붙인 --ros-args를 흡수.
        Node(
            package='domain_bridge', executable='domain_bridge',
            arguments=[PathJoinSubstitution([pkg_share, 'config', 'bridge_host_to_follower.yaml'])],
            output='screen', emulate_tty=True,
        ),
        Node(
            package='domain_bridge', executable='domain_bridge',
            arguments=[PathJoinSubstitution([pkg_share, 'config', 'bridge_host_to_leader.yaml'])],
            output='screen', emulate_tty=True,
        ),
    ])
