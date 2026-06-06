"""Rover -> Host IMU domain bridge (MASTER_PLAN K-2, Step 1).

Runs two ``domain_bridge`` instances on the HOST that subscribe to the rovers'
``/imu`` topics on their own domains and re-publish them onto the host domain
(30) under disambiguated names:

    leader   /imu @ domain 25  ->  /leader/imu   @ domain 30
    follower /imu @ domain 73  ->  /follower/imu @ domain 30

The rovers are NOT modified (K-1). This is purely host-side curation so the
two IMU streams can be co-recorded in one bag alongside host UWB for offline
localization experiments (see record_bag.sh).

Requires the ``domain_bridge`` package (declared as an exec_depend):
    sudo apt install ros-humble-domain-bridge

Run:
    ros2 launch host_localization rover_bridge.launch.py
Verify (on host, domain 30):
    ros2 topic hz /leader/imu
    ros2 topic delay /leader/imu     # must be single-digit ms; large => clock skew
"""

from launch import LaunchDescription
from launch.actions import TimerAction
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('host_localization')

    # Node names come from each config's `name:` field (rover_bridge_leader /
    # rover_bridge_follower), so the two processes do not clash on domain 30.
    # We pass the config path positionally; rclcpp strips the launch-appended
    # --ros-args, leaving the YAML path for domain_bridge to parse.
    return LaunchDescription([
        Node(
            package='domain_bridge',
            executable='domain_bridge',
            arguments=[
                PathJoinSubstitution([pkg_share, 'config', 'bridge_leader.yaml'])],
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='domain_bridge',
            executable='domain_bridge',
            arguments=[
                PathJoinSubstitution([pkg_share, 'config', 'bridge_follower.yaml'])],
            output='screen',
            emulate_tty=True,
        ),
        # 브릿지 기동 ~4s 뒤 1회: /leader·follower/imu 수신 지연(=클럭 skew+지연)을 재서
        # 동기 안 됐으면 WARN 출력(이 콘솔에 노랗게). 브릿지 동작은 막지 않음.
        TimerAction(period=4.0, actions=[
            Node(
                package='host_localization',
                executable='check_imu_sync',
                name='check_imu_sync',
                output='screen',
                emulate_tty=True,
            ),
        ]),
    ])
