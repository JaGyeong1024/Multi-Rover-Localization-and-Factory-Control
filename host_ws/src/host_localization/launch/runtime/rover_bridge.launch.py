"""Rover -> host IMU domain bridges.

Runs two ``domain_bridge`` instances on the host that subscribe to the rovers' ``/imu`` on
their own domains and re-publish them on the host domain (30) under distinct names:

    leader   /imu @ domain 25  ->  /leader/imu   @ domain 30
    follower /imu @ domain 73  ->  /follower/imu @ domain 30

The rovers are not modified. Needs the ``domain_bridge`` package:
    sudo apt install ros-humble-domain-bridge

Run:    ros2 launch host_localization rover_bridge.launch.py
Verify: ros2 topic hz /leader/imu
        ros2 topic delay /leader/imu     # single-digit ms; large = clock skew
"""

from launch import LaunchDescription
from launch.actions import TimerAction
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('host_localization')

    # Node names come from each config's `name:` field, so the two processes do not clash on
    # domain 30. The config path is positional; rclcpp strips the --ros-args launch appends.
    # respawn: domain_bridge can die at startup when several bridges start together (see
    # host_mission/launch/mission_bridge.launch.py).
    return LaunchDescription([
        Node(
            package='domain_bridge',
            executable='domain_bridge',
            arguments=[
                PathJoinSubstitution([pkg_share, 'config', 'bridge_leader.yaml'])],
            output='screen',
            emulate_tty=True,
            respawn=True, respawn_delay=2.0,
        ),
        Node(
            package='domain_bridge',
            executable='domain_bridge',
            arguments=[
                PathJoinSubstitution([pkg_share, 'config', 'bridge_follower.yaml'])],
            output='screen',
            emulate_tty=True,
            respawn=True, respawn_delay=2.0,
        ),
        # Once, 4 s after the bridges start: measures the /leader|follower/imu receive delay
        # (clock skew + latency) and warns if the clocks are not in sync. Does not block.
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
