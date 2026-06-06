"""Launch the FINAL runtime global camera node (full freeze + camera_info).

Intrinsic-capture (geometric-only, exposure AUTO) lives in global_cam.launch.py
(camera_publisher_node); this one is for run time.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('host_localization')
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [pkg_share, 'config', 'global_cam_params.yaml'])),
        Node(
            package='host_localization',
            executable='global_cam_node',
            name='global_cam_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
            emulate_tty=True,
        ),
    ])
