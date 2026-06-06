from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('host_localization')

    return LaunchDescription([
        DeclareLaunchArgument('params_file',
            default_value=PathJoinSubstitution(
                [pkg_share, 'config', 'uwb_config.yaml'])),

        Node(
            package='host_localization',
            executable='uwb_correction_node',
            name='uwb_correction_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
            emulate_tty=True,
        ),
    ])
