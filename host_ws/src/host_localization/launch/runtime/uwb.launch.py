from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('host_localization')

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyUWB'),
        DeclareLaunchArgument('params_file',
            default_value=PathJoinSubstitution(
                [pkg_share, 'config', 'uwb_config.yaml'])),

        Node(
            package='host_localization',
            executable='uwb_reader_node',
            name='uwb_reader_node',
            parameters=[
                LaunchConfiguration('params_file'),
                {'port': LaunchConfiguration('port')},
            ],
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='host_localization',
            executable='anchor_publisher_node',
            name='anchor_publisher_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='host_localization',
            executable='uwb_correction_node',
            name='uwb_correction_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
            emulate_tty=True,
        ),
    ])
