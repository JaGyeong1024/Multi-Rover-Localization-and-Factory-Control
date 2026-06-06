from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('host_localization')

    return LaunchDescription([
        DeclareLaunchArgument('camera_device',    default_value='/dev/video2'),
        DeclareLaunchArgument('detection_method', default_value='color'),
        DeclareLaunchArgument('model_path',       default_value=''),
        DeclareLaunchArgument('calibration_path', default_value=''),
        DeclareLaunchArgument('remap_path',       default_value=''),
        DeclareLaunchArgument('params_file',
            default_value=PathJoinSubstitution(
                [pkg_share, 'config', 'ceiling_tracker_params.yaml'])),

        Node(
            package='host_localization',
            executable='ceiling_tracker_node',
            name='ceiling_tracker_node',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'camera_device':    LaunchConfiguration('camera_device'),
                    'detection_method': LaunchConfiguration('detection_method'),
                    'model_path':       LaunchConfiguration('model_path'),
                    'calibration_path': LaunchConfiguration('calibration_path'),
                    'remap_path':       LaunchConfiguration('remap_path'),
                }
            ],
            output='screen',
            emulate_tty=True,
        ),
    ])
