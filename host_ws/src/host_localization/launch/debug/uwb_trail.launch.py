from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='host_localization',
            executable='uwb_trail_cloud_node',
            name='uwb_trail_cloud_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
