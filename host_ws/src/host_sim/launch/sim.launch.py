"""Launch the 2D rover simulator (stands in for rover bases + host localization).

Example (never on the real domains 25/30/73):
  ROS_DOMAIN_ID=43 ros2 launch host_sim sim.launch.py \
      leader_cmd_topic:=/leader/cmd_vel_out follower_cmd_topic:=/follower/cmd_vel_out
Remap each rover safety node's output (/cmd_vel on the real rover) to these topics.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('host_sim'), 'config', 'sim_default.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('leader_cmd_topic', default_value='/leader/cmd_vel_out'),
        DeclareLaunchArgument('follower_cmd_topic', default_value='/follower/cmd_vel_out'),
        Node(
            package='host_sim', executable='sim_node', name='host_sim', output='screen',
            parameters=[{
                'config_file': LaunchConfiguration('config_file'),
                'leader_cmd_topic': LaunchConfiguration('leader_cmd_topic'),
                'follower_cmd_topic': LaunchConfiguration('follower_cmd_topic'),
            }],
        ),
    ])
