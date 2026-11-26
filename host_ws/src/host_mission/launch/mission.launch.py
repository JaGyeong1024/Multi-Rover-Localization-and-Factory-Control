"""Mission runtime for the real rovers (nav_mode: path).

Coordinator (named goal poses, host-planned PathCommand) + the mission domain
bridges (host 30 <-> leader 25 / follower 73). Localization (estimate pipeline) and
the rovers' rover_drive stack are started separately.

    ros2 launch host_mission mission.launch.py
    ros2 launch host_mission mission.launch.py bridges:=false          # host only
    ros2 launch host_mission mission.launch.py obstacle_mapper:=true   # + shared box map
      (needs /follower/scan + /leader/depth_scan: host_localization rover_bridge.launch.py)
    ros2 topic pub --once /mission/teach_goal std_msgs/String "data: 'leader:pick_gear'"
    ros2 topic pub --once /task/add std_msgs/String "data: 'fetch:gear_01'"
    ros2 topic echo /mission/events

File arguments take a name inside share/host_mission/config or an absolute path. The settings
are mission.yaml with the arena and rover overlays (config/site/arena.yaml,
config/hardware/rovers.yaml) merged over it.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

FILE_ARGS = (('settings_file', 'mission.yaml', 'planner / tolerance / exception settings'),
             ('scenario_file', 'mission_scenario.yaml', 'parts and their goal names'),
             ('goals_file', 'goals.yaml', 'named goal poses (rewritten by /mission/teach_goal)'))


def coordinator_node():
    """The path-mode coordinator with its three config-file parameters."""
    params = {name: LaunchConfiguration(name) for name, _default, _doc in FILE_ARGS}
    return Node(
        package='host_mission', executable='mission_coordinator_node',
        name='mission_coordinator_node', output='screen', emulate_tty=True,
        parameters=[{'nav_mode': 'path', **params}])


def mapper_parameter_files():
    """obstacle_mapper_node parameter files, later wins: algorithm, site keys, sensor mounts."""
    config = os.path.join(get_package_share_directory('host_mission'), 'config')
    return [os.path.join(config, 'obstacle_mapper.yaml'),
            os.path.join(config, 'site', 'obstacle_mapper.yaml'),
            os.path.join(config, 'hardware', 'sensor_mounts.yaml')]


def obstacle_mapper_node(condition):
    """Follower LiDAR + leader depth scan -> /mission/mapped_obstacles."""
    return Node(
        package='host_mission', executable='obstacle_mapper_node',
        name='obstacle_mapper_node', output='screen', emulate_tty=True,
        parameters=[*mapper_parameter_files(),
                    {'settings_file': LaunchConfiguration('settings_file')}],
        condition=condition)


def file_arguments():
    return [DeclareLaunchArgument(name, default_value=default, description=doc)
            for name, default, doc in FILE_ARGS]


def generate_launch_description():
    share = get_package_share_directory('host_mission')
    bridges = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'mission_bridge.launch.py')),
        condition=IfCondition(LaunchConfiguration('bridges')))
    return LaunchDescription([
        *file_arguments(),
        DeclareLaunchArgument('bridges', default_value='true',
                              description='start the mission domain bridges (30 <-> 25 / 73)'),
        DeclareLaunchArgument('obstacle_mapper', default_value='false',
                              description='map boxes from follower LiDAR + leader depth '
                                          '(site keys: config/site/obstacle_mapper.yaml)'),
        coordinator_node(),
        obstacle_mapper_node(IfCondition(LaunchConfiguration('obstacle_mapper'))),
        bridges,
    ])
