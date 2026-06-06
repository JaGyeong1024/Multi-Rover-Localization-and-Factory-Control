"""Mission domain bridges (host 30 <-> rovers 25/73).

Broadcasts /mission/state + /mission/queue to BOTH rover domains, forwards each
robot's mission command (target_cmd, leader pickplace_cmd) to its domain, and returns
nav/pick&place feedback to the host. Run alongside the coordinator on the host.

    ros2 launch host_mission mission_bridge.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

CFG = ('bridge_mission_30_to_25', 'bridge_mission_30_to_73',
       'bridge_mission_25_to_30', 'bridge_mission_73_to_30')


def generate_launch_description():
    share = get_package_share_directory('host_mission')
    bridges = [
        Node(package='domain_bridge', executable='domain_bridge', name=cfg,
             arguments=[os.path.join(share, 'config', cfg + '.yaml')])
        for cfg in CFG
    ]
    return LaunchDescription(bridges)
