"""map.json-based mission sim — same as mock_demo (mock_rover stand-ins) but on the real
demo-space graph (lane_graph_map.json) with map zones/shelves as standby/unloading/parts.
Leader starts at the leader_standby node (32); follower at the loading/drop-off node (22)
— its old dedicated wait zone (node 24) was removed 2026-06-11.

    ros2 launch host_mission map_demo.launch.py
    ros2 topic echo /mission/state          # IDLE -> TO_PICKUP -> LOADING -> ... -> IDLE
    ros2 topic pub --once /task/add std_msgs/String "data: 'fetch:gear_01'"
    ros2 topic pub --once /unload/done std_msgs/Empty "{}"      # operator 하역완료 (in UNLOADING)
    ros2 topic pub --once /estop std_msgs/Bool "data: true"     # E-stop (false to release)
"""
from launch import LaunchDescription
from launch_ros.actions import Node

GRAPH = 'lane_graph_map.json'


def generate_launch_description():
    leader = Node(
        package='host_mission', executable='mock_rover_node', name='rover_nav_leader',
        parameters=[{'robot': 'leader', 'start_x': 1.8, 'start_y': 3.3,
                     'speed': 0.6, 'graph_file': GRAPH}])    # leader_standby (node 32)
    follower = Node(
        package='host_mission', executable='mock_rover_node', name='rover_nav_follower',
        parameters=[{'robot': 'follower', 'start_x': 1.8, 'start_y': 3.8,
                     'speed': 0.6, 'graph_file': GRAPH}])    # homes at drop-off (node 22)
    coordinator = Node(
        package='host_mission', executable='mission_coordinator_node',
        name='mission_coordinator_node',
        parameters=[{'health_gating': False, 'use_nav_feedback': False,
                     'graph_file': GRAPH, 'scenario_file': 'mission_scenario_map.yaml'}])
    pickplace = Node(
        package='host_mission', executable='mock_pickplace_node', name='mock_pnp',
        parameters=[{'duration': 3.0}])
    return LaunchDescription([leader, follower, coordinator, pickplace])
