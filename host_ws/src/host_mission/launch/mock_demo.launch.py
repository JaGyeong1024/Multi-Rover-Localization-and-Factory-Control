"""Mock demo — full host coordination loop with NO real robots or localization stack.
Starts the mission coordinator + two mock_rover stand-ins (leader@STANDBY,
follower@UNLOADING) + mock pick&place on domain 30. The mock rovers republish the real
contract topic /{robot}/global/position, closing the loop. (Rover-side interface
reference = rover_nav_ref; not used here.) Dispatch with:

    ros2 topic pub --once /mission/dispatch std_msgs/String "data: 'fetch:gear_01'"
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    leader = Node(
        package='host_mission', executable='mock_rover_node', name='rover_nav_leader',
        parameters=[{'robot': 'leader', 'start_x': 1.8, 'start_y': 4.3, 'speed': 0.6}])
    follower = Node(
        package='host_mission', executable='mock_rover_node', name='rover_nav_follower',
        parameters=[{'robot': 'follower', 'start_x': 1.8, 'start_y': 3.7, 'speed': 0.6}])
    coordinator = Node(
        package='host_mission', executable='mission_coordinator_node',
        name='mission_coordinator_node',
        parameters=[{'health_gating': False}])   # mock pose ≠ real localization
    pickplace = Node(
        package='host_mission', executable='mock_pickplace_node', name='mock_pickplace',
        parameters=[{'robot': 'leader', 'duration': 3.0}])
    return LaunchDescription([leader, follower, coordinator, pickplace])
