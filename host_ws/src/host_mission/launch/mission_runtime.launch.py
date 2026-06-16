"""Real-rover mission runtime — coordinator on LIVE localization + domain bridges.

mock_demo/map_demo와 달리 mock_rover/mock_pickplace가 **없다**. 위치추정은 estimate
파이프라인(/{robot}/global/position)이, pick&place는 실로버(팀원B 컨트롤러)가 담당하고,
coordinator는 실 /localization/diagnostics로 health gating(default True)한다. 도메인
브릿지(25/73)로 실로버에 target_cmd/pickplace_cmd를 송출하고 feedback을 회수한다.

보통 호스트 통합 런처(run_host.sh)가 위치추정·대시보드와 함께 묶어 띄운다. 단독 기동:

    ros2 launch host_mission mission_runtime.launch.py
    ros2 launch host_mission mission_runtime.launch.py \\
        graph_file:=lane_graph.json scenario_file:=mission_scenario.yaml
    ros2 topic pub --once /mission/dispatch std_msgs/String "data: 'fetch:gear_01'"
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('host_mission')
    graph_file = LaunchConfiguration('graph_file')
    scenario_file = LaunchConfiguration('scenario_file')

    coordinator = Node(
        package='host_mission', executable='mission_coordinator_node',
        name='mission_coordinator_node', output='screen', emulate_tty=True,
        parameters=[{'health_gating': True,
                     'graph_file': graph_file,
                     'scenario_file': scenario_file}])

    # 도메인 브릿지(host 30 <-> rovers 25/73)는 mission_bridge.launch.py 재사용 (단일 진실).
    bridges = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', 'mission_bridge.launch.py')))

    return LaunchDescription([
        DeclareLaunchArgument('graph_file', default_value='lane_graph_map.json',
                              description='config/ 내 lane graph json (실데모공간=map)'),
        DeclareLaunchArgument('scenario_file', default_value='mission_scenario_map.yaml',
                              description='config/ 내 시나리오 yaml (실데모공간=map)'),
        coordinator,
        bridges,
    ])
