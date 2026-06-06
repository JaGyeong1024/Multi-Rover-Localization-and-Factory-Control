"""호스트 위치추정 소스 일괄 기동 (UWB + 천장 카메라 + 로버 IMU 브릿지).

bag 녹화(record_bag.sh)와 estimate.launch.py가 공유하는 **'소스 계층'** — 무엇을 켜는지의
단일 진실. 각 소스는 인자로 토글한다:
    uwb:=true|false      UWB reader+anchor+correction        (uwb.launch.py)
    camera:=true|false   global_cam_node 런타임 파이프라인    (global_cam_runtime.launch.py)
    imu:=true|false      로버 IMU 도메인 브릿지 + 동기 점검    (rover_bridge.launch.py)

예)  ros2 launch host_localization bringup.launch.py            # 전부
     ros2 launch host_localization bringup.launch.py imu:=false # IMU 빼고
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _include(launch_file, toggle):
    """하위 런치를 토글 조건부 + **독립 스코프**로 include.

    하위 런치들이 모두 'params_file'이라는 동일 이름 인자를 선언하므로 scoped GroupAction
    으로 격리해야 한다. 안 그러면 먼저 include된 uwb의 params_file(uwb_config.yaml)이
    뒤따르는 global_cam include로 누수돼, global_cam_node가 uwb_config.yaml을 읽고(자기
    섹션이 없어) 전부 기본값으로 떨어진다(camera_info 미발행·ROI 기본값). scoped=True가
    각 include의 LaunchConfiguration을 그 그룹 안에 가둬 누수를 막는다.
    """
    share = FindPackageShare('host_localization')
    return GroupAction(
        condition=IfCondition(LaunchConfiguration(toggle)),
        scoped=True,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([share, 'launch', launch_file])))],
    )


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('uwb', default_value='true'),
        DeclareLaunchArgument('camera', default_value='true'),
        DeclareLaunchArgument('imu', default_value='true'),
        _include('uwb.launch.py', 'uwb'),
        _include('global_cam_runtime.launch.py', 'camera'),
        _include('rover_bridge.launch.py', 'imu'),
    ])
