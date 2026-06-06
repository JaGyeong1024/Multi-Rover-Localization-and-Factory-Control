"""호스트 위치추정 — 소스 기동(bringup) + 최종 추정 위치 발행.

bringup.launch.py(UWB+CAM+IMU)를 그대로 include하므로 **이 런치 하나로 전부** 올라온다.
localization_node가 가용 입력(현재 UWB corrected)을 모아 /leader·follower/pose 를 발행한다.
토글 인자(uwb/camera/imu)는 bringup으로 그대로 전달.

예)  ros2 launch host_localization estimate.launch.py
     ros2 launch host_localization estimate.launch.py imu:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare('host_localization')
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, 'launch', 'bringup.launch.py'])),
        launch_arguments={
            'uwb': LaunchConfiguration('uwb'),
            'camera': LaunchConfiguration('camera'),
            'imu': LaunchConfiguration('imu'),
        }.items(),
    )
    return LaunchDescription([
        DeclareLaunchArgument('uwb', default_value='true'),
        DeclareLaunchArgument('camera', default_value='true'),
        DeclareLaunchArgument('imu', default_value='true'),
        bringup,
        Node(
            package='host_localization',
            executable='localization_node',
            name='localization_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
