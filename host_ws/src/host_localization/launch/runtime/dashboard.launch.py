"""대시보드 호스트측 서비스 — 웹 브릿지 + 헬스 집계.

  rosbridge_websocket :9090  (roslibjs ↔ ROS — 대시보드 pub/sub)
  rosapi_node                (토픽/타입 조회 보조)
  web_video_server    :8080  (MJPEG: /global_cam/processed/compressed → <img>)
  localization_diagnostics_node → /localization/diagnostics (헬스 타일 + 미션 게이트 공유)

선행(1회): sudo apt install ros-humble-rosbridge-suite ros-humble-web-video-server
Run:  ros2 launch host_localization dashboard.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='rosbridge_server', executable='rosbridge_websocket',
             name='rosbridge_websocket', output='screen', parameters=[{'port': 9090}]),
        Node(package='rosapi', executable='rosapi_node', name='rosapi', output='screen'),
        Node(package='web_video_server', executable='web_video_server',
             name='web_video_server', output='screen',
             parameters=[{'port': 8080, 'address': '0.0.0.0'}]),
        Node(package='host_localization', executable='localization_diagnostics_node',
             name='localization_diagnostics_node', output='screen', emulate_tty=True),
    ])
