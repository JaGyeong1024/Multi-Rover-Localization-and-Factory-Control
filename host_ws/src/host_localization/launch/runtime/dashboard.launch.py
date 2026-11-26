"""Dashboard host services: web bridges and health aggregation.

  rosbridge_websocket :9090  (roslibjs <-> ROS, dashboard pub/sub)
  rosapi_node                (topic / type lookup)
  web_video_server    :8080  (MJPEG: /global_cam/processed/compressed -> <img>)
  localization_diagnostics_node -> /localization/diagnostics (health tiles + mission gate)

Needs once: sudo apt install ros-humble-rosbridge-suite ros-humble-web-video-server
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
