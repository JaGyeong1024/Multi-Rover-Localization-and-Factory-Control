"""Host localization pipeline, all roles in one launch.

  Sources: uwb (/uwb/*), global_cam (/global_cam/*), rover_bridge (/{robot}/imu, joint_states)
  Fusion:  fusion_node (position filter + IMU heading) -> /{robot}/pose
  Output:  global_position (relay -> /{robot}/global/position -> host-to-rover bridges)

Perception (YOLO, publishes /perception/{robot}/bev_pose) runs in `.venv_yolo` with another
interpreter, so it is a separate process (run_perception script) and connects by topic only.
Without it fusion runs on UWB alone.

Run:  ros2 launch host_localization pipeline.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

_SHARE = get_package_share_directory('host_localization')


def _inc(name, **args):
    # Scoped GroupAction per include: sibling launches declare the same 'params_file' argument,
    # and without a scope the first include's value (uwb_config.yaml) leaks into the next one.
    src = PythonLaunchDescriptionSource(os.path.join(_SHARE, 'launch', name))
    inc = IncludeLaunchDescription(src, launch_arguments=args.items() if args else None)
    return GroupAction(scoped=True, actions=[inc])


def generate_launch_description():
    return LaunchDescription([
        _inc('uwb.launch.py'),
        _inc('global_cam_runtime.launch.py'),
        _inc('rover_bridge.launch.py'),
        Node(
            package='host_localization', executable='fusion_node',
            name='fusion_node', output='screen', emulate_tty=True,
            # bev_frames: YOLO can label a rover as the other one, so every box of a frame is
            # assigned by position (frame_assoc.py); needs perception publish_all_boxes:=true.
            # zero_motion / wheel_still: a rover whose wheels stand still keeps its position
            # and stays GOOD through a camera gap (an arm over the other rover can merge both
            # into one box). Moving rovers keep the camera-age rule.
            # Wheel radius and joint names: config/hardware/fusion_wheels.yaml.
            parameters=[os.path.join(_SHARE, 'config', 'hardware', 'fusion_wheels.yaml'),
                        {'leader.bev_frames': True, 'follower.bev_frames': True,
                         'quality.mode': 'camera_age_hold',
                         'leader.zero_motion': True, 'follower.zero_motion': True,
                         'follower.wheel_still': True}],
        ),
        # The relay takes the fused /{robot}/pose, not the raw UWB.
        _inc('global_position.launch.py',
             leader_in='/leader/pose', follower_in='/follower/pose'),
    ])
