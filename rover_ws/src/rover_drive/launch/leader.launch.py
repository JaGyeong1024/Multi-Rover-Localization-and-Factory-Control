"""Leader rover: path tracker, safety gate, depth front distance.

    ros2 launch rover_drive leader.launch.py
    ros2 launch rover_drive leader.launch.py presets:=pivot_slow,xtrack_off   (config/presets/)
    ros2 launch rover_drive leader.launch.py hardware_file:=<other hardware yaml>
    ros2 launch rover_drive leader.launch.py depth:=false watchdog:=true

Parameters: config/leader.yaml, then config/hardware/rover_leader.yaml, then the presets.

Velocity chain:
    path_tracker -> /leader/cmd_vel -> cmd_vel_mux (leader_cmd_mux, output set to
    /cmd_vel_mux) -> safety_gate -> /cmd_vel -> base
depth_front turns /camera/depth/image_raw into /leader/front_distance, which the
gate uses for its front stop (the leader has no LiDAR). The camera driver and
the mux are started by rover_ws/run_leader.sh, not here.
The leader base stops by itself about 0.5 s after the last command, so the
command watchdog is off by default.
"""
from launch import LaunchDescription

from rover_drive.launch_common import (drive_node, hardware_argument, params_argument,
                                       presets_argument, toggle_argument)

_ROBOT = 'leader'


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        params_argument(_ROBOT),
        hardware_argument(_ROBOT),
        presets_argument(_ROBOT),
        toggle_argument('depth', True, 'start depth_front (front distance from the camera)'),
        toggle_argument('watchdog', False, 'start cmd_watchdog (base has its own timeout)'),
        drive_node('path_tracker_node', _ROBOT),
        drive_node('safety_gate_node', _ROBOT),
        drive_node('depth_front_node', _ROBOT, toggle='depth'),
        drive_node('cmd_watchdog_node', _ROBOT, toggle='watchdog'),
    ])
