"""Follower rover: path tracker, safety gate (LiDAR stop), command watchdog.

    ros2 launch rover_drive follower.launch.py
    ros2 launch rover_drive follower.launch.py presets:=pivot_slow,xtrack_off   (config/presets/)
    ros2 launch rover_drive follower.launch.py hardware_file:=<other hardware yaml>
    ros2 launch rover_drive follower.launch.py watchdog:=false

Parameters: config/follower.yaml, then config/hardware/rover_follower.yaml, then the presets.

Velocity chain:
    path_tracker -> /follower/cmd_vel -> safety_gate -> /cmd_vel -> base
The follower base has no command timeout, so cmd_watchdog (its own process)
zeroes /cmd_vel when the gate heartbeat stops.
The YDLIDAR G6 driver is not started here; run it separately with the
config/hardware/ydlidar_g6_follower.yaml (it publishes /scan).
"""
from launch import LaunchDescription

from rover_drive.launch_common import (drive_node, hardware_argument, params_argument,
                                       presets_argument, toggle_argument)

_ROBOT = 'follower'


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        params_argument(_ROBOT),
        hardware_argument(_ROBOT),
        presets_argument(_ROBOT),
        toggle_argument('watchdog', True, 'start cmd_watchdog (base has no command timeout)'),
        drive_node('path_tracker_node', _ROBOT),
        drive_node('safety_gate_node', _ROBOT),
        drive_node('cmd_watchdog_node', _ROBOT, toggle='watchdog'),
    ])
