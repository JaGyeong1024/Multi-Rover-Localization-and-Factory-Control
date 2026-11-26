#!/usr/bin/env python3
#
# Copyright 2022 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Darby Lim
#
# Modified for the leader rover: optional lidar, front depth camera (Astra Mini),
# end-effector USB camera, battery/sensor state relays, stay-pose arguments.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_lidar(context):
    if LaunchConfiguration('start_lidar').perform(context).lower() != 'true':
        return []

    lds_model = os.environ.get('LDS_MODEL')
    if lds_model == 'LDS-01':
        lidar_launch = PathJoinSubstitution(
            [
                FindPackageShare('hls_lfcd_lds_driver'),
                'launch',
                'hlds_laser.launch.py'
            ]
        )
    elif lds_model == 'LDS-02':
        lidar_launch = PathJoinSubstitution(
            [
                FindPackageShare('ld08_driver'),
                'launch',
                'ld08.launch.py'
            ]
        )
    else:
        raise RuntimeError(
            'start_lidar:=true requires LDS_MODEL to be set to LDS-01 or LDS-02.'
        )

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([lidar_launch]),
            launch_arguments={
                'port': LaunchConfiguration('lidar_port'),
                'frame_id': LaunchConfiguration('lidar_frame_id'),
            }.items(),
            condition=IfCondition(LaunchConfiguration('start_lidar')),
        )
    ]


def launch_eef_camera(context):
    if LaunchConfiguration('start_eef_camera_driver').perform(context).lower() != 'true':
        return []

    width = int(LaunchConfiguration('eef_camera_image_width').perform(context))
    height = int(LaunchConfiguration('eef_camera_image_height').perform(context))

    return [
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            namespace='eef_camera',
            name='v4l2_camera',
            output='screen',
            parameters=[{
                'video_device': LaunchConfiguration('eef_camera_video_device').perform(context),
                'camera_frame_id': LaunchConfiguration('eef_camera_frame_id').perform(context),
                'camera_name': LaunchConfiguration('eef_camera_name').perform(context),
                'camera_info_url': LaunchConfiguration('eef_camera_info_url').perform(context),
                'pixel_format': LaunchConfiguration('eef_camera_pixel_format').perform(context),
                'output_encoding': LaunchConfiguration('eef_camera_output_encoding').perform(context),
                'image_size': [width, height],
            }],
        )
    ]


def generate_launch_description():
    start_rviz = LaunchConfiguration('start_rviz')
    prefix = LaunchConfiguration('prefix')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    use_camera_driver_tf = LaunchConfiguration('use_camera_driver_tf')
    start_camera = LaunchConfiguration('start_camera')
    move_to_stay_pose = LaunchConfiguration('move_to_stay_pose')
    stay_pose_joint_trajectory_topic = LaunchConfiguration('stay_pose_joint_trajectory_topic')
    start_state_relays = LaunchConfiguration('start_state_relays')
    use_eef_usb_camera = LaunchConfiguration('use_eef_usb_camera')
    eef_usb_camera_parent = LaunchConfiguration('eef_usb_camera_parent')
    eef_usb_camera_xyz = LaunchConfiguration('eef_usb_camera_xyz')
    eef_usb_camera_rpy = LaunchConfiguration('eef_usb_camera_rpy')

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_rviz',
            default_value='false',
            description='Whether execute rviz2'),

        DeclareLaunchArgument(
            'prefix',
            default_value='',
            description='Prefix of the joint and link names'),

        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='false',
            description='Start robot with fake hardware mirroring command to its states.'),

        DeclareLaunchArgument(
            'use_camera_driver_tf',
            default_value='true',
            description='Let the external camera driver publish camera internal TF frames.'),

        DeclareLaunchArgument(
            'use_eef_usb_camera',
            default_value='true',
            description='Attach the end-effector USB camera frames to robot_description.'),

        DeclareLaunchArgument(
            'eef_usb_camera_parent',
            default_value='dummy_mimic_fix',
            description='Parent link for the end-effector USB camera frame. Use the full link name.'),

        DeclareLaunchArgument(
            'eef_usb_camera_xyz',
            default_value='0.02 0.0 0.065',
            description='End-effector USB camera translation relative to eef_usb_camera_parent.'),

        DeclareLaunchArgument(
            'eef_usb_camera_rpy',
            default_value='0.0 0.0 0.0',
            description='End-effector USB camera rotation relative to eef_usb_camera_parent.'),

        DeclareLaunchArgument(
            'start_camera',
            default_value='true',
            description='Whether to launch the Astra Mini camera driver.'),

        DeclareLaunchArgument(
            'start_eef_camera_driver',
            default_value='true',
            description='Whether to launch the v4l2 end-effector USB camera driver.'),

        DeclareLaunchArgument(
            'eef_camera_video_device',
            default_value='/dev/video0',
            description='Linux video device for the end-effector USB camera.'),

        DeclareLaunchArgument(
            'eef_camera_frame_id',
            default_value='eef_usb_camera_optical_frame',
            description='Frame id used by the end-effector camera images.'),

        DeclareLaunchArgument(
            'eef_camera_pixel_format',
            default_value='YUYV',
            description='V4L2 pixel format requested from the end-effector camera.'),

        DeclareLaunchArgument(
            'eef_camera_output_encoding',
            default_value='rgb8',
            description='ROS image encoding published by the end-effector camera.'),

        DeclareLaunchArgument(
            'eef_camera_image_width',
            default_value='320',
            description='EEF USB camera image width.'),

        DeclareLaunchArgument(
            'eef_camera_image_height',
            default_value='240',
            description='EEF USB camera image height.'),

        DeclareLaunchArgument(
            'eef_camera_name',
            default_value='eef_usb_camera',
            description='Camera name used when loading and saving EEF USB camera calibration.'),

        DeclareLaunchArgument(
            'eef_camera_info_url',
            default_value='',
            description='Camera calibration URL for the EEF USB camera (empty = none).'),

        DeclareLaunchArgument(
            'move_to_stay_pose',
            default_value='true',
            description='Move the manipulator to the saved stay pose after the arm controller starts.'),

        DeclareLaunchArgument(
            'stay_pose_joint_trajectory_topic',
            default_value='/arm_controller/joint_trajectory',
            description='JointTrajectory topic used by the startup stay-pose command.'),

        DeclareLaunchArgument(
            'start_state_relays',
            default_value='true',
            description='Publish /battery_state and /sensor_state from ros2_control /dynamic_joint_states.'),

        DeclareLaunchArgument(
            'start_lidar',
            default_value='false',
            description='Whether to launch a lidar driver alongside the robot bringup.'),

        DeclareLaunchArgument(
            'lidar_port',
            default_value='/dev/ttyUSB0',
            description='Connected USB port for the lidar.'),

        DeclareLaunchArgument(
            'lidar_frame_id',
            default_value='base_scan',
            description='Frame id used by the lidar driver.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/base.launch.py']),
            launch_arguments={
                'start_rviz': start_rviz,
                'prefix': prefix,
                'use_fake_hardware': use_fake_hardware,
                'use_camera_driver_tf': use_camera_driver_tf,
                'use_eef_usb_camera': use_eef_usb_camera,
                'eef_usb_camera_parent': eef_usb_camera_parent,
                'eef_usb_camera_xyz': eef_usb_camera_xyz,
                'eef_usb_camera_rpy': eef_usb_camera_rpy,
                'move_to_stay_pose': move_to_stay_pose,
                'stay_pose_joint_trajectory_topic': stay_pose_joint_trajectory_topic,
            }.items(),
        ),

        Node(
            package='leader_platooning_beacon',
            executable='battery_state_from_dynamic_joint_state_node',
            name='leader_battery_state_relay',
            output='screen',
            parameters=[{
                'dynamic_joint_states_topic': '/dynamic_joint_states',
                'battery_state_topic': '/battery_state',
                'battery_sensor_name': 'battery',
                'publish_period_s': 1.0,
            }],
            condition=IfCondition(start_state_relays),
        ),

        Node(
            package='leader_platooning_beacon',
            executable='sensor_state_from_dynamic_joint_state_node',
            name='leader_sensor_state_relay',
            output='screen',
            parameters=[{
                'dynamic_joint_states_topic': '/dynamic_joint_states',
                'sensor_state_topic': '/sensor_state',
                'battery_sensor_name': 'battery',
                'left_wheel_name': 'wheel_left_joint',
                'right_wheel_name': 'wheel_right_joint',
                'publish_period_s': 0.1,
            }],
            condition=IfCondition(start_state_relays),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/astra_mini.launch.py']),
            condition=IfCondition(start_camera),
        ),

        OpaqueFunction(function=launch_lidar),

        OpaqueFunction(function=launch_eef_camera),
    ])
