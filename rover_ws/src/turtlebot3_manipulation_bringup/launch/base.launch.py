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
# Author: Darby Lim, Hye-jong KIM
#
# Modified for the leader rover: xacro from xacro/, camera and EEF camera arguments,
# arm stay pose sent once after the arm controller starts.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import RegisterEventHandler
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command
from launch.substitutions import FindExecutable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'start_rviz',
            default_value='false',
            description='Whether execute rviz2'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'prefix',
            default_value='',
            description='Prefix of the joint and link names'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Start robot in Gazebo simulation.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='false',
            description='Start robot with fake hardware mirroring command to its states.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'fake_sensor_commands',
            default_value='false',
            description='Enable fake command interfaces for sensors used for simple simulations. \
            Used only if "use_fake_hardware" parameter is true.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_camera_driver_tf',
            default_value='true',
            description='Let the external camera driver publish camera internal TF frames.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_eef_usb_camera',
            default_value='true',
            description='Attach the end-effector USB camera frames to robot_description.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'eef_usb_camera_parent',
            default_value='dummy_mimic_fix',
            description='Parent link for the end-effector USB camera frame. Use the full link name.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'eef_usb_camera_xyz',
            default_value='0.02 0.0 0.065',
            description='End-effector USB camera translation relative to eef_usb_camera_parent.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'eef_usb_camera_rpy',
            default_value='0.0 0.0 0.0',
            description='End-effector USB camera rotation relative to eef_usb_camera_parent.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'move_to_stay_pose',
            default_value='true',
            description='Move the manipulator to the saved stay pose after the arm controller starts.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'stay_pose_joint_trajectory_topic',
            default_value='/arm_controller/joint_trajectory',
            description='JointTrajectory topic used by the startup stay-pose command.'
        )
    )

    start_rviz = LaunchConfiguration('start_rviz')
    prefix = LaunchConfiguration('prefix')
    use_sim = LaunchConfiguration('use_sim')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands')
    use_camera_driver_tf = LaunchConfiguration('use_camera_driver_tf')
    use_eef_usb_camera = LaunchConfiguration('use_eef_usb_camera')
    eef_usb_camera_parent = LaunchConfiguration('eef_usb_camera_parent')
    eef_usb_camera_xyz = LaunchConfiguration('eef_usb_camera_xyz')
    eef_usb_camera_rpy = LaunchConfiguration('eef_usb_camera_rpy')
    move_to_stay_pose = LaunchConfiguration('move_to_stay_pose')
    stay_pose_joint_trajectory_topic = LaunchConfiguration('stay_pose_joint_trajectory_topic')

    urdf_file = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution(
                [
                    FindPackageShare('turtlebot3_manipulation_description'),
                    'xacro',
                    'turtlebot3_manipulation.urdf.xacro'
                ]
            ),
            ' ',
            'prefix:=',
            prefix,
            ' ',
            'use_sim:=',
            use_sim,
            ' ',
            'use_fake_hardware:=',
            use_fake_hardware,
            ' ',
            'fake_sensor_commands:=',
            fake_sensor_commands,
            ' ',
            'use_camera_driver_tf:=',
            use_camera_driver_tf,
            ' ',
            'use_eef_usb_camera:=',
            use_eef_usb_camera,
            ' ',
            'eef_usb_camera_parent:=',
            eef_usb_camera_parent,
            ' ',
            'eef_usb_camera_xyz:=',
            '"',
            eef_usb_camera_xyz,
            '"',
            ' ',
            'eef_usb_camera_rpy:=',
            '"',
            eef_usb_camera_rpy,
            '"',
        ]
    )

    controller_manager_config = PathJoinSubstitution(
        [
            FindPackageShare('turtlebot3_manipulation_bringup'),
            'config',
            'hardware_controller_manager.yaml',
        ]
    )

    rviz_config_file = PathJoinSubstitution(
        [
            FindPackageShare('turtlebot3_manipulation_bringup'),
            'rviz',
            'turtlebot3_manipulation.rviz'
        ]
    )

    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': urdf_file},
            controller_manager_config
        ],
        remappings=[
            ('~/cmd_vel_unstamped', 'cmd_vel'),
            ('~/odom', 'odom')
        ],
        output='both',
        condition=UnlessCondition(use_sim))

    robot_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': urdf_file, 'use_sim_time': use_sim}],
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        condition=IfCondition(start_rviz)
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    diff_drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '-c', '/controller_manager'],
        output='screen',
        condition=UnlessCondition(use_sim)
    )

    imu_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['imu_broadcaster'],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller'],
        output='screen',
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller'],
        output='screen',
    )

    stay_pose_msg = (
        '{joint_names: [joint1, joint2, joint3, joint4], '
        'points: [{positions: [0.104311, 0.027612, -0.001534, -1.638291], '
        'time_from_start: {sec: 3}}]}'
    )

    move_arm_to_stay_pose = ExecuteProcess(
        cmd=[
            FindExecutable(name='ros2'),
            'topic',
            'pub',
            '--time',
            '4',
            '--rate',
            '0.5',
            stay_pose_joint_trajectory_topic,
            'trajectory_msgs/msg/JointTrajectory',
            stay_pose_msg,
        ],
        output='screen',
        condition=IfCondition(move_to_stay_pose),
    )

    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        )
    )

    delay_diff_drive_controller_spawner_after_joint_state_broadcaster_spawner = \
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[diff_drive_controller_spawner],
            )
        )

    delay_imu_broadcaster_spawner_after_joint_state_broadcaster_spawner = \
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[imu_broadcaster_spawner],
            )
        )

    delay_arm_controller_spawner_after_joint_state_broadcaster_spawner = \
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[arm_controller_spawner],
            )
        )

    delay_gripper_controller_spawner_after_joint_state_broadcaster_spawner = \
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[gripper_controller_spawner],
            )
        )

    delay_stay_pose_after_arm_controller_spawner = \
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=arm_controller_spawner,
                on_exit=[
                    TimerAction(
                        period=1.0,
                        actions=[move_arm_to_stay_pose],
                    )
                ],
            )
        )

    nodes = [
        control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        delay_rviz_after_joint_state_broadcaster_spawner,
        delay_diff_drive_controller_spawner_after_joint_state_broadcaster_spawner,
        delay_imu_broadcaster_spawner_after_joint_state_broadcaster_spawner,
        delay_arm_controller_spawner_after_joint_state_broadcaster_spawner,
        delay_gripper_controller_spawner_after_joint_state_broadcaster_spawner,
        delay_stay_pose_after_arm_controller_spawner,
    ]

    return LaunchDescription(declared_arguments + nodes)
