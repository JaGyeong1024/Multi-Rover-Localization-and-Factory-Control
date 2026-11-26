"""v6 pick and place on the running leader bringup (arm already at Stay).

    onnx_probe_cpp (ONNX parity check) -> then:
    v4l2 EEF camera, mp_control_node (ArUco profile + overrides below),
    ros_policy_relay_cpp (v6 residual on mp_control's arm trajectories),
    ros_policy_observer (EEF ArUco -> target topics + 69-D observation),
    ros_trial_gate (waits for Stay and live inputs, sends /mp_control/start once).
The launch shuts down when the gate exits. No serial device is opened here.
"""
from pathlib import Path
import os
import time
import yaml
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, RegisterEventHandler, EmitEvent, LogInfo
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node
from omx_turtlebot3_runtime.launch_support import runtime_python, calibrated_camera


def build(context):
    def value(name):return LaunchConfiguration(name).perform(context)
    execute=value('execute')=='true'
    selected=int(value('marker_id'))
    if selected not in (0,1):raise ValueError('Unsupported marker ID')
    retract_mode=value('retract_mode')
    if retract_mode not in ('cartesian','joint','lift'):raise ValueError('Unsupported handoff mode')
    runtime=Path(get_package_share_directory('omx_turtlebot3_runtime'))
    mp=Path(get_package_share_directory('mp_control'))
    # Model files must sit in one real directory (checksum check); with --symlink-install
    # the installed files are links, so use the directory they point to.
    policy=(Path(get_package_share_directory('omx_turtlebot3_policy'))/'models/omx_turtlebot3_v6'/'actor.onnx').resolve().parent
    python=runtime_python(get_package_prefix('omx_turtlebot3_runtime'),value('python_executable'))
    logs=Path(value('log_dir')).expanduser().absolute()
    if logs.exists():raise ValueError('Use a new log_dir for each trial')
    domain=int(value('ros_domain_id'))
    if not 0<=domain<=232:raise ValueError('Invalid ROS domain ID')
    wrist_offset=float(value('gripper_down_joint4_offset_rad'))
    if not -.4<=wrist_offset<=.4:raise ValueError('Invalid wrist pitch offset')
    env={'ROS_DOMAIN_ID':str(domain),'ROS_LOG_DIR':str(logs/'ros'),'PYTHONNOUSERSITE':'1'}
    config=runtime/'config/runtime.yaml'
    camera_file=runtime/'config/eef_usb_camera.yaml';calibrated_camera(camera_file)
    # mp_control: ArUco profile, overridden for the v6 relay (raw arm topic, handoff via the relay).
    profile=yaml.safe_load((mp/'config/mp_control_aruco_params.yaml').read_text())
    params=profile['mp_control_node']['ros__parameters']
    params.update(auto_start=False,auto_start_on_bbox=False,hold_joint4_during_extension=False,
        pause_before_grasp=value('pause_before_grasp')=='true',
        pause_before_rotate=value('pause_before_rotate')=='true',
        handoff_lift_joint2_delta_rad=-.55 if retract_mode=='lift' else 0.,
        handoff_retract_joint3_delta_rad=0. if retract_mode=='lift' else .65,
        handoff_retract_hold_z=retract_mode=='cartesian',handoff_retract_distance_m=.05,
        handoff_restore_grasp_pose=retract_mode=='lift',handoff_lower_before_release=retract_mode=='lift',
        handoff_lift_joint_tolerance_rad=.10,
        handoff_lift_duration_s=1.8,handoff_place_duration_s=2.,
        pregrasp_preserve_gripper_roll=True,
        pregrasp_move_duration_s=.35,pregrasp_settle_s=.05,pregrasp_republish_period_s=.4,
        pregrasp_joint2_max_step_rad=.15,pregrasp_joint3_max_step_rad=.25,pregrasp_joint4_max_step_rad=.15,
        handoff_rotate_angle_rad=-1.5707963267948966,
        handoff_rotate_odd_angle_deg=-90.,handoff_rotate_even_angle_deg=-90.,
        gripper_down_joint4_offset_rad=wrist_offset,
        joint_trajectory_topic='/arm_controller/joint_trajectory_raw',
        handoff_joint_trajectory_topic='/omx_v6/nominal_joint_trajectory',
        camera_info_topic='/eef_camera/camera_info',eef_camera_info_topic='/eef_camera/camera_info',
        object_width_m=.05,object_height_m=.05,gripper_fallback_object_width_m=.05,
        handoff_grasp_settle_s=2.,aruco_joint1_align_gain=2.,aruco_joint1_align_max_step_rad=.04)
    mp_node=Node(package='mp_control',executable='mp_control_node',output='screen',parameters=[params],additional_env=env)
    relay=Node(package='omx_turtlebot3_control_cpp',executable='ros_policy_relay_cpp',
        output='screen',parameters=[{'model_dir':str(policy),'execute':execute,'max_raw_joint3_delta_rad':.25}],additional_env=env)
    observer=ExecuteProcess(cmd=[python,'-m','omx_turtlebot3_runtime.ros_policy_observer',
        '--config',str(config),'--training',str(policy/'training_config.yaml'),
        '--scene',str(runtime/'assets/scene.xml'),'--log-dir',str(logs/'observer'),
        '--selected-id',str(selected)],
        output='screen',additional_env=env)
    gate_command=[python,'-m','omx_turtlebot3_runtime.ros_trial_gate',
        '--output',str(logs/'gate.jsonl'),'--startup-timeout','45','--mission-timeout',value('mission_timeout_s'),
        '--selected-id',str(selected),'--retract-mode',retract_mode]
    if execute:gate_command.append('--execute')
    if value('resume_held')=='true':gate_command.append('--resume-held')
    gate=ExecuteProcess(cmd=gate_command,output='screen',additional_env=env)
    children=[mp_node,relay,observer,gate]
    camera=None
    if value('start_camera')=='true':
        camera=Node(package='v4l2_camera',executable='v4l2_camera_node',namespace='eef_camera',
            output='screen',additional_env=env,parameters=[{
                'video_device':value('video_device'),'camera_frame_id':'eef_usb_camera_optical_frame',
                'camera_name':'eef_usb_camera','camera_info_url':'file://'+str(camera_file),
                'pixel_format':'YUYV','output_encoding':'rgb8','image_size':[320,240]}])
        children.insert(0,camera)
    # Children start only after the ONNX parity check passes.
    check=ExecuteProcess(cmd=[str(Path(get_package_prefix('omx_turtlebot3_control_cpp'))/'lib/omx_turtlebot3_control_cpp/onnx_probe_cpp'),str(policy)],
        output='screen',additional_env=env)
    def checked(event,context):
        if event.returncode:raise RuntimeError('Original ONNX conformance failed; mission not started')
        logs.mkdir(parents=True,exist_ok=False)
        (logs/'controller_profile.yaml').write_text(yaml.safe_dump(profile,sort_keys=False))
        return children
    def finished(event,context):
        if event.returncode and not context.is_shutdown:raise RuntimeError('Mission gate stopped; inspect gate.jsonl')
        return [EmitEvent(event=Shutdown(reason='Policy trial finished; external hardware bringup keeps its hold'))]
    handlers=[RegisterEventHandler(OnProcessExit(target_action=check,on_exit=checked)),
              RegisterEventHandler(OnProcessExit(target_action=gate,on_exit=finished))]
    for child in [observer,relay,mp_node]:
        def failed(event,context):
            if event.returncode and not context.is_shutdown:raise RuntimeError('Policy trial component failed')
        handlers.append(RegisterEventHandler(OnProcessExit(target_action=child,on_exit=failed)))
    if camera is not None:
        def camera_exit(event,context):
            if not context.is_shutdown:raise RuntimeError('Trial camera exited')
        handlers.append(RegisterEventHandler(OnProcessExit(target_action=camera,on_exit=camera_exit)))
    return [LogInfo(msg='mp_control + C++ ONNX trial; execute='+str(execute)+
        '; requires existing hardware at Stay; no serial device opened by this launch'),*handlers,check]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('execute',default_value='false',choices=['true','false']),
        DeclareLaunchArgument('marker_id',default_value='0',choices=['0','1']),
        DeclareLaunchArgument('resume_held',default_value='false',choices=['true','false']),
        DeclareLaunchArgument('retract_mode',default_value='lift',choices=['lift','cartesian','joint']),
        DeclareLaunchArgument('start_camera',default_value='true',choices=['true','false']),
        DeclareLaunchArgument('video_device',default_value='/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0'),
        DeclareLaunchArgument('python_executable',default_value=''),
        DeclareLaunchArgument('ros_domain_id',default_value=EnvironmentVariable('ROS_DOMAIN_ID',default_value='25')),
        DeclareLaunchArgument('mission_timeout_s',default_value='300'),
        DeclareLaunchArgument('gripper_down_joint4_offset_rad',default_value='0.20'),
        DeclareLaunchArgument('pause_before_grasp',default_value='false',choices=['true','false']),
        DeclareLaunchArgument('pause_before_rotate',default_value='false',choices=['true','false']),
        DeclareLaunchArgument('log_dir',default_value='/tmp/omx_ros_policy_trial_'+str(time.time_ns())),
        OpaqueFunction(function=build)])
