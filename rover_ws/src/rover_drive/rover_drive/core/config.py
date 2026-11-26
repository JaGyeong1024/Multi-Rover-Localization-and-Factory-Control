"""Configuration dataclasses shared by the pure core and the ROS nodes."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DriveConfig:
    """Parameters of the path tracker (``DriveController``)."""

    robot: str = 'follower'
    control_rate_hz: float = 20.0
    feedback_rate_hz: float = 5.0          # nav_feedback, >= 2 Hz required

    max_linear_speed: float = 0.20
    max_angular_speed: float = 1.0
    max_linear_accel: float = 0.30
    max_linear_decel: float = 0.50
    max_linear_jerk: float = 2.0
    max_angular_accel: float = 4.0
    omega_quantum: float = 0.0
    omega_quantize: str = 'round'
    omega_truncate_margin: float = 0.3

    ref_offset_x: float = -0.063           # tracked point ahead of rotation centre [m]
    lookahead_time_s: float = 1.0
    lookahead_min: float = 0.18
    lookahead_max: float = 0.40
    max_curvature: float = 15.0            # [1/m]
    heading_free_rad: float = 0.35
    heading_slow_rad: float = 0.90
    turn_speed: float = 0.04
    runin_heading_gain: float = 0.0
    runin_lookaheads: float = 0.0
    xtrack_integral_gain: float = 0.0
    xtrack_integral_limit: float = 1.5

    max_lateral_accel: float = 0.15
    curvature_preview_m: float = 0.30
    goal_decel: float = 0.15
    approach_speed: float = 0.03
    goal_stop_margin: float = 0.01
    gap_gain: float = 1.5
    gap_resume_margin: float = 0.05
    gap_hard_min: float = 0.08
    # Two gap definitions (never compare one with the other):
    #   free gap = my front edge to the lead's rear surface, measured by my LiDAR
    #              (via lead_gap_topic); wanted = gap_free_target.
    #   path gap = tracked point to the lead's relayed position along my path (centre to
    #              centre); wanted = PathCommand.gap.
    gap_free_target: float = 0.20
    lead_gap_timeout_s: float = 0.3
    gap_bridge: bool = False
    lead_timeout_s: float = 0.5
    lead_off_path_tol: float = 0.30
    lead_back_window: float = 0.5
    degraded_speed: float = 0.05
    quality_default: str = 'GOOD'
    lidar_stop_base: float = 0.12
    lidar_brake_decel: float = 0.50
    lidar_reaction_s: float = 0.20

    pose_timeout_s: float = 0.5
    cmd_timeout_s: float = 1.5
    gate_timeout_s: float = 1.0

    jump_speed_factor: float = 2.0
    jump_slack_m: float = 0.10
    jump_reacquire_s: float = 0.5
    off_path_max: float = 0.25
    stuck_cmd_dist: float = 0.20
    stuck_min_ratio: float = 0.2
    projection_window: float = 0.5
    init_tie_margin: float = 0.05
    default_goal_pos_tol: float = 0.05
    arrival_settle_s: float = 0.4

    # Speed must stay above the host filter's course-over-ground gate, else the heading
    # never anchors.
    align_speed: float = 0.12
    align_max_dist: float = 0.8

    pick_approach_dist: float = 0.6

    pivot_omega: float = 0.5               # rad/s; see PathCmd.pivot_first
    pivot_min_omega: float = 0.1
    pivot_gain: float = 1.0
    pivot_tol: float = 0.05
    pivot_pos_tol: float = 0.10
    pivot_timeout_s: float = 30.0
    # Heading must follow the rotation (monitor.PivotMonitor), else STOPPED STUCK:
    pivot_min_turn_ratio: float = 0.25
    pivot_check_rad: float = 0.8
    pivot_slack_rad: float = 0.5

    face_align: bool = False               # square up to the fixture face (core.face_align)
    face_align_tol_rad: float = 0.026
    face_align_max_rad: float = 0.6
    face_align_window: float = 0.20
    face_align_range: Tuple[float, ...] = (0.15, 0.90)   # [near, far] [m]
    face_align_min_points: int = 8
    face_align_timeout_s: float = 10.0
    face_align_memory_s: float = 15.0

    cmd_vel_topic: str = ''
    safety_state_topic: str = ''
    approach_topic: str = ''
    lead_gap_topic: str = ''
    imu_topic: str = '/imu'
    depth_scan_topic: str = ''


@dataclass
class GateConfig:
    """Parameters of the safety gate (``SafetyGate``)."""

    robot: str = 'follower'
    gate_rate_hz: float = 50.0
    state_rate_hz: float = 5.0
    alive_rate_hz: float = 10.0

    angular_sign: float = 1.0              # +1 / -1, final output only
    max_linear_speed: float = 0.20
    max_reverse_speed: float = 0.0
    max_angular_speed: float = 1.0

    input_timeout_s: float = 0.5
    pose_timeout_s: float = 0.5
    cmd_timeout_s: float = 1.5

    align_pass_speed: float = 0.13
    align_pass_omega: float = 0.02

    lidar_enabled: bool = True
    scan_timeout_s: float = 0.5
    lidar_stop_base: float = 0.12
    lidar_to_front: float = 0.08
    lidar_brake_decel: float = 0.50
    lidar_reaction_s: float = 0.20
    lidar_resume_s: float = 1.0
    lidar_front_half_angle: float = 0.60
    lidar_half_width: float = 0.15
    lidar_min_range: float = 0.05
    lidar_yaw_offset: float = -0.5236      # scan angle 0 vs body +x [rad]
    lidar_ignore_sectors: Tuple[float, ...] = (0.0, 0.0)  # body-frame [from, to] pairs [rad]
    lidar_min_valid_beams: int = 5
    lidar_blind_speed: float = 0.10
    lidar_exclude_lead: bool = True
    lidar_lead_radius: float = 0.30
    lidar_lead_min_points: int = 10
    gap_hard_min: float = 0.08
    lead_timeout_s: float = 0.5
    front_lost_stops: bool = True
    front_lost_near_m: float = 0.35
    front_memory: bool = False
    front_path: bool = False
    heading_free_rad: float = 0.35
    projection_window: float = 0.5
    init_tie_margin: float = 0.05
    depth_col_min: float = 0.30
    depth_col_max: float = 0.70
    depth_hfov: float = 1.019
    front_pick_exempt: bool = True
    approach_timeout_s: float = 1.0

    input_topic: str = ''
    output_topic: str = '/cmd_vel'
    estop_topic: str = '/estop'
    scan_topic: str = '/scan'
    front_distance_topic: str = ''
    depth_scan_topic: str = ''
    safety_state_topic: str = ''
    alive_topic: str = ''
    approach_topic: str = ''
    lead_gap_topic: str = ''


@dataclass
class WatchdogConfig:
    """Parameters of the command watchdog (``cmd_watchdog_node``)."""

    robot: str = 'follower'
    watchdog_timeout_s: float = 0.5
    watchdog_rate_hz: float = 10.0
    output_topic: str = '/cmd_vel'
    alive_topic: str = ''


@dataclass
class DepthFrontConfig:
    """Parameters of the depth front distance node (``depth_front_node``)."""

    robot: str = 'leader'
    depth_topic: str = '/camera/depth/image_raw'
    front_distance_topic: str = ''
    depth_rate_hz: float = 10.0
    depth_row_min: float = 0.35
    depth_row_max: float = 0.55
    depth_col_min: float = 0.30
    depth_col_max: float = 0.70
    depth_valid_min_m: float = 0.30
    depth_valid_max_m: float = 6.0
    depth_percentile: float = 5.0
    depth_min_valid_ratio: float = 0.30
    depth_scan_enabled: bool = True
    depth_scan_row_min: float = 0.30
    depth_scan_row_max: float = 0.48
    depth_scan_topic: str = ''
    depth_scan_rate_hz: float = 10.0
    depth_scan_frame: str = 'camera_depth_frame'   # x forward, y left
    depth_camera_info_topic: str = '/camera/depth/camera_info'
    depth_hfov: float = 1.019
    depth_scan_beams: int = 80
    depth_scan_min_valid_ratio: float = 0.20
