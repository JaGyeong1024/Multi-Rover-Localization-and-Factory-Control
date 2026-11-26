// mp_control_node: leader grasp sequence and handoff onto the follower.
// Target from /target/* (here the v6 ArUco observer) -> joint pregrasp -> EEF
// approach -> gripper close -> joint1 turn to the follower deck -> release ->
// Stay. Progress on /mp_control/status, cargo events on /cargo/events.
// The base only receives zero Twists (base_cmd_vel_topic) while it works.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include "mp_control/level_retraction.hpp"

#include <builtin_interfaces/msg/duration.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace mp_control
{
namespace
{
template<typename T>
T clampValue(T value, T min_value, T max_value)
{
  return std::max(min_value, std::min(value, max_value));
}

double vectorNorm(double x, double y, double z)
{
  return std::sqrt(x * x + y * y + z * z);
}

double normalizeAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}
}  // namespace

class MpControlNode final : public rclcpp::Node
{
#ifdef MP_CONTROL_TESTING
  friend struct ExtensionWristTestAccess;
#endif
public:
  using GripperCommand = control_msgs::action::GripperCommand;
  using GripperGoalHandle = rclcpp_action::ClientGoalHandle<GripperCommand>;
  using Trigger = std_srvs::srv::Trigger;

  explicit MpControlNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("mp_control_node", options),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    readParameters();

    const auto sensor_qos = rclcpp::SensorDataQoS();
    const auto default_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    const auto init_bbox_qos =
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    const auto status_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local();
    const auto current_id_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    bbox_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      bbox_topic_, default_qos,
      [this](const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg) { onBbox(msg); });
    if (use_fallback_bbox_for_control_ && !fallback_bbox_topic_.empty()) {
      init_bbox_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        fallback_bbox_topic_, init_bbox_qos,
        [this](const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg) { onBbox(msg); });
    }
    eef_bbox_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      eef_bbox_topic_, default_qos,
      [this](const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg) { onEefBbox(msg); });
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic_, sensor_qos,
      [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) { onDepth(msg); });
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, sensor_qos,
      [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) { onCameraInfo(msg); });
    eef_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      eef_camera_info_topic_, sensor_qos,
      [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) { onEefCameraInfo(msg); });
    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic_, default_qos,
      [this](const sensor_msgs::msg::JointState::ConstSharedPtr msg) { onJointState(msg); });
    aruco_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      aruco_pose_topic_, default_qos,
      [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) { onArucoPose(msg); });
    start_sub_ = create_subscription<std_msgs::msg::Bool>(
      start_topic_, default_qos,
      [this](const std_msgs::msg::Bool::ConstSharedPtr msg) {
        if (msg->data) {
          startSequence();
        }
      });
    resume_held_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/mp_control/resume_held_handoff", default_qos,
      [this](const std_msgs::msg::Bool::ConstSharedPtr msg) {
        if (msg->data) startSequence(true);
      });
    cancel_sub_ = create_subscription<std_msgs::msg::Bool>(
      cancel_topic_, default_qos,
      [this](const std_msgs::msg::Bool::ConstSharedPtr msg) {
        if (msg->data) {
          cancelSequence("cancel requested");
        }
      });
    continue_grasp_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/mp_control/continue_grasp", default_qos,
      [this](std_msgs::msg::Bool::ConstSharedPtr msg) {
        if (msg->data && active_ && joint_pregrasp_done_) grasp_resume_requested_ = true;
      });
    continue_handoff_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/mp_control/continue_handoff", default_qos,
      [this](std_msgs::msg::Bool::ConstSharedPtr msg) {
        if (msg->data && active_ && isHandoffStage(stage_)) handoff_resume_requested_ = true;
      });
    close_range_ready_sub_ = create_subscription<std_msgs::msg::Bool>(
      close_range_ready_topic_, init_bbox_qos,
      [this](const std_msgs::msg::Bool::ConstSharedPtr msg) { onCloseRangeReady(msg); });
    if (!stable_object_topic_.empty()) {
      stable_object_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
        stable_object_topic_, default_qos,
        [this](const geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {
          onStableObjectPoint(msg);
        });
    }

    twist_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(twist_topic_, default_qos);
    base_cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(base_cmd_vel_topic_, default_qos);
    joint_trajectory_pub_ =
      create_publisher<trajectory_msgs::msg::JointTrajectory>(
      joint_trajectory_topic_, default_qos);
    handoff_joint_trajectory_pub_ =
      create_publisher<trajectory_msgs::msg::JointTrajectory>(
      handoff_joint_trajectory_topic_, default_qos);
    status_pub_ = create_publisher<std_msgs::msg::String>(status_topic_, status_qos);
    cargo_event_pub_ = create_publisher<std_msgs::msg::String>(cargo_event_topic_, default_qos);
    cargo_current_id_pub_ =
      create_publisher<std_msgs::msg::String>(cargo_current_id_topic_, current_id_qos);
    eef_init_bbox_pub_ =
      create_publisher<std_msgs::msg::Float32MultiArray>(eef_init_bbox_topic_, init_bbox_qos);
    eef_auto_init_enable_pub_ =
      create_publisher<std_msgs::msg::Bool>(eef_auto_init_enable_topic_, init_bbox_qos);
    base_hold_pub_ =
      create_publisher<std_msgs::msg::Bool>(base_hold_topic_, init_bbox_qos);
    gripper_client_ = rclcpp_action::create_client<GripperCommand>(this, gripper_action_name_);
    servo_start_client_ = create_client<Trigger>("/servo_node/start_servo");
    servo_stop_client_ = create_client<Trigger>("/servo_node/stop_servo");

    if (auto_start_) {
      startSequence();
    }

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, command_rate_hz_));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(period),
      [this]() { update(); });

    RCLCPP_INFO(
      get_logger(),
      "mp_control started: bbox=%s depth=%s camera_info=%s eef_bbox=%s eef_camera_info=%s twist=%s joint_trajectory=%s handoff_joint_trajectory=%s target_frame=%s eef_frame=%s auto_start_on_bbox=%s eef_refinement=%s",
      bbox_topic_.c_str(), depth_topic_.c_str(), camera_info_topic_.c_str(),
      eef_bbox_topic_.c_str(), eef_camera_info_topic_.c_str(), twist_topic_.c_str(),
      joint_trajectory_topic_.c_str(), handoff_joint_trajectory_topic_.c_str(),
      target_frame_.c_str(), end_effector_frame_.c_str(),
      auto_start_on_bbox_ ? "true" : "false",
      use_eef_refinement_ ? "true" : "false");

    publishStatus("ready; publish std_msgs/Bool true to " + start_topic_ + " to start grasping", true);
  }

private:
  enum class GraspStage
  {
    DEPTH_APPROACH,
    TRIANGULATION_EXTEND,
    EEF_REFINE,
    HANDOFF_RETRACT,
    HANDOFF_ROTATE,
    HANDOFF_PLACE,
    HANDOFF_RELEASE,
    HANDOFF_STAY
  };

  struct Bbox
  {
    double x{0.0};
    double y{0.0};
    double width{0.0};
    double height{0.0};
    rclcpp::Time stamp;
  };

  struct CameraInfo
  {
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::string frame_id;
  };

  struct GripperTarget
  {
    double position{0.0};
    double object_width_m{0.0};
    double target_gap_m{0.0};
    bool measured{false};
  };

  struct Ray
  {
    tf2::Vector3 origin;
    tf2::Vector3 direction;
  };

  struct RpyError
  {
    double roll{0.0};
    double pitch{0.0};
    double yaw{0.0};
    double norm{0.0};
    bool ready{true};
    bool using_stay_roll{false};
  };

  struct PoseStatus
  {
    std::string frame;
    bool valid{false};
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double roll{0.0};
    double pitch{0.0};
    double yaw{0.0};
    std::string reason;
  };

  struct VisualGraspState
  {
    bool front_fresh{false};
    bool eef_fresh{false};
    double front_age_s{std::numeric_limits<double>::infinity()};
    double eef_age_s{std::numeric_limits<double>::infinity()};
  };

  struct EefForwardJointProgress
  {
    bool available{false};
    bool joint2_complete{false};
    bool joint3_complete{false};
    bool joint4_complete{true};
    bool complete{false};
    double joint2_progress_rad{0.0};
    double joint2_required_rad{0.0};
    double joint3_progress_rad{0.0};
    double joint3_required_rad{0.0};
    double joint4_error_rad{0.0};
    double joint4_target_rad{0.0};
  };

  struct PlanarIkPose
  {
    double x{0.0};
    double z{0.0};
    double pitch{0.0};
  };

  struct PlanarIkResult
  {
    bool success{false};
    std::array<double, 4> controller_target{};
    PlanarIkPose target_pose;
    PlanarIkPose solved_pose;
    double error_norm{std::numeric_limits<double>::infinity()};
    int iterations{0};
    std::string reason;
  };

  struct DepthStats
  {
    double depth_m{0.0};
    double fill_ratio{0.0};
    double std_m{0.0};
    int sample_count{0};
    int total_count{0};
    std::string source;
  };

  void readParameters()
  {
    bbox_topic_ = declare_parameter<std::string>("bbox_topic", "/target/tracked_bbox");
    fallback_bbox_topic_ = declare_parameter<std::string>("fallback_bbox_topic", "/target/init_bbox");
    eef_bbox_topic_ = declare_parameter<std::string>("eef_bbox_topic", "/target/eef_tracked_bbox");
    eef_init_bbox_topic_ = declare_parameter<std::string>("eef_init_bbox_topic", "/target/eef_init_bbox");
    depth_topic_ = declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    camera_info_topic_ = declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");
    eef_camera_info_topic_ = declare_parameter<std::string>("eef_camera_info_topic", "/eef_camera/camera_info");
    eef_auto_init_enable_topic_ = declare_parameter<std::string>("eef_auto_init_enable_topic", "/target/eef_auto_init_enable");
	    base_hold_topic_ = declare_parameter<std::string>("base_hold_topic", "/target/base_hold");
    close_range_ready_topic_ = declare_parameter<std::string>("close_range_ready_topic", "/target/close_range_ready");
    stable_object_topic_ = declare_parameter<std::string>("stable_object_topic", "/target/object_in_base");
	    twist_topic_ = declare_parameter<std::string>("twist_topic", "/servo_node/delta_twist_cmds");
	    base_cmd_vel_topic_ = declare_parameter<std::string>("base_cmd_vel_topic", "/cmd_vel");
	    joint_state_topic_ = declare_parameter<std::string>("joint_state_topic", "/joint_states");
	    joint_trajectory_topic_ = declare_parameter<std::string>("joint_trajectory_topic", "/arm_controller/joint_trajectory");
    handoff_joint_trajectory_topic_ = declare_parameter<std::string>("handoff_joint_trajectory_topic", "/arm_controller/joint_trajectory");
	    start_topic_ = declare_parameter<std::string>("start_topic", "/mp_control/start");
    cancel_topic_ = declare_parameter<std::string>("cancel_topic", "/mp_control/cancel");
    status_topic_ = declare_parameter<std::string>("status_topic", "/mp_control/status");
    cargo_event_topic_ = declare_parameter<std::string>("cargo_event_topic", "/cargo/events");
    cargo_current_id_topic_ = declare_parameter<std::string>("cargo_current_id_topic", "/cargo/current_id");
    cargo_id_prefix_ = declare_parameter<std::string>("cargo_id_prefix", "PKG");
    cargo_sequence_next_ = declare_parameter<int>("cargo_sequence_start", 1);
    gripper_action_name_ = declare_parameter<std::string>("gripper_action_name", "/gripper_controller/gripper_cmd");
    target_frame_ = declare_parameter<std::string>("target_frame", "base_link");
    end_effector_frame_ = declare_parameter<std::string>("end_effector_frame", "end_effector_link");
    joint4_pose_frame_ = declare_parameter<std::string>("joint4_pose_frame", "link5");
    gripper_pose_frame_ = declare_parameter<std::string>("gripper_pose_frame", end_effector_frame_);
    camera_frame_override_ = declare_parameter<std::string>("camera_frame_override", "");
    eef_camera_frame_override_ = declare_parameter<std::string>("eef_camera_frame_override", "eef_usb_camera_optical_frame");
    allow_eef_camera_info_fallback_ = declare_parameter<bool>("allow_eef_camera_info_fallback", true);
    eef_camera_fallback_width_px_ = declare_parameter<int>("eef_camera_fallback_width_px", 640);
    eef_camera_fallback_height_px_ = declare_parameter<int>("eef_camera_fallback_height_px", 480);
    eef_camera_fallback_fx_ = declare_parameter<double>("eef_camera_fallback_fx", 554.0);
    eef_camera_fallback_fy_ = declare_parameter<double>("eef_camera_fallback_fy", 554.0);
    aruco_pose_topic_ = declare_parameter<std::string>("aruco_pose_topic", "/target/aruco_pose");
    aruco_center_align_enabled_ = declare_parameter<bool>("aruco_center_align_enabled", true);
    aruco_center_pose_timeout_s_ = declare_parameter<double>("aruco_center_pose_timeout_s", 0.5);
    aruco_gripper_center_x_offset_m_ = declare_parameter<double>("aruco_gripper_center_x_offset_m", 0.0);
    aruco_gripper_center_y_offset_m_ = declare_parameter<double>("aruco_gripper_center_y_offset_m", 0.0);
    aruco_center_x_tolerance_m_ = declare_parameter<double>("aruco_center_x_tolerance_m", 0.012);
    aruco_center_y_tolerance_m_ = declare_parameter<double>("aruco_center_y_tolerance_m", 0.015);
    aruco_require_center_before_grasp_ = declare_parameter<bool>("aruco_require_center_before_grasp", true);
    aruco_joint1_align_gain_ = declare_parameter<double>("aruco_joint1_align_gain", 0.6);
    aruco_joint1_align_max_step_rad_ = declare_parameter<double>("aruco_joint1_align_max_step_rad", 0.020);
    aruco_joint1_align_sign_ = declare_parameter<double>("aruco_joint1_align_sign", -1.0);
    aruco_center_lateral_gain_ = declare_parameter<double>("aruco_center_lateral_gain", 0.6);
    aruco_center_max_linear_speed_ = declare_parameter<double>("aruco_center_max_linear_speed", 0.010);
    release_base_hold_after_handoff_ = declare_parameter<bool>("release_base_hold_after_handoff", true);
    auto_start_ = declare_parameter<bool>("auto_start", false);
    auto_start_on_bbox_ = declare_parameter<bool>("auto_start_on_bbox", false);
    use_fallback_bbox_for_control_ = declare_parameter<bool>("use_fallback_bbox_for_control", false);
    start_servo_on_start_ = declare_parameter<bool>("start_servo_on_start", true);
    open_gripper_on_start_ = declare_parameter<bool>("open_gripper_on_start", true);
    close_gripper_on_arrival_ = declare_parameter<bool>("close_gripper_on_arrival", true);
    pause_before_grasp_ = declare_parameter<bool>("pause_before_grasp", false);
    pause_before_rotate_ = declare_parameter<bool>("pause_before_rotate", false);
    require_visual_grasp_confirmation_ = declare_parameter<bool>("require_visual_grasp_confirmation", true);
    use_eef_refinement_ = declare_parameter<bool>("use_eef_refinement", true);
    wait_for_base_approach_ = declare_parameter<bool>("wait_for_base_approach", false);
    auto_init_eef_tracker_from_object_ = declare_parameter<bool>("auto_init_eef_tracker_from_object", true);
    use_depthless_triangulation_ = declare_parameter<bool>("use_depthless_triangulation", false);
	    use_color_triangulation_after_min_depth_ = declare_parameter<bool>("use_color_triangulation_after_min_depth", false);
    require_close_range_ready_for_pregrasp_ = declare_parameter<bool>("require_close_range_ready_for_pregrasp", true);
	    use_joint_pregrasp_ = declare_parameter<bool>("use_joint_pregrasp", true);
	    command_rate_hz_ = declare_parameter<double>("command_rate_hz", 20.0);
    max_target_age_s_ = declare_parameter<double>("max_target_age_s", 0.6);
    close_range_ready_max_age_s_ = declare_parameter<double>("close_range_ready_max_age_s", max_target_age_s_);
    linear_gain_ = declare_parameter<double>("linear_gain", 0.9);
    max_linear_speed_ = declare_parameter<double>("max_linear_speed", 0.025);
    position_tolerance_m_ = declare_parameter<double>("position_tolerance_m", 0.035);
    close_after_stable_cycles_ = declare_parameter<int>("close_after_stable_cycles", 8);
    grasp_completion_front_max_age_s_ = declare_parameter<double>("grasp_completion_front_max_age_s", max_target_age_s_);
    grasp_completion_eef_lost_timeout_s_ = declare_parameter<double>("grasp_completion_eef_lost_timeout_s", max_target_age_s_);
    depth_roi_radius_px_ = declare_parameter<int>("depth_roi_radius_px", 5);
    depth_unit_scale_ = declare_parameter<double>("depth_unit_scale", 0.001);
    min_valid_depth_m_ = declare_parameter<double>("min_valid_depth_m", 0.12);
    max_valid_depth_m_ = declare_parameter<double>("max_valid_depth_m", 1.2);
    depth_std_max_m_ = declare_parameter<double>("depth_std_max_m", 0.08);
    depth_min_fill_ratio_ = declare_parameter<double>("depth_min_fill_ratio", 0.25);
    depth_jump_limit_m_ = declare_parameter<double>("depth_jump_limit_m", 0.10);
    stable_depth_frames_ = declare_parameter<int>("stable_depth_frames", 3);
    min_depth_samples_ = declare_parameter<int>("min_depth_samples", 30);
    grasp_offset_x_ = declare_parameter<double>("grasp_offset_x", 0.0);
    grasp_offset_y_ = declare_parameter<double>("grasp_offset_y", 0.0);
    grasp_offset_z_ = declare_parameter<double>("grasp_offset_z", 0.0);
    eef_refinement_switch_distance_m_ = declare_parameter<double>("eef_refinement_switch_distance_m", 0.12);
    eef_refinement_start_depth_m_ = declare_parameter<double>("eef_refinement_start_depth_m", min_valid_depth_m_);
    eef_yolo_pre_enable_depth_m_ = declare_parameter<double>("eef_yolo_pre_enable_depth_m", eef_refinement_start_depth_m_);
    min_depth_handoff_margin_m_ = declare_parameter<double>("min_depth_handoff_margin_m", 0.02);
    min_depth_handoff_bbox_area_ratio_ = declare_parameter<double>("min_depth_handoff_bbox_area_ratio", 0.30);
    min_depth_handoff_bbox_height_ratio_ = declare_parameter<double>("min_depth_handoff_bbox_height_ratio", 0.75);
    eef_refinement_start_object_x_m_ = declare_parameter<double>("eef_refinement_start_object_x_m", 0.50);
    arm_start_max_error_m_ = declare_parameter<double>("arm_start_max_error_m", 0.40);
    arm_start_max_object_x_m_ = declare_parameter<double>("arm_start_max_object_x_m", 0.60);
    object_height_m_ = declare_parameter<double>("object_height_m", 0.10);
    eef_init_bbox_min_size_px_ = declare_parameter<double>("eef_init_bbox_min_size_px", 32.0);
    eef_init_bbox_max_size_px_ = declare_parameter<double>("eef_init_bbox_max_size_px", 180.0);
    eef_init_bbox_padding_scale_ = declare_parameter<double>("eef_init_bbox_padding_scale", 1.6);
    eef_init_bbox_republish_period_s_ = declare_parameter<double>("eef_init_bbox_republish_period_s", 0.5);
    eef_final_depth_m_ = declare_parameter<double>("eef_final_depth_m", 0.08);
    object_pregrasp_standoff_m_ = declare_parameter<double>("object_pregrasp_standoff_m", 0.08);
    object_pregrasp_min_z_m_ = declare_parameter<double>("object_pregrasp_min_z_m", 0.50);
    object_pregrasp_lower_standoff_m_ = declare_parameter<double>("object_pregrasp_lower_standoff_m", 0.02);
    object_pregrasp_min_lower_z_m_ = declare_parameter<double>("object_pregrasp_min_lower_z_m", 0.12);
	    object_pregrasp_enable_lowering_ = declare_parameter<bool>("object_pregrasp_enable_lowering", false);
	    joint_pregrasp_ready_positions_ = declare_parameter<std::vector<double>>("pregrasp_ready_joint_positions", std::vector<double>{0.0, 0.65, -0.85, -1.20});
	    joint_pregrasp_preserve_gripper_roll_ = declare_parameter<bool>("pregrasp_preserve_gripper_roll", true);
    hold_joint4_during_extension_ = declare_parameter<bool>("hold_joint4_during_extension", false);
	    pregrasp_roll_joint2_weight_ = declare_parameter<double>("pregrasp_roll_joint2_weight", 1.0);
	    pregrasp_roll_joint3_weight_ = declare_parameter<double>("pregrasp_roll_joint3_weight", 1.0);
	    pregrasp_roll_joint4_weight_ = declare_parameter<double>("pregrasp_roll_joint4_weight", 1.0);
	    joint_pregrasp_reverse_joint3_delta_ = declare_parameter<bool>("pregrasp_reverse_joint3_delta", true);
	    joint_pregrasp_hold_current_duration_s_ = declare_parameter<double>("pregrasp_hold_current_duration_s", 0.0);
	    joint_pregrasp_move_duration_s_ = declare_parameter<double>("pregrasp_move_duration_s", 1.2);
	    joint_pregrasp_joint3_lead_enabled_ = declare_parameter<bool>("pregrasp_joint3_lead_enabled", false);
	    joint_pregrasp_joint3_lead_duration_ratio_ = declare_parameter<double>("pregrasp_joint3_lead_duration_ratio", 0.45);
	    joint_pregrasp_joint3_lead_fraction_ = declare_parameter<double>("pregrasp_joint3_lead_fraction", 1.0);
	    joint_pregrasp_sync_steps_ = declare_parameter<int>("pregrasp_sync_steps", 1);
	    joint_pregrasp_settle_s_ = declare_parameter<double>("pregrasp_settle_s", 0.5);
	    joint_pregrasp_tolerance_rad_ = declare_parameter<double>("pregrasp_joint_tolerance_rad", 0.04);
	    joint_pregrasp_republish_period_s_ = declare_parameter<double>("pregrasp_republish_period_s", 1.0);
      joint_pregrasp_joint2_max_step_rad_ = declare_parameter<double>("pregrasp_joint2_max_step_rad", 0.05);
	    joint_pregrasp_joint3_max_step_rad_ = declare_parameter<double>("pregrasp_joint3_max_step_rad", 0.06);
      joint_pregrasp_joint4_max_step_rad_ = declare_parameter<double>("pregrasp_joint4_max_step_rad", 0.04);
	    joint_pregrasp_min_positions_ = declare_parameter<std::vector<double>>("pregrasp_joint_min_positions", std::vector<double>{-3.14, -1.79, -0.94, -1.79});
	    joint_pregrasp_max_positions_ = declare_parameter<std::vector<double>>("pregrasp_joint_max_positions", std::vector<double>{3.14, 1.57, 1.38, 2.04});
	    eef_center_tolerance_px_ = declare_parameter<double>("eef_center_tolerance_px", 18.0);
    eef_close_tolerance_px_ = declare_parameter<double>("eef_close_tolerance_px", eef_center_tolerance_px_);
    eef_depth_tolerance_m_ = declare_parameter<double>("eef_depth_tolerance_m", 0.018);
    eef_refine_lateral_gain_ = declare_parameter<double>("eef_refine_lateral_gain", 0.8);
    eef_refine_depth_gain_ = declare_parameter<double>("eef_refine_depth_gain", 0.5);
    eef_refine_max_linear_speed_ = declare_parameter<double>("eef_refine_max_linear_speed", 0.012);
    use_eef_rpy_refinement_ = declare_parameter<bool>("use_eef_rpy_refinement", true);
    eef_hold_current_rpy_ = declare_parameter<bool>("eef_hold_current_rpy", true);
    eef_hold_stay_roll_ = declare_parameter<bool>("eef_hold_stay_roll", false);
    eef_target_roll_rad_ = declare_parameter<double>("eef_target_roll_rad", 0.0);
    eef_target_pitch_rad_ = declare_parameter<double>("eef_target_pitch_rad", 0.0);
    eef_target_yaw_rad_ = declare_parameter<double>("eef_target_yaw_rad", 0.0);
    eef_rpy_tolerance_rad_ = declare_parameter<double>("eef_rpy_tolerance_rad", 0.12);
    eef_rpy_gain_ = declare_parameter<double>("eef_rpy_gain", 0.8);
    eef_refine_max_angular_speed_ = declare_parameter<double>("eef_refine_max_angular_speed", 0.25);
    eef_forward_after_align_ = declare_parameter<bool>("eef_forward_after_align", true);
    eef_forward_distance_m_ = declare_parameter<double>("eef_forward_distance_m", 0.05);
    eef_forward_speed_mps_ = declare_parameter<double>("eef_forward_speed_mps", 0.012);
    eef_forward_fixed_duration_s_ = declare_parameter<double>("eef_forward_fixed_duration_s", 0.0);
    eef_forward_gate_timeout_s_ = declare_parameter<double>("eef_forward_gate_timeout_s", 8.0);
    eef_forward_start_tolerance_px_ = declare_parameter<double>("eef_forward_start_tolerance_px", eef_close_tolerance_px_);
    eef_forward_use_joint_nudge_ = declare_parameter<bool>("eef_forward_use_joint_nudge", false);
    eef_forward_joint2_delta_rad_ = declare_parameter<double>("eef_forward_joint2_delta_rad", 0.025);
    eef_forward_joint3_delta_rad_ = declare_parameter<double>("eef_forward_joint3_delta_rad", -0.035);
    eef_forward_joint_nudge_duration_s_ = declare_parameter<double>("eef_forward_joint_nudge_duration_s", 0.75);
    eef_forward_joint_nudge_period_s_ = declare_parameter<double>("eef_forward_joint_nudge_period_s", 0.80);
    eef_forward_joint3_first_duration_ratio_ = declare_parameter<double>("eef_forward_joint3_first_duration_ratio", 0.65);
    eef_forward_joint4_after_joint3_complete_ = declare_parameter<bool>("eef_forward_joint4_after_joint3_complete", true);
    eef_forward_joint3_complete_delta_rad_ = declare_parameter<double>("eef_forward_joint3_complete_delta_rad", 0.14);
    eef_forward_joint3_complete_tolerance_rad_ = declare_parameter<double>("eef_forward_joint3_complete_tolerance_rad", 0.015);
    eef_forward_joint4_finish_tolerance_rad_ = declare_parameter<double>("eef_forward_joint4_finish_tolerance_rad", 0.015);
    eef_forward_use_bezier_ik_ = declare_parameter<bool>("eef_forward_use_bezier_ik", true);
    eef_forward_use_analytic_ik_ = declare_parameter<bool>("eef_forward_use_analytic_ik", true);
    eef_forward_ik_link2_x_m_ = declare_parameter<double>("eef_forward_ik_link2_x_m", 0.024);
    eef_forward_ik_link2_z_m_ = declare_parameter<double>("eef_forward_ik_link2_z_m", 0.128);
    eef_forward_ik_link3_m_ = declare_parameter<double>("eef_forward_ik_link3_m", 0.124);
    eef_forward_ik_tool_m_ = declare_parameter<double>("eef_forward_ik_tool_m", 0.126);
    eef_forward_ik_pitch_weight_ = declare_parameter<double>("eef_forward_ik_pitch_weight", 0.35);
    eef_forward_ik_damping_ = declare_parameter<double>("eef_forward_ik_damping", 0.03);
    eef_forward_ik_iterations_ = declare_parameter<int>("eef_forward_ik_iterations", 16);
    eef_forward_bezier_total_duration_s_ = declare_parameter<double>("eef_forward_bezier_total_duration_s", 5.0);
    eef_forward_bezier_progress_step_ = declare_parameter<double>("eef_forward_bezier_progress_step", 0.08);
    eef_forward_bezier_joint2_total_delta_rad_ = declare_parameter<double>("eef_forward_bezier_joint2_total_delta_rad", std::abs(eef_forward_joint2_delta_rad_) * 5.0);
    eef_forward_bezier_joint3_total_delta_rad_ = declare_parameter<double>("eef_forward_bezier_joint3_total_delta_rad", eef_forward_joint3_complete_delta_rad_);
    eef_forward_bezier_joint2_max_step_rad_ = declare_parameter<double>("eef_forward_bezier_joint2_max_step_rad", std::max(0.005, std::abs(eef_forward_joint2_delta_rad_)));
    eef_forward_bezier_joint3_max_step_rad_ = declare_parameter<double>("eef_forward_bezier_joint3_max_step_rad", std::max(0.005, std::min(0.020, std::abs(eef_forward_joint3_delta_rad_))));
    eef_forward_bezier_joint4_max_step_rad_ = declare_parameter<double>("eef_forward_bezier_joint4_max_step_rad", std::max(0.0, eef_forward_joint4_max_delta_rad_));
    eef_forward_bezier_preserve_roll_ = declare_parameter<bool>("eef_forward_bezier_preserve_roll", true);
    eef_forward_roll_joint2_weight_ = declare_parameter<double>("eef_forward_roll_joint2_weight", 0.0);
    eef_forward_roll_joint3_weight_ = declare_parameter<double>("eef_forward_roll_joint3_weight", 1.0);
    eef_forward_roll_joint4_weight_ = declare_parameter<double>("eef_forward_roll_joint4_weight", 1.0);
    eef_forward_joint4_rpy_roll_gain_ = declare_parameter<double>("eef_forward_joint4_rpy_roll_gain", 0.0);
    eef_forward_joint4_rpy_roll_max_delta_rad_ = declare_parameter<double>("eef_forward_joint4_rpy_roll_max_delta_rad", 0.0);
    eef_forward_joint4_max_delta_rad_ = declare_parameter<double>("eef_forward_joint4_max_delta_rad", 0.0);
    eef_forward_joint4_ground_parallel_limit_rad_ = declare_parameter<double>("eef_forward_joint4_ground_parallel_limit_rad", -1.05);
    eef_forward_joint4_ground_limit_tolerance_rad_ = declare_parameter<double>("eef_forward_joint4_ground_limit_tolerance_rad", 0.01);
    eef_forward_joint4_down_positive_ = declare_parameter<bool>("eef_forward_joint4_down_positive", true);
    gripper_down_joint4_offset_rad_ = declare_parameter<double>("gripper_down_joint4_offset_rad", 0.0);
    close_on_front_bbox_shrink_ = declare_parameter<bool>("close_on_front_bbox_shrink", false);
    front_bbox_close_area_ratio_ = declare_parameter<double>("front_bbox_close_area_ratio", 0.60);
    close_on_eef_bbox_shrink_ = declare_parameter<bool>("close_on_eef_bbox_shrink", false);
    eef_bbox_close_area_ratio_ = declare_parameter<double>("eef_bbox_close_area_ratio", 0.60);
    eef_forward_min_advance_before_close_m_ = declare_parameter<double>("eef_forward_min_advance_before_close_m", 0.025);
    close_after_full_eef_forward_extension_ = declare_parameter<bool>("close_after_full_eef_forward_extension", false);
    handoff_after_grasp_ = declare_parameter<bool>("handoff_after_grasp", false);
    handoff_grasp_settle_s_ = declare_parameter<double>("handoff_grasp_settle_s", 0.5);
    handoff_lift_joint2_delta_rad_ = declare_parameter<double>("handoff_lift_joint2_delta_rad", 0.25);
    handoff_retract_joint3_delta_rad_ = declare_parameter<double>("handoff_retract_joint3_delta_rad", 0.0);
    handoff_retract_hold_z_ = declare_parameter<bool>("handoff_retract_hold_z", false);
    handoff_restore_grasp_pose_ = declare_parameter<bool>("handoff_restore_grasp_pose", false);
    handoff_lift_joint_tolerance_rad_ = declare_parameter<double>("handoff_lift_joint_tolerance_rad", 0.10);
    if (!std::isfinite(handoff_lift_joint_tolerance_rad_) || handoff_lift_joint_tolerance_rad_ <= 0.0 ||
        handoff_lift_joint_tolerance_rad_ > 0.15)
      throw std::invalid_argument("Invalid lift settling tolerance");
    handoff_retract_distance_m_ = declare_parameter<double>("handoff_retract_distance_m", 0.05);
    handoff_retract_z_tolerance_m_ = declare_parameter<double>("handoff_retract_z_tolerance_m", 0.005);
    handoff_lower_before_release_ = declare_parameter<bool>("handoff_lower_before_release", true);
    handoff_place_joint2_delta_rad_ = declare_parameter<double>("handoff_place_joint2_delta_rad", -0.20);
    handoff_joint_move_duration_s_ = declare_parameter<double>("handoff_joint_move_duration_s", 1.0);
    handoff_place_duration_s_ = declare_parameter<double>("handoff_place_duration_s", handoff_joint_move_duration_s_);
    handoff_joint_settle_s_ = declare_parameter<double>("handoff_joint_settle_s", 0.4);
    handoff_republish_period_s_ = declare_parameter<double>("handoff_republish_period_s", 0.25);
    handoff_rotate_angle_rad_ = declare_parameter<double>("handoff_rotate_angle_rad", 3.14159265358979323846);
    handoff_rotate_odd_angle_deg_ =
      declare_parameter<double>("handoff_rotate_odd_angle_deg", handoff_rotate_angle_rad_ * 180.0 / M_PI);
    handoff_rotate_even_angle_deg_ =
      declare_parameter<double>("handoff_rotate_even_angle_deg", handoff_rotate_angle_rad_ * 180.0 / M_PI);
    handoff_center_joint1_before_rotate_ = declare_parameter<bool>("handoff_center_joint1_before_rotate", true);
    handoff_joint1_center_target_rad_ = declare_parameter<double>("handoff_joint1_center_target_rad", 0.0);
    handoff_joint1_center_tolerance_rad_ = declare_parameter<double>("handoff_joint1_center_tolerance_rad", 0.03);
    handoff_rotate_angular_speed_rad_s_ = declare_parameter<double>("handoff_rotate_angular_speed_rad_s", 0.45);
    handoff_release_settle_s_ = declare_parameter<double>("handoff_release_settle_s", 0.5);
    handoff_lift_duration_s_ = declare_parameter<double>("handoff_lift_duration_s", 1.0);
    handoff_stay_joint_positions_ = declare_parameter<std::vector<double>>("handoff_stay_joint_positions", std::vector<double>{0.104311, 0.027612, -0.001534, -1.638291});
    triangulation_extend_x_m_ = declare_parameter<double>("triangulation_extend_x_m", 0.25);
    triangulation_extend_y_m_ = declare_parameter<double>("triangulation_extend_y_m", 0.0);
    triangulation_extend_z_m_ = declare_parameter<double>("triangulation_extend_z_m", 0.12);
    triangulation_extend_tolerance_m_ = declare_parameter<double>("triangulation_extend_tolerance_m", 0.025);
    triangulation_extend_gain_ = declare_parameter<double>("triangulation_extend_gain", 0.7);
    triangulation_extend_max_speed_ = declare_parameter<double>("triangulation_extend_max_speed", 0.015);
    triangulation_min_range_m_ = declare_parameter<double>("triangulation_min_range_m", 0.06);
    triangulation_max_range_m_ = declare_parameter<double>("triangulation_max_range_m", 1.0);
    triangulation_max_ray_gap_m_ = declare_parameter<double>("triangulation_max_ray_gap_m", 0.10);
    use_eef_front_camera_extrinsic_override_ = declare_parameter<bool>("use_eef_front_camera_extrinsic_override", false);
    eef_front_camera_offset_x_m_ = declare_parameter<double>("eef_front_camera_offset_x_m", 0.0);
    eef_front_camera_offset_y_m_ = declare_parameter<double>("eef_front_camera_offset_y_m", 0.0);
    eef_front_camera_offset_z_m_ = declare_parameter<double>("eef_front_camera_offset_z_m", 0.0);
    color_triangulation_base_stop_object_x_m_ = declare_parameter<double>("color_triangulation_base_stop_object_x_m", 0.30);
    color_triangulation_min_object_x_m_ = declare_parameter<double>("color_triangulation_min_object_x_m", 0.05);
    gripper_open_position_ = declare_parameter<double>("gripper_open_position", 0.025);
    gripper_close_position_ = declare_parameter<double>("gripper_close_position", -0.015);
    gripper_max_effort_ = declare_parameter<double>("gripper_max_effort", -1.0);
    gripper_width_control_enabled_ = declare_parameter<bool>("gripper_width_control_enabled", true);
    gripper_fallback_object_width_m_ = declare_parameter<double>("gripper_fallback_object_width_m", 0.06);
    gripper_finger_home_half_gap_m_ = declare_parameter<double>("gripper_finger_home_half_gap_m", 0.021);
    gripper_pre_grasp_clearance_m_ = declare_parameter<double>("gripper_pre_grasp_clearance_m", 0.012);
    gripper_grasp_compression_m_ = declare_parameter<double>("gripper_grasp_compression_m", 0.002);
    gripper_grasp_clearance_m_ = declare_parameter<double>("gripper_grasp_clearance_m", 0.0);
    gripper_grasp_width_scale_ = declare_parameter<double>("gripper_grasp_width_scale", 1.0);
    gripper_min_position_ = declare_parameter<double>("gripper_min_position", -0.010);
    gripper_max_position_ = declare_parameter<double>("gripper_max_position", 0.019);
    gripper_min_measured_object_width_m_ = declare_parameter<double>("gripper_min_measured_object_width_m", 0.01);
    gripper_max_measured_object_width_m_ = declare_parameter<double>("gripper_max_measured_object_width_m", 0.08);

    command_rate_hz_ = std::max(1.0, command_rate_hz_);
    close_range_ready_max_age_s_ = std::max(0.1, close_range_ready_max_age_s_);
    max_linear_speed_ = std::max(0.0, max_linear_speed_);
    position_tolerance_m_ = std::max(0.005, position_tolerance_m_);
    close_after_stable_cycles_ = std::max(1, close_after_stable_cycles_);
    grasp_completion_front_max_age_s_ =
      std::max(0.1, grasp_completion_front_max_age_s_);
    grasp_completion_eef_lost_timeout_s_ =
      std::max(0.1, grasp_completion_eef_lost_timeout_s_);
    depth_roi_radius_px_ = std::max(0, depth_roi_radius_px_);
    depth_std_max_m_ = std::max(0.0, depth_std_max_m_);
    depth_min_fill_ratio_ = clampValue(depth_min_fill_ratio_, 0.0, 1.0);
    depth_jump_limit_m_ = std::max(0.0, depth_jump_limit_m_);
    stable_depth_frames_ = std::max(1, stable_depth_frames_);
    min_depth_samples_ = std::max(1, min_depth_samples_);
    eef_refinement_switch_distance_m_ = std::max(0.01, eef_refinement_switch_distance_m_);
    eef_refinement_start_depth_m_ = std::max(min_valid_depth_m_, eef_refinement_start_depth_m_);
    eef_yolo_pre_enable_depth_m_ =
      std::max(eef_refinement_start_depth_m_, eef_yolo_pre_enable_depth_m_);
    min_depth_handoff_margin_m_ = std::max(0.0, min_depth_handoff_margin_m_);
    min_depth_handoff_bbox_area_ratio_ =
      clampValue(min_depth_handoff_bbox_area_ratio_, 0.0, 1.0);
    min_depth_handoff_bbox_height_ratio_ =
      clampValue(min_depth_handoff_bbox_height_ratio_, 0.0, 1.0);
    eef_refinement_start_object_x_m_ = std::max(0.01, eef_refinement_start_object_x_m_);
    arm_start_max_error_m_ = std::max(0.05, arm_start_max_error_m_);
    arm_start_max_object_x_m_ = std::max(0.05, arm_start_max_object_x_m_);
    object_height_m_ = std::max(0.01, object_height_m_);
    eef_init_bbox_min_size_px_ = std::max(2.0, eef_init_bbox_min_size_px_);
    eef_init_bbox_max_size_px_ = std::max(eef_init_bbox_min_size_px_, eef_init_bbox_max_size_px_);
    eef_init_bbox_padding_scale_ = std::max(1.0, eef_init_bbox_padding_scale_);
    eef_init_bbox_republish_period_s_ = std::max(0.1, eef_init_bbox_republish_period_s_);
    eef_final_depth_m_ = std::max(0.0, eef_final_depth_m_);
    object_pregrasp_standoff_m_ = std::max(0.0, object_pregrasp_standoff_m_);
    object_pregrasp_min_z_m_ = std::max(0.0, object_pregrasp_min_z_m_);
	    object_pregrasp_lower_standoff_m_ = std::max(0.0, object_pregrasp_lower_standoff_m_);
	    object_pregrasp_min_lower_z_m_ = std::max(0.0, object_pregrasp_min_lower_z_m_);
	    joint_pregrasp_ready_positions_ = normalizedJointVector(joint_pregrasp_ready_positions_, {0.0, 0.65, -0.85, -1.20});
	    joint_pregrasp_min_positions_ = normalizedJointVector(joint_pregrasp_min_positions_, {-3.14, -1.79, -0.94, -1.79});
	    joint_pregrasp_max_positions_ = normalizedJointVector(joint_pregrasp_max_positions_, {3.14, 1.57, 1.38, 2.04});
	    for (std::size_t i = 0; i < arm_joint_names_.size(); ++i) {
	      if (joint_pregrasp_min_positions_[i] > joint_pregrasp_max_positions_[i]) {
	        std::swap(joint_pregrasp_min_positions_[i], joint_pregrasp_max_positions_[i]);
	      }
	    }
	    joint_pregrasp_hold_current_duration_s_ = std::max(0.0, joint_pregrasp_hold_current_duration_s_);
	    joint_pregrasp_move_duration_s_ = std::max(0.2, joint_pregrasp_move_duration_s_);
	    joint_pregrasp_joint3_lead_duration_ratio_ = clampValue(joint_pregrasp_joint3_lead_duration_ratio_, 0.1, 0.8);
	    joint_pregrasp_joint3_lead_fraction_ = clampValue(joint_pregrasp_joint3_lead_fraction_, 0.1, 1.0);
	    joint_pregrasp_sync_steps_ = std::max(1, joint_pregrasp_sync_steps_);
	    joint_pregrasp_settle_s_ = std::max(0.0, joint_pregrasp_settle_s_);
	    joint_pregrasp_tolerance_rad_ = std::max(0.001, joint_pregrasp_tolerance_rad_);
	    joint_pregrasp_republish_period_s_ = std::max(0.2, joint_pregrasp_republish_period_s_);
	    joint_pregrasp_joint3_max_step_rad_ = std::max(0.0, joint_pregrasp_joint3_max_step_rad_);
	    eef_center_tolerance_px_ = std::max(1.0, eef_center_tolerance_px_);
    eef_close_tolerance_px_ = std::max(eef_center_tolerance_px_, eef_close_tolerance_px_);
    eef_depth_tolerance_m_ = std::max(0.001, eef_depth_tolerance_m_);
    eef_refine_max_linear_speed_ = std::max(0.0, eef_refine_max_linear_speed_);
    eef_rpy_tolerance_rad_ = std::max(0.001, eef_rpy_tolerance_rad_);
    eef_rpy_gain_ = std::max(0.0, eef_rpy_gain_);
    eef_refine_max_angular_speed_ = std::max(0.0, eef_refine_max_angular_speed_);
    eef_forward_distance_m_ = std::max(0.0, eef_forward_distance_m_);
    eef_forward_speed_mps_ = std::max(0.0, eef_forward_speed_mps_);
    eef_forward_fixed_duration_s_ = std::max(0.0, eef_forward_fixed_duration_s_);
    eef_forward_gate_timeout_s_ = std::max(0.0, eef_forward_gate_timeout_s_);
    eef_forward_start_tolerance_px_ = std::max(eef_close_tolerance_px_, eef_forward_start_tolerance_px_);
    eef_forward_joint_nudge_duration_s_ = std::max(0.1, eef_forward_joint_nudge_duration_s_);
    eef_forward_joint_nudge_period_s_ = std::max(0.05, eef_forward_joint_nudge_period_s_);
    eef_forward_joint3_first_duration_ratio_ = clampValue(eef_forward_joint3_first_duration_ratio_, 0.1, 0.9);
    eef_forward_joint3_complete_delta_rad_ = std::max(0.0, std::abs(eef_forward_joint3_complete_delta_rad_));
    eef_forward_joint3_complete_tolerance_rad_ = clampValue(eef_forward_joint3_complete_tolerance_rad_, 0.001, 0.10);
    eef_forward_joint4_finish_tolerance_rad_ = clampValue(eef_forward_joint4_finish_tolerance_rad_, 0.001, 0.10);
    if (std::abs(eef_forward_roll_joint4_weight_) < 1.0e-6) {
      eef_forward_roll_joint4_weight_ = 1.0;
    }
    if (std::abs(pregrasp_roll_joint4_weight_) < 1.0e-6) {
      pregrasp_roll_joint4_weight_ = 1.0;
    }
    eef_forward_ik_link2_x_m_ = std::max(0.0, eef_forward_ik_link2_x_m_);
    eef_forward_ik_link2_z_m_ = std::max(0.0, eef_forward_ik_link2_z_m_);
    eef_forward_ik_link3_m_ = std::max(0.01, eef_forward_ik_link3_m_);
    eef_forward_ik_tool_m_ = std::max(0.0, eef_forward_ik_tool_m_);
    eef_forward_ik_pitch_weight_ = clampValue(eef_forward_ik_pitch_weight_, 0.0, 2.0);
    eef_forward_ik_damping_ = clampValue(eef_forward_ik_damping_, 1.0e-4, 0.5);
    eef_forward_ik_iterations_ = std::max(1, eef_forward_ik_iterations_);
    eef_forward_joint4_rpy_roll_gain_ = std::max(0.0, eef_forward_joint4_rpy_roll_gain_);
    eef_forward_joint4_rpy_roll_max_delta_rad_ = std::max(0.0, eef_forward_joint4_rpy_roll_max_delta_rad_);
    eef_forward_joint4_max_delta_rad_ = std::max(0.0, eef_forward_joint4_max_delta_rad_);
    eef_forward_joint4_ground_parallel_limit_rad_ = clampValue(
      eef_forward_joint4_ground_parallel_limit_rad_,
      joint_pregrasp_min_positions_[3],
      joint_pregrasp_max_positions_[3]);
    eef_forward_joint4_ground_limit_tolerance_rad_ = clampValue(eef_forward_joint4_ground_limit_tolerance_rad_, 0.0, 0.10);
    front_bbox_close_area_ratio_ = clampValue(front_bbox_close_area_ratio_, 0.05, 1.0);
    eef_bbox_close_area_ratio_ = clampValue(eef_bbox_close_area_ratio_, 0.05, 1.0);
    eef_forward_min_advance_before_close_m_ = clampValue(eef_forward_min_advance_before_close_m_, 0.0, eef_forward_distance_m_);
    handoff_joint_move_duration_s_ = std::max(0.1, handoff_joint_move_duration_s_);
    handoff_place_duration_s_ = std::max(0.2, handoff_place_duration_s_);
    handoff_retract_joint3_delta_rad_ = clampValue(handoff_retract_joint3_delta_rad_, 0.0, 1.0);
    handoff_retract_z_tolerance_m_ = clampValue(handoff_retract_z_tolerance_m_, 0.001, 0.01);
    handoff_joint_settle_s_ = std::max(0.0, handoff_joint_settle_s_);
    handoff_grasp_settle_s_ = std::max(0.0, handoff_grasp_settle_s_);
    handoff_lift_duration_s_ = std::max(0.2, handoff_lift_duration_s_);
    handoff_joint1_center_target_rad_ = clampValue(
      handoff_joint1_center_target_rad_,
      joint_pregrasp_min_positions_[0],
      joint_pregrasp_max_positions_[0]);
    handoff_joint1_center_tolerance_rad_ = clampValue(handoff_joint1_center_tolerance_rad_, 0.001, 0.30);
    handoff_rotate_angular_speed_rad_s_ = std::max(0.05, std::abs(handoff_rotate_angular_speed_rad_s_));
    handoff_release_settle_s_ = std::max(0.0, handoff_release_settle_s_);
    handoff_stay_joint_positions_ = normalizedJointVector(handoff_stay_joint_positions_, {0.104311, 0.027612, -0.001534, -1.638291});
    triangulation_extend_tolerance_m_ = std::max(0.005, triangulation_extend_tolerance_m_);
    triangulation_extend_gain_ = std::max(0.0, triangulation_extend_gain_);
    triangulation_extend_max_speed_ = std::max(0.0, triangulation_extend_max_speed_);
    triangulation_min_range_m_ = std::max(0.01, triangulation_min_range_m_);
    triangulation_max_range_m_ = std::max(triangulation_min_range_m_, triangulation_max_range_m_);
    triangulation_max_ray_gap_m_ = std::max(0.005, triangulation_max_ray_gap_m_);
    if (!std::isfinite(eef_front_camera_offset_x_m_)) {
      eef_front_camera_offset_x_m_ = 0.0;
    }
    if (!std::isfinite(eef_front_camera_offset_y_m_)) {
      eef_front_camera_offset_y_m_ = 0.0;
    }
    if (!std::isfinite(eef_front_camera_offset_z_m_)) {
      eef_front_camera_offset_z_m_ = 0.0;
    }
    color_triangulation_base_stop_object_x_m_ = std::max(0.01, color_triangulation_base_stop_object_x_m_);
    color_triangulation_min_object_x_m_ = clampValue(color_triangulation_min_object_x_m_, 0.0, color_triangulation_base_stop_object_x_m_);
    cargo_sequence_next_ = std::max(1, cargo_sequence_next_);
    if (gripper_min_position_ > gripper_max_position_) {
      std::swap(gripper_min_position_, gripper_max_position_);
    }
    if (gripper_min_measured_object_width_m_ > gripper_max_measured_object_width_m_) {
      std::swap(gripper_min_measured_object_width_m_, gripper_max_measured_object_width_m_);
    }
    eef_camera_fallback_width_px_ = std::max(1, eef_camera_fallback_width_px_);
    eef_camera_fallback_height_px_ = std::max(1, eef_camera_fallback_height_px_);
    eef_camera_fallback_fx_ = std::max(1.0, eef_camera_fallback_fx_);
    eef_camera_fallback_fy_ = std::max(1.0, eef_camera_fallback_fy_);
    gripper_open_position_ = clampValue(gripper_open_position_, gripper_min_position_, gripper_max_position_);
    gripper_close_position_ = clampValue(gripper_close_position_, gripper_min_position_, gripper_max_position_);
    gripper_fallback_object_width_m_ = std::max(0.0, gripper_fallback_object_width_m_);
    gripper_finger_home_half_gap_m_ = std::max(0.0, gripper_finger_home_half_gap_m_);
    gripper_pre_grasp_clearance_m_ = std::max(0.0, gripper_pre_grasp_clearance_m_);
    gripper_grasp_compression_m_ = std::max(0.0, gripper_grasp_compression_m_);
    gripper_grasp_clearance_m_ = std::max(0.0, gripper_grasp_clearance_m_);
    gripper_grasp_width_scale_ = std::max(0.0, gripper_grasp_width_scale_);
  }

  void onBbox(const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg)
  {
    if (msg->data.size() < 4) {
      return;
    }
    const double width = msg->data[2];
    const double height = msg->data[3];
    if (width <= 1.0 || height <= 1.0) {
      return;
    }

    bool should_auto_start = false;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      latest_bbox_ = Bbox{
        static_cast<double>(msg->data[0]),
        static_cast<double>(msg->data[1]),
        width,
        height,
        now()};
      should_auto_start = auto_start_on_bbox_ && !active_ && !done_ && !safety_abort_latched_;
    }

    if (should_auto_start) {
      startSequence();
    }
  }

  void onDepth(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (use_color_triangulation_after_min_depth_ && min_depth_reached_) {
      return;
    }
    latest_depth_ = msg;
  }

  void onEefBbox(const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg)
  {
    if (msg->data.size() < 4) {
      return;
    }
    const double width = msg->data[2];
    const double height = msg->data[3];
    if (width <= 1.0 || height <= 1.0) {
      return;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_eef_bbox_ = Bbox{
      static_cast<double>(msg->data[0]),
      static_cast<double>(msg->data[1]),
      width,
      height,
      now()};
  }

  void onCloseRangeReady(const std_msgs::msg::Bool::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    close_range_ready_ = msg->data;
    close_range_ready_stamp_ = now();
  }

  void onStableObjectPoint(const geometry_msgs::msg::PointStamped::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_stable_object_point_ = *msg;
    latest_stable_object_stamp_ = now();
  }

  void onArucoPose(const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_aruco_pose_ = *msg;
    latest_aruco_pose_stamp_ = now();
  }

  void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {
    CameraInfo info;
    info.fx = msg->k[0];
    info.fy = msg->k[4];
    info.cx = msg->k[2];
    info.cy = msg->k[5];
    info.width = msg->width;
    info.height = msg->height;
    info.frame_id = msg->header.frame_id;
    if (info.fx <= 1.0 || info.fy <= 1.0) {
      return;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_camera_info_ = info;
  }

  void onEefCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {
    CameraInfo info;
    info.fx = msg->k[0];
    info.fy = msg->k[4];
    info.cx = msg->k[2];
    info.cy = msg->k[5];
    info.width = msg->width;
    info.height = msg->height;
    info.frame_id = msg->header.frame_id;
    if (info.fx <= 1.0 || info.fy <= 1.0) {
      return;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_eef_camera_info_ = info;
  }

  void onJointState(const sensor_msgs::msg::JointState::ConstSharedPtr msg)
  {
    if (msg->name.empty() || msg->position.size() < msg->name.size()) {
      return;
    }

    std::array<double, 4> positions{};
    std::array<bool, 4> found{false, false, false, false};
    std::array<double, 4> velocities{};
    std::array<bool, 4> velocity_found{};
    std::optional<double> gripper_position;
    for (std::size_t i = 0; i < msg->name.size(); ++i) {
      if (msg->name[i] == "gripper_left_joint" && std::isfinite(msg->position[i]))
        gripper_position = msg->position[i];
      for (std::size_t joint_index = 0; joint_index < arm_joint_names_.size(); ++joint_index) {
        if (msg->name[i] == arm_joint_names_[joint_index]) {
          positions[joint_index] = msg->position[i];
          found[joint_index] = true;
          if (i < msg->velocity.size() && std::isfinite(msg->velocity[i])) {
            velocities[joint_index] = msg->velocity[i];
            velocity_found[joint_index] = true;
          }
          break;
        }
      }
    }

    if (!std::all_of(found.begin(), found.end(), [](bool value) {return value;})) {
      return;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_arm_joint_positions_ = positions;
    latest_gripper_position_ = gripper_position;
    latest_arm_joint_velocities_ =
      std::all_of(velocity_found.begin(), velocity_found.end(), [](bool value) {return value;}) ?
      std::make_optional(velocities) : std::nullopt;
    latest_joint_state_stamp_ = now();
  }

  void startSequence(bool resume_held = false)
  {
    if (safety_abort_latched_) {
      active_ = false;
      done_ = false;
      publishStop();
      publishBaseStop();
      publishBaseHold(true);
      publishEefAutoInitEnable(false);
      if (!safety_abort_ignore_reported_) {
        safety_abort_ignore_reported_ = true;
        publishStatus(
          "start ignored after EEF forward safety abort; publish true to /mp_control/cancel or relaunch before retry",
          true);
      }
      return;
    }

    if (active_ && !done_) {
      publishStatus("start ignored because sequence already active", false);
      return;
    }

    // Explicit operator recovery after an interrupted grasp. Never reopen a
    // held object or restart perception/pregrasp while resuming the handoff.
    if (resume_held) {
      const auto q = latestArmJointPositions();
      bool finger_closed = false;
      {
        std::lock_guard<std::mutex> lock(data_mutex_);
        finger_closed = latest_gripper_position_ &&
          std::isfinite(*latest_gripper_position_) &&
          *latest_gripper_position_ < gripper_max_position_ - .001;
      }
      if (!q || !armStopped() || !finger_closed ||
          !std::all_of(q->begin(), q->end(), [](double v) {return std::isfinite(v);})) {
        publishStatus("held handoff refused: requires fresh stopped joints and closed gripper", true);
        return;
      }
    }

    active_ = true;
    done_ = false;
    safety_abort_ignore_reported_ = false;
    close_sent_ = false;
    eef_bbox_seen_after_close_ = false;
    open_sent_ = false;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      eef_refinement_requested_ = false;
      latest_eef_bbox_.reset();
      latest_stable_object_point_.reset();
      latest_stable_object_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      close_range_ready_ = false;
      close_range_ready_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      close_range_pregrasp_latched_ = false;
      resetDepthFilterStateLocked();
    }
    object_pregrasp_horizontal_done_ = false;
    joint_pregrasp_sent_ = false;
    joint_pregrasp_done_ = false;
    extension_joint4_reference_.reset();
    grasp_resume_requested_ = false;
    handoff_resume_requested_ = false;
    pregrasp_pause_reported_ = false;
    initial_aruco_joint1_aligned_ = false;
    joint_pregrasp_target_.reset();
    joint_pregrasp_controller_target_.reset();
    joint_pregrasp_controller_step_target_.reset();
    joint_pregrasp_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    joint_pregrasp_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    eef_forward_last_joint_nudge_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_stage_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_lift_duration_s_ = std::max(0.1, handoff_lift_duration_s_);
    handoff_lift_controller_target_.reset();
    handoff_place_controller_target_.reset();
    handoff_stay_controller_target_.reset();
    handoff_retract_start_joint_positions_.reset();
    handoff_grasp_controller_pose_.reset();
    handoff_joint1_target_rad_.reset();
    handoff_joint1_centering_ = false;
    handoff_joint1_centered_for_rotate_ = false;
    resetEefRefinementMotionState();
    eef_refinement_object_in_target_.reset();
    eef_refinement_use_bbox_center_ = false;
    min_depth_reached_ = false;
    latest_depth_object_in_target_.reset();
    latest_depth_object_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    servo_start_requested_ = false;
    publishEefAutoInitEnable(true);
    publishBaseHold(false);
    last_eef_init_bbox_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    stable_cycles_ = 0;
    stage_ = GraspStage::DEPTH_APPROACH;
    handoff_pick_sequence_index_ += 1;
    assignCargoId();
    publishCargoEvent("assigned", true);
    if (resume_held) {
      close_sent_ = true;
      startHandoffSequence("operator confirmed held object; resuming without opening or regrasping");
      return;
    }
    if (open_gripper_on_start_) {
      sendGripperOpenForObject();
      open_sent_ = true;
    }
    if (start_servo_on_start_) {
      startMoveItServo();
    }
    publishStatus("grasp sequence started", true);
  }

  void cancelSequence(const std::string & reason)
  {
    safety_abort_latched_ = false;
    safety_abort_ignore_reported_ = false;
    active_ = false;
    done_ = false;
    eef_bbox_seen_after_close_ = false;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      eef_refinement_requested_ = false;
      latest_eef_bbox_.reset();
      latest_stable_object_point_.reset();
      latest_stable_object_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      close_range_ready_ = false;
      close_range_ready_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      close_range_pregrasp_latched_ = false;
      resetDepthFilterStateLocked();
    }
    object_pregrasp_horizontal_done_ = false;
    joint_pregrasp_sent_ = false;
    joint_pregrasp_done_ = false;
    extension_joint4_reference_.reset();
    initial_aruco_joint1_aligned_ = false;
    joint_pregrasp_target_.reset();
    joint_pregrasp_controller_target_.reset();
    joint_pregrasp_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    joint_pregrasp_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    eef_forward_last_joint_nudge_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_stage_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_lift_controller_target_.reset();
    handoff_place_controller_target_.reset();
    handoff_stay_controller_target_.reset();
    handoff_joint1_target_rad_.reset();
    handoff_joint1_centering_ = false;
    handoff_joint1_centered_for_rotate_ = false;
    resetEefRefinementMotionState();
    eef_refinement_object_in_target_.reset();
    eef_refinement_use_bbox_center_ = false;
    min_depth_reached_ = false;
    latest_depth_object_in_target_.reset();
    latest_depth_object_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    servo_start_requested_ = false;
    publishEefAutoInitEnable(false);
    publishBaseHold(false);
    stable_cycles_ = 0;
    stage_ = GraspStage::DEPTH_APPROACH;
    publishStop();
    publishBaseStop();
    publishCargoEvent("cancelled", true);
    publishStatus(reason, true);
  }

  void update()
  {
    if (!active_ || done_) {
      return;
    }

    if (isHandoffStage(stage_)) {
      updateHandoff();
      return;
    }

    // Admission to pregrasp already accepted the selected target. Camera
    // occlusion while the wrist moves must not re-enter the approach branch.
    // The committed joint path still requires fresh measured joint feedback;
    // cancelSequence and the policy relay watchdog remain in control.
    if (use_joint_pregrasp_ && joint_pregrasp_sent_) {
      publishBaseHold(true);
      publishBaseStop();
      publishStop();
      if (updateJointPregrasp()) finishJointPregrasp();
      return;
    }

    geometry_msgs::msg::TransformStamped eef_tf;
    try {
      eef_tf = tf_buffer_.lookupTransform(target_frame_, end_effector_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      publishStop();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "end-effector TF unavailable: %s", ex.what());
      return;
    }
    const auto gripper_tf = gripperPoseTransformOrEef(eef_tf);
    captureEefStayRollReference(gripper_tf);

    if (stage_ == GraspStage::EEF_REFINE) {
      publishBaseHold(true);
      publishBaseStop();
      if (!eef_refinement_object_in_target_) {
        eef_refinement_object_in_target_ = latestDepthObjectInTarget();
      }
      if (!eef_refinement_object_in_target_) {
        publishStop();
        publishStatus("EEF refinement active; holding base while waiting for stored object target");
        return;
      }
      updateEefRefinement(*eef_refinement_object_in_target_, eef_tf);
      return;
    }

    std::string object_block_reason;
    auto maybe_object = latestStableObjectInTarget();
    if (maybe_object) {
      object_block_reason = "stable object point from " + stable_object_topic_;
    } else {
      maybe_object = estimateObjectPoint(&object_block_reason);
    }
    bool using_latched_depth_for_pregrasp = false;
    bool using_visual_bbox_for_pregrasp = false;
    bool command_published = false;
    if (useColorTriangulationAfterMinDepth()) {
      std::string color_reason;
      auto front_size_object = estimateObjectPointFromFrontBboxSize();
      auto depth_reference = front_size_object ? front_size_object :
        (maybe_object ? maybe_object : latestDepthObjectInTarget());
      const bool front_bbox_size_close_ready = isFrontBboxCloseBySize();
      const bool eef_bbox_ready = shouldHoldForEefRefinement();
      const bool front_size_object_close =
        front_size_object &&
        objectGoalXForPregrasp(*front_size_object) <=
        color_triangulation_base_stop_object_x_m_;

      if (front_size_object_close) {
        close_range_pregrasp_latched_ = true;
      }
      if (depth_reference) {
        prepareEefColorTriangulation(*depth_reference, &color_reason);
      } else {
        publishEefAutoInitEnable(true);
        color_reason = "depth limit reached; end-effector YOLO enabled";
      }

      auto maybe_color_object = triangulateObjectPoint(&color_reason, false);
      if (maybe_color_object &&
          maybe_color_object->point.x < color_triangulation_min_object_x_m_) {
        std::ostringstream reason;
        reason << "color triangulation object_x=" << maybe_color_object->point.x
               << " below " << color_triangulation_min_object_x_m_;
        color_reason = reason.str();
        maybe_color_object.reset();
      }

      if (!maybe_color_object) {
        const bool visual_reference_close =
          isVisualPregraspReferenceClose(front_size_object, depth_reference);

        const bool front_size_close_ready =
          close_range_pregrasp_latched_ ||
          front_size_object_close;

        if (front_size_close_ready) {
          maybe_object = front_size_object ? front_size_object : depth_reference;
          if (close_range_pregrasp_latched_ || front_size_object_close || !maybe_object) {
            maybe_object = makeVisualPregraspObject(eef_tf);
            using_visual_bbox_for_pregrasp = true;
          }

          using_latched_depth_for_pregrasp = true;
          object_block_reason.clear();
          std::ostringstream status;
          status << "color triangulation unavailable; using close visual bbox for EEF pregrasp"
                 << " object_x=";
          if (front_size_object) {
            status << (front_size_object->point.x + grasp_offset_x_);
          } else if (maybe_object) {
            status << (maybe_object->point.x + grasp_offset_x_);
          } else {
            status << "unavailable";
          }
          status << " front_bbox_size_ready=" << (front_bbox_size_close_ready ? "true" : "false")
                 << " target_stop_x=" << color_triangulation_base_stop_object_x_m_
                 << " eef_bbox_ready=" << (eef_bbox_ready ? "true" : "false")
                 << " visual_reference_close=" << (visual_reference_close ? "true" : "false")
                 << " visual_bbox_fallback=" << (using_visual_bbox_for_pregrasp ? "true" : "false")
                 << ": " << color_reason;
          publishStatus(status.str());
        } else {
          stable_cycles_ = 0;
          publishBaseHold(false);
          publishStop();
          std::ostringstream status;
          status << "after depth limit; waiting for close-range color triangulation: "
                 << color_reason
                 << "; base tracker fallback approach allowed toward object_x="
                 << color_triangulation_base_stop_object_x_m_;
          publishStatus(status.str());
          return;
        }
      } else {
        const double color_goal_x = maybe_color_object->point.x + grasp_offset_x_;
        const bool visual_reference_close =
          isVisualPregraspReferenceClose(front_size_object, depth_reference);

        if (color_goal_x <= color_triangulation_base_stop_object_x_m_ ||
            front_size_object_close) {
          close_range_pregrasp_latched_ = true;
        }

        const bool front_size_close_ready =
          close_range_pregrasp_latched_ ||
          front_size_object_close;

        if (color_goal_x > color_triangulation_base_stop_object_x_m_ && front_size_close_ready) {
          maybe_object = makeVisualPregraspObject(eef_tf);
          using_visual_bbox_for_pregrasp = true;
          using_latched_depth_for_pregrasp = true;

          object_block_reason.clear();
          std::ostringstream status;
          status << "color triangulation still far object_x=" << color_goal_x
                 << "; using close visual bbox for EEF pregrasp"
                 << " front_size_object_x=";
          if (front_size_object) {
            status << (front_size_object->point.x + grasp_offset_x_);
          } else if (maybe_object) {
            status << (maybe_object->point.x + grasp_offset_x_);
          } else {
            status << "unavailable";
          }
          status << " front_bbox_size_ready=" << (front_bbox_size_close_ready ? "true" : "false")
                 << " target_stop_x=" << color_triangulation_base_stop_object_x_m_
                 << " eef_bbox_ready=" << (eef_bbox_ready ? "true" : "false")
                 << " visual_reference_close=" << (visual_reference_close ? "true" : "false")
                 << " visual_bbox_fallback=" << (using_visual_bbox_for_pregrasp ? "true" : "false");
          publishStatus(status.str());
        } else {
          maybe_object = maybe_color_object;
          object_block_reason.clear();
        }
      }

      if (!maybe_object) {
        stable_cycles_ = 0;
        publishBaseHold(false);
        publishStop();
        std::ostringstream status;
        status << "after depth limit; waiting for close-range color triangulation: "
               << color_reason
               << "; base tracker fallback approach allowed toward object_x="
               << color_triangulation_base_stop_object_x_m_;
        publishStatus(status.str());
        return;
      }

      const double color_goal_x = maybe_object->point.x + grasp_offset_x_;
      if (color_goal_x <= color_triangulation_base_stop_object_x_m_ ||
          front_size_object_close) {
        close_range_pregrasp_latched_ = true;
      }

      if (!close_range_pregrasp_latched_ &&
          !using_latched_depth_for_pregrasp &&
          color_goal_x > color_triangulation_base_stop_object_x_m_) {
        stable_cycles_ = 0;
        publishBaseHold(false);
        publishStop();
        std::ostringstream status;
        status << "color triangulation approach: object_x=" << color_goal_x
              << " target_stop_x=" << color_triangulation_base_stop_object_x_m_
              << "; depth ignored after " << eef_refinement_start_depth_m_ << "m";
        publishStatus(status.str());
        return;
      }

      if (close_range_pregrasp_latched_) {
        using_latched_depth_for_pregrasp = true;
        publishBaseHold(true);
        publishBaseStop();
      }
    }

    if (!maybe_object && use_depthless_triangulation_ &&
        !use_color_triangulation_after_min_depth_ &&
        isDepthUnavailableReason(object_block_reason)) {
      publishBaseHold(true);
      maybe_object = estimateObjectPointByTriangulation(
        eef_tf, &object_block_reason, &command_published);
    }

    if (!maybe_object) {
      if (require_close_range_ready_for_pregrasp_ && !closeRangeReadyFresh()) {
        publishBaseHold(false);
        publishStatus(
          "waiting for close-range ready; base released for depth_perception approach: " +
          object_block_reason);
        return;
      }
      
      if (use_eef_refinement_ &&
          (!use_color_triangulation_after_min_depth_ || useColorTriangulationAfterMinDepth()) &&
          shouldHoldForEefRefinement()) {
        publishBaseHold(true);
      }
      if (!command_published) {
        publishStop();
      }
      publishStatus("waiting for " + object_block_reason);
      return;
    }

    const auto & object = *maybe_object;

    if (stage_ == GraspStage::EEF_REFINE) {
      updateEefRefinement(object, eef_tf);
      return;
    }

    const double goal_x = object.point.x + grasp_offset_x_;
    const double goal_y = object.point.y + grasp_offset_y_;
    const double goal_z = object.point.z + grasp_offset_z_;

    const double eef_x = eef_tf.transform.translation.x;
    const double eef_y = eef_tf.transform.translation.y;
    const double eef_z = eef_tf.transform.translation.z;

    const double err_x = goal_x - eef_x;
    const double err_y = goal_y - eef_y;
    const double err_z = goal_z - eef_z;
    const double err_norm = vectorNorm(err_x, err_y, err_z);

    const bool object_x_ready_for_eef = use_color_triangulation_after_min_depth_ ?
      (using_latched_depth_for_pregrasp || goal_x <= color_triangulation_base_stop_object_x_m_) :
      goal_x <= eef_refinement_start_object_x_m_;
    const bool depth_ready_for_eef = use_color_triangulation_after_min_depth_ ?
      false :
      shouldStartEefRefinementByDepth();
    const bool eef_candidate =
      using_latched_depth_for_pregrasp ||
      shouldHoldForEefRefinement() ||
      depth_ready_for_eef ||
      object_x_ready_for_eef ||
      err_norm <= eef_refinement_switch_distance_m_;
    const bool use_eef_now =
      use_eef_refinement_ &&
      (!use_color_triangulation_after_min_depth_ || object_x_ready_for_eef) &&
      eef_candidate;
	    if (use_eef_now) {
      if (require_close_range_ready_for_pregrasp_ && !closeRangeReadyFresh()) {
        if (!object_x_ready_for_eef) {
          publishBaseHold(false);
          publishStop();
          std::ostringstream status;
          status << "waiting for close-range ready from depth_perception before pregrasp"
                 << ": goal_x=" << goal_x
                 << " target_stop_x=" << color_triangulation_base_stop_object_x_m_
                 << " err_norm=" << err_norm
                 << " object_x_ready=" << (object_x_ready_for_eef ? "true" : "false")
                 << " depth_ready=" << (depth_ready_for_eef ? "true" : "false");
          publishStatus(status.str());
          return;
        }

        publishBaseHold(true);
        publishBaseStop();
        std::ostringstream status;
        status << "local close-range stop by object_x; depth_perception close_range_ready missing/stale"
               << ": goal_x=" << goal_x
               << " target_stop_x=" << color_triangulation_base_stop_object_x_m_
               << " err_norm=" << err_norm
               << " object_x_ready=true"
               << " depth_ready=" << (depth_ready_for_eef ? "true" : "false");
        publishStatus(status.str());
      }
	      publishBaseHold(true);
	      publishBaseStop();

        if (use_joint_pregrasp_ && !initial_aruco_joint1_aligned_) {
          publishStop();
          publishBaseStop();

          if (!updateArucoJoint1OnlyAlignment()) {
            return;
          }

          initial_aruco_joint1_aligned_ = true;
        }

	      const bool joint_pregrasp_ready =
	        !use_joint_pregrasp_ || updateJointPregrasp();
	      if (!joint_pregrasp_ready) {
	        return;
	      }

        if (use_joint_pregrasp_) {
          publishStop();
          publishBaseStop();
          finishJointPregrasp();
          return;
        }

        if (!startMoveItServo()) {
          publishStatus("waiting for MoveIt Servo start before EEF refinement");
          return;
        }

        if (!use_joint_pregrasp_) {
          bool extension_cmd_published = false;
          if (!moveArmToObjectPregraspPose(eef_tf, object, &extension_cmd_published)) {
            return;
          }
        }

        if (!prepareEefRefinement(object)) {
          return;
        }
              eef_refinement_object_in_target_ = object;
      eef_refinement_use_bbox_center_ = using_visual_bbox_for_pregrasp;
      stage_ = GraspStage::EEF_REFINE;
      stable_cycles_ = 0;
      captureEefRpyReference(gripper_tf);
      publishStatus(
        "base stopped, depth+EEF ready; arm extended toward object and switching to refinement" +
        poseStatusSuffix(eef_tf),
        true);
      updateEefRefinement(object, eef_tf);
      return;
    }

    const double forward_err = std::max(0.0, err_x);
    if (wait_for_base_approach_ &&
        (forward_err > arm_start_max_error_m_ || goal_x > arm_start_max_object_x_m_)) {
      stable_cycles_ = 0;
      publishBaseHold(false);
      publishStop();
      std::ostringstream status;
      status << "waiting for base approach before arm motion: goal_x=" << goal_x
             << " forward_err=" << forward_err
             << " err_norm=" << err_norm;
      publishStatus(status.str());
      return;
    }

    if (err_norm <= position_tolerance_m_) {
      stable_cycles_ += 1;
      publishStop();
      if (stable_cycles_ >= close_after_stable_cycles_) {
        closeAndCompleteWhenVisualReady(
          "grasp target reached; width-aware gripper command sent",
          "gripper close commanded; waiting for front-only visual grasp confirmation");
      } else {
        publishStatus("holding near target before closing");
      }
      return;
    }

    stable_cycles_ = 0;
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = target_frame_;
    cmd.twist.linear.x = clampValue(linear_gain_ * err_x, -max_linear_speed_, max_linear_speed_);
    cmd.twist.linear.y = clampValue(linear_gain_ * err_y, -max_linear_speed_, max_linear_speed_);
    cmd.twist.linear.z = clampValue(linear_gain_ * err_z, -max_linear_speed_, max_linear_speed_);
    twist_pub_->publish(cmd);

    std::ostringstream status;
    status << "approach err=(" << err_x << ", " << err_y << ", " << err_z
           << ") norm=" << err_norm;
    publishStatus(status.str());
  }

  bool useColorTriangulationAfterMinDepth()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return use_color_triangulation_after_min_depth_ && min_depth_reached_;
  }

  std::optional<geometry_msgs::msg::PointStamped> latestDepthObjectInTarget()
  {
    const auto stamp = now();
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (use_color_triangulation_after_min_depth_ && min_depth_reached_) {
      return std::nullopt;
    }
    if (!latest_depth_object_in_target_ ||
        latest_depth_object_stamp_.nanoseconds() == 0 ||
        (stamp - latest_depth_object_stamp_).seconds() > max_target_age_s_) {
      return std::nullopt;
    }
    return latest_depth_object_in_target_;
  }

  std::optional<geometry_msgs::msg::PointStamped> latestStableObjectInTarget()
  {
    const auto stamp = now();
    std::optional<geometry_msgs::msg::PointStamped> object;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!latest_stable_object_point_ ||
          latest_stable_object_stamp_.nanoseconds() == 0 ||
          (stamp - latest_stable_object_stamp_).seconds() > max_target_age_s_) {
        return std::nullopt;
      }
      object = latest_stable_object_point_;
    }

    if (!object) {
      return std::nullopt;
    }
    if (object->header.frame_id.empty() || object->header.frame_id == target_frame_) {
      object->header.frame_id = target_frame_;
      return object;
    }

    try {
      return tf_buffer_.transform(*object, target_frame_);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "stable object TF transform failed from %s to %s: %s",
        object->header.frame_id.c_str(), target_frame_.c_str(), ex.what());
      return std::nullopt;
    }
  }

  std::optional<geometry_msgs::msg::PointStamped> estimateObjectPointFromFrontBboxSize()
  {
    Bbox bbox;
    CameraInfo info;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!latest_bbox_ || !latest_camera_info_) {
        return std::nullopt;
      }
      bbox = *latest_bbox_;
      info = *latest_camera_info_;
    }

    if ((now() - bbox.stamp).seconds() > max_target_age_s_ ||
        !shouldHandoffByBboxSize(bbox, info)) {
      return std::nullopt;
    }

    const auto range_m = estimateRangeFromBboxSize(bbox, info);
    if (!range_m) {
      return std::nullopt;
    }

    const double u = bbox.x + 0.5 * bbox.width;
    const double v = bbox.y + 0.5 * bbox.height;
    geometry_msgs::msg::PointStamped object_camera;
    object_camera.header.stamp = builtin_interfaces::msg::Time();
    object_camera.header.frame_id = camera_frame_override_.empty() ? info.frame_id : camera_frame_override_;
    object_camera.point.z = *range_m;
    object_camera.point.x = (u - info.cx) * (*range_m) / info.fx;
    object_camera.point.y = (v - info.cy) * (*range_m) / info.fy;

    try {
      auto object_in_target = tf_buffer_.transform(object_camera, target_frame_);
      rememberObjectDepth(*range_m);
      rememberDepthObjectPoint(object_in_target);
      return object_in_target;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "front bbox size object TF transform failed: %s", ex.what());
      return std::nullopt;
    }
  }

  std::optional<geometry_msgs::msg::PoseStamped> latestFreshArucoPose()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!latest_aruco_pose_) {
      return std::nullopt;
    }

    if (latest_aruco_pose_stamp_.nanoseconds() == 0 ||
        (now() - latest_aruco_pose_stamp_).seconds() > aruco_center_pose_timeout_s_) {
      return std::nullopt;
    }

    return latest_aruco_pose_;
  }

  bool alignEefToArucoX()
  {
    if (!aruco_center_align_enabled_) {
      return true;
    }

    const auto aruco_pose = latestFreshArucoPose();
    if (!aruco_pose) {
      publishStop();
      publishStatus("aruco x align: waiting for fresh /target/aruco_pose");
      return false;
    }

    const double err_x =
      aruco_pose->pose.position.x - aruco_gripper_center_x_offset_m_;

    if (std::abs(err_x) <= aruco_center_x_tolerance_m_) {
      publishStop();
      publishStatus(
        "aruco x align complete: error_x=" + std::to_string(err_x),
        true);
      return true;
    }

    const std::string camera_frame =
      eef_camera_frame_override_.empty() ?
      aruco_pose->header.frame_id :
      eef_camera_frame_override_;

    geometry_msgs::msg::TransformStamped camera_tf_msg;
    try {
      camera_tf_msg =
        tf_buffer_.lookupTransform(target_frame_, camera_frame, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      publishStop();
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "aruco x align TF unavailable: %s", ex.what());
      return false;
    }

    tf2::Transform camera_to_target;
    tf2::fromMsg(camera_tf_msg.transform, camera_to_target);

    // EEF camera x-axis direction only. y/z correction disabled.
    tf2::Vector3 linear_camera(
      aruco_center_lateral_gain_ * err_x,
      0.0,
      0.0);

    tf2::Vector3 linear_target = camera_to_target.getBasis() * linear_camera;

    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = target_frame_;

    cmd.twist.linear.x = clampValue(
      linear_target.x(),
      -aruco_center_max_linear_speed_,
      aruco_center_max_linear_speed_);
    cmd.twist.linear.y = clampValue(
      linear_target.y(),
      -aruco_center_max_linear_speed_,
      aruco_center_max_linear_speed_);
    cmd.twist.linear.z = clampValue(
      linear_target.z(),
      -aruco_center_max_linear_speed_,
      aruco_center_max_linear_speed_);

    twist_pub_->publish(cmd);

    publishStatus(
      "aruco x aligning: error_x=" + std::to_string(err_x) +
      " cmd=(" +
      std::to_string(cmd.twist.linear.x) + ", " +
      std::to_string(cmd.twist.linear.y) + ", " +
      std::to_string(cmd.twist.linear.z) + ")");

    return false;
  }

  bool isFrontBboxCloseBySize()
  {
    Bbox bbox;
    CameraInfo info;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!latest_bbox_ || !latest_camera_info_) {
        return false;
      }
      bbox = *latest_bbox_;
      info = *latest_camera_info_;
    }

    return (now() - bbox.stamp).seconds() <= max_target_age_s_ &&
      shouldHandoffByBboxSize(bbox, info);
  }

  bool prepareEefColorTriangulation(
    const geometry_msgs::msg::PointStamped & object_in_target,
    std::string * block_reason)
  {
    CameraInfo info;
    std::optional<Bbox> eef_bbox;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (latest_eef_camera_info_) {
        info = *latest_eef_camera_info_;
      } else {
        auto fallback_info = fallbackEefCameraInfo();
        if (!fallback_info) {
          setBlockReason(block_reason, "end-effector camera info");
          return false;
        }
        info = *fallback_info;
      }
      eef_bbox = latest_eef_bbox_;
    }

    publishEefAutoInitEnable(true);

    const auto stamp = now();
    if (eef_bbox && (stamp - eef_bbox->stamp).seconds() <= max_target_age_s_) {
      return true;
    }

    if (auto_init_eef_tracker_from_object_) {
      if (last_eef_init_bbox_stamp_.nanoseconds() == 0 ||
          (stamp - last_eef_init_bbox_stamp_).seconds() >= eef_init_bbox_republish_period_s_) {
        if (publishProjectedEefInitBbox(object_in_target, info)) {
          last_eef_init_bbox_stamp_ = stamp;
        }
      }
    }

    setBlockReason(block_reason, "fresh end-effector bbox for color triangulation");
    return false;
  }

  bool shouldHoldForEefRefinement()
  {
    const auto stamp = now();
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (eef_refinement_requested_) {
      return true;
    }
    return latest_eef_bbox_ &&
      (stamp - latest_eef_bbox_->stamp).seconds() <= max_target_age_s_;
  }

  bool closeRangeReadyFresh()
  {
    const auto stamp = now();
    std::lock_guard<std::mutex> lock(data_mutex_);
    return close_range_ready_ &&
      close_range_ready_stamp_.nanoseconds() != 0 &&
      (stamp - close_range_ready_stamp_).seconds() <= close_range_ready_max_age_s_;
  }

  double objectGoalXForPregrasp(const geometry_msgs::msg::PointStamped & object) const
  {
    return object.point.x + grasp_offset_x_;
  }

  bool isVisualPregraspReferenceClose(
    const std::optional<geometry_msgs::msg::PointStamped> & front_size_object,
    const std::optional<geometry_msgs::msg::PointStamped> & depth_reference) const
  {
    const auto & reference = front_size_object ? front_size_object : depth_reference;
    return reference &&
      objectGoalXForPregrasp(*reference) <= color_triangulation_base_stop_object_x_m_;
  }

  geometry_msgs::msg::PointStamped makeVisualPregraspObject(
    const geometry_msgs::msg::TransformStamped & eef_tf)
  {
    geometry_msgs::msg::PointStamped object;
    object.header.stamp = now();
    object.header.frame_id = target_frame_;
    object.point.x = std::max(
      color_triangulation_min_object_x_m_,
      color_triangulation_base_stop_object_x_m_ - grasp_offset_x_);
    object.point.y = eef_tf.transform.translation.y - grasp_offset_y_;
    object.point.z = eef_tf.transform.translation.z - grasp_offset_z_;
    return object;
  }

  bool shouldStartEefRefinementByDepth()
  {
    const auto stamp = now();
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (use_color_triangulation_after_min_depth_ && min_depth_reached_) {
      return false;
    }
    return latest_object_depth_m_ &&
      latest_object_depth_stamp_.nanoseconds() != 0 &&
      (stamp - latest_object_depth_stamp_).seconds() <= max_target_age_s_ &&
      *latest_object_depth_m_ <= eef_refinement_start_depth_m_;
  }

  bool prepareEefRefinement(const geometry_msgs::msg::PointStamped & object_in_target)
  {
    CameraInfo info;
    std::optional<Bbox> eef_bbox;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      eef_refinement_requested_ = true;
      if (latest_eef_camera_info_) {
        info = *latest_eef_camera_info_;
      } else {
        auto fallback_info = fallbackEefCameraInfo();
        if (!fallback_info) {
          publishStop();
          publishStatus("waiting for end-effector camera info before near-field refinement");
          return false;
        }
        info = *fallback_info;
      }
      eef_bbox = latest_eef_bbox_;
    }
    publishEefAutoInitEnable(true);

    if (eef_bbox && (now() - eef_bbox->stamp).seconds() <= max_target_age_s_) {
      return true;
    }

    if (!auto_init_eef_tracker_from_object_) {
      publishStop();
      publishStatus("waiting for end-effector visual feature bbox near object");
      return false;
    }

    const auto stamp = now();
    if (last_eef_init_bbox_stamp_.nanoseconds() == 0 ||
        (stamp - last_eef_init_bbox_stamp_).seconds() >= eef_init_bbox_republish_period_s_) {
      if (publishProjectedEefInitBbox(object_in_target, info)) {
        last_eef_init_bbox_stamp_ = stamp;
      }
    }

    publishStop();
    publishStatus("waiting for end-effector visual feature bbox near object");
    return false;
  }

  bool publishProjectedEefInitBbox(
    const geometry_msgs::msg::PointStamped & object_in_target,
    const CameraInfo & info)
  {
    const std::string eef_camera_frame =
      eef_camera_frame_override_.empty() ? info.frame_id : eef_camera_frame_override_;

    geometry_msgs::msg::PointStamped object_latest = object_in_target;
    object_latest.header.stamp = builtin_interfaces::msg::Time();
    object_latest.point.x += grasp_offset_x_;
    object_latest.point.y += grasp_offset_y_;
    object_latest.point.z += grasp_offset_z_;

    geometry_msgs::msg::PointStamped object_in_eef_camera;
    try {
      object_in_eef_camera = tf_buffer_.transform(object_latest, eef_camera_frame);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "end-effector init bbox TF unavailable: %s", ex.what());
      return false;
    }

    const double z_m = object_in_eef_camera.point.z;
    if (!std::isfinite(z_m) || z_m <= 0.01) {
      return false;
    }

    const double u = info.fx * object_in_eef_camera.point.x / z_m + info.cx;
    const double v = info.fy * object_in_eef_camera.point.y / z_m + info.cy;
    if (!std::isfinite(u) || !std::isfinite(v)) {
      return false;
    }

    const double image_width = std::max(1.0, static_cast<double>(info.width));
    const double image_height = std::max(1.0, static_cast<double>(info.height));
    if (u < 0.0 || u >= image_width || v < 0.0 || v >= image_height) {
      return false;
    }

    std::optional<double> measured_width;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      measured_width = latest_object_width_m_;
    }
    const double width_m = measured_width.value_or(gripper_fallback_object_width_m_);
    const double bbox_w = clampValue(
      eef_init_bbox_padding_scale_ * width_m * info.fx / z_m,
      eef_init_bbox_min_size_px_, eef_init_bbox_max_size_px_);
    const double bbox_h = clampValue(
      eef_init_bbox_padding_scale_ * object_height_m_ * info.fy / z_m,
      eef_init_bbox_min_size_px_, eef_init_bbox_max_size_px_);

    const double x = clampValue(u - 0.5 * bbox_w, 0.0, image_width - 1.0);
    const double y = clampValue(v - 0.5 * bbox_h, 0.0, image_height - 1.0);
    const double w = std::min(bbox_w, image_width - x);
    const double h = std::min(bbox_h, image_height - y);
    if (w < 2.0 || h < 2.0) {
      return false;
    }

    std_msgs::msg::Float32MultiArray msg;
    msg.data = {
      static_cast<float>(x),
      static_cast<float>(y),
      static_cast<float>(w),
      static_cast<float>(h)};
    eef_init_bbox_pub_->publish(msg);
    return true;
  }

  void updateEefRefinement(
    const geometry_msgs::msg::PointStamped & object_in_target,
    const geometry_msgs::msg::TransformStamped & eef_tf)
  {
    publishBaseHold(true);
    publishBaseStop();

    if (continueAfterCloseUntilEefBboxLost(
        eef_tf,
        "eef camera refined grasp reached; width-aware gripper command sent"))
    {
      return;
    }

    const auto gripper_tf = gripperPoseTransformOrEef(eef_tf);
    captureEefRpyReference(gripper_tf);
    const RpyError rpy_error = computeEefRpyError(gripper_tf);

    if (eef_forward_advance_active_) {
      updateEefForwardAdvance(eef_tf, rpy_error);
      return;
    }

    if (!prepareEefRefinement(object_in_target)) {
      if (canStartEefForwardFromFrontBboxFallback()) {
        startEefForwardAdvance(
          eef_tf,
          rpy_error,
          "front bbox close-size fallback; EEF bbox unavailable, starting fixed-pose forward advance");
      }
      return;
    }

    Bbox bbox;
    CameraInfo info;
    bool missing_eef_bbox = false;
    bool missing_eef_camera_info = false;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!latest_eef_bbox_) {
        missing_eef_bbox = true;
      } else {
        bbox = *latest_eef_bbox_;
      }
      if (latest_eef_camera_info_) {
        info = *latest_eef_camera_info_;
      } else {
        auto fallback_info = fallbackEefCameraInfo();
        if (!fallback_info) {
          missing_eef_camera_info = true;
        } else {
          info = *fallback_info;
        }
      }
    }

    if (missing_eef_bbox) {
      if (canStartEefForwardFromFrontBboxFallback()) {
        startEefForwardAdvance(
          eef_tf,
          rpy_error,
          "front bbox close-size fallback; EEF bbox missing, starting fixed-pose forward advance");
        return;
      }
      publishStop();
      publishStatus("waiting for end-effector visual feature bbox");
      return;
    }

    if (missing_eef_camera_info) {
      publishStop();
      publishStatus("waiting for end-effector camera info");
      return;
    }

    if ((now() - bbox.stamp).seconds() > max_target_age_s_ && !eef_forward_advance_active_) {
      if (canStartEefForwardFromFrontBboxFallback()) {
        startEefForwardAdvance(
          eef_tf,
          rpy_error,
          "front bbox close-size fallback; EEF bbox stale, starting fixed-pose forward advance");
        return;
      }
      publishStop();
      publishStatus("waiting for fresh end-effector visual feature bbox");
      return;
    }

    const std::string eef_camera_frame =
      eef_camera_frame_override_.empty() ? info.frame_id : eef_camera_frame_override_;

    geometry_msgs::msg::PointStamped object_latest = object_in_target;
    object_latest.header.stamp = builtin_interfaces::msg::Time();

    geometry_msgs::msg::PointStamped object_in_eef_camera;
    try {
      object_in_eef_camera = tf_buffer_.transform(object_latest, eef_camera_frame);
    } catch (const tf2::TransformException & ex) {
      publishStop();
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "end-effector camera refinement TF unavailable: %s", ex.what());
      return;
    }

    const double bbox_u = bbox.x + 0.5 * bbox.width;
    const double bbox_v = bbox.y + 0.5 * bbox.height;
    const bool projected_object_range_valid =
      std::isfinite(object_in_eef_camera.point.z) &&
      object_in_eef_camera.point.z > 0.0;
    const double projected_u = projected_object_range_valid ?
      info.fx * object_in_eef_camera.point.x / object_in_eef_camera.point.z + info.cx :
      std::numeric_limits<double>::quiet_NaN();
    const double projected_v = projected_object_range_valid ?
      info.fy * object_in_eef_camera.point.y / object_in_eef_camera.point.z + info.cy :
      std::numeric_limits<double>::quiet_NaN();
    const bool projected_center_valid =
      !eef_refinement_use_bbox_center_ &&
      std::isfinite(projected_u) && std::isfinite(projected_v) &&
      projected_u >= 0.0 && projected_u < static_cast<double>(info.width) &&
      projected_v >= 0.0 && projected_v < static_cast<double>(info.height);
    const double u = projected_center_valid ? projected_u : bbox_u;
    const double v = projected_center_valid ? projected_v : bbox_v;
    const double err_u_px = u - info.cx;
    const double err_v_px = v - info.cy;
    const double z_m = projected_center_valid ?
      clampValue(
        object_in_eef_camera.point.z,
        eef_final_depth_m_,
        eef_refinement_start_depth_m_) :
      eef_refinement_switch_distance_m_;
    const double lateral_m = err_u_px * z_m / info.fx;
    const double vertical_m = err_v_px * z_m / info.fy;

    const double center_tolerance_px = scaledEefTolerancePx(eef_center_tolerance_px_, info);
    const double close_tolerance_px = scaledEefTolerancePx(eef_close_tolerance_px_, info);
    const double forward_start_tolerance_px =
      scaledEefTolerancePx(eef_forward_start_tolerance_px_, info);
    const bool centered =
      std::abs(err_u_px) <= center_tolerance_px &&
      std::abs(err_v_px) <= center_tolerance_px;
    const bool close_ready =
      std::abs(err_u_px) <= close_tolerance_px &&
      std::abs(err_v_px) <= close_tolerance_px;
    const bool forward_ready =
      std::abs(err_u_px) <= forward_start_tolerance_px &&
      std::abs(err_v_px) <= forward_start_tolerance_px;
    const bool pose_ready = close_ready && rpy_error.ready;
    const bool forward_pose_ready = forward_ready && rpy_error.ready;

    if (forward_pose_ready) {
      if (eef_forward_after_align_ &&
          eef_forward_distance_m_ > 0.0 &&
          eef_forward_speed_mps_ > 0.0 &&
          !eef_forward_advance_active_) {
        startEefForwardAdvance(
          eef_tf,
          rpy_error,
          "eef pixel+rpy aligned; starting fixed-pose forward advance before grasp",
          forward_start_tolerance_px,
          close_tolerance_px);
        return;
      }

      if (!pose_ready) {
        stable_cycles_ = 0;
        publishStatus("eef forward-ready but not close-ready; waiting for fixed-pose advance path");
        return;
      }

      stable_cycles_ += 1;
      publishStop();
      if (stable_cycles_ >= close_after_stable_cycles_) {
        closeAndCompleteWhenVisualReady(
          "eef camera refined grasp reached after fixed-pose forward advance; width-aware gripper command sent",
          "gripper close commanded after EEF refinement; waiting for front-only visual grasp confirmation");
      } else {
        std::ostringstream hold_status;
        hold_status << "eef camera close-ready; holding before closing"
                    << " pixel_err=(" << err_u_px << ", " << err_v_px << ")"
                    << " rpy_err=(" << rpy_error.roll << ", " << rpy_error.pitch
                    << ", " << rpy_error.yaw << ")"
                    << " roll_ref=" << rpyReferenceMode(rpy_error)
                    << " rpy_frame=" << rpyControlFrame()
                    << " rpy_ready=" << (rpy_error.ready ? "true" : "false")
                    << " strict_centered=" << (centered ? "true" : "false")
                    << poseStatusSuffix(eef_tf);
        publishStatus(hold_status.str());
      }
      return;
    }

    stable_cycles_ = 0;
    eef_forward_advance_active_ = false;
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = target_frame_;
    tf2::Transform camera_to_target;
    try {
      geometry_msgs::msg::TransformStamped camera_tf_msg =
        tf_buffer_.lookupTransform(target_frame_, eef_camera_frame, tf2::TimePointZero);
      tf2::fromMsg(camera_tf_msg.transform, camera_to_target);
    } catch (const tf2::TransformException & ex) {
      publishStop();
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "end-effector camera command TF unavailable: %s", ex.what());
      return;
    }
    const tf2::Vector3 linear_camera(
      eef_refine_lateral_gain_ * lateral_m,
      eef_refine_lateral_gain_ * vertical_m,
      0.0);
    const tf2::Vector3 linear_target = camera_to_target.getBasis() * linear_camera;
    cmd.twist.linear.x = clampValue(
      linear_target.x(), -eef_refine_max_linear_speed_, eef_refine_max_linear_speed_);
    cmd.twist.linear.y = clampValue(
      linear_target.y(), -eef_refine_max_linear_speed_, eef_refine_max_linear_speed_);
    cmd.twist.linear.z = clampValue(
      linear_target.z(), -eef_refine_max_linear_speed_, eef_refine_max_linear_speed_);
    applyEefRpyCorrection(rpy_error, cmd);
    twist_pub_->publish(cmd);

    std::ostringstream status;
    status << "eef refine pixel_err=(" << err_u_px << ", " << err_v_px << ")"
           << " feature_err=(" << (bbox_u - u) << ", " << (bbox_v - v) << ")"
           << " feature_source=" << (projected_center_valid ? "front_projection" : "eef_bbox")
           << " range_cmd=disabled"
           << " close_tol_px=" << close_tolerance_px
           << " forward_tol_px=" << forward_start_tolerance_px
           << " rpy_err=(" << rpy_error.roll << ", " << rpy_error.pitch
           << ", " << rpy_error.yaw << ")"
           << " roll_ref=" << rpyReferenceMode(rpy_error)
           << " rpy_frame=" << rpyControlFrame()
           << " rpy_ready=" << (rpy_error.ready ? "true" : "false")
           << " eef_resolution=" << info.width << "x" << info.height
           << " cmd_frame=" << target_frame_
           << poseStatusSuffix(eef_tf);
    publishStatus(status.str());
  }

  bool canStartEefForwardFromFrontBboxFallback()
  {
    if (!eef_forward_after_align_ ||
        eef_forward_distance_m_ <= 0.0 ||
        eef_forward_speed_mps_ <= 0.0 ||
        eef_forward_advance_active_) {
      return false;
    }
    auto front_size_object = estimateObjectPointFromFrontBboxSize();
    auto depth_reference = front_size_object ? front_size_object : latestDepthObjectInTarget();
    return isFrontBboxCloseBySize() &&
      latestFreshFrontBboxArea().has_value() &&
      isVisualPregraspReferenceClose(front_size_object, depth_reference);
  }

  void startEefForwardAdvance(
    const geometry_msgs::msg::TransformStamped & eef_tf,
    const RpyError & rpy_error,
    const std::string & reason,
    const std::optional<double> forward_tolerance_px = std::nullopt,
    const std::optional<double> close_tolerance_px = std::nullopt)
  {
    eef_forward_advance_active_ = true;
    eef_forward_start_stamp_ = now();
    eef_forward_start_x_m_ = eef_tf.transform.translation.x;
    eef_forward_start_joint_positions_ = latestArmJointPositions();
    if (hold_joint4_during_extension_ && !extension_joint4_reference_ && eef_forward_start_joint_positions_) {
      extension_joint4_reference_ = (*eef_forward_start_joint_positions_)[3];
    }
    front_bbox_area_at_eef_forward_start_ = latestFreshFrontBboxArea();
    eef_bbox_area_at_eef_forward_start_ = latestFreshEefBboxArea();
    stable_cycles_ = 0;
    publishEefForwardAdvanceCommand(rpy_error, false);

    std::ostringstream status;
    status << reason
           << " distance=" << eef_forward_distance_m_
           << " speed=" << eef_forward_speed_mps_
           << " start_joints="
           << (eef_forward_start_joint_positions_ ?
             formatJointArray(*eef_forward_start_joint_positions_) : "unavailable")
           << " front_bbox_start_area=" << front_bbox_area_at_eef_forward_start_.value_or(-1.0)
           << " eef_bbox_start_area=" << eef_bbox_area_at_eef_forward_start_.value_or(-1.0)
           << " front_close_area_ratio=" << front_bbox_close_area_ratio_
           << " eef_close_area_ratio=" << eef_bbox_close_area_ratio_;
    if (forward_tolerance_px) {
      status << " forward_tol_px=" << *forward_tolerance_px;
    }
    if (close_tolerance_px) {
      status << " close_tol_px=" << *close_tolerance_px;
    }
    status << " rpy_err=(" << rpy_error.roll << ", " << rpy_error.pitch
           << ", " << rpy_error.yaw << ")"
           << " rpy_ready=" << (rpy_error.ready ? "true" : "false")
           << " roll_ref=" << rpyReferenceMode(rpy_error)
           << " rpy_frame=" << rpyControlFrame()
           << " cmd_frame=" << target_frame_
           << poseStatusSuffix(eef_tf);
    publishStatus(status.str(), true);
  }

  void updateEefForwardAdvance(
    const geometry_msgs::msg::TransformStamped & eef_tf,
    const RpyError & rpy_error)
  {
    const double advanced_x = std::max(
      0.0, eef_tf.transform.translation.x - eef_forward_start_x_m_);
    const double elapsed_s =
      eef_forward_start_stamp_.nanoseconds() == 0 ?
      0.0 :
      (now() - eef_forward_start_stamp_).seconds();
    const auto front_area_ratio = frontBboxAreaRatioFromForwardStart();
    const auto eef_area_ratio = eefBboxAreaRatioFromForwardStart();
    const auto joint_progress = eefForwardJointProgress();
    const bool joint_progress_gate_enabled = eef_forward_use_joint_nudge_;
    const bool joint4_only_gate_active =
      joint_progress_gate_enabled &&
      eef_forward_joint4_after_joint3_complete_ &&
      joint_progress.available &&
      joint_progress.joint3_complete &&
      !joint_progress.joint4_complete;
    if (joint4_only_gate_active) {
      abortEefForwardAdvance(
        "EEF forward aborted: joint4-only loop detected; possible collision or joint stall",
        advanced_x,
        elapsed_s,
        joint_progress,
        eef_tf);
      return;
    }

    const bool distance_extension_complete =
      advanced_x >= eef_forward_distance_m_;
    const bool min_advance_complete =
      advanced_x >= eef_forward_min_advance_before_close_m_;
    const bool joint3_extension_complete =
      joint_progress_gate_enabled &&
      joint_progress.available &&
      joint_progress.joint3_complete;
    const bool all_joints_extended =
      joint_progress_gate_enabled &&
      joint_progress.available &&
      joint_progress.complete;
    const bool analytic_ik_extension_complete =
      eef_forward_use_analytic_ik_ &&
      joint_progress_gate_enabled &&
      min_advance_complete &&
      (!close_after_full_eef_forward_extension_ || distance_extension_complete);
    const bool joint_based_extension_complete =
      eef_forward_use_analytic_ik_ ?
      (analytic_ik_extension_complete && all_joints_extended) :
      all_joints_extended;
    const bool arm_extension_complete =
      distance_extension_complete || joint_based_extension_complete;
    const bool continue_joint3_after_required =
      !eef_forward_use_analytic_ik_ &&
      joint_progress_gate_enabled &&
      joint_progress.available &&
      joint_progress.joint3_complete &&
      !min_advance_complete &&
      !distance_extension_complete;
    const bool fixed_duration_mode = eef_forward_fixed_duration_s_ > 0.0;
    const bool elapsed_enough_for_close =
      !fixed_duration_mode || elapsed_s >= eef_forward_fixed_duration_s_;
    const bool advanced_enough_for_visual_close =
      min_advance_complete && elapsed_enough_for_close;
    const bool forward_stalled =
      eef_forward_gate_timeout_s_ > 0.0 &&
      elapsed_s >= eef_forward_gate_timeout_s_ &&
      !arm_extension_complete;
    if (forward_stalled) {
      abortEefForwardAdvance(
        "EEF forward aborted: arm extension gate timeout; possible collision or joint stall",
        advanced_x,
        elapsed_s,
        joint_progress,
        eef_tf);
      return;
    }

    if (arm_extension_complete && advanced_enough_for_visual_close &&
        (shouldCloseOnFrontBboxShrink() || shouldCloseOnEefBboxShrink())) {
      stable_cycles_ += 1;
      publishStop();
      if (stable_cycles_ >= close_after_stable_cycles_) {
        closeAndCompleteWhenVisualReady(
          "bbox shrank to close threshold; width-aware gripper command sent",
          "gripper close commanded after bbox shrink; waiting for visual confirmation");
      } else {
        std::ostringstream status;
        status << "bbox shrink close-ready; holding before closing"
               << " front_ratio=" << front_area_ratio.value_or(-1.0)
               << " front_threshold=" << front_bbox_close_area_ratio_
               << " eef_ratio=" << eef_area_ratio.value_or(-1.0)
               << " eef_threshold=" << eef_bbox_close_area_ratio_
               << " advanced_x=" << advanced_x
               << " min_close_advance=" << eef_forward_min_advance_before_close_m_
               << " elapsed=" << elapsed_s
               << " fixed_duration=" << eef_forward_fixed_duration_s_
               << formatEefForwardJointProgress(joint_progress);
        publishStatus(status.str());
      }
      return;
    }
    const bool should_continue_forward =
      (fixed_duration_mode && elapsed_s < eef_forward_fixed_duration_s_);
    if (should_continue_forward) {
      publishEefForwardAdvanceCommand(rpy_error, continue_joint3_after_required);
      std::ostringstream status;
      status << "eef aligned; advancing forward before grasp: advanced_x="
             << advanced_x << "/" << eef_forward_distance_m_
             << " elapsed=" << elapsed_s
             << "/" << eef_forward_fixed_duration_s_
             << " front_bbox_area_ratio=" << front_area_ratio.value_or(-1.0)
             << " front_close_ratio_threshold=" << front_bbox_close_area_ratio_
             << " eef_bbox_area_ratio=" << eef_area_ratio.value_or(-1.0)
             << " eef_close_ratio_threshold=" << eef_bbox_close_area_ratio_
             << " min_close_advance=" << eef_forward_min_advance_before_close_m_
             << " advanced_enough_for_visual_close="
             << (advanced_enough_for_visual_close ? "true" : "false")
             << " fixed_duration_mode="
             << (fixed_duration_mode ? "true" : "false")
             << " distance_extension_complete="
             << (distance_extension_complete ? "true" : "false")
             << " min_advance_complete="
             << (min_advance_complete ? "true" : "false")
             << " joint3_extension_complete="
             << (joint3_extension_complete ? "true" : "false")
             << " joint_based_extension_complete="
             << (joint_based_extension_complete ? "true" : "false")
             << " analytic_ik_extension_complete="
             << (analytic_ik_extension_complete ? "true" : "false")
             << " continue_joint3_after_required="
             << (continue_joint3_after_required ? "true" : "false")
             << " arm_extension_complete="
             << (arm_extension_complete ? "true" : "false")
             << " gate_timeout=" << eef_forward_gate_timeout_s_
             << formatEefForwardJointProgress(joint_progress)
             << " rpy_err=(" << rpy_error.roll << ", " << rpy_error.pitch
             << ", " << rpy_error.yaw << ")"
             << " rpy_ready=" << (rpy_error.ready ? "true" : "false")
             << " roll_ref=" << rpyReferenceMode(rpy_error)
             << " rpy_frame=" << rpyControlFrame()
             << " cmd_frame=" << target_frame_
             << poseStatusSuffix(eef_tf);
      publishStatus(status.str());
      return;
    }

    stable_cycles_ += 1;
    publishStop();
    if (stable_cycles_ >= close_after_stable_cycles_) {
      if (close_after_full_eef_forward_extension_) {
        if (close_gripper_on_arrival_ && !close_sent_) {
          sendGripperGraspForObject();
          close_sent_ = true;
        }
        completeGrasp(
          "eef full forward extension reached; width-aware gripper command sent; " +
          visualGraspStateText(currentVisualGraspState()));
      } else {
        closeAndCompleteWhenVisualReady(
          "eef fixed-pose forward advance complete; width-aware gripper command sent",
          "gripper close commanded after EEF forward advance; waiting for front-only visual grasp confirmation");
      }
    } else {
      std::ostringstream status;
      status << "eef forward advance complete; holding before closing"
             << " advanced_x=" << advanced_x << "/" << eef_forward_distance_m_
             << " elapsed=" << elapsed_s
             << "/" << eef_forward_fixed_duration_s_
             << " distance_extension_complete="
             << (distance_extension_complete ? "true" : "false")
             << " min_advance_complete="
             << (min_advance_complete ? "true" : "false")
             << " joint3_extension_complete="
             << (joint3_extension_complete ? "true" : "false")
             << " joint_based_extension_complete="
             << (joint_based_extension_complete ? "true" : "false")
             << " analytic_ik_extension_complete="
             << (analytic_ik_extension_complete ? "true" : "false")
             << " arm_extension_complete="
             << (arm_extension_complete ? "true" : "false")
             << " close_after_full_eef_forward_extension="
             << (close_after_full_eef_forward_extension_ ? "true" : "false")
             << formatEefForwardJointProgress(joint_progress)
             << " rpy_err=(" << rpy_error.roll << ", " << rpy_error.pitch
             << ", " << rpy_error.yaw << ")"
             << " rpy_ready=" << (rpy_error.ready ? "true" : "false")
             << " roll_ref=" << rpyReferenceMode(rpy_error)
             << " rpy_frame=" << rpyControlFrame()
             << poseStatusSuffix(eef_tf);
      publishStatus(status.str());
    }
  }

  bool isDepthUnavailableReason(const std::string & reason) const
  {
    return reason.rfind("valid depth inside bbox", 0) == 0 ||
      reason.rfind("depth image on ", 0) == 0;
  }

  std::optional<geometry_msgs::msg::PointStamped> estimateObjectPointByTriangulation(
    const geometry_msgs::msg::TransformStamped & eef_tf,
    std::string * block_reason,
    bool * command_published)
  {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      eef_refinement_requested_ = true;
    }
    publishEefAutoInitEnable(true);

    if (stage_ != GraspStage::EEF_REFINE &&
        !moveArmToTriangulationPose(eef_tf, command_published)) {
      setBlockReason(block_reason, "arm extension for stereo triangulation");
      return std::nullopt;
    }

    return triangulateObjectPoint(block_reason);
  }

  std::vector<double> normalizedJointVector(
    const std::vector<double> & values,
    const std::array<double, 4> & fallback) const
  {
    if (values.size() != arm_joint_names_.size()) {
      return std::vector<double>(fallback.begin(), fallback.end());
    }
    return values;
  }

  builtin_interfaces::msg::Duration durationFromSeconds(double seconds) const
  {
    builtin_interfaces::msg::Duration duration;
    const auto safe_seconds = std::max(0.0, seconds);
    const auto whole_seconds = static_cast<std::int32_t>(std::floor(safe_seconds));
    duration.sec = whole_seconds;
    duration.nanosec =
      static_cast<std::uint32_t>(std::round((safe_seconds - whole_seconds) * 1e9));
    if (duration.nanosec >= 1000000000U) {
      duration.sec += 1;
      duration.nanosec -= 1000000000U;
    }
    return duration;
  }

  std::string formatJointArray(const std::array<double, 4> & values) const
  {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      out << values[i];
    }
    out << "]";
    return out.str();
  }

  std::optional<std::array<double, 4>> latestArmJointPositions()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!latest_arm_joint_positions_ ||
        latest_joint_state_stamp_.nanoseconds() == 0 ||
        (now() - latest_joint_state_stamp_).seconds() > max_target_age_s_) {
      return std::nullopt;
    }
    return latest_arm_joint_positions_;
  }

  std::array<double, 4> clampJointPregraspTarget(const std::array<double, 4> & positions) const
  {
    std::array<double, 4> clamped = positions;
    for (std::size_t i = 0; i < clamped.size(); ++i) {
      clamped[i] = clampValue(
        clamped[i], joint_pregrasp_min_positions_[i], joint_pregrasp_max_positions_[i]);
    }
    return clamped;
  }

  double gripperRollProxyFromJoints(const std::array<double, 4> & joints) const
  {
    return
      pregrasp_roll_joint2_weight_ * joints[1] +
      pregrasp_roll_joint3_weight_ * joints[2] +
      pregrasp_roll_joint4_weight_ * joints[3];
  }

  double joint4ForPreservedGripperRoll(
    const std::array<double, 4> & reference,
    const std::array<double, 4> & target) const
  {
    return (
      gripperRollProxyFromJoints(reference) -
      pregrasp_roll_joint2_weight_ * target[1] -
      pregrasp_roll_joint3_weight_ * target[2]) /
      pregrasp_roll_joint4_weight_;
  }

  double eefForwardGripperRollProxyFromJoints(const std::array<double, 4> & joints) const
  {
    return
      eef_forward_roll_joint2_weight_ * joints[1] +
      eef_forward_roll_joint3_weight_ * joints[2] +
      eef_forward_roll_joint4_weight_ * joints[3];
  }

    double joint4ForPreservedEefForwardRoll(
      const std::array<double, 4> & reference,
      const std::array<double, 4> & target) const
    {
      if (std::abs(eef_forward_roll_joint4_weight_) < 1.0e-6) {
        return reference[3] -
          eef_forward_roll_joint2_weight_ * (target[1] - reference[1]) -
          eef_forward_roll_joint3_weight_ * (target[2] - reference[2]);
      }

      const double reference_roll_proxy = eefForwardGripperRollProxyFromJoints(reference);
      return (
        reference_roll_proxy -
        eef_forward_roll_joint2_weight_ * target[1] -
        eef_forward_roll_joint3_weight_ * target[2]) /
        eef_forward_roll_joint4_weight_;
    }

  double eefForwardJoint3Direction() const
  {
    return eef_forward_joint3_delta_rad_ < 0.0 ? -1.0 : 1.0;
  }

  double eefForwardEffectiveJoint3RequiredRad(
    const std::array<double, 4> & start) const
  {
    const double requested = std::max(0.0, eef_forward_joint3_complete_delta_rad_);
    if (requested <= 0.0) {
      return 0.0;
    }

    const double direction = eefForwardJoint3Direction();
    const double requested_target = start[2] + direction * requested;
    const double clamped_target = clampValue(
      requested_target,
      joint_pregrasp_min_positions_[2],
      joint_pregrasp_max_positions_[2]);
    return std::abs(clamped_target - start[2]);
  }

  double bezierEase01(double t) const
  {
    t = clampValue(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
  }

  double clampStep(double delta, double max_abs_step) const
  {
    if (max_abs_step <= 0.0) {
      return delta;
    }
    return clampValue(delta, -max_abs_step, max_abs_step);
  }

  std::string enforceEefForwardMonotonicTarget(
    const std::array<double, 4> & current,
    const std::array<double, 4> & start,
    const std::array<double, 4> & reference,
    std::array<double, 4> & target) const
  {
    std::string reason = "none";
    for (std::size_t i = 1; i <= 3; ++i) {
      const double planned_delta = reference[i] - start[i];
      const double target_delta = target[i] - current[i];
      if (
        std::abs(planned_delta) > 1.0e-6 &&
        std::abs(target_delta) > 1.0e-6 &&
        planned_delta * target_delta < 0.0)
      {
        target[i] = current[i];
        if (reason == "none") {
          reason.clear();
        } else {
          reason += ",";
        }
        reason += arm_joint_names_[i];
        reason += ":hold_reverse";
      }
    }
    return reason;
  }

  PlanarIkPose eefForwardIkPoseFromJoints(const std::array<double, 4> & joints) const
  {
    const double link2_len =
      std::hypot(eef_forward_ik_link2_x_m_, eef_forward_ik_link2_z_m_);
    const double link2_angle =
      std::atan2(eef_forward_ik_link2_z_m_, eef_forward_ik_link2_x_m_);
    const double theta2 = joints[1];
    const double theta3 = joints[1] + joints[2];
    const double theta4 = joints[1] + joints[2] + joints[3];

    PlanarIkPose pose;
    pose.x =
      link2_len * std::cos(theta2 + link2_angle) +
      eef_forward_ik_link3_m_ * std::cos(theta3) +
      eef_forward_ik_tool_m_ * std::cos(theta4);
    pose.z =
      link2_len * std::sin(theta2 + link2_angle) +
      eef_forward_ik_link3_m_ * std::sin(theta3) +
      eef_forward_ik_tool_m_ * std::sin(theta4);
    pose.pitch = normalizeAngle(theta4);
    return pose;
  }

  bool solveLinear3x3(
    std::array<std::array<double, 3>, 3> matrix,
    std::array<double, 3> rhs,
    std::array<double, 3> * solution) const
  {
    for (int pivot = 0; pivot < 3; ++pivot) {
      int pivot_row = pivot;
      double pivot_abs = std::abs(matrix[pivot][pivot]);
      for (int row = pivot + 1; row < 3; ++row) {
        const double candidate_abs = std::abs(matrix[row][pivot]);
        if (candidate_abs > pivot_abs) {
          pivot_abs = candidate_abs;
          pivot_row = row;
        }
      }
      if (pivot_abs < 1.0e-9) {
        return false;
      }
      if (pivot_row != pivot) {
        std::swap(matrix[pivot], matrix[pivot_row]);
        std::swap(rhs[pivot], rhs[pivot_row]);
      }

      const double pivot_value = matrix[pivot][pivot];
      for (int col = pivot; col < 3; ++col) {
        matrix[pivot][col] /= pivot_value;
      }
      rhs[pivot] /= pivot_value;

      for (int row = 0; row < 3; ++row) {
        if (row == pivot) {
          continue;
        }
        const double factor = matrix[row][pivot];
        for (int col = pivot; col < 3; ++col) {
          matrix[row][col] -= factor * matrix[pivot][col];
        }
        rhs[row] -= factor * rhs[pivot];
      }
    }

    *solution = rhs;
    return true;
  }

  PlanarIkResult solveEefForwardPlanarIk(
    const std::array<double, 4> & start,
    const std::array<double, 4> & current,
    double target_advance_m) const
  {
    PlanarIkResult result;
    result.controller_target = current;

    const PlanarIkPose start_pose = eefForwardIkPoseFromJoints(start);
    result.target_pose = start_pose;
    result.target_pose.x += target_advance_m;

    std::array<double, 4> solution = current;
    solution[0] = current[0];
    const double lambda2 = eef_forward_ik_damping_ * eef_forward_ik_damping_;
    const std::array<double, 3> row_weights{
      1.0,
      1.0,
      eef_forward_ik_pitch_weight_};

    for (int iter = 0; iter < eef_forward_ik_iterations_; ++iter) {
      const PlanarIkPose pose = eefForwardIkPoseFromJoints(solution);
      std::array<double, 3> error{
        result.target_pose.x - pose.x,
        result.target_pose.z - pose.z,
        normalizeAngle(result.target_pose.pitch - pose.pitch)};
      result.error_norm = std::hypot(error[0], error[1]);
      result.solved_pose = pose;
      result.iterations = iter + 1;
      if (result.error_norm < 0.002 && std::abs(error[2]) < 0.04) {
        result.success = true;
        result.reason = "converged";
        break;
      }

      std::array<std::array<double, 3>, 3> jacobian{};
      constexpr double eps = 1.0e-4;
      for (int col = 0; col < 3; ++col) {
        std::array<double, 4> perturbed = solution;
        perturbed[static_cast<std::size_t>(col + 1)] += eps;
        const PlanarIkPose perturbed_pose = eefForwardIkPoseFromJoints(perturbed);
        jacobian[0][col] = (perturbed_pose.x - pose.x) / eps;
        jacobian[1][col] = (perturbed_pose.z - pose.z) / eps;
        jacobian[2][col] = normalizeAngle(perturbed_pose.pitch - pose.pitch) / eps;
      }

      std::array<std::array<double, 3>, 3> weighted_jacobian{};
      std::array<double, 3> weighted_error{};
      for (int row = 0; row < 3; ++row) {
        weighted_error[row] = error[row] * row_weights[row];
        for (int col = 0; col < 3; ++col) {
          weighted_jacobian[row][col] = jacobian[row][col] * row_weights[row];
        }
      }

      std::array<std::array<double, 3>, 3> normal{};
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          for (int k = 0; k < 3; ++k) {
            normal[row][col] += weighted_jacobian[row][k] * weighted_jacobian[col][k];
          }
        }
        normal[row][row] += lambda2;
      }

      std::array<double, 3> intermediate{};
      if (!solveLinear3x3(normal, weighted_error, &intermediate)) {
        result.reason = "singular_dls_matrix";
        break;
      }

      std::array<double, 3> delta{};
      for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
          delta[col] += weighted_jacobian[row][col] * intermediate[row];
        }
      }

      bool changed = false;
      for (int col = 0; col < 3; ++col) {
        const std::size_t joint_index = static_cast<std::size_t>(col + 1);
        const double limited_delta = clampStep(delta[col], 0.05);
        const double next_position = clampValue(
          solution[joint_index] + limited_delta,
          joint_pregrasp_min_positions_[joint_index],
          joint_pregrasp_max_positions_[joint_index]);
        changed = changed || std::abs(next_position - solution[joint_index]) > 1.0e-6;
        solution[joint_index] = next_position;
      }
      if (!changed) {
        result.reason = "joint_limits";
        break;
      }
    }

    if (result.reason.empty()) {
      result.reason = result.success ? "converged" : "iteration_limit";
    }
    result.controller_target = clampJointPregraspTarget(solution);
    result.solved_pose = eefForwardIkPoseFromJoints(result.controller_target);
    result.error_norm = std::hypot(
      result.target_pose.x - result.solved_pose.x,
      result.target_pose.z - result.solved_pose.z);
    return result;
  }

  double stepJointWithoutReversingTowardLimit(
    double current,
    double delta,
    double min_value,
    double max_value) const
  {
    if (delta > 0.0) {
      return current >= max_value ? current : std::min(current + delta, max_value);
    }
    if (delta < 0.0) {
      return current <= min_value ? current : std::max(current + delta, min_value);
    }
    return current;
  }

  EefForwardJointProgress eefForwardJointProgressFromCurrent(
    const std::array<double, 4> & current) const
  {
    EefForwardJointProgress progress;
    if (!eef_forward_start_joint_positions_) {
      return progress;
    }

    progress.available = true;
    const auto & start = *eef_forward_start_joint_positions_;
    progress.joint3_required_rad = eefForwardEffectiveJoint3RequiredRad(start);
    const double direction = eefForwardJoint3Direction();
    progress.joint3_progress_rad =
      std::max(0.0, direction * (current[2] - start[2]));
    progress.joint3_complete =
      progress.joint3_required_rad <= 0.0 ||
      progress.joint3_progress_rad + eef_forward_joint3_complete_tolerance_rad_ >=
      progress.joint3_required_rad;

    const double joint2_direction =
      eef_forward_bezier_joint2_total_delta_rad_ >= 0.0 ? 1.0 : -1.0;

    const double requested_joint2_target =
      start[1] + joint2_direction * std::abs(eef_forward_bezier_joint2_total_delta_rad_);

    const double clamped_joint2_target = clampValue(
      requested_joint2_target,
      joint_pregrasp_min_positions_[1],
      joint_pregrasp_max_positions_[1]);

    progress.joint2_required_rad = std::abs(clamped_joint2_target - start[1]);

    progress.joint2_progress_rad =
      std::max(0.0, joint2_direction * (current[1] - start[1]));

    progress.joint2_complete =
      progress.joint2_required_rad <= 0.0 ||
      progress.joint2_progress_rad + eef_forward_joint3_complete_tolerance_rad_ >=
      progress.joint2_required_rad;

    if (hold_joint4_during_extension_ || !eef_forward_joint4_after_joint3_complete_) {
      progress.joint4_target_rad = hold_joint4_during_extension_ ?
        extension_joint4_reference_.value_or(start[3]) : current[3];
      progress.joint4_error_rad = progress.joint4_target_rad - current[3];
      progress.joint4_complete = std::abs(progress.joint4_error_rad) <= eef_forward_joint4_finish_tolerance_rad_;
      progress.complete =
        progress.joint2_complete &&
        progress.joint3_complete &&
        progress.joint4_complete;
      return progress;
    }

    std::array<double, 4> final_roll_target = current;
    final_roll_target[2] =
      start[2] + direction * progress.joint3_required_rad;
    if (joint_pregrasp_preserve_gripper_roll_) {
      progress.joint4_target_rad =
        joint4ForPreservedEefForwardRoll(start, final_roll_target);
    } else {
      progress.joint4_target_rad = current[3];
    }
    progress.joint4_target_rad += gripper_down_joint4_offset_rad_;
    progress.joint4_target_rad = clampValue(
      progress.joint4_target_rad, joint_pregrasp_min_positions_[3], joint_pregrasp_max_positions_[3]);
    progress.joint4_error_rad = progress.joint4_target_rad - current[3];
    progress.joint4_complete =
      progress.joint3_complete && 
      std::abs(progress.joint4_error_rad) <= eef_forward_joint4_finish_tolerance_rad_;
    progress.complete = progress.joint2_complete && progress.joint3_complete && progress.joint4_complete;
    return progress;
  }

  EefForwardJointProgress eefForwardJointProgress()
  {
    const auto current = latestArmJointPositions();
    if (!current) {
      return EefForwardJointProgress{};
    }
    return eefForwardJointProgressFromCurrent(*current);
  }

  std::string formatEefForwardJointProgress(const EefForwardJointProgress & progress) const
  {
    std::ostringstream out;
    out << " joint_progress_available=" << (progress.available ? "true" : "false")
        << " joint2_progress=" << progress.joint2_progress_rad
        << "/" << progress.joint2_required_rad
        << " joint2_complete=" << (progress.joint2_complete ? "true" : "false")
        << " joint3_progress=" << progress.joint3_progress_rad
        << "/" << progress.joint3_required_rad
        << " joint3_complete=" << (progress.joint3_complete ? "true" : "false")
        << " joint4_target=" << progress.joint4_target_rad
        << " joint4_error=" << progress.joint4_error_rad
        << " joint4_complete=" << (progress.joint4_complete ? "true" : "false")
        << " arm_extension_complete=" << (progress.complete ? "true" : "false");
    return out.str();
  }

  double pregraspToolPitchTargetRad() const
  {
    return normalizeAngle(
      joint_pregrasp_ready_positions_[1] +
      joint_pregrasp_ready_positions_[2] +
      joint_pregrasp_ready_positions_[3] +
      gripper_down_joint4_offset_rad_);
  }

  double joint4ForPregraspToolPitch(
    const std::array<double, 4> & target,
    double desired_tool_pitch) const
  {
    return normalizeAngle(desired_tool_pitch - target[1] - target[2]);
  }

  void holdExtensionJoint4(
    std::array<double, 4> & target,
    const std::array<double, 4> & current) const
  {
    if (hold_joint4_during_extension_) {
      target[3] = extension_joint4_reference_.value_or(current[3]);
    }
  }

  std::array<double, 4> jointPregraspControllerTargetFromCurrent(
    const std::array<double, 4> & current) const
  {
    (void)current;

    std::array<double, 4> target{};
    for (std::size_t i = 0; i < target.size(); ++i) {
      target[i] = joint_pregrasp_ready_positions_[i];
    }
    target[0] = current[0];

    if (hold_joint4_during_extension_) {
      holdExtensionJoint4(target, current);
      return clampJointPregraspTarget(target);
    }

    const double desired_tool_pitch = pregraspToolPitchTargetRad();
    const double desired_joint4 =
      joint4ForPregraspToolPitch(target, desired_tool_pitch);
    target[3] = current[3] +
      clampStep(desired_joint4 - current[3], joint_pregrasp_joint4_max_step_rad_);
    return clampJointPregraspTarget(target);
  }

  std::array<double, 4> jointPregraspRawTargetFromControllerTarget(
    const std::array<double, 4> & current,
    const std::array<double, 4> & controller_target) const
  {
    std::array<double, 4> target = controller_target;
    holdExtensionJoint4(target, current);
    if (joint_pregrasp_reverse_joint3_delta_) {
      target[2] = current[2] - (controller_target[2] - current[2]);
    }
    return clampJointPregraspTarget(target);
  }

  std::optional<double> latestArucoCenterXError()
  {
    if (!aruco_center_align_enabled_) {
      return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!latest_aruco_pose_) {
      return std::nullopt;
    }

    if (latest_aruco_pose_stamp_.nanoseconds() == 0 ||
        (now() - latest_aruco_pose_stamp_).seconds() > aruco_center_pose_timeout_s_) {
      return std::nullopt;
    }

    return latest_aruco_pose_->pose.position.x - aruco_gripper_center_x_offset_m_;
  }

  std::optional<double> latestArucoCenterYError()
  {
    if (!aruco_center_align_enabled_) {
      return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!latest_aruco_pose_) {
      return std::nullopt;
    }

    if (latest_aruco_pose_stamp_.nanoseconds() == 0 ||
        (now() - latest_aruco_pose_stamp_).seconds() > aruco_center_pose_timeout_s_) {
      return std::nullopt;
    }

    return latest_aruco_pose_->pose.position.y - aruco_gripper_center_y_offset_m_;
  }

  void applyArucoJoint1Correction(
    const std::array<double, 4> & current,
    std::array<double, 4> * target) 
  {
    if (!target || !aruco_center_align_enabled_) {
      return;
    }

    const auto error_x_opt = latestArucoCenterXError();
    if (!error_x_opt) {
      return;
    }

    const double error_x = *error_x_opt;

    if (std::abs(error_x) <= aruco_center_x_tolerance_m_) {
      return;
    }

    const double raw_delta =
      aruco_joint1_align_sign_ * aruco_joint1_align_gain_ * error_x;

    const double joint1_delta =
      clampStep(raw_delta, aruco_joint1_align_max_step_rad_);

    (*target)[0] = clampValue(
      current[0] + joint1_delta,
      joint_pregrasp_min_positions_[0],
      joint_pregrasp_max_positions_[0]);
  }

  bool arucoAlignedForClose()
  {
    if (!aruco_center_align_enabled_ || !aruco_require_center_before_grasp_) {
      return true;
    }

    const auto error_x_opt = latestArucoCenterXError();
    if (!error_x_opt) {
      publishStatus("aruco close blocked: waiting for fresh /target/aruco_pose");
      return false;
    }

    const double error_x = *error_x_opt;
    const auto error_y_opt = latestArucoCenterYError();
    const double error_y = error_y_opt.value_or(0.0);

    const bool x_ok = std::abs(error_x) <= aruco_center_x_tolerance_m_;
    const bool y_ok = std::abs(error_y) <= aruco_center_y_tolerance_m_;

    if (!x_ok || !y_ok) {
      std::ostringstream status;
      status << "aruco close blocked: center not aligned"
             << " error_x=" << error_x
             << "/" << aruco_center_x_tolerance_m_
             << " error_y=" << error_y
             << "/" << aruco_center_y_tolerance_m_;
      publishStatus(status.str());
      return false;
    }

    return true;
  }

  bool updateArucoJoint1OnlyAlignment()
  {
    if (!aruco_center_align_enabled_) {
      return true;
    }

    const auto error_x_opt = latestArucoCenterXError();
    if (!error_x_opt) {
      publishStatus("aruco joint1-only align: waiting for fresh /target/aruco_pose");
      return false;
    }

    const double error_x = *error_x_opt;

    if (std::abs(error_x) <= aruco_center_x_tolerance_m_) {
      publishStatus(
        "aruco joint1-only align complete: error_x=" + std::to_string(error_x),
        true);
      return true;
    }

    const auto current = latestArmJointPositions();
    if (!current) {
      publishStatus("aruco joint1-only align: waiting for joint_states");
      return false;
    }

    std::array<double, 4> target = *current;

    const double raw_delta =
      aruco_joint1_align_sign_ * aruco_joint1_align_gain_ * error_x;

    const double joint1_delta =
      clampStep(raw_delta, aruco_joint1_align_max_step_rad_);

    target[0] = clampValue(
      (*current)[0] + joint1_delta,
      joint_pregrasp_min_positions_[0],
      joint_pregrasp_max_positions_[0]);

    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = now();
    msg.joint_names.assign(arm_joint_names_.begin(), arm_joint_names_.end());

    // Keep joint2, joint3 and joint4 at their current values.
    appendJointTrajectoryPoint(msg, target, 0.13);
    joint_trajectory_pub_->publish(msg);

    publishStatus(
      "aruco joint1-only aligning: error_x=" + std::to_string(error_x) +
      " current=" + formatJointArray(*current) +
      " target=" + formatJointArray(target));

    return false;
  }

  std::array<double, 4> jointPregraspControllerStepTarget(
    const std::array<double, 4> & current,
    const std::array<double, 4> & final_target,
    bool * joint3_step_active = nullptr) 
  {
    if (joint3_step_active) {
      *joint3_step_active = false;
    }

    const double desired_tool_pitch = pregraspToolPitchTargetRad();

    std::array<double, 4> target = final_target;
    const double joint3_delta = final_target[2] - current[2];
    const double joint3_abs_delta = std::abs(joint3_delta);

    if (joint3_abs_delta > joint_pregrasp_tolerance_rad_) {
      const double joint3_step =
        joint_pregrasp_joint3_max_step_rad_ > 1.0e-9 ?
        std::min(joint3_abs_delta, joint_pregrasp_joint3_max_step_rad_) :
        joint3_abs_delta;

      const double ratio =
        clampValue(joint3_step / joint3_abs_delta, 0.0, 1.0);

      for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] = current[i] + (final_target[i] - current[i]) * ratio;
      }
      target[1] = current[1] +
        clampStep(target[1] - current[1], joint_pregrasp_joint2_max_step_rad_);
      const double desired_joint4 =
        joint4ForPregraspToolPitch(target, desired_tool_pitch);
      target[3] = current[3] +
        clampStep(desired_joint4 - current[3], joint_pregrasp_joint4_max_step_rad_);
      holdExtensionJoint4(target, current);

      if (joint3_step_active) {
        *joint3_step_active = true;
      }

      target[0] = current[0]; // ignore aruco correction during joint3 step for smoother motion

      return clampJointPregraspTarget(target);
    }

    target = final_target;
    target[1] = current[1] +
      clampStep(final_target[1] - current[1], joint_pregrasp_joint2_max_step_rad_);
    const double desired_joint4 =
      joint4ForPregraspToolPitch(target, desired_tool_pitch);
    target[3] = current[3] +
      clampStep(desired_joint4 - current[3], joint_pregrasp_joint4_max_step_rad_);
    holdExtensionJoint4(target, current);

    target[0] = current[0]; // ignore aruco correction during joint2 step for smoother motion

    return clampJointPregraspTarget(target);
  }

  std::array<double, 4> jointNudgeRawTargetFromControllerTarget(
    const std::array<double, 4> & current,
    const std::array<double, 4> & controller_target) const
  {
    std::array<double, 4> target = controller_target;
    holdExtensionJoint4(target, current);
    if (joint_pregrasp_reverse_joint3_delta_) {
      target[2] = current[2] - (controller_target[2] - current[2]);
    }
    target[0] = clampValue(target[0], joint_pregrasp_min_positions_[0], joint_pregrasp_max_positions_[0]);
    target[1] = clampValue(target[1], joint_pregrasp_min_positions_[1], joint_pregrasp_max_positions_[1]);
    target[2] = clampValue(target[2], joint_pregrasp_min_positions_[2], joint_pregrasp_max_positions_[2]);
    target[3] = clampValue(target[3], joint_pregrasp_min_positions_[3], joint_pregrasp_max_positions_[3]);
    return target;
  }

  void appendJointTrajectoryPoint(
    trajectory_msgs::msg::JointTrajectory & msg,
    const std::array<double, 4> & positions,
    double time_from_start_s) const
  {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.assign(positions.begin(), positions.end());
    point.time_from_start = durationFromSeconds(time_from_start_s);
    msg.points.push_back(point);
  }

  trajectory_msgs::msg::JointTrajectory makeJointPregraspTrajectory(
    const std::array<double, 4> & current,
    const std::array<double, 4> & target)
  {
    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = now();
    msg.joint_names.assign(arm_joint_names_.begin(), arm_joint_names_.end());

    auto append_held = [&](std::array<double, 4> positions, double stamp) {
      holdExtensionJoint4(positions, current);
      appendJointTrajectoryPoint(msg, positions, stamp);
    };

    double trajectory_time_s = 0.0;
    if (joint_pregrasp_hold_current_duration_s_ > 0.0) {
      trajectory_time_s += joint_pregrasp_hold_current_duration_s_;
      append_held(clampJointPregraspTarget(current), trajectory_time_s);
    }

    const bool use_joint3_lead =
      joint_pregrasp_joint3_lead_enabled_ &&
      std::abs(target[2] - current[2]) > joint_pregrasp_tolerance_rad_;
    if (use_joint3_lead) {
      std::array<double, 4> joint3_lead = current;
      joint3_lead[2] = current[2] +
        (target[2] - current[2]) * joint_pregrasp_joint3_lead_fraction_;
      append_held(
        clampJointPregraspTarget(joint3_lead),
        trajectory_time_s +
        joint_pregrasp_move_duration_s_ * joint_pregrasp_joint3_lead_duration_ratio_);
    }

    const int sync_steps = std::max(1, joint_pregrasp_sync_steps_);
    for (int step = 1; step <= sync_steps; ++step) {
      const double ratio = static_cast<double>(step) / static_cast<double>(sync_steps);
      std::array<double, 4> waypoint{};
      for (std::size_t i = 0; i < waypoint.size(); ++i) {
        waypoint[i] = current[i] + (target[i] - current[i]) * ratio;
      }
      append_held(
        clampJointPregraspTarget(waypoint),
        trajectory_time_s + joint_pregrasp_move_duration_s_ * ratio);
    }

    return msg;
  }

  void publishJointPregraspTrajectory(
    const std::array<double, 4> & current,
    const std::array<double, 4> & target)
  {
    joint_trajectory_pub_->publish(makeJointPregraspTrajectory(current, target));
  }

  double maxJointError(
    const std::array<double, 4> & current,
    const std::array<double, 4> & target) const
  {
    double max_error = 0.0;
    for (std::size_t i = 0; i < current.size(); ++i) {
      max_error = std::max(max_error, std::abs(current[i] - target[i]));
    }
    return max_error;
  }

  std::array<double, 4> jointErrors(
    const std::array<double, 4> & current,
    const std::array<double, 4> & target) const
  {
    std::array<double, 4> errors{};
    for (std::size_t i = 0; i < current.size(); ++i) {
      errors[i] = target[i] - current[i];
    }
    return errors;
  }

  std::array<double, 3> rpyFromTransform(
    const geometry_msgs::msg::TransformStamped & transform_msg) const
  {
    tf2::Quaternion q;
    tf2::fromMsg(transform_msg.transform.rotation, q);
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    return {roll, pitch, yaw};
  }

  PoseStatus poseStatusFromTransform(
    const std::string & frame,
    const geometry_msgs::msg::TransformStamped & transform_msg) const
  {
    const auto rpy = rpyFromTransform(transform_msg);
    PoseStatus pose;
    pose.frame = frame;
    pose.valid = true;
    pose.x = transform_msg.transform.translation.x;
    pose.y = transform_msg.transform.translation.y;
    pose.z = transform_msg.transform.translation.z;
    pose.roll = rpy[0];
    pose.pitch = rpy[1];
    pose.yaw = rpy[2];
    return pose;
  }

  PoseStatus lookupPoseStatus(const std::string & frame)
  {
    PoseStatus pose;
    pose.frame = frame;
    if (frame.empty()) {
      pose.reason = "empty_frame";
      return pose;
    }

    try {
      const auto transform_msg =
        tf_buffer_.lookupTransform(target_frame_, frame, tf2::TimePointZero);
      return poseStatusFromTransform(frame, transform_msg);
    } catch (const tf2::TransformException & ex) {
      pose.reason = ex.what();
      return pose;
    }
  }

  std::string formatPoseStatus(const PoseStatus & pose) const
  {
    std::ostringstream out;
    out << pose.frame << ":";
    if (!pose.valid) {
      out << "unavailable";
      if (!pose.reason.empty()) {
        out << "(" << pose.reason << ")";
      }
      return out.str();
    }

    out << std::fixed << std::setprecision(3)
        << "xyz=(" << pose.x << ", " << pose.y << ", " << pose.z << ")"
        << " rpy=(" << pose.roll << ", " << pose.pitch << ", " << pose.yaw << ")";
    return out.str();
  }

  std::string poseStatusSuffix(const geometry_msgs::msg::TransformStamped & eef_tf)
  {
    std::ostringstream out;
    out << " eef_pose="
        << formatPoseStatus(poseStatusFromTransform(end_effector_frame_, eef_tf))
        << " joint4_pose=" << formatPoseStatus(lookupPoseStatus(joint4_pose_frame_));

    if (gripper_pose_frame_ == end_effector_frame_) {
      out << " gripper_pose="
          << formatPoseStatus(poseStatusFromTransform(gripper_pose_frame_, eef_tf));
    } else {
      out << " gripper_pose=" << formatPoseStatus(lookupPoseStatus(gripper_pose_frame_));
    }
    return out.str();
  }

  std::string poseStatusSuffix()
  {
    std::ostringstream out;
    out << " eef_pose=" << formatPoseStatus(lookupPoseStatus(end_effector_frame_))
        << " joint4_pose=" << formatPoseStatus(lookupPoseStatus(joint4_pose_frame_))
        << " gripper_pose=" << formatPoseStatus(lookupPoseStatus(rpyControlFrame()));
    return out.str();
  }

  geometry_msgs::msg::TransformStamped gripperPoseTransformOrEef(
    const geometry_msgs::msg::TransformStamped & eef_tf)
  {
    if (gripper_pose_frame_.empty() || gripper_pose_frame_ == end_effector_frame_) {
      return eef_tf;
    }

    try {
      return tf_buffer_.lookupTransform(target_frame_, gripper_pose_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "gripper pose TF unavailable, falling back to %s for RPY control: %s",
        end_effector_frame_.c_str(), ex.what());
      return eef_tf;
    }
  }

  std::string rpyControlFrame() const
  {
    return gripper_pose_frame_.empty() ? end_effector_frame_ : gripper_pose_frame_;
  }

  void resetEefRefinementMotionState()
  {
    eef_rpy_reference_.reset();
    eef_stay_roll_reference_.reset();
    eef_forward_advance_active_ = false;
    eef_forward_start_x_m_ = 0.0;
    eef_forward_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    eef_forward_start_joint_positions_.reset();
    eef_forward_last_joint_nudge_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    front_bbox_area_at_eef_forward_start_.reset();
    eef_bbox_area_at_eef_forward_start_.reset();
  }

  void abortEefForwardAdvance(
    const std::string & reason,
    double advanced_x,
    double elapsed_s,
    const EefForwardJointProgress & joint_progress,
    const geometry_msgs::msg::TransformStamped & eef_tf)
  {
    safety_abort_latched_ = true;
    safety_abort_ignore_reported_ = false;
    active_ = false;
    done_ = false;
    close_sent_ = false;
    open_sent_ = false;
    eef_forward_advance_active_ = false;
    stable_cycles_ = 0;
    object_pregrasp_horizontal_done_ = false;
    joint_pregrasp_sent_ = false;
    joint_pregrasp_done_ = false;
    extension_joint4_reference_.reset();
    joint_pregrasp_target_.reset();
    joint_pregrasp_controller_target_.reset();
    joint_pregrasp_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    joint_pregrasp_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    eef_forward_last_joint_nudge_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    eef_forward_start_joint_positions_.reset();
    handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_lift_controller_target_.reset();
    handoff_place_controller_target_.reset();
    handoff_stay_controller_target_.reset();
    handoff_joint1_target_rad_.reset();
    handoff_joint1_centering_ = false;
    handoff_joint1_centered_for_rotate_ = false;
    publishStop();
    publishBaseStop();
    publishBaseHold(true);
    publishEefAutoInitEnable(false);

    std::ostringstream status;
    status << reason
           << " advanced_x=" << advanced_x
           << "/" << eef_forward_distance_m_
           << " min_close_advance=" << eef_forward_min_advance_before_close_m_
           << " elapsed=" << elapsed_s
           << "/" << eef_forward_gate_timeout_s_
           << formatEefForwardJointProgress(joint_progress)
           << poseStatusSuffix(eef_tf);
    publishStatus(status.str(), true);
  }

  void captureEefStayRollReference(const geometry_msgs::msg::TransformStamped & eef_tf)
  {
    if (!use_eef_rpy_refinement_ || !eef_hold_stay_roll_ || eef_stay_roll_reference_) {
      return;
    }
    eef_stay_roll_reference_ = rpyFromTransform(eef_tf)[0];
  }

  void captureEefRpyReference(const geometry_msgs::msg::TransformStamped & eef_tf)
  {
    if (!use_eef_rpy_refinement_ || !eef_hold_current_rpy_ || eef_rpy_reference_) {
      return;
    }
    eef_rpy_reference_ = rpyFromTransform(eef_tf);
  }

  RpyError computeEefRpyError(const geometry_msgs::msg::TransformStamped & eef_tf) const
  {
    RpyError error;
    if (!use_eef_rpy_refinement_) {
      return error;
    }

    const auto current = rpyFromTransform(eef_tf);
    std::array<double, 3> target{
      eef_target_roll_rad_,
      eef_target_pitch_rad_,
      eef_target_yaw_rad_};
    if (eef_hold_current_rpy_) {
      target = eef_rpy_reference_.value_or(current);
    }
    if (eef_hold_stay_roll_) {
      target[0] = eef_stay_roll_reference_.value_or(current[0]);
      error.using_stay_roll = eef_stay_roll_reference_.has_value();
    }

    error.roll = normalizeAngle(target[0] - current[0]);
    error.pitch = normalizeAngle(target[1] - current[1]);
    error.yaw = normalizeAngle(target[2] - current[2]);
    error.norm = vectorNorm(error.roll, error.pitch, error.yaw);
    error.ready =
      std::abs(error.roll) <= eef_rpy_tolerance_rad_ &&
      std::abs(error.pitch) <= eef_rpy_tolerance_rad_ &&
      std::abs(error.yaw) <= eef_rpy_tolerance_rad_;
    return error;
  }

  std::string rpyReferenceMode(const RpyError & rpy_error) const
  {
    if (!use_eef_rpy_refinement_) {
      return "disabled";
    }
    if (rpy_error.using_stay_roll) {
      return "stay_roll";
    }
    return eef_hold_current_rpy_ ? "current_rpy" : "configured_rpy";
  }

  void applyEefRpyCorrection(
    const RpyError & rpy_error,
    geometry_msgs::msg::TwistStamped & cmd) const
  {
    if (!use_eef_rpy_refinement_) {
      return;
    }
    cmd.twist.angular.x = clampValue(
      eef_rpy_gain_ * rpy_error.roll,
      -eef_refine_max_angular_speed_, eef_refine_max_angular_speed_);
    cmd.twist.angular.y = clampValue(
      eef_rpy_gain_ * rpy_error.pitch,
      -eef_refine_max_angular_speed_, eef_refine_max_angular_speed_);
    cmd.twist.angular.z = clampValue(
      eef_rpy_gain_ * rpy_error.yaw,
      -eef_refine_max_angular_speed_, eef_refine_max_angular_speed_);
  }

  void publishEefForwardAdvanceCommand(
    const RpyError & rpy_error,
    bool allow_joint_nudge_after_joint3_complete)
  {
    if (
      eef_forward_use_joint_nudge_ &&
      publishEefForwardJointNudge(rpy_error, allow_joint_nudge_after_joint3_complete))
    {
      return;
    }

    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = target_frame_;
    cmd.twist.linear.x = eef_forward_speed_mps_;
    cmd.twist.linear.y = 0.0;
    cmd.twist.linear.z = 0.0;
    applyEefRpyCorrection(rpy_error, cmd);
    twist_pub_->publish(cmd);
  }

  bool publishEefForwardAnalyticIkNudge(
    const RpyError & rpy_error,
    bool allow_after_joint3_complete)
  {
    const auto stamp = now();
    const double effective_nudge_period_s = std::max(
      eef_forward_joint_nudge_period_s_,
      eef_forward_joint_nudge_duration_s_);
    if (eef_forward_last_joint_nudge_stamp_.nanoseconds() != 0 &&
        (stamp - eef_forward_last_joint_nudge_stamp_).seconds() <
        effective_nudge_period_s) {
      return true;
    }

    const auto current = latestArmJointPositions();
    if (!current) {
      return false;
    }

    if (!eef_forward_start_joint_positions_) {
      eef_forward_start_joint_positions_ = *current;
    }
    if (hold_joint4_during_extension_ && !extension_joint4_reference_) {
      extension_joint4_reference_ = (*current)[3];
    }
    const auto & start = *eef_forward_start_joint_positions_;
    const auto progress = eefForwardJointProgressFromCurrent(*current);

    const PlanarIkPose start_pose = eefForwardIkPoseFromJoints(start);
    const PlanarIkPose current_pose = eefForwardIkPoseFromJoints(*current);
    const double current_advance_m =
      eef_forward_distance_m_ > 1.0e-6 ?
      clampValue(current_pose.x - start_pose.x, 0.0, eef_forward_distance_m_) :
      0.0;
    const double current_progress_ratio =
      eef_forward_distance_m_ > 1.0e-6 ?
      clampValue(current_advance_m / eef_forward_distance_m_, 0.0, 1.0) :
      1.0;
    const double elapsed_s =
      eef_forward_start_stamp_.nanoseconds() == 0 ?
      0.0 :
      (stamp - eef_forward_start_stamp_).seconds();
    const double time_progress_ratio = eef_forward_bezier_total_duration_s_ > 1.0e-6 ?
      clampValue(elapsed_s / eef_forward_bezier_total_duration_s_, 0.0, 1.0) :
      1.0;
    const double target_progress_ratio = clampValue(
      std::max(current_progress_ratio + eef_forward_bezier_progress_step_, time_progress_ratio),
      0.0,
      1.0);
    const double target_advance_m = eef_forward_distance_m_ * target_progress_ratio;

    const auto ik = solveEefForwardPlanarIk(start, *current, target_advance_m);
    const double eased_progress = bezierEase01(target_progress_ratio);
    const double joint2_total_direction =
      eef_forward_joint2_delta_rad_ < 0.0 ? -1.0 : 1.0;
    const double joint3_direction = eefForwardJoint3Direction();
    std::array<double, 4> coupled_waypoint = start;
    coupled_waypoint[1] = start[1] +
      joint2_total_direction * std::abs(eef_forward_bezier_joint2_total_delta_rad_) * eased_progress;
    coupled_waypoint[2] = start[2] +
      joint3_direction * std::abs(eef_forward_bezier_joint3_total_delta_rad_) * eased_progress;
    coupled_waypoint[3] = start[3] -
      ((coupled_waypoint[1] - start[1]) + (coupled_waypoint[2] - start[2]));
    coupled_waypoint = clampJointPregraspTarget(coupled_waypoint);

    std::array<double, 4> synchronized_ik_target = ik.controller_target;
    bool ik_joint_sync_fill = false;
    std::string ik_joint_sync_reason = "none";
    for (std::size_t i = 1; i <= 3; ++i) {
      const double ik_delta = synchronized_ik_target[i] - (*current)[i];
      const double coupled_delta = coupled_waypoint[i] - (*current)[i];
      const bool coupled_active = std::abs(coupled_delta) > 1.0e-6;
      const bool ik_missing = std::abs(ik_delta) < 1.0e-6;
      const bool ik_opposes_coupled =
        coupled_active && !ik_missing && ik_delta * coupled_delta < 0.0;
      if (coupled_active && (ik_missing || ik_opposes_coupled)) {
        synchronized_ik_target[i] = coupled_waypoint[i];
        ik_joint_sync_fill = true;
        if (ik_joint_sync_reason == "none") {
          ik_joint_sync_reason.clear();
        } else {
          ik_joint_sync_reason += ",";
        }
        ik_joint_sync_reason += arm_joint_names_[i];
        ik_joint_sync_reason += ik_missing ? ":missing" : ":opposite";
      }
    }
    const std::string ik_monotonic_hold_reason =
      enforceEefForwardMonotonicTarget(*current, start, coupled_waypoint, synchronized_ik_target);

    // std::array<double, 4> controller_target = *current;
    // controller_target[1] = (*current)[1] + clampStep(
    //   synchronized_ik_target[1] - (*current)[1],
    //   eef_forward_bezier_joint2_max_step_rad_);
    // controller_target[2] = (*current)[2] + clampStep(
    //   synchronized_ik_target[2] - (*current)[2],
    //   eef_forward_bezier_joint3_max_step_rad_);
    // controller_target[3] = (*current)[3] + clampStep(
    //   coupled_waypoint[3] - (*current)[3],
    //   eef_forward_bezier_joint4_max_step_rad_);

    std::array<double, 4> controller_target = *current;

    const double next_joint2 = (*current)[1] + clampStep(
      synchronized_ik_target[1] - (*current)[1],
      eef_forward_bezier_joint2_max_step_rad_);

    const double next_joint3 = (*current)[2] + clampStep(
      synchronized_ik_target[2] - (*current)[2],
      eef_forward_bezier_joint3_max_step_rad_);

    controller_target[1] = next_joint2;
    controller_target[2] = next_joint3;

    const bool joint2_or_joint3_active =
      std::abs(next_joint2 - (*current)[1]) > 1.0e-6 ||
      std::abs(next_joint3 - (*current)[2]) > 1.0e-6;

    if (joint2_or_joint3_active) {
      controller_target[3] = (*current)[3] + clampStep(
        coupled_waypoint[3] - (*current)[3],
        eef_forward_bezier_joint4_max_step_rad_);
    } 
    else {
      controller_target[3] = (*current)[3];
    }

    holdExtensionJoint4(controller_target, *current);
    controller_target = clampJointPregraspTarget(controller_target);
    const std::string controller_monotonic_hold_reason =
      enforceEefForwardMonotonicTarget(*current, start, coupled_waypoint, controller_target);
    holdExtensionJoint4(controller_target, *current);

    const auto raw_target =
      jointNudgeRawTargetFromControllerTarget(*current, controller_target);
    const double controller_joint2_delta = controller_target[1] - (*current)[1];
    const double controller_joint3_delta = controller_target[2] - (*current)[2];
    const double controller_joint4_delta = controller_target[3] - (*current)[3];
    const double raw_joint3_delta = raw_target[2] - (*current)[2];

    const bool joint2_active = std::abs(controller_joint2_delta) > 1.0e-6;
    const bool joint3_active = std::abs(controller_joint3_delta) > 1.0e-6;
    const bool joint4_active = std::abs(controller_joint4_delta) > 1.0e-6;
    std::string joint_stage;
    if (joint2_active) {
      joint_stage += "joint2";
    }
    if (joint3_active) {
      joint_stage += joint_stage.empty() ? "joint3" : ",joint3";
    }
    if (joint4_active) {
      joint_stage += joint_stage.empty() ? "joint4" : ",joint4";
    }
    if (joint_stage.empty()) {
      joint_stage = "none";
    }

    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = stamp;
    msg.joint_names.assign(arm_joint_names_.begin(), arm_joint_names_.end());
    appendJointTrajectoryPoint(msg, raw_target, eef_forward_joint_nudge_duration_s_);
    joint_trajectory_pub_->publish(msg);
    eef_forward_last_joint_nudge_stamp_ = stamp;

    std::ostringstream status;
    status << "eef forward analytic ik nudge: current=" << formatJointArray(*current)
           << " start=" << formatJointArray(start)
           << " ik_solution=" << formatJointArray(ik.controller_target)
           << " synchronized_ik_target=" << formatJointArray(synchronized_ik_target)
           << " coupled_waypoint=" << formatJointArray(coupled_waypoint)
           << " controller_target=" << formatJointArray(controller_target)
           << " raw_target=" << formatJointArray(raw_target)
           << " simultaneous_joints=" << joint_stage
           << " ik_joint_sync_fill=" << (ik_joint_sync_fill ? "true" : "false")
           << " ik_joint_sync_reason=" << ik_joint_sync_reason
           << " ik_monotonic_hold=" << ik_monotonic_hold_reason
           << " controller_monotonic_hold=" << controller_monotonic_hold_reason
           << " allow_after_joint3_complete="
           << (allow_after_joint3_complete ? "true" : "false")
           << " target_advance=" << target_advance_m
           << "/" << eef_forward_distance_m_
           << " progress=(" << current_progress_ratio << " -> "
           << target_progress_ratio << ")"
           << " elapsed=" << elapsed_s
           << "/" << eef_forward_bezier_total_duration_s_
           << " ik_target=(x=" << ik.target_pose.x
           << ", z=" << ik.target_pose.z
           << ", pitch=" << ik.target_pose.pitch << ")"
           << " ik_solved=(x=" << ik.solved_pose.x
           << ", z=" << ik.solved_pose.z
           << ", pitch=" << ik.solved_pose.pitch << ")"
           << " ik_error=" << ik.error_norm
           << " ik_reason=" << ik.reason
           << " ik_iterations=" << ik.iterations
           << formatEefForwardJointProgress(progress)
           << " controller_joint2_delta=" << controller_joint2_delta
           << " controller_joint3_delta=" << controller_joint3_delta
           << " controller_joint4_delta=" << controller_joint4_delta
           << " raw_joint3_delta=" << raw_joint3_delta
           << " joint3_reverse="
           << (joint_pregrasp_reverse_joint3_delta_ ? "true" : "false")
           << " max_step=(" << eef_forward_bezier_joint2_max_step_rad_
           << ", " << eef_forward_bezier_joint3_max_step_rad_
           << ", " << eef_forward_bezier_joint4_max_step_rad_ << ")"
           << " rpy_roll_err=" << rpy_error.roll
           << " rpy_frame=" << rpyControlFrame()
           << " duration=" << eef_forward_joint_nudge_duration_s_
           << " configured_period=" << eef_forward_joint_nudge_period_s_
           << " effective_period=" << effective_nudge_period_s
           << poseStatusSuffix();
    publishStatus(status.str());
    return true;
  }

  bool publishEefForwardBezierIkNudge(
    const RpyError & rpy_error,
    bool allow_after_joint3_complete)
  {
    const auto stamp = now();
    const double effective_nudge_period_s = std::max(
      eef_forward_joint_nudge_period_s_,
      eef_forward_joint_nudge_duration_s_);
    if (eef_forward_last_joint_nudge_stamp_.nanoseconds() != 0 &&
        (stamp - eef_forward_last_joint_nudge_stamp_).seconds() <
        effective_nudge_period_s) {
      return true;
    }

    const auto current = latestArmJointPositions();
    if (!current) {
      return false;
    }

    if (!eef_forward_start_joint_positions_) {
      eef_forward_start_joint_positions_ = *current;
    }
    if (hold_joint4_during_extension_ && !extension_joint4_reference_) {
      extension_joint4_reference_ = (*current)[3];
    }
    const auto & start = *eef_forward_start_joint_positions_;
    const auto progress = eefForwardJointProgressFromCurrent(*current);

    const double joint2_total_direction =
      eef_forward_joint2_delta_rad_ < 0.0 ? -1.0 : 1.0;
    const double joint3_direction = eefForwardJoint3Direction();
    std::array<double, 4> controller_goal = start;
    controller_goal[1] = start[1] +
      joint2_total_direction * std::abs(eef_forward_bezier_joint2_total_delta_rad_);
    controller_goal[2] = start[2] +
      joint3_direction * std::abs(eef_forward_bezier_joint3_total_delta_rad_);
    if (!hold_joint4_during_extension_ && eef_forward_bezier_preserve_roll_ && joint_pregrasp_preserve_gripper_roll_) {
      controller_goal[3] = joint4ForPreservedEefForwardRoll(start, controller_goal);
    }
    controller_goal[0] =
      clampValue(controller_goal[0], joint_pregrasp_min_positions_[0], joint_pregrasp_max_positions_[0]);
    controller_goal[1] =
      clampValue(controller_goal[1], joint_pregrasp_min_positions_[1], joint_pregrasp_max_positions_[1]);
    controller_goal[2] =
      clampValue(controller_goal[2], joint_pregrasp_min_positions_[2], joint_pregrasp_max_positions_[2]);
    controller_goal[3] =
      clampValue(controller_goal[3], joint_pregrasp_min_positions_[3], joint_pregrasp_max_positions_[3]);

    const double joint3_total = std::abs(controller_goal[2] - start[2]);
    const double joint3_progress_ratio = joint3_total > 1.0e-6 ?
      clampValue(
        joint3_direction * ((*current)[2] - start[2]) / joint3_total,
        0.0,
        1.0) :
      1.0;
    const double elapsed_s =
      eef_forward_start_stamp_.nanoseconds() == 0 ?
      0.0 :
      (stamp - eef_forward_start_stamp_).seconds();
    const double time_progress_ratio = eef_forward_bezier_total_duration_s_ > 1.0e-6 ?
      clampValue(elapsed_s / eef_forward_bezier_total_duration_s_, 0.0, 1.0) :
      1.0;
    const double base_progress_ratio =
      std::max(joint3_progress_ratio, time_progress_ratio);
    const double target_progress_ratio = clampValue(
      base_progress_ratio + eef_forward_bezier_progress_step_,
      0.0,
      1.0);
    const double eased = bezierEase01(target_progress_ratio);

    std::array<double, 4> bezier_waypoint = start;
    for (std::size_t i = 0; i < bezier_waypoint.size(); ++i) {
      bezier_waypoint[i] = start[i] + (controller_goal[i] - start[i]) * eased;
    }

    std::array<double, 4> controller_target = *current;
    controller_target[1] = (*current)[1] + clampStep(
      bezier_waypoint[1] - (*current)[1],
      eef_forward_bezier_joint2_max_step_rad_);
    controller_target[2] = (*current)[2] + clampStep(
      bezier_waypoint[2] - (*current)[2],
      eef_forward_bezier_joint3_max_step_rad_);
    controller_target[3] = (*current)[3] + clampStep(
      bezier_waypoint[3] - (*current)[3],
      eef_forward_bezier_joint4_max_step_rad_);
    controller_target[0] =
      clampValue(controller_target[0], joint_pregrasp_min_positions_[0], joint_pregrasp_max_positions_[0]);
    controller_target[1] =
      clampValue(controller_target[1], joint_pregrasp_min_positions_[1], joint_pregrasp_max_positions_[1]);
    controller_target[2] =
      clampValue(controller_target[2], joint_pregrasp_min_positions_[2], joint_pregrasp_max_positions_[2]);
    controller_target[3] =
      clampValue(controller_target[3], joint_pregrasp_min_positions_[3], joint_pregrasp_max_positions_[3]);
    const std::string controller_monotonic_hold_reason =
      enforceEefForwardMonotonicTarget(*current, start, bezier_waypoint, controller_target);
    holdExtensionJoint4(controller_target, *current);

    const auto raw_target =
      jointNudgeRawTargetFromControllerTarget(*current, controller_target);
    const double controller_joint2_delta = controller_target[1] - (*current)[1];
    const double controller_joint3_delta = controller_target[2] - (*current)[2];
    const double controller_joint4_delta = controller_target[3] - (*current)[3];
    const double raw_joint3_delta = raw_target[2] - (*current)[2];

    const bool joint2_active = std::abs(controller_joint2_delta) > 1.0e-6;
    const bool joint3_active = std::abs(controller_joint3_delta) > 1.0e-6;
    const bool joint4_active = std::abs(controller_joint4_delta) > 1.0e-6;
    std::string joint_stage;
    if (joint2_active) {
      joint_stage += "joint2";
    }
    if (joint3_active) {
      joint_stage += joint_stage.empty() ? "joint3" : ",joint3";
    }
    if (joint4_active) {
      joint_stage += joint_stage.empty() ? "joint4" : ",joint4";
    }
    if (joint_stage.empty()) {
      joint_stage = "none";
    }

    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = stamp;
    msg.joint_names.assign(arm_joint_names_.begin(), arm_joint_names_.end());
    appendJointTrajectoryPoint(msg, raw_target, eef_forward_joint_nudge_duration_s_);
    joint_trajectory_pub_->publish(msg);
    eef_forward_last_joint_nudge_stamp_ = stamp;

    std::ostringstream status;
    status << "eef forward bezier ik nudge: current=" << formatJointArray(*current)
           << " start=" << formatJointArray(start)
           << " controller_goal=" << formatJointArray(controller_goal)
           << " bezier_waypoint=" << formatJointArray(bezier_waypoint)
           << " raw_target=" << formatJointArray(raw_target)
           << " controller_target=" << formatJointArray(controller_target)
           << " simultaneous_joints=" << joint_stage
           << " controller_monotonic_hold=" << controller_monotonic_hold_reason
           << " allow_after_joint3_complete="
           << (allow_after_joint3_complete ? "true" : "false")
           << " progress=(" << base_progress_ratio << " -> "
           << target_progress_ratio << ", eased=" << eased << ")"
           << " elapsed=" << elapsed_s
           << "/" << eef_forward_bezier_total_duration_s_
           << formatEefForwardJointProgress(progress)
           << " controller_joint2_delta=" << controller_joint2_delta
           << " controller_joint3_delta=" << controller_joint3_delta
           << " controller_joint4_delta=" << controller_joint4_delta
           << " raw_joint3_delta=" << raw_joint3_delta
           << " max_step=(" << eef_forward_bezier_joint2_max_step_rad_
           << ", " << eef_forward_bezier_joint3_max_step_rad_
           << ", " << eef_forward_bezier_joint4_max_step_rad_ << ")"
           << " preserve_roll="
           << (eef_forward_bezier_preserve_roll_ ? "true" : "false")
           << " rpy_roll_err=" << rpy_error.roll
           << " rpy_frame=" << rpyControlFrame()
           << " duration=" << eef_forward_joint_nudge_duration_s_
           << " configured_period=" << eef_forward_joint_nudge_period_s_
           << " effective_period=" << effective_nudge_period_s
           << poseStatusSuffix();
    publishStatus(status.str());
    return true;
  }

  bool publishEefForwardJointNudge(
    const RpyError & rpy_error,
    bool allow_after_joint3_complete)
  {
    if (eef_forward_use_analytic_ik_) {
      return publishEefForwardAnalyticIkNudge(rpy_error, allow_after_joint3_complete);
    }
    if (eef_forward_use_bezier_ik_) {
      return publishEefForwardBezierIkNudge(rpy_error, allow_after_joint3_complete);
    }

    publishStatus("eef forward joint nudge disabled: legacy direct joint-delta path is commented out");
    return false;
  }

  void finishJointPregrasp()
  {
    if (pause_before_grasp_ && !grasp_resume_requested_) {
      publishStatus("pregrasp_ready: arm deployed; checking grasp height before closing", !pregrasp_pause_reported_);
      pregrasp_pause_reported_ = true;
      return;
    }
    if (close_gripper_on_arrival_ && !close_sent_) {
      sendGripperGraspForObject();
      close_sent_ = true;
    }
    completeGrasp("joint pregrasp complete; gripper closed; starting inward retraction");
  }

  bool updateJointPregrasp()
  {
    if (joint_pregrasp_done_) {
      return true;
    }

    const auto stamp = now();
    if (joint_pregrasp_sent_) {
      const double elapsed_s = (stamp - joint_pregrasp_start_stamp_).seconds();
      const double wait_s =
        joint_pregrasp_hold_current_duration_s_ +
        joint_pregrasp_move_duration_s_ +
        joint_pregrasp_settle_s_;

      const auto current = latestArmJointPositions();
      const bool can_check_reached =
        current && joint_pregrasp_controller_target_;

      double max_error = -1.0;
      bool pregrasp_reached = false;

      if (can_check_reached) {
        const auto final_errors =
          jointErrors(*current, *joint_pregrasp_controller_target_);

        const bool joint1_reached =
          std::abs(final_errors[0]) <= joint_pregrasp_tolerance_rad_;
        const bool joint2_reached =
          std::abs(final_errors[1]) <= joint_pregrasp_tolerance_rad_;
        const bool joint3_reached =
          std::abs(final_errors[2]) <= joint_pregrasp_tolerance_rad_;

        bool joint4_step_reached = true;
        double joint4_step_error = 0.0;

        if (joint_pregrasp_controller_step_target_) {
          joint4_step_error =
            (*joint_pregrasp_controller_step_target_)[3] - (*current)[3];
          joint4_step_reached =
            std::abs(joint4_step_error) <= joint_pregrasp_tolerance_rad_;
        }

        max_error = std::max({
          std::abs(final_errors[0]),
          std::abs(final_errors[1]),
          std::abs(final_errors[2]),
          std::abs(joint4_step_error)
        });

        pregrasp_reached =
          joint1_reached &&
          joint2_reached &&
          joint3_reached &&
          joint4_step_reached;
      }

      if (elapsed_s >= wait_s && can_check_reached && pregrasp_reached) {
        joint_pregrasp_done_ = true;
        std::ostringstream status;
        status << "joint pregrasp complete; primary joints reached and joint4 step reached"
              << " max_checked_err=" << max_error
              << " <= " << joint_pregrasp_tolerance_rad_
              << "; switching to EEF refinement";
        publishStatus(status.str(), true);
        return true;
      }

      if (elapsed_s >= wait_s && can_check_reached &&
          (stamp - joint_pregrasp_last_publish_stamp_).seconds() >=
          std::max(joint_pregrasp_republish_period_s_,
            joint_pregrasp_hold_current_duration_s_ + joint_pregrasp_move_duration_s_ +
            joint_pregrasp_settle_s_)) {
        bool joint3_step_active = false;
        const auto controller_step_target =
          jointPregraspControllerStepTarget(
          *current, *joint_pregrasp_controller_target_, &joint3_step_active);
        const auto raw_target =
          jointPregraspRawTargetFromControllerTarget(*current, controller_step_target);
        publishJointPregraspTrajectory(*current, raw_target);
        joint_pregrasp_target_ = raw_target;
        joint_pregrasp_controller_step_target_ = controller_step_target;
        joint_pregrasp_last_publish_stamp_ = stamp;
      }

      std::ostringstream status;
      status << "waiting for joint pregrasp trajectory: elapsed="
             << elapsed_s << "/" << wait_s;
      if (can_check_reached) {
        const auto errors = jointErrors(*current, *joint_pregrasp_controller_target_);
        status << " max_joint_err=" << max_error
               << " tolerance=" << joint_pregrasp_tolerance_rad_
               << " joint_err=" << formatJointArray(errors)
               << " current=" << formatJointArray(*current)
               << " controller_target=" << formatJointArray(*joint_pregrasp_controller_target_)
               << " joint3_remaining="
               << ((*joint_pregrasp_controller_target_)[2] - (*current)[2])
               << " joint3_max_step=" << joint_pregrasp_joint3_max_step_rad_
               << poseStatusSuffix();
        if (joint_pregrasp_controller_step_target_) {
          const auto step_errors =
            jointErrors(*current, *joint_pregrasp_controller_step_target_);
          status << " controller_step_target="
                 << formatJointArray(*joint_pregrasp_controller_step_target_)
                 << " step_joint_err=" << formatJointArray(step_errors);
        }
        if (joint_pregrasp_target_) {
          status << " raw_target=" << formatJointArray(*joint_pregrasp_target_);
        }
      } else {
        status << " waiting_for_joint_state_or_target";
      }
      publishStatus(status.str());
      return false;
    }

    auto current = latestArmJointPositions();
    if (!current) {
      publishStatus("waiting for arm joint_states before joint pregrasp");
      return false;
    }

    if (hold_joint4_during_extension_ && !extension_joint4_reference_) {
      extension_joint4_reference_ = (*current)[3];
      RCLCPP_INFO(get_logger(), "Joint4 camera hold enabled for extension: %.6f rad", *extension_joint4_reference_);
    }
    const auto controller_target = jointPregraspControllerTargetFromCurrent(*current);
    bool joint3_step_active = false;
    const auto controller_step_target =
      jointPregraspControllerStepTarget(*current, controller_target, &joint3_step_active);
    const auto raw_target =
      jointPregraspRawTargetFromControllerTarget(*current, controller_step_target);
    publishJointPregraspTrajectory(*current, raw_target);
    joint_pregrasp_sent_ = true;
    joint_pregrasp_target_ = raw_target;
    joint_pregrasp_controller_target_ = controller_target;
    joint_pregrasp_controller_step_target_ = controller_step_target;
    joint_pregrasp_start_stamp_ = stamp;
    joint_pregrasp_last_publish_stamp_ = stamp;
    object_pregrasp_horizontal_done_ = true;

    std::ostringstream status;
    status << "moving arm to joint pregrasp: current=" << formatJointArray(*current)
           << " raw_target=" << formatJointArray(raw_target)
           << " controller_target=" << formatJointArray(controller_target)
           << " controller_step_target=" << formatJointArray(controller_step_target)
           << " joint3_step_active=" << (joint3_step_active ? "true" : "false")
           << " joint3_max_step=" << joint_pregrasp_joint3_max_step_rad_
           << " preserve_gripper_roll="
           << (joint_pregrasp_preserve_gripper_roll_ ? "true" : "false")
           << " hold_joint4_during_extension=" << (hold_joint4_during_extension_ ? "true" : "false")
           << " joint4_hold_reference=" << extension_joint4_reference_.value_or((*current)[3])
           << " pregrasp_roll_weights=(" << pregrasp_roll_joint2_weight_ << ", "
           << pregrasp_roll_joint3_weight_ << ", "
           << pregrasp_roll_joint4_weight_ << ")"
           << " joint4_down_offset=" << gripper_down_joint4_offset_rad_
           << " sync_steps=" << joint_pregrasp_sync_steps_
           << " joint3_lead_enabled="
           << (joint_pregrasp_joint3_lead_enabled_ ? "true" : "false")
           << " joint3_lead_duration_ratio=" << joint_pregrasp_joint3_lead_duration_ratio_
           << " joint3_lead_fraction=" << joint_pregrasp_joint3_lead_fraction_
           << " joint3_reverse_delta="
           << (joint_pregrasp_reverse_joint3_delta_ ? "true" : "false")
           << poseStatusSuffix();
    publishStatus(status.str(), true);
    return false;
  }

  bool moveArmToObjectPregraspPose(
    const geometry_msgs::msg::TransformStamped & eef_tf,
    const geometry_msgs::msg::PointStamped & object_in_target,
    bool * command_published)
  {
    const double target_x = object_in_target.point.x - object_pregrasp_standoff_m_ + grasp_offset_x_;
    const double target_y = object_in_target.point.y + grasp_offset_y_;

    const double eef_x = eef_tf.transform.translation.x;
    const double eef_y = eef_tf.transform.translation.y;
    const double eef_z = eef_tf.transform.translation.z;
    const double horizontal_err = vectorNorm(target_x - eef_x, target_y - eef_y, 0.0);
    const double safe_target_z = std::max(eef_z, object_pregrasp_min_z_m_);
    if (!object_pregrasp_horizontal_done_ &&
        horizontal_err <= triangulation_extend_tolerance_m_ &&
        std::abs(safe_target_z - eef_z) <= triangulation_extend_tolerance_m_) {
      object_pregrasp_horizontal_done_ = true;
    }
    if (object_pregrasp_horizontal_done_ && !object_pregrasp_enable_lowering_) {
      return true;
    }

    const double target_z = object_pregrasp_horizontal_done_ && object_pregrasp_enable_lowering_ ?
      std::max(
        object_in_target.point.z + grasp_offset_z_ + object_pregrasp_lower_standoff_m_,
        object_pregrasp_min_lower_z_m_) :
      safe_target_z;
    const double err_x = target_x - eef_x;
    const double err_y = target_y - eef_y;
    const double err_z = target_z - eef_z;
    const double err_norm = vectorNorm(err_x, err_y, err_z);

    if (err_norm <= triangulation_extend_tolerance_m_) {
      return true;
    }

    stage_ = GraspStage::TRIANGULATION_EXTEND;
    stable_cycles_ = 0;

    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = target_frame_;
    cmd.twist.linear.x = clampValue(
      triangulation_extend_gain_ * err_x,
      -triangulation_extend_max_speed_, triangulation_extend_max_speed_);
    cmd.twist.linear.y = clampValue(
      triangulation_extend_gain_ * err_y,
      -triangulation_extend_max_speed_, triangulation_extend_max_speed_);
    cmd.twist.linear.z = clampValue(
      triangulation_extend_gain_ * err_z,
      -triangulation_extend_max_speed_, triangulation_extend_max_speed_);
    twist_pub_->publish(cmd);
    if (command_published) {
      *command_published = true;
    }

    std::ostringstream status;
    status << (object_pregrasp_horizontal_done_ ?
      "lowering arm toward depth object: target=(" :
      "extending arm horizontally toward depth object: target=(")
           << target_x << ", " << target_y << ", " << target_z
           << ") err=(" << err_x << ", " << err_y << ", " << err_z
           << ") norm=" << err_norm;
    publishStatus(status.str());
    return false;
  }

  bool moveArmToTriangulationPose(
    const geometry_msgs::msg::TransformStamped & eef_tf,
    bool * command_published)
  {
    const double eef_x = eef_tf.transform.translation.x;
    const double eef_y = eef_tf.transform.translation.y;
    const double eef_z = eef_tf.transform.translation.z;
    const double err_x = triangulation_extend_x_m_ - eef_x;
    const double err_y = triangulation_extend_y_m_ - eef_y;
    const double err_z = triangulation_extend_z_m_ - eef_z;
    const double err_norm = vectorNorm(err_x, err_y, err_z);

    if (err_norm <= triangulation_extend_tolerance_m_) {
      if (stage_ == GraspStage::TRIANGULATION_EXTEND) {
        stage_ = GraspStage::DEPTH_APPROACH;
      }
      return true;
    }

    stage_ = GraspStage::TRIANGULATION_EXTEND;
    stable_cycles_ = 0;

    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = target_frame_;
    cmd.twist.linear.x = clampValue(
      triangulation_extend_gain_ * err_x,
      -triangulation_extend_max_speed_, triangulation_extend_max_speed_);
    cmd.twist.linear.y = clampValue(
      triangulation_extend_gain_ * err_y,
      -triangulation_extend_max_speed_, triangulation_extend_max_speed_);
    cmd.twist.linear.z = clampValue(
      triangulation_extend_gain_ * err_z,
      -triangulation_extend_max_speed_, triangulation_extend_max_speed_);
    twist_pub_->publish(cmd);
    if (command_published) {
      *command_published = true;
    }

    std::ostringstream status;
    status << "extending arm for stereo triangulation: err=("
           << err_x << ", " << err_y << ", " << err_z
           << ") norm=" << err_norm;
    publishStatus(status.str());
    return false;
  }

  std::optional<geometry_msgs::msg::PointStamped> triangulateObjectPoint(
    std::string * block_reason,
    bool publish_triangulation_status = true)
  {
    Bbox front_bbox;
    Bbox eef_bbox;
    CameraInfo front_info;
    CameraInfo eef_info;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!latest_bbox_) {
        setBlockReason(block_reason, bboxInputDescription());
        return std::nullopt;
      }
      if (!latest_camera_info_) {
        setBlockReason(block_reason, "front camera info on " + camera_info_topic_);
        return std::nullopt;
      }
      if (!latest_eef_bbox_) {
        setBlockReason(block_reason, "end-effector bbox for stereo triangulation");
        return std::nullopt;
      }
      front_bbox = *latest_bbox_;
      eef_bbox = *latest_eef_bbox_;
      front_info = *latest_camera_info_;
      if (latest_eef_camera_info_) {
        eef_info = *latest_eef_camera_info_;
      } else {
        auto fallback_info = fallbackEefCameraInfo();
        if (!fallback_info) {
          setBlockReason(block_reason, "end-effector camera info");
          return std::nullopt;
        }
        eef_info = *fallback_info;
      }
    }

    const auto stamp = now();
    if ((stamp - front_bbox.stamp).seconds() > max_target_age_s_) {
      setBlockReason(block_reason, "fresh " + bboxInputDescription());
      return std::nullopt;
    }
    if ((stamp - eef_bbox.stamp).seconds() > max_target_age_s_) {
      setBlockReason(block_reason, "fresh end-effector bbox for stereo triangulation");
      return std::nullopt;
    }

    const std::string front_frame =
      camera_frame_override_.empty() ? front_info.frame_id : camera_frame_override_;
    const std::string eef_frame =
      eef_camera_frame_override_.empty() ? eef_info.frame_id : eef_camera_frame_override_;
    const double front_u = front_bbox.x + 0.5 * front_bbox.width;
    const double front_v = front_bbox.y + 0.5 * front_bbox.height;
    const double eef_u = eef_bbox.x + 0.5 * eef_bbox.width;
    const double eef_v = eef_bbox.y + 0.5 * eef_bbox.height;

    auto front_ray = cameraPixelRayInTarget(front_frame, front_info, front_u, front_v, block_reason);
    if (!front_ray) {
      return std::nullopt;
    }
    auto eef_ray = cameraPixelRayInTarget(eef_frame, eef_info, eef_u, eef_v, block_reason);
    if (!eef_ray) {
      return std::nullopt;
    }
    if (use_eef_front_camera_extrinsic_override_) {
      eef_ray->origin = front_ray->origin + tf2::Vector3(
        eef_front_camera_offset_x_m_,
        eef_front_camera_offset_y_m_,
        eef_front_camera_offset_z_m_);
    }

    const tf2::Vector3 w0 = front_ray->origin - eef_ray->origin;
    const double a = front_ray->direction.dot(front_ray->direction);
    const double b = front_ray->direction.dot(eef_ray->direction);
    const double c = eef_ray->direction.dot(eef_ray->direction);
    const double d = front_ray->direction.dot(w0);
    const double e = eef_ray->direction.dot(w0);
    const double denom = a * c - b * b;
    if (std::abs(denom) < 1e-6) {
      setBlockReason(block_reason, "non-parallel camera rays for stereo triangulation");
      return std::nullopt;
    }

    const double front_range = (b * e - c * d) / denom;
    const double eef_range = (a * e - b * d) / denom;
    if (front_range < triangulation_min_range_m_ || front_range > triangulation_max_range_m_ ||
        eef_range < triangulation_min_range_m_ || eef_range > triangulation_max_range_m_) {
      std::ostringstream reason;
      reason << "triangulation range front=" << front_range << " eef=" << eef_range;
      setBlockReason(block_reason, reason.str());
      return std::nullopt;
    }

    const tf2::Vector3 front_point = front_ray->origin + front_ray->direction * front_range;
    const tf2::Vector3 eef_point = eef_ray->origin + eef_ray->direction * eef_range;
    const double ray_gap = (front_point - eef_point).length();
    if (ray_gap > triangulation_max_ray_gap_m_) {
      std::ostringstream reason;
      reason << "triangulation ray gap " << ray_gap;
      setBlockReason(block_reason, reason.str());
      return std::nullopt;
    }

    const tf2::Vector3 object = (front_point + eef_point) * 0.5;
    if (!std::isfinite(object.x()) || !std::isfinite(object.y()) || !std::isfinite(object.z())) {
      setBlockReason(block_reason, "finite stereo triangulation result");
      return std::nullopt;
    }

    // EEF RGB is alignment-only; gripper width is updated only from front depth.

    geometry_msgs::msg::PointStamped point;
    point.header.stamp = stamp;
    point.header.frame_id = target_frame_;
    point.point.x = object.x();
    point.point.y = object.y();
    point.point.z = object.z();

    if (publish_triangulation_status) {
      std::ostringstream status;
      status << "stereo triangulation object=(" << point.point.x << ", "
             << point.point.y << ", " << point.point.z
             << ") ray_gap=" << ray_gap
             << " eef_front_offset=(" << eef_front_camera_offset_x_m_ << ", "
             << eef_front_camera_offset_y_m_ << ", " << eef_front_camera_offset_z_m_ << ")"
             << " offset_override=" << (use_eef_front_camera_extrinsic_override_ ? "true" : "false");
      publishStatus(status.str());
    }
    return point;
  }

  std::optional<Ray> cameraPixelRayInTarget(
    const std::string & camera_frame,
    const CameraInfo & info,
    double u,
    double v,
    std::string * block_reason)
  {
    if (camera_frame.empty()) {
      setBlockReason(block_reason, "camera frame for stereo triangulation");
      return std::nullopt;
    }

    geometry_msgs::msg::TransformStamped transform_msg;
    try {
      transform_msg = tf_buffer_.lookupTransform(target_frame_, camera_frame, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      setBlockReason(block_reason, "TF from " + camera_frame + " to " + target_frame_);
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "stereo triangulation TF unavailable: %s", ex.what());
      return std::nullopt;
    }

    tf2::Transform transform;
    tf2::fromMsg(transform_msg.transform, transform);

    tf2::Vector3 ray_camera((u - info.cx) / info.fx, (v - info.cy) / info.fy, 1.0);
    if (ray_camera.length2() < 1e-9) {
      setBlockReason(block_reason, "valid pixel ray for stereo triangulation");
      return std::nullopt;
    }
    ray_camera.normalize();

    Ray ray;
    ray.origin = transform.getOrigin();
    ray.direction = transform.getBasis() * ray_camera;
    if (ray.direction.length2() < 1e-9) {
      setBlockReason(block_reason, "valid target-frame ray for stereo triangulation");
      return std::nullopt;
    }
    ray.direction.normalize();
    return ray;
  }

  std::optional<geometry_msgs::msg::PointStamped> estimateObjectPoint(
    std::string * block_reason = nullptr)
  {
    Bbox bbox;
    CameraInfo info;
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!latest_bbox_) {
        setBlockReason(block_reason, bboxInputDescription());
        return std::nullopt;
      }
      if (!latest_camera_info_) {
        setBlockReason(block_reason, "camera info on " + camera_info_topic_);
        return std::nullopt;
      }
      if (use_color_triangulation_after_min_depth_ && min_depth_reached_) {
        setBlockReason(block_reason, "depth disabled after 0.47m handoff; using color triangulation");
        return std::nullopt;
      }
      if (!latest_depth_) {
        setBlockReason(block_reason, "depth image on " + depth_topic_);
        return std::nullopt;
      }
      bbox = *latest_bbox_;
      info = *latest_camera_info_;
      depth = latest_depth_;
    }

    if ((now() - bbox.stamp).seconds() > max_target_age_s_) {
      setBlockReason(block_reason, "fresh " + bboxInputDescription());
      return std::nullopt;
    }

    const double u = bbox.x + 0.5 * bbox.width;
    const double v = bbox.y + 0.5 * bbox.height;
    if (use_color_triangulation_after_min_depth_ && shouldHandoffByBboxSize(bbox, info)) {
      const double handoff_depth =
        estimateRangeFromBboxSize(bbox, info).value_or(eef_refinement_start_depth_m_);
      rememberObjectDepth(handoff_depth);
      maybePreEnableEefYolo(handoff_depth);

      geometry_msgs::msg::PointStamped object_camera;
      object_camera.header.stamp = depth->header.stamp;
      object_camera.header.frame_id =
        camera_frame_override_.empty() ? info.frame_id : camera_frame_override_;
      object_camera.point.z = handoff_depth;
      object_camera.point.x = (u - info.cx) * handoff_depth / info.fx;
      object_camera.point.y = (v - info.cy) * handoff_depth / info.fy;

      try {
        rememberDepthObjectPoint(tf_buffer_.transform(object_camera, target_frame_));
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "bbox-size handoff TF transform failed: %s", ex.what());
      }

      std::ostringstream reason;
      reason << "minimum depth reached by bbox size"
             << " area_ratio=" << bboxAreaRatio(bbox, info)
             << " height_ratio=" << bboxHeightRatio(bbox, info)
             << " estimated_range=" << handoff_depth
             << "; switching to color triangulation";
      setBlockReason(block_reason, reason.str());
      return std::nullopt;
    }

    std::string depth_reject_reason;
    auto depth_m = robustDepthInBbox(*depth, info, bbox, &depth_reject_reason);
    if (!depth_m) {
      depth_m = medianDepthAt(*depth, info, u, v, &depth_reject_reason);
    }
    if (!depth_m) {
      if (use_color_triangulation_after_min_depth_) {
        const auto near_limit_depth = nearLimitDepthInBbox(*depth, info, bbox);
        if (near_limit_depth) {
          rememberObjectDepth(*near_limit_depth);
          maybePreEnableEefYolo(*near_limit_depth);

          geometry_msgs::msg::PointStamped object_camera;
          object_camera.header.stamp = depth->header.stamp;
          object_camera.header.frame_id =
            camera_frame_override_.empty() ? info.frame_id : camera_frame_override_;
          object_camera.point.z = *near_limit_depth;
          object_camera.point.x = (u - info.cx) * (*near_limit_depth) / info.fx;
          object_camera.point.y = (v - info.cy) * (*near_limit_depth) / info.fy;

          try {
            rememberDepthObjectPoint(tf_buffer_.transform(object_camera, target_frame_));
          } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 1000,
              "near-limit object TF transform failed: %s", ex.what());
          }

          std::ostringstream reason;
          reason << "minimum depth reached by near-limit sample "
                 << *near_limit_depth << "m; switching to color triangulation";
          setBlockReason(block_reason, reason.str());
          return std::nullopt;
        }
      }

      std::ostringstream reason;
      reason << "valid depth inside bbox in ["
             << min_valid_depth_m_ << ", " << max_valid_depth_m_ << "] m";
      if (!depth_reject_reason.empty()) {
        reason << "; " << depth_reject_reason;
      }
      setBlockReason(block_reason, reason.str());
      return std::nullopt;
    }
    rememberObjectDepth(*depth_m);
    maybePreEnableEefYolo(*depth_m);
    rememberMeasuredObjectWidth(bbox.width * (*depth_m) / info.fx);

    geometry_msgs::msg::PointStamped object_camera;
    object_camera.header.stamp = depth->header.stamp;
    object_camera.header.frame_id = camera_frame_override_.empty() ? info.frame_id : camera_frame_override_;
    object_camera.point.z = *depth_m;
    object_camera.point.x = (u - info.cx) * (*depth_m) / info.fx;
    object_camera.point.y = (v - info.cy) * (*depth_m) / info.fy;

    try {
      auto object_in_target = tf_buffer_.transform(object_camera, target_frame_);
      rememberDepthObjectPoint(object_in_target);
      return object_in_target;
    } catch (const tf2::TransformException & ex) {
      setBlockReason(block_reason, "TF from " + object_camera.header.frame_id + " to " + target_frame_);
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "object TF transform failed: %s", ex.what());
      return std::nullopt;
    }
  }

  void setBlockReason(std::string * block_reason, const std::string & reason) const
  {
    if (block_reason) {
      *block_reason = reason;
    }
  }

  std::string bboxInputDescription() const
  {
    std::string description = "front bbox on " + bbox_topic_;
    if (use_fallback_bbox_for_control_ && !fallback_bbox_topic_.empty()) {
      description += " or fallback bbox on " + fallback_bbox_topic_;
    }
    return description;
  }

  std::optional<double> medianDepthAt(
    const sensor_msgs::msg::Image & depth,
    const CameraInfo & info,
    double color_u,
    double color_v,
    std::string * reject_reason = nullptr)
  {
    const double image_width = info.width > 0 ? static_cast<double>(info.width) : static_cast<double>(depth.width);
    const double image_height = info.height > 0 ? static_cast<double>(info.height) : static_cast<double>(depth.height);
    const int center_col = static_cast<int>(std::lround(color_u * static_cast<double>(depth.width) / std::max(1.0, image_width)));
    const int center_row = static_cast<int>(std::lround(color_v * static_cast<double>(depth.height) / std::max(1.0, image_height)));

    std::vector<double> samples;
    const int row_begin = clampValue(center_row - depth_roi_radius_px_, 0, static_cast<int>(depth.height) - 1);
    const int row_end = clampValue(center_row + depth_roi_radius_px_, 0, static_cast<int>(depth.height) - 1);
    const int col_begin = clampValue(center_col - depth_roi_radius_px_, 0, static_cast<int>(depth.width) - 1);
    const int col_end = clampValue(center_col + depth_roi_radius_px_, 0, static_cast<int>(depth.width) - 1);

    for (int row = row_begin; row <= row_end; ++row) {
      for (int col = col_begin; col <= col_end; ++col) {
        const auto meters = depthPixelMeters(depth, row, col);
        if (meters && *meters >= min_valid_depth_m_ && *meters <= max_valid_depth_m_) {
          samples.push_back(*meters);
        }
      }
    }

    if (samples.empty()) {
      if (reject_reason) {
        *reject_reason = "center depth rejected: no valid samples";
      }
      return std::nullopt;
    }
    std::sort(samples.begin(), samples.end());
    const double depth_m = samples[samples.size() / 2];
    const int total_count =
      std::max(1, (row_end - row_begin + 1) * (col_end - col_begin + 1));
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
      static_cast<double>(samples.size());
    double variance = 0.0;
    for (const double sample : samples) {
      const double diff = sample - mean;
      variance += diff * diff;
    }
    variance /= static_cast<double>(samples.size());

    DepthStats stats;
    stats.depth_m = depth_m;
    stats.fill_ratio = static_cast<double>(samples.size()) / static_cast<double>(total_count);
    stats.std_m = std::sqrt(std::max(0.0, variance));
    stats.sample_count = static_cast<int>(samples.size());
    stats.total_count = total_count;
    stats.source = "center";
    return acceptDepthStats(stats, reject_reason);
  }

  std::optional<double> robustDepthInBbox(
    const sensor_msgs::msg::Image & depth,
    const CameraInfo & info,
    const Bbox & bbox,
    std::string * reject_reason = nullptr)
  {
    const double image_width =
      info.width > 0 ? static_cast<double>(info.width) : static_cast<double>(depth.width);
    const double image_height =
      info.height > 0 ? static_cast<double>(info.height) : static_cast<double>(depth.height);

    const int col_begin = clampValue(
      static_cast<int>(std::floor(bbox.x * static_cast<double>(depth.width) /
      std::max(1.0, image_width))),
      0,
      static_cast<int>(depth.width) - 1);
    const int col_end = clampValue(
      static_cast<int>(std::ceil((bbox.x + bbox.width) * static_cast<double>(depth.width) /
      std::max(1.0, image_width))),
      0,
      static_cast<int>(depth.width) - 1);
    const int row_begin = clampValue(
      static_cast<int>(std::floor(bbox.y * static_cast<double>(depth.height) /
      std::max(1.0, image_height))),
      0,
      static_cast<int>(depth.height) - 1);
    const int row_end = clampValue(
      static_cast<int>(std::ceil((bbox.y + bbox.height) * static_cast<double>(depth.height) /
      std::max(1.0, image_height))),
      0,
      static_cast<int>(depth.height) - 1);

    if (row_end < row_begin || col_end < col_begin) {
      if (reject_reason) {
        *reject_reason = "bbox depth rejected: empty ROI";
      }
      return std::nullopt;
    }

    const int roi_width = std::max(1, col_end - col_begin + 1);
    const int roi_height = std::max(1, row_end - row_begin + 1);
    const int step = std::max(1, std::min(roi_width, roi_height) / 24);

    std::vector<double> samples;
    int total_count = 0;
    samples.reserve(static_cast<size_t>((roi_width / step + 1) * (roi_height / step + 1)));
    for (int row = row_begin; row <= row_end; row += step) {
      for (int col = col_begin; col <= col_end; col += step) {
        ++total_count;
        const auto meters = depthPixelMeters(depth, row, col);
        if (meters && *meters >= min_valid_depth_m_ && *meters <= max_valid_depth_m_) {
          samples.push_back(*meters);
        }
      }
    }

    if (samples.empty()) {
      if (reject_reason) {
        *reject_reason = "bbox depth rejected: no valid samples";
      }
      return std::nullopt;
    }

    std::sort(samples.begin(), samples.end());
    const size_t index = std::min(samples.size() - 1, samples.size() / 4);
    const double depth_m = samples[index];
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
      static_cast<double>(samples.size());
    double variance = 0.0;
    for (const double sample : samples) {
      const double diff = sample - mean;
      variance += diff * diff;
    }
    variance /= static_cast<double>(samples.size());

    DepthStats stats;
    stats.depth_m = depth_m;
    stats.fill_ratio = static_cast<double>(samples.size()) /
      static_cast<double>(std::max(1, total_count));
    stats.std_m = std::sqrt(std::max(0.0, variance));
    stats.sample_count = static_cast<int>(samples.size());
    stats.total_count = std::max(1, total_count);
    stats.source = "bbox";
    return acceptDepthStats(stats, reject_reason);
  }

  std::optional<double> acceptDepthStats(
    const DepthStats & stats,
    std::string * reject_reason = nullptr)
  {
    auto make_status = [&stats](const std::string & prefix) {
        std::ostringstream status;
        status << prefix
               << " source=" << stats.source
               << " depth=" << stats.depth_m
               << " samples=" << stats.sample_count << "/" << stats.total_count
               << " fill=" << stats.fill_ratio
               << " std=" << stats.std_m;
        return status.str();
      };

    if (stats.sample_count < min_depth_samples_) {
      const auto reason = make_status(
        "mp_control depth rejected: sample_count below min_depth_samples=" +
        std::to_string(min_depth_samples_));
      setBlockReason(reject_reason, reason);
      publishStatus(reason);
      return std::nullopt;
    }
    if (stats.fill_ratio < depth_min_fill_ratio_) {
      const auto reason = make_status(
        "mp_control depth rejected: fill_ratio below depth_min_fill_ratio=" +
        std::to_string(depth_min_fill_ratio_));
      setBlockReason(reject_reason, reason);
      publishStatus(reason);
      return std::nullopt;
    }
    if (depth_std_max_m_ > 0.0 && stats.std_m > depth_std_max_m_) {
      const auto reason = make_status(
        "mp_control depth rejected: std above depth_std_max_m=" +
        std::to_string(depth_std_max_m_));
      setBlockReason(reject_reason, reason);
      publishStatus(reason);
      return std::nullopt;
    }

    std::string status;
    std::optional<double> accepted_depth;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (depth_jump_limit_m_ > 0.0 && depth_filter_last_accepted_m_) {
        const double jump = std::abs(stats.depth_m - *depth_filter_last_accepted_m_);
        if (jump > depth_jump_limit_m_) {
          std::ostringstream reason;
          reason << make_status("mp_control depth rejected: jump above depth_jump_limit_m")
                 << " jump=" << jump
                 << " limit=" << depth_jump_limit_m_
                 << " last_accepted=" << *depth_filter_last_accepted_m_;
          status = reason.str();
          last_depth_filter_status_ = status;
        }
      }

      if (status.empty()) {
        const double stable_tolerance = std::max(0.01, 0.5 * std::max(0.001, depth_jump_limit_m_));
        if (!depth_filter_pending_m_ ||
            std::abs(stats.depth_m - *depth_filter_pending_m_) > stable_tolerance) {
          depth_filter_pending_m_ = stats.depth_m;
          depth_filter_stable_count_ = 1;
        } else {
          depth_filter_pending_m_ =
            0.65 * (*depth_filter_pending_m_) + 0.35 * stats.depth_m;
          depth_filter_stable_count_ =
            std::min(stable_depth_frames_, depth_filter_stable_count_ + 1);
        }

        if (depth_filter_stable_count_ < stable_depth_frames_) {
          std::ostringstream waiting;
          waiting << make_status("mp_control depth waiting for stable frames")
                  << " stable=" << depth_filter_stable_count_
                  << "/" << stable_depth_frames_
                  << " pending=" << *depth_filter_pending_m_;
          status = waiting.str();
        } else {
          accepted_depth = *depth_filter_pending_m_;
          depth_filter_last_accepted_m_ = *accepted_depth;
          std::ostringstream accepted;
          accepted << make_status("mp_control depth accepted")
                   << " stable=" << depth_filter_stable_count_
                   << "/" << stable_depth_frames_
                   << " accepted=" << *accepted_depth;
          status = accepted.str();
        }
        last_depth_filter_status_ = status;
      }
    }

    setBlockReason(reject_reason, status);
    publishStatus(status);
    return accepted_depth;
  }

  void resetDepthFilterStateLocked()
  {
    depth_filter_last_accepted_m_.reset();
    depth_filter_pending_m_.reset();
    depth_filter_stable_count_ = 0;
    last_depth_filter_status_.clear();
  }

  std::optional<double> nearLimitDepthInBbox(
    const sensor_msgs::msg::Image & depth,
    const CameraInfo & info,
    const Bbox & bbox) const
  {
    const double image_width =
      info.width > 0 ? static_cast<double>(info.width) : static_cast<double>(depth.width);
    const double image_height =
      info.height > 0 ? static_cast<double>(info.height) : static_cast<double>(depth.height);

    const int col_begin = clampValue(
      static_cast<int>(std::floor(bbox.x * static_cast<double>(depth.width) /
      std::max(1.0, image_width))),
      0,
      static_cast<int>(depth.width) - 1);
    const int col_end = clampValue(
      static_cast<int>(std::ceil((bbox.x + bbox.width) * static_cast<double>(depth.width) /
      std::max(1.0, image_width))),
      0,
      static_cast<int>(depth.width) - 1);
    const int row_begin = clampValue(
      static_cast<int>(std::floor(bbox.y * static_cast<double>(depth.height) /
      std::max(1.0, image_height))),
      0,
      static_cast<int>(depth.height) - 1);
    const int row_end = clampValue(
      static_cast<int>(std::ceil((bbox.y + bbox.height) * static_cast<double>(depth.height) /
      std::max(1.0, image_height))),
      0,
      static_cast<int>(depth.height) - 1);

    if (row_end < row_begin || col_end < col_begin) {
      return std::nullopt;
    }

    const int roi_width = std::max(1, col_end - col_begin + 1);
    const int roi_height = std::max(1, row_end - row_begin + 1);
    const int step = std::max(1, std::min(roi_width, roi_height) / 24);
    const double handoff_depth = eef_refinement_start_depth_m_ + min_depth_handoff_margin_m_;

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>((roi_width / step + 1) * (roi_height / step + 1)));
    for (int row = row_begin; row <= row_end; row += step) {
      for (int col = col_begin; col <= col_end; col += step) {
        const auto meters = depthPixelMeters(depth, row, col);
        if (meters && *meters > 0.02 && *meters <= handoff_depth) {
          samples.push_back(*meters);
        }
      }
    }

    if (samples.size() < 5U) {
      return std::nullopt;
    }

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  }

  std::optional<double> estimateRangeFromBboxSize(
    const Bbox & bbox,
    const CameraInfo & info) const
  {
    if (bbox.height <= 2.0 || info.fy <= 0.0 || object_height_m_ <= 0.0) {
      return std::nullopt;
    }

    const double range_m = object_height_m_ * info.fy / bbox.height;
    if (!std::isfinite(range_m) || range_m <= 0.0) {
      return std::nullopt;
    }

    return clampValue(range_m, color_triangulation_min_object_x_m_, eef_refinement_start_depth_m_);
  }

  double bboxAreaRatio(const Bbox & bbox, const CameraInfo & info) const
  {
    const double image_width = std::max(1.0, static_cast<double>(info.width));
    const double image_height = std::max(1.0, static_cast<double>(info.height));
    return std::max(0.0, bbox.width) * std::max(0.0, bbox.height) /
      (image_width * image_height);
  }

  double bboxHeightRatio(const Bbox & bbox, const CameraInfo & info) const
  {
    const double image_height = std::max(1.0, static_cast<double>(info.height));
    return std::max(0.0, bbox.height) / image_height;
  }

  bool shouldHandoffByBboxSize(const Bbox & bbox, const CameraInfo & info) const
  {
    if (min_depth_handoff_bbox_area_ratio_ <= 0.0 &&
        min_depth_handoff_bbox_height_ratio_ <= 0.0) {
      return false;
    }

    const double area_ratio = bboxAreaRatio(bbox, info);
    const double height_ratio = bboxHeightRatio(bbox, info);
    return
      (min_depth_handoff_bbox_area_ratio_ > 0.0 &&
      area_ratio >= min_depth_handoff_bbox_area_ratio_) ||
      (min_depth_handoff_bbox_height_ratio_ > 0.0 &&
      height_ratio >= min_depth_handoff_bbox_height_ratio_);
  }

  double scaledEefTolerancePx(double tolerance_px, const CameraInfo & info) const
  {
    const double baseline_width = static_cast<double>(std::max(1, eef_camera_fallback_width_px_));
    const double baseline_height = static_cast<double>(std::max(1, eef_camera_fallback_height_px_));
    const double width_scale = static_cast<double>(std::max<std::uint32_t>(1U, info.width)) / baseline_width;
    const double height_scale = static_cast<double>(std::max<std::uint32_t>(1U, info.height)) / baseline_height;
    return std::max(1.0, tolerance_px * std::min(width_scale, height_scale));
  }

  std::optional<double> depthPixelMeters(const sensor_msgs::msg::Image & depth, int row, int col) const
  {
    if (row < 0 || col < 0 || row >= static_cast<int>(depth.height) || col >= static_cast<int>(depth.width)) {
      return std::nullopt;
    }

    if (depth.encoding == sensor_msgs::image_encodings::TYPE_16UC1 || depth.encoding == "16UC1") {
      const auto offset = static_cast<size_t>(row) * depth.step + static_cast<size_t>(col) * sizeof(std::uint16_t);
      if (offset + sizeof(std::uint16_t) > depth.data.size()) {
        return std::nullopt;
      }
      std::uint16_t raw = 0;
      std::memcpy(&raw, depth.data.data() + offset, sizeof(raw));
      if (raw == 0) {
        return std::nullopt;
      }
      return static_cast<double>(raw) * depth_unit_scale_;
    }

    if (depth.encoding == sensor_msgs::image_encodings::TYPE_32FC1 || depth.encoding == "32FC1") {
      const auto offset = static_cast<size_t>(row) * depth.step + static_cast<size_t>(col) * sizeof(float);
      if (offset + sizeof(float) > depth.data.size()) {
        return std::nullopt;
      }
      float raw = 0.0F;
      std::memcpy(&raw, depth.data.data() + offset, sizeof(raw));
      if (!std::isfinite(raw) || raw <= 0.0F) {
        return std::nullopt;
      }
      return static_cast<double>(raw);
    }

    return std::nullopt;
  }

  void rememberMeasuredObjectWidth(double width_m)
  {
    if (!std::isfinite(width_m) ||
        width_m < gripper_min_measured_object_width_m_ ||
        width_m > gripper_max_measured_object_width_m_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "ignoring object width estimate %.3f m outside [%.3f, %.3f] m",
        width_m, gripper_min_measured_object_width_m_, gripper_max_measured_object_width_m_);
      return;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_object_width_m_ = width_m;
  }

  void rememberObjectDepth(double depth_m)
  {
    if (!std::isfinite(depth_m)) {
      return;
    }
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_object_depth_m_ = depth_m;
    latest_object_depth_stamp_ = now();
    if (use_color_triangulation_after_min_depth_ &&
        depth_m <= eef_refinement_start_depth_m_ + min_depth_handoff_margin_m_) {
      min_depth_reached_ = true;
    }
  }

  void maybePreEnableEefYolo(double depth_m)
  {
    if (!use_eef_refinement_ ||
        !use_color_triangulation_after_min_depth_ ||
        !std::isfinite(depth_m) ||
        depth_m > eef_yolo_pre_enable_depth_m_) {
      return;
    }

    publishEefAutoInitEnable(true);
  }

  void rememberDepthObjectPoint(const geometry_msgs::msg::PointStamped & object_in_target)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (use_color_triangulation_after_min_depth_ && min_depth_reached_) {
      return;
    }
    latest_depth_object_in_target_ = object_in_target;
    latest_depth_object_stamp_ = now();
  }

  double gripperPositionForGap(double gap_m) const
  {
    const double joint_position =
      0.05 * std::max(0.0, gap_m);
    return clampValue(joint_position, gripper_min_position_, gripper_max_position_);
  }

  GripperTarget makeGripperTarget(bool open)
  {
    if (!gripper_width_control_enabled_) {
      return GripperTarget{
        open ? gripper_open_position_ : gripper_close_position_,
        0.0,
        0.0,
        false};
    }

    std::optional<double> measured_width;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      measured_width = latest_object_width_m_;
    }

    const double object_width_m =
      measured_width.value_or(gripper_fallback_object_width_m_);
    const double target_gap_m = open ?
      object_width_m + gripper_pre_grasp_clearance_m_ :
      finalGraspGapForObjectWidth(object_width_m);

    return GripperTarget{
      gripperPositionForGap(target_gap_m),
      object_width_m,
      target_gap_m,
      measured_width.has_value()};
  }

  double finalGraspGapForObjectWidth(double object_width_m) const
  {
    const double scaled_width = object_width_m * gripper_grasp_width_scale_;
    if (gripper_grasp_clearance_m_ > 0.0) {
      return scaled_width + gripper_grasp_clearance_m_;
    }
    return std::max(0.0, scaled_width - gripper_grasp_compression_m_);
  }

  VisualGraspState currentVisualGraspState()
  {
    VisualGraspState state;
    const auto stamp = now();
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (latest_bbox_) {
      state.front_age_s = (stamp - latest_bbox_->stamp).seconds();
      state.front_fresh = state.front_age_s <= grasp_completion_front_max_age_s_;
    }
    if (latest_eef_bbox_) {
      state.eef_age_s = (stamp - latest_eef_bbox_->stamp).seconds();
      state.eef_fresh = state.eef_age_s <= grasp_completion_eef_lost_timeout_s_;
    }
    return state;
  }

  std::string visualGraspStateText(const VisualGraspState & state) const
  {
    std::ostringstream status;
    status << "front_bbox=" << (state.front_fresh ? "fresh" : "missing")
           << " age=" << state.front_age_s
           << "s, eef_bbox=" << (state.eef_fresh ? "visible" : "lost")
           << " age=" << state.eef_age_s << "s";
    return status.str();
  }

  bool visualGraspConfirmed(const VisualGraspState & state) const
  {
    return close_sent_ && !state.front_fresh && eef_bbox_seen_after_close_ && !state.eef_fresh;
  }

  bool visualCloseTriggerReady(const VisualGraspState & state) const
  {
    return !state.front_fresh && eef_bbox_seen_after_close_ && !state.eef_fresh;
  }

  double bboxArea(const Bbox & bbox) const
  {
    return std::max(0.0, bbox.width) * std::max(0.0, bbox.height);
  }

  std::optional<double> latestFreshFrontBboxArea() 
  {
    const auto stamp = now();
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!latest_bbox_) {
      return std::nullopt;
    }
    if ((stamp - latest_bbox_->stamp).seconds() > grasp_completion_front_max_age_s_) {
      return std::nullopt;
    }
    return bboxArea(*latest_bbox_);
  }

  std::optional<double> latestFreshEefBboxArea()
  {
    const auto stamp = now();
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!latest_eef_bbox_) {
      return std::nullopt;
    }
    if ((stamp - latest_eef_bbox_->stamp).seconds() > grasp_completion_eef_lost_timeout_s_) {
      return std::nullopt;
    }
    return bboxArea(*latest_eef_bbox_);
  }

  std::optional<double> frontBboxAreaRatioFromForwardStart()
  {
    if (!front_bbox_area_at_eef_forward_start_ ||
        *front_bbox_area_at_eef_forward_start_ <= 1.0) {
      return std::nullopt;
    }
    const auto area = latestFreshFrontBboxArea();
    if (!area) {
      return std::nullopt;
    }
    return *area / *front_bbox_area_at_eef_forward_start_;
  }

  std::optional<double> eefBboxAreaRatioFromForwardStart()
  {
    if (!eef_bbox_area_at_eef_forward_start_ ||
        *eef_bbox_area_at_eef_forward_start_ <= 1.0) {
      return std::nullopt;
    }
    const auto area = latestFreshEefBboxArea();
    if (!area) {
      return std::nullopt;
    }
    return *area / *eef_bbox_area_at_eef_forward_start_;
  }

  bool shouldCloseOnFrontBboxShrink()
  {
    if (!close_on_front_bbox_shrink_ || close_sent_) {
      return false;
    }
    const auto ratio = frontBboxAreaRatioFromForwardStart();
    return ratio && *ratio > 0.0 && *ratio <= 1.0 &&
      *ratio <= front_bbox_close_area_ratio_;
  }

  bool shouldCloseOnEefBboxShrink()
  {
    if (!close_on_eef_bbox_shrink_ || close_sent_) {
      return false;
    }
    const auto ratio = eefBboxAreaRatioFromForwardStart();
    return ratio && *ratio > 0.0 && *ratio <= 1.0 &&
      *ratio <= eef_bbox_close_area_ratio_;
  }

  void completeGrasp(const std::string & status_text)
  {
    publishCargoEvent("picked", true);
    if (handoff_after_grasp_) {
      startHandoffSequence(status_text);
      return;
    }
    done_ = true;
    active_ = false;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      eef_refinement_requested_ = false;
    }
    publishEefAutoInitEnable(false);
    publishBaseHold(false);
    publishStatus(status_text, true);
  }

  bool isHandoffStage(GraspStage stage) const
  {
    return
      stage == GraspStage::HANDOFF_RETRACT ||
      stage == GraspStage::HANDOFF_ROTATE ||
      stage == GraspStage::HANDOFF_PLACE ||
      stage == GraspStage::HANDOFF_RELEASE ||
      stage == GraspStage::HANDOFF_STAY;
  }

  void startHandoffSequence(const std::string & grasp_status)
  {
    const auto stamp = now();
    handoff_retract_start_joint_positions_.reset();
    handoff_rotation_stable_since_.reset();

    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      eef_refinement_requested_ = false;
    }
    publishEefAutoInitEnable(false);
    publishBaseHold(true);
    publishStop();
    publishBaseStop();
    stopMoveItServo();
    handoff_lift_controller_target_.reset();
    handoff_place_controller_target_.reset();
    handoff_stay_controller_target_.reset();
    handoff_joint1_target_rad_.reset();
    handoff_joint1_centering_ = false;
    handoff_joint1_centered_for_rotate_ = false;
    stage_ = GraspStage::HANDOFF_RETRACT;
    handoff_stage_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

    publishStatus(
      grasp_status + (handoff_restore_grasp_pose_ ?
        "; starting handoff: lift -> joint1 rotate -> restore grasp height -> open -> Stay" :
        "; starting handoff sequence: close -> wait -> joint1 rotate -> wait -> open"),
      true);
  }

  void updateHandoff()
  {
    publishBaseHold(true);
    publishBaseStop();

    switch (stage_) {
      case GraspStage::HANDOFF_RETRACT:
        updateHandoffRetract();
        return;
      case GraspStage::HANDOFF_ROTATE:
        updateHandoffRotate();
        return;
      case GraspStage::HANDOFF_PLACE:
        updateHandoffPlace();
        return;
      case GraspStage::HANDOFF_RELEASE:
        updateHandoffRelease();
        return;
      case GraspStage::HANDOFF_STAY:
        updateHandoffStay();
        return;
      default:
        return;
    }
  }

  std::array<double, 4> clampHandoffTarget(const std::array<double, 4> & positions) const
  {
    std::array<double, 4> clamped = positions;
    clamped[0] = clampValue(clamped[0], joint_pregrasp_min_positions_[0], joint_pregrasp_max_positions_[0]);
    clamped[1] = clampValue(clamped[1], joint_pregrasp_min_positions_[1], joint_pregrasp_max_positions_[1]);
    clamped[3] = clampValue(clamped[3], joint_pregrasp_min_positions_[3], joint_pregrasp_max_positions_[3]);
    return clamped;
  }

  void publishHandoffJointTrajectory(
    const std::array<double, 4> & controller_target,
    bool preserve_current_non_joint1 = false,
    double duration_override_s = -1.0)
  {
    const auto current = latestArmJointPositions();
    if (!current) {
      publishStatus("handoff waiting for joint states before joint trajectory");
      return;
    }

    auto target = clampHandoffTarget(controller_target);
    // target already captures the pitch joints at entry. Re-reading them on
    // every retry would ratchet tracking error and change the carried height.

    trajectory_msgs::msg::JointTrajectory msg;
    if (preserve_current_non_joint1) msg.header.frame_id = "mp_control_pitch_hold";
    const auto stamp = now();
    msg.header.stamp = stamp;
    msg.joint_names.assign(arm_joint_names_.begin(), arm_joint_names_.end());
    const double duration_s = duration_override_s > 0.0 ?
      duration_override_s :
      handoff_joint_move_duration_s_;

    appendJointTrajectoryPoint(msg, target, duration_s);
    handoff_joint_trajectory_pub_->publish(msg);
    handoff_last_publish_stamp_ = stamp;

    std::ostringstream status;
    status << "handoff joint trajectory published: controller_target="
           << formatJointArray(target)
           << " direct_topic=" << handoff_joint_trajectory_topic_
           << " duration=" << duration_s
           << " preserve_current_non_joint1="
           << (preserve_current_non_joint1 ? "true" : "false");
    publishStatus(status.str());
  }

  void maybeRepublishHandoffJointTrajectory(
    const std::optional<std::array<double, 4>> & controller_target,
    bool preserve_current_non_joint1 = false,
    double duration_override_s = -1.0)
  {
    if (!controller_target || handoff_republish_period_s_ <= 0.0) {
      return;
    }
    const auto stamp = now();
    if (
      handoff_last_publish_stamp_.nanoseconds() != 0 &&
      (stamp - handoff_last_publish_stamp_).seconds() < handoff_republish_period_s_)
    {
      return;
    }
    publishHandoffJointTrajectory(*controller_target, preserve_current_non_joint1, duration_override_s);
  }

  double handoffRotateAngleForCurrentPick() const
  {
    const bool odd_pick = (handoff_pick_sequence_index_ % 2) == 1;
    const double angle_deg = odd_pick ?
      handoff_rotate_odd_angle_deg_ :
      handoff_rotate_even_angle_deg_;

    return angle_deg * M_PI / 180.0;
  }

  double chooseHandoffJoint1Target(double current_joint1) const
  {
    const double min_joint1 = joint_pregrasp_min_positions_[0];
    const double max_joint1 = joint_pregrasp_max_positions_[0];
    const double rotate_base =
      handoff_center_joint1_before_rotate_ ?
      handoff_joint1_center_target_rad_ :
      current_joint1;

    const double rotate_angle = handoffRotateAngleForCurrentPick();

    return clampValue(rotate_base + rotate_angle, min_joint1, max_joint1);
  }

  std::array<double, 4> handoffRetractTarget(const std::array<double, 4> & current) const
  {
    if (handoff_retract_hold_z_) return levelRetractionPoints(current).back().q;
    auto target = current;
    target[1] += handoff_lift_joint2_delta_rad_;
    target[2] = clampValue(current[2] + handoff_retract_joint3_delta_rad_,
      joint_pregrasp_min_positions_[2], joint_pregrasp_max_positions_[2]);
    target = clampHandoffTarget(target);
    if (handoff_restore_grasp_pose_ || handoff_retract_joint3_delta_rad_ > 0.0)
      target[3] = joint4ForPreservedGripperRoll(current, target);
    return clampHandoffTarget(target);
  }

  level_retraction::Geometry levelRetractionGeometry() const
  {
    return {eef_forward_ik_link2_x_m_, eef_forward_ik_link2_z_m_,
      eef_forward_ik_link3_m_, eef_forward_ik_tool_m_};
  }

  std::vector<level_retraction::Point> levelRetractionPoints(
    const std::array<double, 4> & current) const
  {
    std::array<double, 4> low{}, high{};
    std::copy_n(joint_pregrasp_min_positions_.begin(), 4, low.begin());
    std::copy_n(joint_pregrasp_max_positions_.begin(), 4, high.begin());
    return level_retraction::plan(current, handoff_retract_distance_m_,
      handoff_lift_duration_s_, 36, levelRetractionGeometry(), low, high);
  }

  void abortLevelRetraction(const std::string & reason)
  {
    // Replace the queued multi-point curve with a measured hold even when this
    // node is used without the ONNX trial gate. Never release the gripper here.
    const auto measured = latestArmJointPositions();
    if (measured && std::all_of(measured->begin(), measured->end(),
        [](double q) {return std::isfinite(q);}))
      publishHandoffJointTrajectory(*measured, true, .2);
    cancelSequence("aborted: " + reason);
    publishBaseHold(true);
    safety_abort_latched_ = true;
  }

  void updateHandoffRetract()
  {
    const auto stamp = now();

    if (handoff_stage_start_stamp_.nanoseconds() == 0) {
      handoff_stage_start_stamp_ = stamp;
      handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

      publishStop();
      publishBaseStop();

      publishStatus(
        handoff_restore_grasp_pose_ ? "handoff lift settle: gripper closed; waiting before lifting" :
        "handoff grasp settle: gripper closed; waiting before inward retraction",
        true);
      return;
    }

    publishStop();
    publishBaseStop();

    const double elapsed_s = (stamp - handoff_stage_start_stamp_).seconds();

    if (!handoff_lift_controller_target_) {
      if (elapsed_s < handoff_grasp_settle_s_) {
        publishStatus(
          "handoff grasp settle: waiting before retraction elapsed=" +
          std::to_string(elapsed_s) +
          "/" + std::to_string(handoff_grasp_settle_s_));
        return;
      }

      const auto current = latestArmJointPositions();
      if (!current) {
        publishStatus("handoff retract waiting for joint states");
        return;
      }
      handoff_grasp_controller_pose_ = *current;

      try {
        if (handoff_retract_hold_z_) {
          const auto points = levelRetractionPoints(*current);
          trajectory_msgs::msg::JointTrajectory trajectory;
          trajectory.header.stamp = stamp;
          trajectory.header.frame_id = "mp_control_retract_z_hold";
          trajectory.joint_names.assign(arm_joint_names_.begin(), arm_joint_names_.end());
          for (const auto & point : points)
            appendJointTrajectoryPoint(trajectory, point.q, point.seconds);
          handoff_retract_start_joint_positions_ = *current;
          handoff_lift_controller_target_ = points.back().q;
          handoff_joint_trajectory_pub_->publish(trajectory);
        } else {
          handoff_lift_controller_target_ = handoffRetractTarget(*current);
          publishHandoffJointTrajectory(*handoff_lift_controller_target_, handoff_restore_grasp_pose_, handoff_lift_duration_s_);
        }
      } catch (const std::exception & error) {
        abortLevelRetraction(std::string("no constant-Z retract path: ") + error.what());
        return;
      }
      handoff_retract_stable_since_.reset();

      handoff_stage_start_stamp_ = stamp;
      handoff_last_publish_stamp_ = stamp;

      publishStatus(
        std::string(handoff_restore_grasp_pose_ ? "handoff lift: raising before joint1 turn; target=" :
          "handoff retract: moving toward body before joint1 turn; target=") +
        formatJointArray(*handoff_lift_controller_target_) +
        " joint2_delta_rad=" + std::to_string(handoff_lift_joint2_delta_rad_),
        true);
      return;
    }

    const double lift_elapsed_s = (stamp - handoff_stage_start_stamp_).seconds();
    if (handoff_retract_hold_z_ && handoff_retract_start_joint_positions_) {
      const auto measured = latestArmJointPositions();
      if (!measured) return;
      const auto geometry = levelRetractionGeometry();
      const auto actual = level_retraction::forward(*measured, geometry);
      const auto origin = level_retraction::forward(*handoff_retract_start_joint_positions_, geometry);
      const double z_error = actual.z - origin.z;
      if (!std::isfinite(z_error) || std::abs(z_error) > handoff_retract_z_tolerance_m_) {
        abortLevelRetraction("constant-Z retract tracking deviation=" + std::to_string(z_error));
        return;
      }
      if (!std::isfinite(actual.radius) || actual.radius > origin.radius + handoff_retract_z_tolerance_m_) {
        abortLevelRetraction("Cartesian retraction moved outward instead of toward body");
        return;
      }
    }
    if (lift_elapsed_s < handoff_lift_duration_s_ + handoff_joint_settle_s_) {
      publishStatus(
        std::string(handoff_restore_grasp_pose_ ? "handoff lift: waiting for upward motion elapsed=" :
          "handoff retract: waiting for inward motion elapsed=") +
        std::to_string(lift_elapsed_s) +
        "/" + std::to_string(handoff_lift_duration_s_ + handoff_joint_settle_s_));
      return;
    }

    double max_error = -1.0;
    const bool exact_target_reached = handoffTargetReached(handoff_lift_controller_target_, &max_error);
    // A stationary loaded arm can stop a few degrees short of a lift target.
    // Accept its measured lifted pose; rotation will freeze those actual pitch
    // joints. An explicit continue request also advances a stopped lift.
    const bool measured_lift_ready = handoff_restore_grasp_pose_ && max_error >= 0.0 &&
      (max_error <= handoff_lift_joint_tolerance_rad_ || handoff_resume_requested_);
    if ((!exact_target_reached && !measured_lift_ready) || !armStopped()) {
      handoff_retract_stable_since_.reset();
      if (handoff_retract_hold_z_ && lift_elapsed_s > handoff_lift_duration_s_ + handoff_joint_settle_s_ + 3.0) {
        abortLevelRetraction("constant-Z retraction did not settle; gripper remains closed");
        return;
      }
      if (!handoff_retract_hold_z_ && max_error > joint_pregrasp_tolerance_rad_)
        maybeRepublishHandoffJointTrajectory(handoff_lift_controller_target_, handoff_restore_grasp_pose_, 1.0);
      publishStatus(std::string(handoff_restore_grasp_pose_ ? "handoff lift: waiting for lifted pose and stopped joints; max_error=" :
        "handoff retract: waiting for compact pose and stopped joints; max_error=") + std::to_string(max_error));
      return;
    }
    if (!handoff_retract_stable_since_) handoff_retract_stable_since_ = stamp;
    if ((stamp - *handoff_retract_stable_since_).seconds() < .3) return;
    if (pause_before_rotate_ && !handoff_resume_requested_) {
      publishStatus("retract_ready: arm folded toward body; checking box clearance before rotation");
      return;
    }

    stage_ = GraspStage::HANDOFF_ROTATE;
    handoff_stage_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

    publishStatus(
      handoff_restore_grasp_pose_ ? "handoff rotate: measured lift complete; starting joint1 turn" :
      "handoff rotate: measured inward retraction complete; starting joint1 turn",
      true);
  }

  void updateHandoffRotate()
  {
    const auto stamp = now();
    if (handoff_stage_start_stamp_.nanoseconds() == 0) {
      const auto current = latestArmJointPositions();
      if (!current) {
        publishStatus("handoff joint1 turn waiting for joint states");
        return;
      }
      std::array<double, 4> target = *current;
      const double initial_joint1 = (*current)[0];
      if (
        handoff_center_joint1_before_rotate_ &&
        !handoff_joint1_centered_for_rotate_ &&
        std::abs(initial_joint1 - handoff_joint1_center_target_rad_) >
        handoff_joint1_center_tolerance_rad_)
      {
        target[0] = handoff_joint1_center_target_rad_;
        handoff_lift_controller_target_ = clampHandoffTarget(target);
        handoff_joint1_centering_ = true;
        publishHandoffJointTrajectory(*handoff_lift_controller_target_, true);
        handoff_stage_start_stamp_ = stamp;
        const double center_delta = (*handoff_lift_controller_target_)[0] - initial_joint1;
        const double center_delta_deg = center_delta * 180.0 / M_PI;
        publishStatus(
          "handoff joint1 center: aligning joint1 before same-direction 180deg turn; target=" +
          formatJointArray(*handoff_lift_controller_target_) +
          " joint1_delta_deg=" + std::to_string(center_delta_deg),
          true);
        return;
      }

      handoff_joint1_centering_ = false;
      handoff_joint1_centered_for_rotate_ = true;
      target[0] = chooseHandoffJoint1Target(initial_joint1);
      handoff_joint1_target_rad_ = target[0];
      handoff_lift_controller_target_ = clampHandoffTarget(target);
      publishHandoffJointTrajectory(*handoff_lift_controller_target_, true);
      handoff_stage_start_stamp_ = stamp;
      const double planned_delta = (*handoff_lift_controller_target_)[0] - initial_joint1;
      const double planned_delta_deg = planned_delta * 180.0 / M_PI;
      const double selected_rotate_deg = handoffRotateAngleForCurrentPick() * 180.0 / M_PI;
      const bool odd_pick = (handoff_pick_sequence_index_ % 2) == 1;
      // publishStatus(
      //   "handoff joint1 turn: rotating manipulator joint1 toward follower; target=" +
      //   formatJointArray(*handoff_lift_controller_target_) +
      //   " joint1_delta_deg=" + std::to_string(planned_delta_deg),
      //   true);
      // return;
      publishStatus(
        "handoff joint1 turn: rotating manipulator joint1 toward follower; pick_index=" +
        std::to_string(handoff_pick_sequence_index_) +
        " parity=" + std::string(odd_pick ? "odd" : "even") +
        " selected_rotate_deg=" + std::to_string(selected_rotate_deg) +
        " target=" + formatJointArray(*handoff_lift_controller_target_) +
        " joint1_delta_deg=" + std::to_string(planned_delta_deg),
        true);
      return;
    }

    publishStop();
    publishBaseStop();
    maybeRepublishHandoffJointTrajectory(handoff_lift_controller_target_, true);
    const double rotate_elapsed_s = (stamp - handoff_stage_start_stamp_).seconds();
    const double rotate_wait_s = handoff_joint_move_duration_s_ + handoff_joint_settle_s_;

    if (rotate_elapsed_s < rotate_wait_s) {
      publishStatus(
        "handoff joint1 turn: waiting for timed settle elapsed=" +
        std::to_string(rotate_elapsed_s) +
        "/" + std::to_string(rotate_wait_s));
      return;
    }

    const auto current = latestArmJointPositions();
    if (!current) {
      publishStatus("handoff joint1 turn: waiting for joint states before gripper open");
      return;
    }

    double target_joint1 = (*current)[0];
    if (handoff_lift_controller_target_) {
      target_joint1 = (*handoff_lift_controller_target_)[0];
    } else if (handoff_joint1_target_rad_) {
      target_joint1 = *handoff_joint1_target_rad_;
    }

    const double joint1_error = normalizeAngle(target_joint1 - (*current)[0]);
    const double joint1_error_abs = std::abs(joint1_error);

    if (joint1_error_abs > handoff_joint1_center_tolerance_rad_) {
      handoff_rotation_stable_since_.reset();
      maybeRepublishHandoffJointTrajectory(handoff_lift_controller_target_, true);
      publishStatus(
        "handoff joint1 turn: waiting for joint1 target before gripper open"
        " current=" + std::to_string((*current)[0]) +
        " target=" + std::to_string(target_joint1) +
        " error=" + std::to_string(joint1_error) +
        " tolerance=" + std::to_string(handoff_joint1_center_tolerance_rad_));
      return;
    }

    if (handoff_joint1_centering_) {
      handoff_joint1_centering_ = false;
      handoff_joint1_centered_for_rotate_ = true;
      handoff_lift_controller_target_.reset();
      handoff_stage_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      publishStatus(
        "handoff joint1 center complete; starting same-direction joint1 turn",
        true);
      return;
    }

    if (!armStopped()) {handoff_rotation_stable_since_.reset(); return;}
    if (!handoff_rotation_stable_since_) handoff_rotation_stable_since_ = stamp;
    if ((stamp - *handoff_rotation_stable_since_).seconds() < .2) return;
    if (!handoff_lower_before_release_) {
      stage_ = GraspStage::HANDOFF_RELEASE;
      handoff_stage_start_stamp_ = stamp;
      handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      sendGripperOpenForObject();
      publishStatus("handoff rotation complete: releasing at carried height; no additional lowering", true);
      return;
    }
    stage_ = GraspStage::HANDOFF_PLACE;
    handoff_stage_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_place_controller_target_.reset();
    publishStatus("handoff rotate complete; lowering arm before release", true);
  }

  void updateHandoffPlace()
  {
    const auto stamp = now();
    if (handoff_stage_start_stamp_.nanoseconds() == 0) {
      std::array<double, 4> target{};
      if (handoff_restore_grasp_pose_) {
        if (!handoff_grasp_controller_pose_ || !handoff_joint1_target_rad_) {
          publishStatus("handoff lower waiting for captured grasp pose and completed turn");
          return;
        }
        target = *handoff_grasp_controller_pose_;
        target[0] = *handoff_joint1_target_rad_;
      } else if (handoff_lift_controller_target_) {
        target = *handoff_lift_controller_target_;
      } else if (handoff_joint1_target_rad_) {
        const auto current = latestArmJointPositions();
        if (!current) {
          publishStatus("handoff place waiting for joint states");
          return;
        }
        target = *current;
        target[0] = *handoff_joint1_target_rad_;
      } else {
        const auto current = latestArmJointPositions();
        if (!current) {
          publishStatus("handoff place waiting for joint states");
          return;
        }
        target = *current;
      }
      if (!handoff_restore_grasp_pose_) target[1] += handoff_place_joint2_delta_rad_;
      handoff_place_controller_target_ = clampHandoffTarget(target);
      handoff_place_stable_since_.reset();
      publishHandoffJointTrajectory(
        *handoff_place_controller_target_,
        handoff_restore_grasp_pose_ || std::abs(handoff_place_joint2_delta_rad_) <= 1.0e-9,
        handoff_place_duration_s_);
      handoff_stage_start_stamp_ = stamp;
      publishStatus(
        "handoff place: settling before release; target=" +
        formatJointArray(*handoff_place_controller_target_), true);
      return;
    }

    publishStop();
    publishBaseStop();
    if ((stamp - handoff_stage_start_stamp_).seconds() <
        handoff_place_duration_s_ + handoff_joint_settle_s_) return;
    double max_error = -1.0;
    if (!handoffTargetReached(handoff_place_controller_target_, &max_error) || !armStopped()) {
      handoff_place_stable_since_.reset();
      if (max_error > joint_pregrasp_tolerance_rad_)
        maybeRepublishHandoffJointTrajectory(handoff_place_controller_target_, handoff_restore_grasp_pose_, 1.0);
      publishStatus("handoff place: waiting for measured lowered pose and stopped joints; max_error=" + std::to_string(max_error));
      return;
    }
    if (!handoff_place_stable_since_) handoff_place_stable_since_ = stamp;
    if ((stamp - *handoff_place_stable_since_).seconds() < 0.3) return;
    stage_ = GraspStage::HANDOFF_RELEASE;
    handoff_stage_start_stamp_ = stamp;
    handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    sendGripperOpenForObject();
    publishStatus("handoff place complete: lowered pose reached and stationary; opening gripper", true);
  }

  void updateHandoffRelease()
  {
    publishStop();
    publishBaseStop();
    if ((now() - handoff_stage_start_stamp_).seconds() < handoff_release_settle_s_) {
      publishStatus("handoff release: waiting for gripper open settle");
      return;
    }
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!latest_gripper_position_ || latest_joint_state_stamp_.nanoseconds() == 0 ||
        (now() - latest_joint_state_stamp_).seconds() > .2 ||
        *latest_gripper_position_ < gripper_max_position_ - .001) {
        publishStatus("handoff release: waiting for measured gripper opening before Stay return");
        return;
      }
    }
    publishCargoEvent("placed", true);

    stage_ = GraspStage::HANDOFF_STAY;
    handoff_stage_start_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_last_publish_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    handoff_stay_controller_target_.reset();
    publishStatus("handoff release complete; returning manipulator to stay pose", true);
  }

  void updateHandoffStay()
  {
    const auto stamp = now();

    if (handoff_stage_start_stamp_.nanoseconds() == 0) {
      std::array<double, 4> target{};
      for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] = handoff_stay_joint_positions_[i];
      }

      handoff_stay_controller_target_ = clampHandoffTarget(target);

      // Initial command: give the arm enough time to return to stay pose.
      publishHandoffJointTrajectory(*handoff_stay_controller_target_);

      handoff_stage_start_stamp_ = stamp;
      handoff_last_publish_stamp_ = stamp;

      publishStatus(
        "handoff stay: moving manipulator back to saved stay pose; target=" +
        formatJointArray(*handoff_stay_controller_target_),
        true);
      return;
    }

    publishStop();
    publishBaseStop();

    const double elapsed_s = (stamp - handoff_stage_start_stamp_).seconds();
    const double nominal_wait_s =
      handoff_joint_move_duration_s_ + handoff_joint_settle_s_;

    double max_error = -1.0;
    const bool reached = handoffTargetReached(
      handoff_stay_controller_target_,
      &max_error);

    if (elapsed_s < nominal_wait_s) {
      publishStatus(
        "handoff stay: waiting for manipulator stay pose settle"
        " elapsed=" + std::to_string(elapsed_s) +
        "/" + std::to_string(nominal_wait_s) +
        " max_joint_err=" + std::to_string(max_error) +
        " tolerance=" + std::to_string(joint_pregrasp_tolerance_rad_));
      return;
    }

    if (!reached) {
      // Corrective command after the nominal trajectory time.
      // Do not restart a 7-second trajectory forever; use a short correction.
      constexpr double stay_retry_duration_s = 1.0;

      maybeRepublishHandoffJointTrajectory(
        handoff_stay_controller_target_,
        false,
        stay_retry_duration_s);

      publishStatus(
        "handoff stay: target not reached; retrying stay pose"
        " max_joint_err=" + std::to_string(max_error) +
        " tolerance=" + std::to_string(joint_pregrasp_tolerance_rad_) +
        " retry_duration=" + std::to_string(stay_retry_duration_s) +
        " target=" +
        (handoff_stay_controller_target_ ?
          formatJointArray(*handoff_stay_controller_target_) :
          std::string("unavailable")));

      return;
    }

    done_ = true;
    active_ = false;
    eef_refinement_object_in_target_.reset();

    publishBaseHold(false);
    publishBaseStop();
    publishCargoEvent("loaded", true);

    publishStatus(
      "cargo_loaded: placed on follower side by joint1 turn; arm reached stay pose; "
      "max_joint_err=" + std::to_string(max_error) +
      "; base hold kept",
      true);
  }

  bool armStopped()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return latest_arm_joint_velocities_ && latest_joint_state_stamp_.nanoseconds() != 0 &&
      (now() - latest_joint_state_stamp_).seconds() <= 0.2 &&
      std::all_of(latest_arm_joint_velocities_->begin(), latest_arm_joint_velocities_->end(),
        [](double velocity) {return std::abs(velocity) <= 0.03;});
  }

  bool handoffTargetReached(
    const std::optional<std::array<double, 4>> & controller_target,
    double * max_error_out = nullptr)
  {
    if (!controller_target) {
      if (max_error_out) {
        *max_error_out = -1.0;
      }
      return false;
    }

    const auto current = latestArmJointPositions();
    if (!current) {
      if (max_error_out) {
        *max_error_out = -1.0;
      }
      return false;
    }

    const auto target = clampHandoffTarget(*controller_target);
    const double max_error = maxJointError(*current, target);

    if (max_error_out) {
      *max_error_out = max_error;
    }

    return max_error <= joint_pregrasp_tolerance_rad_;
  }

  bool closeAndCompleteWhenVisualReady(
    const std::string & complete_status,
    const std::string & waiting_status)
  {
    if (!require_visual_grasp_confirmation_) {
      if (close_gripper_on_arrival_ && !close_sent_) {
        sendGripperGraspForObject();
        close_sent_ = true;
      }
      completeGrasp(complete_status);
      return true;
    }

    const auto visual_state = currentVisualGraspState();
    if (visual_state.eef_fresh) {
      eef_bbox_seen_after_close_ = true;
    }

    if (close_gripper_on_arrival_ && !close_sent_) {
      if (!visualCloseTriggerReady(visual_state)) {
        publishStop();
        publishStatus(
          "waiting for visual close trigger before gripper close: " +
          visualGraspStateText(visual_state) +
          ", eef_seen_before_close=" +
          (eef_bbox_seen_after_close_ ? "true" : "false") +
          "; need front occluded and eef lost");
        return false;
      }

      sendGripperGraspForObject();
      close_sent_ = true;
      completeGrasp(
        complete_status +
        "; visual close trigger: front bbox occluded and eef bbox lost");
      return true;
    }

    if (visualGraspConfirmed(visual_state)) {
      completeGrasp(
        complete_status +
        "; visual confirmation: front bbox occluded and eef bbox lost");
      return true;
    }

    publishStop();
    publishStatus(waiting_status + ": " + visualGraspStateText(visual_state));
    return false;
  }

  bool continueAfterCloseUntilEefBboxLost(
    const geometry_msgs::msg::TransformStamped & eef_tf,
    const std::string & complete_status)
  {
    if (!close_sent_ || !require_visual_grasp_confirmation_) {
      return false;
    }

    const auto visual_state = currentVisualGraspState();
    if (visual_state.eef_fresh) {
      eef_bbox_seen_after_close_ = true;
    }
    if (visualGraspConfirmed(visual_state)) {
      completeGrasp(
        complete_status +
        "; visual confirmation: front bbox occluded and eef bbox lost");
      return true;
    }

    if (visual_state.eef_fresh && eef_forward_speed_mps_ > 0.0) {
      const auto joint_progress = eefForwardJointProgress();
      publishStop();
      publishStatus(
        "eef bbox still visible after gripper close; holding for safety before handoff: " +
        visualGraspStateText(visual_state) +
        formatEefForwardJointProgress(joint_progress) +
        " rpy_frame=" + rpyControlFrame() +
        poseStatusSuffix(eef_tf));
      return true;
    }

    publishStop();
    publishStatus(
      "waiting for visual grasp confirmation after gripper close: " +
      visualGraspStateText(visual_state) +
      ", eef_seen_after_close=" + (eef_bbox_seen_after_close_ ? "true" : "false"));
    return true;
  }

  void sendGripperOpenForObject()
  {
    RCLCPP_INFO(
      get_logger(),
      "gripper open target: fully open, position=%.4f m",
      gripper_max_position_);
    sendGripper(gripper_max_position_);
  }

  void sendGripperGraspForObject()
  {
    const auto target = makeGripperTarget(false);
    if (gripper_width_control_enabled_) {
      RCLCPP_INFO(
        get_logger(),
        "gripper grasp target: object_width=%.3f m (%s), gap=%.3f m, position=%.4f m",
        target.object_width_m,
        target.measured ? "measured" : "fallback",
        target.target_gap_m,
        target.position);
    }
    sendGripper(target.position);
  }

  void sendGripper(double position)
  {
    position = clampValue(position, gripper_min_position_, gripper_max_position_);
    if (!gripper_client_->wait_for_action_server(std::chrono::milliseconds(100))) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "gripper action server unavailable: %s", gripper_action_name_.c_str());
      return;
    }

    GripperCommand::Goal goal;
    goal.command.position = position;
    goal.command.max_effort = gripper_max_effort_;

    auto options = rclcpp_action::Client<GripperCommand>::SendGoalOptions();
    options.result_callback = [this](const GripperGoalHandle::WrappedResult & result) {
      if (result.code != rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_WARN(get_logger(), "gripper action finished with non-success result");
      }
    };
    gripper_client_->async_send_goal(goal, options);
  }

  bool startMoveItServo()
  {
    if (servo_start_requested_) {
      return true;
    }
    if (!servo_start_client_->wait_for_service(std::chrono::milliseconds(500))) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "MoveIt Servo start service unavailable: /servo_node/start_servo");
      return false;
    }

    servo_start_requested_ = true;
    servo_start_client_->async_send_request(
      std::make_shared<Trigger::Request>(),
      [this](rclcpp::Client<Trigger>::SharedFuture future) {
        const auto response = future.get();
        if (!response->success) {
          RCLCPP_WARN(
            get_logger(), "MoveIt Servo start request returned false: %s",
            response->message.c_str());
        }
      });
    return true;
  }

  bool stopMoveItServo()
  {
    if (!servo_stop_client_) {
      publishStatus("handoff: MoveIt Servo stop client is not initialized");
      return false;
    }

    if (!servo_stop_client_->wait_for_service(std::chrono::milliseconds(200))) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "MoveIt Servo stop service unavailable: /servo_node/stop_servo");
      publishStatus("handoff: /servo_node/stop_servo unavailable; continuing joint1 turn");
      return false;
    }

    servo_stop_client_->async_send_request(
      std::make_shared<Trigger::Request>(),
      [this](rclcpp::Client<Trigger>::SharedFuture future) {
        const auto response = future.get();
        if (!response->success) {
          RCLCPP_WARN(
            get_logger(), "MoveIt Servo stop request returned false: %s",
            response->message.c_str());
        } else {
          RCLCPP_INFO(get_logger(), "MoveIt Servo stopped before handoff");
        }
      });

    servo_start_requested_ = false;
    publishStatus("handoff: MoveIt Servo stop requested before joint1 turn", true);
    return true;
  }

  void publishStop()
  {
    geometry_msgs::msg::TwistStamped stop;
    stop.header.stamp = now();
    stop.header.frame_id = target_frame_;
    twist_pub_->publish(stop);
  }

  void publishBaseStop()
  {
    if (!base_cmd_vel_pub_) {
      return;
    }
    geometry_msgs::msg::Twist stop;
    base_cmd_vel_pub_->publish(stop);
  }

  void publishEefAutoInitEnable(bool enabled)
  {
    if (!eef_auto_init_enable_pub_) {
      return;
    }
    std_msgs::msg::Bool msg;
    msg.data = enabled;
    eef_auto_init_enable_pub_->publish(msg);
  }

  void publishBaseHold(bool enabled)
  {
    if (!base_hold_pub_) {
      return;
    }
    std_msgs::msg::Bool msg;
    msg.data = enabled;
    base_hold_pub_->publish(msg);
  }

  void publishStatus(const std::string & text, bool force = false)
  {
    const auto stamp = now();
    if (!force && last_status_stamp_.nanoseconds() != 0 &&
        (stamp - last_status_stamp_).seconds() < 0.5) {
      return;
    }
    std_msgs::msg::String msg;
    msg.data = text;
    status_pub_->publish(msg);
    last_status_stamp_ = stamp;
  }

  std::optional<CameraInfo> fallbackEefCameraInfo() const
  {
    if (!allow_eef_camera_info_fallback_) {
      return std::nullopt;
    }

    CameraInfo info;
    info.fx = eef_camera_fallback_fx_;
    info.fy = eef_camera_fallback_fy_;
    info.width = static_cast<std::uint32_t>(eef_camera_fallback_width_px_);
    info.height = static_cast<std::uint32_t>(eef_camera_fallback_height_px_);
    info.cx = 0.5 * static_cast<double>(info.width);
    info.cy = 0.5 * static_cast<double>(info.height);
    info.frame_id =
      eef_camera_frame_override_.empty() ? "eef_usb_camera_optical_frame" :
      eef_camera_frame_override_;
    return info;
  }

  void assignCargoId()
  {
    std::ostringstream id;
    id << cargo_id_prefix_ << "-" << std::setw(6) << std::setfill('0') << cargo_sequence_next_++;
    current_cargo_id_ = id.str();
    std_msgs::msg::String msg;
    msg.data = current_cargo_id_;
    cargo_current_id_pub_->publish(msg);
  }

  void publishCargoEvent(const std::string & event, bool force_current = false)
  {
    if (current_cargo_id_.empty() || force_current) {
      std_msgs::msg::String current;
      current.data = current_cargo_id_;
      cargo_current_id_pub_->publish(current);
    }

    const auto stamp = now();
    const auto stamp_ns = stamp.nanoseconds();
    std_msgs::msg::String msg;
    std::ostringstream event_json;
    event_json << "{\"cargo_id\":\"" << current_cargo_id_
               << "\",\"event\":\"" << event
               << "\",\"stamp\":{\"sec\":" << stamp_ns / 1000000000LL
               << ",\"nanosec\":" << stamp_ns % 1000000000LL
               << "}}";
    msg.data = event_json.str();
    cargo_event_pub_->publish(msg);
  }

  std::string bbox_topic_;
  std::string fallback_bbox_topic_;
  std::string eef_bbox_topic_;
  std::string eef_init_bbox_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string eef_camera_info_topic_;
	  std::string eef_auto_init_enable_topic_;
	  std::string base_hold_topic_;
  std::string close_range_ready_topic_;
  std::string stable_object_topic_;
	  std::string twist_topic_;
	  std::string base_cmd_vel_topic_;
	  std::string joint_state_topic_;
	  std::string joint_trajectory_topic_;
  std::string handoff_joint_trajectory_topic_;
	  std::string start_topic_;
  std::string cancel_topic_;
  std::string status_topic_;
  std::string cargo_event_topic_;
  std::string cargo_current_id_topic_;
  std::string cargo_id_prefix_;
  std::string gripper_action_name_;
  std::string target_frame_;
  std::string end_effector_frame_;
  std::string joint4_pose_frame_;
  std::string gripper_pose_frame_;
  std::string camera_frame_override_;
  std::string eef_camera_frame_override_;
  std::string aruco_pose_topic_{"/target/aruco_pose"};

  bool auto_start_{false};
  bool auto_start_on_bbox_{false};
  bool use_fallback_bbox_for_control_{false};
  bool start_servo_on_start_{true};
  bool open_gripper_on_start_{true};
  bool close_gripper_on_arrival_{true};
  bool require_visual_grasp_confirmation_{true};
  bool use_eef_refinement_{true};
  bool wait_for_base_approach_{false};
  bool auto_init_eef_tracker_from_object_{true};
	  bool allow_eef_camera_info_fallback_{true};
	  bool use_depthless_triangulation_{false};
	  bool use_color_triangulation_after_min_depth_{false};
  bool require_close_range_ready_for_pregrasp_{true};
  bool aruco_center_align_enabled_{true};
  bool initial_aruco_joint1_aligned_{false};
  bool aruco_require_center_before_grasp_{true};
  bool release_base_hold_after_handoff_{true};
	  bool use_joint_pregrasp_{true};
	  double command_rate_hz_{20.0};
  double max_target_age_s_{0.6};
  double close_range_ready_max_age_s_{0.6};
  double linear_gain_{0.9};
  double max_linear_speed_{0.025};
  double position_tolerance_m_{0.035};
  int close_after_stable_cycles_{8};
  double grasp_completion_front_max_age_s_{0.6};
  double grasp_completion_eef_lost_timeout_s_{0.6};
  int depth_roi_radius_px_{5};
  double depth_unit_scale_{0.001};
  double min_valid_depth_m_{0.12};
  double max_valid_depth_m_{1.2};
  double depth_std_max_m_{0.08};
  double depth_min_fill_ratio_{0.25};
  double depth_jump_limit_m_{0.10};
  int stable_depth_frames_{3};
  int min_depth_samples_{30};
  double grasp_offset_x_{0.0};
  double grasp_offset_y_{0.0};
  double grasp_offset_z_{0.0};
  double eef_refinement_switch_distance_m_{0.12};
  double eef_refinement_start_depth_m_{0.47};
  double eef_yolo_pre_enable_depth_m_{0.47};
  double min_depth_handoff_margin_m_{0.02};
  double min_depth_handoff_bbox_area_ratio_{0.30};
  double min_depth_handoff_bbox_height_ratio_{0.75};
  double eef_refinement_start_object_x_m_{0.50};
  double arm_start_max_error_m_{0.40};
  double arm_start_max_object_x_m_{0.60};
  double object_height_m_{0.10};
  double eef_init_bbox_min_size_px_{32.0};
  double eef_init_bbox_max_size_px_{180.0};
  double eef_init_bbox_padding_scale_{1.6};
  double eef_init_bbox_republish_period_s_{0.5};
  double eef_final_depth_m_{0.08};
  double aruco_center_pose_timeout_s_{0.5};
  double aruco_gripper_center_x_offset_m_{0.0};
  double aruco_gripper_center_y_offset_m_{0.0};
  double aruco_center_x_tolerance_m_{0.012};
  double aruco_center_y_tolerance_m_{0.015};
  double aruco_joint1_align_gain_{0.6};
  double aruco_joint1_align_max_step_rad_{0.020};
  double aruco_joint1_align_sign_{-1.0};
  double aruco_center_lateral_gain_{0.6};
  double aruco_center_max_linear_speed_{0.010};
  double object_pregrasp_standoff_m_{0.08};
  double object_pregrasp_min_z_m_{0.50};
	  double object_pregrasp_lower_standoff_m_{0.02};
	  double object_pregrasp_min_lower_z_m_{0.12};
	  bool object_pregrasp_enable_lowering_{false};
	  std::array<std::string, 4> arm_joint_names_{{"joint1", "joint2", "joint3", "joint4"}};
	  std::vector<double> joint_pregrasp_ready_positions_{0.0, 0.65, -0.85, -1.20};
	  bool joint_pregrasp_preserve_gripper_roll_{true};
  bool hold_joint4_during_extension_{false};
  std::optional<double> extension_joint4_reference_;
  double pregrasp_roll_joint2_weight_{1.0};
  double pregrasp_roll_joint3_weight_{1.0};
  double pregrasp_roll_joint4_weight_{1.0};
	  bool joint_pregrasp_reverse_joint3_delta_{true};
	  double joint_pregrasp_hold_current_duration_s_{0.0};
	  double joint_pregrasp_move_duration_s_{1.2};
	  bool joint_pregrasp_joint3_lead_enabled_{false};
	  double joint_pregrasp_joint3_lead_duration_ratio_{0.45};
	  double joint_pregrasp_joint3_lead_fraction_{1.0};
	  int joint_pregrasp_sync_steps_{1};
	  double joint_pregrasp_settle_s_{0.5};
	  double joint_pregrasp_tolerance_rad_{0.04};
	  double joint_pregrasp_republish_period_s_{1.0};
    double joint_pregrasp_joint2_max_step_rad_{0.05};
	  double joint_pregrasp_joint3_max_step_rad_{0.06};
    double joint_pregrasp_joint4_max_step_rad_{0.04};
	  std::vector<double> joint_pregrasp_min_positions_{-3.14, -1.79, -0.94, -1.79};
	  std::vector<double> joint_pregrasp_max_positions_{3.14, 1.57, 1.38, 2.04};
	  double eef_center_tolerance_px_{18.0};
  double eef_close_tolerance_px_{18.0};
  double eef_depth_tolerance_m_{0.018};
  double eef_refine_lateral_gain_{0.8};
  double eef_refine_depth_gain_{0.5};
  double eef_refine_max_linear_speed_{0.012};
  bool use_eef_rpy_refinement_{true};
  bool eef_hold_current_rpy_{true};
  bool eef_hold_stay_roll_{false};
  double eef_target_roll_rad_{0.0};
  double eef_target_pitch_rad_{0.0};
  double eef_target_yaw_rad_{0.0};
  double eef_rpy_tolerance_rad_{0.12};
  double eef_rpy_gain_{0.8};
  double eef_refine_max_angular_speed_{0.25};
  bool eef_forward_after_align_{true};
  double eef_forward_distance_m_{0.05};
  double eef_forward_speed_mps_{0.012};
  double eef_forward_fixed_duration_s_{0.0};
  double eef_forward_gate_timeout_s_{8.0};
  double eef_forward_start_tolerance_px_{18.0};
  bool eef_forward_use_joint_nudge_{false};
  double eef_forward_joint2_delta_rad_{0.025};
  double eef_forward_joint3_delta_rad_{-0.035};
  double eef_forward_joint_nudge_duration_s_{0.75};
  double eef_forward_joint_nudge_period_s_{0.80};
  double eef_forward_joint3_first_duration_ratio_{0.65};
  bool eef_forward_joint4_after_joint3_complete_{true};
  double eef_forward_joint3_complete_delta_rad_{0.14};
  double eef_forward_joint3_complete_tolerance_rad_{0.015};
  double eef_forward_joint4_finish_tolerance_rad_{0.015};
  bool eef_forward_use_bezier_ik_{true};
  bool eef_forward_use_analytic_ik_{true};
  double eef_forward_ik_link2_x_m_{0.024};
  double eef_forward_ik_link2_z_m_{0.128};
  double eef_forward_ik_link3_m_{0.124};
  double eef_forward_ik_tool_m_{0.126};
  double eef_forward_ik_pitch_weight_{0.35};
  double eef_forward_ik_damping_{0.03};
  int eef_forward_ik_iterations_{16};
  double eef_forward_bezier_total_duration_s_{5.0};
  double eef_forward_bezier_progress_step_{0.08};
  double eef_forward_bezier_joint2_total_delta_rad_{0.10};
  double eef_forward_bezier_joint3_total_delta_rad_{0.14};
  double eef_forward_bezier_joint2_max_step_rad_{0.02};
  double eef_forward_bezier_joint3_max_step_rad_{0.02};
  double eef_forward_bezier_joint4_max_step_rad_{0.0};
  bool eef_forward_bezier_preserve_roll_{true};
  double eef_forward_roll_joint2_weight_{0.0};
  double eef_forward_roll_joint3_weight_{1.0};
  double eef_forward_roll_joint4_weight_{1.0};
  double eef_forward_joint4_rpy_roll_gain_{0.0};
  double eef_forward_joint4_rpy_roll_max_delta_rad_{0.0};
  double eef_forward_joint4_max_delta_rad_{0.0};
  double eef_forward_joint4_ground_parallel_limit_rad_{-1.05};
  double eef_forward_joint4_ground_limit_tolerance_rad_{0.01};
  bool eef_forward_joint4_down_positive_{true};
  double gripper_down_joint4_offset_rad_{0.0};
  bool close_on_front_bbox_shrink_{false};
  double front_bbox_close_area_ratio_{0.60};
  bool close_on_eef_bbox_shrink_{false};
  double eef_bbox_close_area_ratio_{0.60};
  double eef_forward_min_advance_before_close_m_{0.025};
  bool close_after_full_eef_forward_extension_{false};
  bool handoff_after_grasp_{false};
  double handoff_lift_joint2_delta_rad_{0.25};
  double handoff_retract_joint3_delta_rad_{0.0};
  double handoff_retract_distance_m_{0.05};
  bool handoff_retract_hold_z_{false};
  bool handoff_restore_grasp_pose_{false};
  double handoff_lift_joint_tolerance_rad_{0.10};
  bool handoff_lower_before_release_{true};
  double handoff_retract_z_tolerance_m_{0.005};
  double handoff_place_duration_s_{1.0};
  double handoff_grasp_settle_s_{0.5};
  double handoff_place_joint2_delta_rad_{-0.20};
  double handoff_joint_move_duration_s_{1.0};
  double handoff_joint_settle_s_{0.4};
  double handoff_republish_period_s_{0.25};
  double handoff_rotate_angle_rad_{3.14159265358979323846};
  double handoff_rotate_odd_angle_deg_{180.0};
  double handoff_rotate_even_angle_deg_{180.0};
  int handoff_pick_sequence_index_{0};
  bool handoff_center_joint1_before_rotate_{true};
  double handoff_joint1_center_target_rad_{0.0};
  double handoff_joint1_center_tolerance_rad_{0.03};
  double handoff_rotate_angular_speed_rad_s_{0.45};
  double handoff_release_settle_s_{0.5};
  double handoff_lift_duration_s_{1.0};
  std::vector<double> handoff_stay_joint_positions_{0.104311, 0.027612, -0.001534, -1.638291};
  double triangulation_extend_x_m_{0.25};
  double triangulation_extend_y_m_{0.0};
  double triangulation_extend_z_m_{0.12};
  double triangulation_extend_tolerance_m_{0.025};
  double triangulation_extend_gain_{0.7};
  double triangulation_extend_max_speed_{0.015};
  double triangulation_min_range_m_{0.06};
  double triangulation_max_range_m_{1.0};
  double triangulation_max_ray_gap_m_{0.10};
  bool use_eef_front_camera_extrinsic_override_{false};
  double eef_front_camera_offset_x_m_{0.0};
  double eef_front_camera_offset_y_m_{0.0};
  double eef_front_camera_offset_z_m_{0.0};
  double color_triangulation_base_stop_object_x_m_{0.30};
  double color_triangulation_min_object_x_m_{0.05};
  int eef_camera_fallback_width_px_{640};
  int eef_camera_fallback_height_px_{480};
  double eef_camera_fallback_fx_{554.0};
  double eef_camera_fallback_fy_{554.0};
  double gripper_open_position_{0.025};
  double gripper_close_position_{-0.015};
  double gripper_max_effort_{-1.0};
  bool gripper_width_control_enabled_{true};
  double gripper_fallback_object_width_m_{0.06};
  double gripper_finger_home_half_gap_m_{0.021};
  double gripper_pre_grasp_clearance_m_{0.012};
  double gripper_grasp_compression_m_{0.002};
  double gripper_grasp_clearance_m_{0.0};
  double gripper_grasp_width_scale_{1.0};
  double gripper_min_position_{-0.010};
  double gripper_max_position_{0.019};
  double gripper_min_measured_object_width_m_{0.01};
  double gripper_max_measured_object_width_m_{0.08};
  int cargo_sequence_next_{1};
  std::string current_cargo_id_;

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr bbox_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr init_bbox_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr eef_bbox_sub_;
	  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
	  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
	  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr eef_camera_info_sub_;
	  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr resume_held_sub_;
	  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr cancel_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr continue_grasp_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr continue_handoff_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr close_range_ready_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr stable_object_sub_;
	  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
	  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr base_cmd_vel_pub_;
	  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr handoff_joint_trajectory_pub_;
	  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cargo_event_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cargo_current_id_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr eef_init_bbox_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr eef_auto_init_enable_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr base_hold_pub_;
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_client_;
  rclcpp::Client<Trigger>::SharedPtr servo_start_client_;
  rclcpp::Client<Trigger>::SharedPtr servo_stop_client_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr aruco_pose_sub_;
  rclcpp::Time latest_aruco_pose_stamp_{0, 0, RCL_ROS_TIME};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex data_mutex_;
  std::optional<Bbox> latest_bbox_;
  std::optional<Bbox> latest_eef_bbox_;
  std::optional<CameraInfo> latest_camera_info_;
  std::optional<CameraInfo> latest_eef_camera_info_;
  std::optional<double> latest_object_width_m_;
  std::optional<double> latest_object_depth_m_;
  rclcpp::Time latest_object_depth_stamp_;
  std::optional<geometry_msgs::msg::PointStamped> latest_depth_object_in_target_;
	  rclcpp::Time latest_depth_object_stamp_;
  std::optional<geometry_msgs::msg::PointStamped> latest_stable_object_point_;
  std::optional<geometry_msgs::msg::PoseStamped> latest_aruco_pose_;
  rclcpp::Time latest_stable_object_stamp_;
  std::optional<double> depth_filter_last_accepted_m_;
  std::optional<double> depth_filter_pending_m_;
  int depth_filter_stable_count_{0};
  std::string last_depth_filter_status_;
  std::optional<geometry_msgs::msg::PointStamped> eef_refinement_object_in_target_;
  bool eef_refinement_use_bbox_center_{false};
	  sensor_msgs::msg::Image::ConstSharedPtr latest_depth_;
	  std::optional<std::array<double, 4>> latest_arm_joint_positions_;
	  rclcpp::Time latest_joint_state_stamp_;

	  bool active_{false};
  bool done_{false};
  bool open_sent_{false};
  bool close_sent_{false};
  bool pause_before_grasp_{false};
  bool pause_before_rotate_{false};
  bool handoff_resume_requested_{false};
  bool grasp_resume_requested_{false};
  bool pregrasp_pause_reported_{false};
  bool eef_bbox_seen_after_close_{false};
	  bool eef_refinement_requested_{false};
	  bool object_pregrasp_horizontal_done_{false};
	  bool joint_pregrasp_sent_{false};
	  bool joint_pregrasp_done_{false};
  bool close_range_ready_{false};
  rclcpp::Time close_range_ready_stamp_;
  bool close_range_pregrasp_latched_{false};
  bool eef_forward_advance_active_{false};
  bool safety_abort_latched_{false};
  bool safety_abort_ignore_reported_{false};
  std::optional<double> front_bbox_area_at_eef_forward_start_;
  std::optional<double> eef_bbox_area_at_eef_forward_start_;
  std::optional<std::array<double, 4>> eef_forward_start_joint_positions_;
	  bool min_depth_reached_{false};
  bool servo_start_requested_{false};
  std::optional<std::array<double, 4>> joint_pregrasp_target_;
  std::optional<std::array<double, 4>> joint_pregrasp_controller_target_;
  std::optional<std::array<double, 4>> joint_pregrasp_controller_step_target_;
  std::optional<std::array<double, 3>> eef_rpy_reference_;
  std::optional<double> eef_stay_roll_reference_;
	  GraspStage stage_{GraspStage::DEPTH_APPROACH};
	  int stable_cycles_{0};
	  rclcpp::Time last_status_stamp_;
	  rclcpp::Time last_eef_init_bbox_stamp_;
  rclcpp::Time joint_pregrasp_start_stamp_;
  rclcpp::Time joint_pregrasp_last_publish_stamp_;
  rclcpp::Time eef_forward_last_joint_nudge_stamp_;
  rclcpp::Time eef_forward_start_stamp_;
  rclcpp::Time handoff_stage_start_stamp_;
  rclcpp::Time handoff_last_publish_stamp_;
  std::optional<std::array<double, 4>> handoff_lift_controller_target_;
  std::optional<std::array<double, 4>> handoff_place_controller_target_;
  std::optional<std::array<double, 4>> latest_arm_joint_velocities_;
  std::optional<double> latest_gripper_position_;
  std::optional<rclcpp::Time> handoff_place_stable_since_;
  std::optional<rclcpp::Time> handoff_retract_stable_since_;
  std::optional<rclcpp::Time> handoff_rotation_stable_since_;
  std::optional<std::array<double, 4>> handoff_retract_start_joint_positions_;
  std::optional<std::array<double, 4>> handoff_grasp_controller_pose_;
  std::optional<std::array<double, 4>> handoff_stay_controller_target_;
  std::optional<double> handoff_joint1_target_rad_;
  bool handoff_joint1_centering_{false};
  bool handoff_joint1_centered_for_rotate_{false};
  double eef_forward_start_x_m_{0.0};
	};

}  // namespace mp_control

#ifndef MP_CONTROL_TESTING
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<mp_control::MpControlNode>());
  rclcpp::shutdown();
  return 0;
}
#endif
