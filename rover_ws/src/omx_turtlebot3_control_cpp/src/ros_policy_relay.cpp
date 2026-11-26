// PPO residual on the robot's existing, empirically tuned mp_control path.
// ros2_control remains the single OpenCR owner. No serial port is opened here.
#include "omx_turtlebot3_control_cpp/control.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <iostream>

class PolicyRelay : public rclcpp::Node {
  using Trajectory=trajectory_msgs::msg::JointTrajectory;
 public:
  PolicyRelay() : Node("omx_v6_ros_policy_relay"), actor_(declare_parameter<std::string>("model_dir", "")) {
    execute_=declare_parameter<bool>("execute",false);
    max_raw_joint3_delta_=declare_parameter<double>("max_raw_joint3_delta_rad",.17);
    if(!std::isfinite(max_raw_joint3_delta_)||max_raw_joint3_delta_<=0||max_raw_joint3_delta_>.25)
      throw std::runtime_error("Invalid raw joint3 step bound");
    input_=declare_parameter<std::string>("input_topic","/omx_v6/nominal_joint_trajectory");
    output_=declare_parameter<std::string>("output_topic","/arm_controller/joint_trajectory");
    if(input_==output_) throw std::runtime_error("Relay topic loop");
    out_=create_publisher<Trajectory>(output_,10);
    cancel_=create_publisher<std_msgs::msg::Bool>("/mp_control/cancel",10);
    actions_=create_publisher<std_msgs::msg::Float64MultiArray>("/omx_v6/policy_action",10);
    goals_=create_publisher<std_msgs::msg::Float64MultiArray>("/omx_v6/policy_goal",10);
    ready_=create_publisher<std_msgs::msg::Bool>("/omx_v6/policy_ready",rclcpp::QoS(1).transient_local());
    joints_=create_subscription<sensor_msgs::msg::JointState>("/joint_states",rclcpp::SensorDataQoS(),[this](sensor_msgs::msg::JointState::ConstSharedPtr m){
      for(size_t i=0;i<4;++i) {
        auto it=std::find(m->name.begin(),m->name.end(),"joint"+std::to_string(i+1));
        if(it==m->name.end()||size_t(it-m->name.begin())>=m->position.size()) return;
        q_[i]=m->position[it-m->name.begin()]; if(!std::isfinite(q_[i])) return;
      }
      joint_time_=omx_v6::now();
    });
    observation_=create_subscription<std_msgs::msg::Float32MultiArray>("/omx_v6/observation",10,[this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m){
      if(m->data.size()!=69 || !std::all_of(m->data.begin(),m->data.end(),[](float x){return std::isfinite(x);})) return;
      std::copy_n(m->data.begin(),69,obs_.begin()); obs_time_=omx_v6::now();
      std_msgs::msg::Bool ready;ready.data=!fault_;ready_->publish(ready);
    });
    status_=create_subscription<std_msgs::msg::String>("/mp_control/status",rclcpp::QoS(10).transient_local(),[this](std_msgs::msg::String::ConstSharedPtr m){
      if(m->data.find("cargo_loaded:")!=std::string::npos) {RCLCPP_INFO(get_logger(),"MISSION_COMPLETED; arm remains at Stay");rclcpp::shutdown();}
    });
    reference_=create_subscription<Trajectory>(input_,10,[this](Trajectory::ConstSharedPtr m){forward(*m);});
    cancel_input_=create_subscription<std_msgs::msg::Bool>("/omx_v6/policy_cancel",10,[this](std_msgs::msg::Bool::ConstSharedPtr m){if(m->data)stop("Operator/gate cancel");});
    raw_reference_=create_subscription<Trajectory>("/arm_controller/joint_trajectory_raw",10,[this](Trajectory::ConstSharedPtr m){
      if(omx_v6::now()-joint_time_>.2 && !started_) return;
      if(m->joint_names!=std::vector<std::string>{"joint1","joint2","joint3","joint4"}) {stop("Unexpected raw joint order");return;}
      std::vector<double> signature;
      for(const auto &p:m->points) {
        if(p.positions.size()!=4) {stop("Invalid raw point");return;}
        signature.push_back(std::round(p.positions[2]*1e6)/1e6);
      }
      if(signature!=raw_signature_) {raw_signature_=signature;raw_joint3_reference_=q_[2];}
      auto transformed=*m;
      for(auto &p:transformed.points) {
        p.positions[2]=std::clamp(raw_joint3_reference_+std::clamp(raw_joint3_reference_-p.positions[2],-max_raw_joint3_delta_,max_raw_joint3_delta_),-2.70,1.38);
        if(p.velocities.size()==4) p.velocities[2]=-p.velocities[2];
        if(p.accelerations.size()==4) p.accelerations[2]=-p.accelerations[2];
        if(p.effort.size()==4) p.effort[2]=-p.effort[2];
      }
      forward(transformed);
    });
    timer_=create_wall_timer(std::chrono::milliseconds(50),[this]{
      if(started_&&!fault_&&(omx_v6::now()-joint_time_>.2||omx_v6::now()-obs_time_>.5)) stop("Live policy/joint input expired");
    });
    RCLCPP_INFO(get_logger(),"Original v6 ONNX: 973 parity vectors passed. execute=%d; mp_control reference + bounded C++ residual; no serial ownership",execute_);
  }
 private:
  void stop(const std::string &reason) {
    if(fault_) return;fault_=true;
    if(execute_) {std_msgs::msg::Bool c;c.data=true;cancel_->publish(c);}
    if(execute_&&omx_v6::now()-joint_time_<.2) {
      Trajectory hold;hold.joint_names={"joint1","joint2","joint3","joint4"};
      trajectory_msgs::msg::JointTrajectoryPoint p;p.positions.assign(q_.begin(),q_.end());p.time_from_start=rclcpp::Duration::from_seconds(.3);hold.points.push_back(p);out_->publish(hold);
    }
    RCLCPP_ERROR(get_logger(),"POLICY_RELAY_STOPPED: %s",reason.c_str());
  }
  void forward(const Trajectory &reference) {
    if(fault_) return;
    if(omx_v6::now()-joint_time_>.2||omx_v6::now()-obs_time_>.5) {
      if(started_) stop("Active reference lost live observations");
      else RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,"Waiting for live observations; no actuator trajectory sent");
      return;
    }
    if(reference.joint_names!=std::vector<std::string>{"joint1","joint2","joint3","joint4"}||reference.points.empty()||reference.points.size()>128) {stop("Unexpected reference trajectory contract");return;}
    try {
      // Each incoming legacy segment receives one frozen v6 residual step.
      // This is an explicit reference-relay trial, not the 9-phase runtime's IK.
      const auto network_action=actor_.infer(obs_);
      auto action=network_action;
      // These paths constrain height/pitch explicitly. Keep the coordinated
      // pitch-joint curve intact; yaw residuals cannot alter its Z coordinate.
      if(reference.header.frame_id=="mp_control_retract_z_hold" ||
         reference.header.frame_id=="mp_control_pitch_hold")
        for(size_t i=1;i<4;++i) action[i]=0;
      auto output=reference;
      const std::array<double,4> low{-3.14,-1.5,-2.70,-1.79},high{3.14,1.5,1.38,2.04};
      bool rotation=obs_[22]>.5; // phase one-hot starts at observation index 17; phase 5.
      for(auto &p:output.points) {
        if(p.positions.size()!=4) throw std::runtime_error("Invalid reference point");
        const double duration=p.time_from_start.sec+p.time_from_start.nanosec*1e-9;
        if(!std::isfinite(duration)||duration<=0) throw std::runtime_error("Invalid reference duration");
        for(size_t i=0;i<4;++i) {
          const double original=p.positions[i];
          if(!std::isfinite(original)||original<low[i]||original>high[i]) throw std::runtime_error("Reference outside mp_control motor-interface limits");
          if(std::abs(original-q_[i])>.9*duration+.05) throw std::runtime_error("Reference rate exceeds trial envelope");
          // v6 outputs bounded residuals, never absolute joint angles.
          if(!(rotation&&i>0)) p.positions[i]=std::clamp(original+.012*.12*action[i],low[i],high[i]);
        }
      }
      std_msgs::msg::Float64MultiArray a,g;a.data.assign(action.begin(),action.end());g.data=reference.points.back().positions;
      actions_->publish(a);goals_->publish(g);
      const double stamp=omx_v6::now();
      double change=1e30;
      if(last_nominal_.size()==4) {
        change=0;
        for(size_t i=0;i<4;++i) change=std::max(change,std::abs(g.data[i]-last_nominal_[i]));
      }
      // Reissuing an almost identical 1 s spline every 50 ms continually
      // restarts its acceleration and can leave the motor below one tick.
      // Let that segment run, retaining live inference/feedback/watchdogs.
      const bool dispatch=change>.01 || stamp-last_dispatch_>=last_duration_;
      if(execute_&&dispatch) {
        out_->publish(output);last_nominal_=g.data;last_dispatch_=stamp;
        const auto &d=reference.points.back().time_from_start;
        last_duration_=std::max(.05,d.sec+d.nanosec*1e-9);
      }
      started_=true;++count_;
      RCLCPP_INFO(get_logger(),"POLICY_REFERENCE %lu action=[%.6f,%.6f,%.6f,%.6f] nominal=[%.6f,%.6f,%.6f,%.6f] command=[%.6f,%.6f,%.6f,%.6f] executed=%d",count_,action[0],action[1],action[2],action[3],reference.points.back().positions[0],reference.points.back().positions[1],reference.points.back().positions[2],reference.points.back().positions[3],output.points.back().positions[0],output.points.back().positions[1],output.points.back().positions[2],output.points.back().positions[3],execute_&&dispatch);
    } catch(const std::exception &e) {stop(e.what());}
  }
  omx_v6::Actor actor_;omx_v6::Observation obs_{};omx_v6::Joints q_{};
  double max_raw_joint3_delta_{.17};
  double joint_time_{-1e30},obs_time_{-1e30};bool execute_{},fault_{},started_{};uint64_t count_{};
  std::string input_,output_;
  std::vector<double> raw_signature_;double raw_joint3_reference_{};
  std::vector<double> last_nominal_;double last_dispatch_{-1e30},last_duration_{};
  rclcpp::Publisher<Trajectory>::SharedPtr out_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr cancel_,ready_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr actions_,goals_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joints_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr observation_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr cancel_input_;
  rclcpp::Subscription<Trajectory>::SharedPtr reference_,raw_reference_;rclcpp::TimerBase::SharedPtr timer_;
};
int main(int argc,char **argv) {
  rclcpp::init(argc,argv);
  try {rclcpp::spin(std::make_shared<PolicyRelay>());}
  catch(const std::exception &e) {std::cerr<<e.what()<<std::endl;if(rclcpp::ok())rclcpp::shutdown();return 1;}
  if(rclcpp::ok())rclcpp::shutdown();return 0;
}
