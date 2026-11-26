// Copyright 2026 ktj
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "control_msgs/msg/dynamic_joint_state.hpp"
#include "leader_platooning_beacon/dynamic_joint_state_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "turtlebot3_msgs/msg/sensor_state.hpp"

using namespace std::chrono_literals;

class SensorStateFromDynamicJointState : public rclcpp::Node
{
public:
  SensorStateFromDynamicJointState()
  : Node("sensor_state_from_dynamic_joint_state")
  {
    const auto input_topic = declare_parameter<std::string>(
      "dynamic_joint_states_topic", "/dynamic_joint_states");
    const auto output_topic = declare_parameter<std::string>(
      "sensor_state_topic", "/sensor_state");
    battery_sensor_name_ = declare_parameter<std::string>("battery_sensor_name", "battery");
    left_wheel_name_ = declare_parameter<std::string>("left_wheel_name", "wheel_left_joint");
    right_wheel_name_ = declare_parameter<std::string>("right_wheel_name", "wheel_right_joint");
    publish_period_s_ = declare_parameter<double>("publish_period_s", 0.1);
    encoder_ticks_per_revolution_ = declare_parameter<double>("encoder_ticks_per_revolution", 4096.0);

    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort().durability_volatile();
    sensor_pub_ = create_publisher<turtlebot3_msgs::msg::SensorState>(output_topic, pub_qos);

    auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local();
    dynamic_state_sub_ = create_subscription<control_msgs::msg::DynamicJointState>(
      input_topic,
      sub_qos,
      std::bind(&SensorStateFromDynamicJointState::dynamicStateCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "sensor_state relay started: %s -> %s",
      input_topic.c_str(),
      output_topic.c_str());
  }

private:
  int32_t radiansToEncoderTicks(const double radians) const
  {
    if (!std::isfinite(radians) || encoder_ticks_per_revolution_ <= 0.0) {
      return 0;
    }

    const auto ticks = radians * encoder_ticks_per_revolution_ / (2.0 * M_PI);
    return static_cast<int32_t>(std::llround(ticks));
  }

  void dynamicStateCallback(const control_msgs::msg::DynamicJointState::SharedPtr msg)
  {
    const auto now = get_clock()->now();
    if (last_publish_time_.nanoseconds() > 0 &&
      (now - last_publish_time_).seconds() < publish_period_s_)
    {
      return;
    }

    turtlebot3_msgs::msg::SensorState sensor;
    sensor.header = msg->header;

    sensor.bumper = 0;
    sensor.cliff = 0.0f;
    sensor.sonar = 0.0f;
    sensor.illumination = 0.0f;
    sensor.led = 0;
    sensor.button = 0;
    sensor.torque = true;
    sensor.left_encoder = radiansToEncoderTicks(
      leader_platooning_beacon::find_state_value(*msg, left_wheel_name_, "position").value_or(0.0));
    sensor.right_encoder = radiansToEncoderTicks(
      leader_platooning_beacon::find_state_value(*msg, right_wheel_name_, "position").value_or(0.0));
    sensor.battery = static_cast<float>(
      leader_platooning_beacon::find_state_value(*msg, battery_sensor_name_, "voltage").value_or(
        std::numeric_limits<float>::quiet_NaN()));

    sensor_pub_->publish(sensor);
    last_publish_time_ = now;
  }

  std::string battery_sensor_name_;
  std::string left_wheel_name_;
  std::string right_wheel_name_;
  double publish_period_s_;
  double encoder_ticks_per_revolution_;
  rclcpp::Time last_publish_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Publisher<turtlebot3_msgs::msg::SensorState>::SharedPtr sensor_pub_;
  rclcpp::Subscription<control_msgs::msg::DynamicJointState>::SharedPtr dynamic_state_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorStateFromDynamicJointState>());
  rclcpp::shutdown();
  return 0;
}
