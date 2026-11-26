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
#include <limits>
#include <memory>
#include <string>

#include "control_msgs/msg/dynamic_joint_state.hpp"
#include "leader_platooning_beacon/dynamic_joint_state_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/battery_state.hpp"

using namespace std::chrono_literals;

class BatteryStateFromDynamicJointState : public rclcpp::Node
{
public:
  BatteryStateFromDynamicJointState()
  : Node("battery_state_from_dynamic_joint_state")
  {
    const auto input_topic = declare_parameter<std::string>(
      "dynamic_joint_states_topic", "/dynamic_joint_states");
    const auto output_topic = declare_parameter<std::string>(
      "battery_state_topic", "/battery_state");
    battery_sensor_name_ = declare_parameter<std::string>("battery_sensor_name", "battery");
    publish_period_s_ = declare_parameter<double>("publish_period_s", 1.0);

    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort().durability_volatile();
    battery_pub_ = create_publisher<sensor_msgs::msg::BatteryState>(output_topic, pub_qos);

    auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local();
    dynamic_state_sub_ = create_subscription<control_msgs::msg::DynamicJointState>(
      input_topic,
      sub_qos,
      std::bind(&BatteryStateFromDynamicJointState::dynamicStateCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "battery relay started: %s[%s] -> %s",
      input_topic.c_str(),
      battery_sensor_name_.c_str(),
      output_topic.c_str());
  }

private:
  void dynamicStateCallback(const control_msgs::msg::DynamicJointState::SharedPtr msg)
  {
    const auto now = get_clock()->now();
    if (last_publish_time_.nanoseconds() > 0 &&
      (now - last_publish_time_).seconds() < publish_period_s_)
    {
      return;
    }

    const auto battery_interfaces =
      leader_platooning_beacon::find_joint_interfaces(*msg, battery_sensor_name_);
    if (!battery_interfaces) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,
        "battery sensor '%s' is missing from /dynamic_joint_states",
        battery_sensor_name_.c_str());
      return;
    }

    const auto & interfaces = **battery_interfaces;
    const auto voltage = leader_platooning_beacon::find_interface_value(interfaces, "voltage");
    const auto percentage = leader_platooning_beacon::find_interface_value(interfaces, "percentage");
    const auto design_capacity =
      leader_platooning_beacon::find_interface_value(interfaces, "design_capacity");
    const auto present = leader_platooning_beacon::find_interface_value(interfaces, "present");

    if (!voltage && !percentage) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,
        "battery sensor '%s' has neither voltage nor percentage interface",
        battery_sensor_name_.c_str());
      return;
    }

    sensor_msgs::msg::BatteryState battery;
    battery.header.stamp = msg->header.stamp;
    battery.header.frame_id = msg->header.frame_id;
    battery.voltage = voltage.value_or(std::numeric_limits<float>::quiet_NaN());
    battery.temperature = 0.0;
    battery.current = 0.0;
    battery.charge = 0.0;
    battery.capacity = 0.0;
    battery.design_capacity = design_capacity.value_or(0.0);
    battery.percentage = percentage.value_or(std::numeric_limits<float>::quiet_NaN());
    battery.power_supply_status = sensor_msgs::msg::BatteryState::POWER_SUPPLY_STATUS_UNKNOWN;
    battery.power_supply_health = sensor_msgs::msg::BatteryState::POWER_SUPPLY_HEALTH_UNKNOWN;
    battery.power_supply_technology = sensor_msgs::msg::BatteryState::POWER_SUPPLY_TECHNOLOGY_UNKNOWN;
    battery.present = present.value_or(0.0) > 0.5;

    battery_pub_->publish(battery);
    last_publish_time_ = now;
  }

  std::string battery_sensor_name_;
  double publish_period_s_;
  rclcpp::Time last_publish_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Publisher<sensor_msgs::msg::BatteryState>::SharedPtr battery_pub_;
  rclcpp::Subscription<control_msgs::msg::DynamicJointState>::SharedPtr dynamic_state_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BatteryStateFromDynamicJointState>());
  rclcpp::shutdown();
  return 0;
}
