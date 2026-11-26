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

#ifndef LEADER_PLATOONING_BEACON__DYNAMIC_JOINT_STATE_UTILS_HPP_
#define LEADER_PLATOONING_BEACON__DYNAMIC_JOINT_STATE_UTILS_HPP_

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include "control_msgs/msg/dynamic_joint_state.hpp"

namespace leader_platooning_beacon
{

inline std::optional<double> find_interface_value(
  const control_msgs::msg::InterfaceValue & interfaces,
  const std::string & interface_name)
{
  const auto it = std::find(
    interfaces.interface_names.begin(),
    interfaces.interface_names.end(),
    interface_name);
  if (it == interfaces.interface_names.end()) {
    return std::nullopt;
  }

  const auto index = static_cast<size_t>(std::distance(interfaces.interface_names.begin(), it));
  if (index >= interfaces.values.size() || !std::isfinite(interfaces.values[index])) {
    return std::nullopt;
  }
  return interfaces.values[index];
}

inline std::optional<const control_msgs::msg::InterfaceValue *> find_joint_interfaces(
  const control_msgs::msg::DynamicJointState & msg,
  const std::string & joint_or_sensor_name)
{
  const auto it = std::find(
    msg.joint_names.begin(),
    msg.joint_names.end(),
    joint_or_sensor_name);
  if (it == msg.joint_names.end()) {
    return std::nullopt;
  }

  const auto index = static_cast<size_t>(std::distance(msg.joint_names.begin(), it));
  if (index >= msg.interface_values.size()) {
    return std::nullopt;
  }
  return &msg.interface_values[index];
}

inline std::optional<double> find_state_value(
  const control_msgs::msg::DynamicJointState & msg,
  const std::string & joint_or_sensor_name,
  const std::string & interface_name)
{
  const auto interfaces = find_joint_interfaces(msg, joint_or_sensor_name);
  if (!interfaces) {
    return std::nullopt;
  }
  return find_interface_value(**interfaces, interface_name);
}

}  // namespace leader_platooning_beacon

#endif  // LEADER_PLATOONING_BEACON__DYNAMIC_JOINT_STATE_UTILS_HPP_
