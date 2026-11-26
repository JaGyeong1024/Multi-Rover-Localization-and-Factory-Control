cmake_minimum_required(VERSION 3.8)
project(leader_platooning_beacon)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(turtlebot3_msgs REQUIRED)

include_directories(include)

# /dynamic_joint_states (ros2_control) -> /battery_state
add_executable(
  battery_state_from_dynamic_joint_state_node
  src/battery_state_from_dynamic_joint_state_node.cpp
)
ament_target_dependencies(
  battery_state_from_dynamic_joint_state_node
  rclcpp
  sensor_msgs
  control_msgs
)

# /dynamic_joint_states (ros2_control) -> /sensor_state
add_executable(
  sensor_state_from_dynamic_joint_state_node
  src/sensor_state_from_dynamic_joint_state_node.cpp
)
ament_target_dependencies(
  sensor_state_from_dynamic_joint_state_node
  rclcpp
  control_msgs
  turtlebot3_msgs
)

install(TARGETS
  battery_state_from_dynamic_joint_state_node
  sensor_state_from_dynamic_joint_state_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

ament_package()
