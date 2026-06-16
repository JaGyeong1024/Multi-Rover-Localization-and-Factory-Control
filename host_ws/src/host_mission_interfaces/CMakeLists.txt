cmake_minimum_required(VERSION 3.8)
project(host_mission_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Part.msg"
  "msg/MissionItem.msg"
  "msg/MissionQueue.msg"
  "msg/RobotStatus.msg"
  "msg/TargetCommand.msg"
  "msg/NavFeedback.msg"
  "msg/PickPlaceCommand.msg"
  "msg/PickPlaceStatus.msg"
  "msg/LeaderRoute.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
