"""Host -> rover global position output (domain 30 -> 25 / 73).

global_position_node publishes `/{robot}/global/position` (geometry_msgs/PoseStamped) on the
host domain 30; two domain_bridge instances forward it to the leader (25) and follower (73),
where it is the rover's control input. The input is the fused /{robot}/pose; override
`leader_in` / `follower_in` to feed another estimator (the output topics and types stay).

Run:    ros2 launch host_localization global_position.launch.py
Verify: ROS_DOMAIN_ID=73 ros2 topic echo /follower/global/position --once
        ROS_DOMAIN_ID=25 ros2 topic echo /leader/global/position --once
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('host_localization')
    return LaunchDescription([
        # Input topics of the output relay (the fused pose by default).
        DeclareLaunchArgument('leader_in', default_value='/leader/pose'),
        DeclareLaunchArgument('follower_in', default_value='/follower/pose'),
        DeclareLaunchArgument('leader_vel', default_value='/leader/velocity'),
        DeclareLaunchArgument('follower_vel', default_value='/follower/velocity'),
        Node(
            package='host_localization',
            executable='global_position_node',
            name='global_position_node',
            output='screen',
            emulate_tty=True,
            parameters=[{                          # output rate: the node's rate parameter
                'leader.pose_in': LaunchConfiguration('leader_in'),
                'follower.pose_in': LaunchConfiguration('follower_in'),
                'leader.vel_in': LaunchConfiguration('leader_vel'),
                'follower.vel_in': LaunchConfiguration('follower_vel'),
            }],
        ),
        # respawn: domain_bridge 0.5.0 can die at startup when several bridges start together
        # (see host_mission/launch/mission_bridge.launch.py); a dead bridge = rover gets no pose.
        # The config path is positional; rclcpp strips the --ros-args launch appends.
        Node(
            package='domain_bridge', executable='domain_bridge',
            arguments=[PathJoinSubstitution([pkg_share, 'config', 'bridge_host_to_follower.yaml'])],
            output='screen', emulate_tty=True,
            respawn=True, respawn_delay=2.0,
        ),
        Node(
            package='domain_bridge', executable='domain_bridge',
            arguments=[PathJoinSubstitution([pkg_share, 'config', 'bridge_host_to_leader.yaml'])],
            output='screen', emulate_tty=True,
            respawn=True, respawn_delay=2.0,
        ),
    ])
