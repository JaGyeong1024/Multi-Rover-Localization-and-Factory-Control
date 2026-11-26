"""Final global pose/velocity output: `/{robot}/global/position` (PoseStamped, z=0) and
`/{robot}/global/velocity` (TwistStamped), frame uwb_global. The latest estimate is
re-published at `rate` Hz with the source stamp; rovers get it through the domain_bridge.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped


def _stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class GlobalPositionNode(Node):
    def __init__(self):
        super().__init__('global_position_node')
        rate = self.declare_parameter('rate', 30.0).value           # output rate [Hz]
        # Stale guard: stop re-publishing once the source stamp is older than this.
        self.stale_timeout = float(self.declare_parameter('stale_timeout', 0.5).value)  # [s]
        self.frame = self.declare_parameter('frame_id', 'uwb_global').value
        robots = self.declare_parameter('robots', ['leader', 'follower']).value

        self.robots = {}
        for name in robots:
            pose_in = self.declare_parameter(f'{name}.pose_in', f'/{name}/pose').value
            vel_in = self.declare_parameter(f'{name}.vel_in', f'/{name}/velocity').value
            st = {
                'pose': None, 'vel': None,
                # outputs (/global/*) use the default RELIABLE QoS
                'pose_pub': self.create_publisher(PoseStamped, f'/{name}/global/position', 10),
                'vel_pub': self.create_publisher(TwistStamped, f'/{name}/global/velocity', 10),
            }
            # inputs match the estimator: BEST_EFFORT (high rate, latest value wins)
            self.create_subscription(PoseStamped, pose_in,
                                     lambda m, s=st: s.__setitem__('pose', m), qos_profile_sensor_data)
            self.create_subscription(TwistStamped, vel_in,
                                     lambda m, s=st: s.__setitem__('vel', m), qos_profile_sensor_data)
            self.robots[name] = st
            self.get_logger().info(
                f'{name}: {pose_in} + {vel_in} -> /{name}/global/{{position,velocity}} @ {rate}Hz')

        self.create_timer(1.0 / float(rate), self._on_timer)

    def _on_timer(self):
        """Re-publish the latest estimate with its source stamp; missing/stale sources are skipped."""
        now = self.get_clock().now().nanoseconds * 1e-9
        for st in self.robots.values():
            m = st['pose']
            if m is not None and now - _stamp_s(m.header.stamp) <= self.stale_timeout:
                out = PoseStamped()
                out.header = m.header                          # keep source stamp
                out.pose.position.x = m.pose.position.x
                out.pose.position.y = m.pose.position.y
                out.pose.position.z = 0.0                      # 2D
                out.pose.orientation = m.pose.orientation      # keep heading
                st['pose_pub'].publish(out)
            v = st['vel']
            if v is not None and now - _stamp_s(v.header.stamp) <= self.stale_timeout:
                out = TwistStamped()
                out.header = v.header
                out.twist = v.twist
                st['vel_pub'].publish(out)


def main():
    rclpy.init()
    node = GlobalPositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
