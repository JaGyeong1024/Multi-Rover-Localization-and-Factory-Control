#!/usr/bin/env python3
"""rover_nav_ref — ROVER-SIDE INTERFACE REFERENCE for the planning & control teammate.

Shows ONLY the host↔rover contract: what the rover RECEIVES from the host and what it
SENDS back. The actual route planning (over the lane graph) and the cmd_vel controller
are the teammate's job — left as TODO(teammate B) below. This is NOT host runtime and
NOT the test mock (the host's runnable stand-in is host_mission/mock_rover_node).

Contract (full per-mode spec: docs/rover_interface_contract.md):
  IN   /{robot}/target_cmd       TargetCommand   target_node + mode(GOTO|HOLD|STOP|RELEASE)
  IN   /{robot}/global/position  PoseStamped     host localization = your pose feedback
  OUT  /{robot}/nav_feedback     NavFeedback     state + echo cmd_id
  OUT  /{robot}/cmd_vel          Twist           your controller output (rover-owned)
  DEADMAN: TargetCommand header.stamp stale (> deadman s) -> STOP.
"""
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from host_mission_interfaces.msg import NavFeedback, TargetCommand


class RoverNavRef(Node):
    def __init__(self):
        super().__init__('rover_nav_ref')
        self.robot = self.declare_parameter('robot', 'leader').value
        self.deadman = float(self.declare_parameter('deadman', 0.5).value)

        self.pose = None        # latest /{robot}/global/position (control feedback)
        self.cmd = None         # latest TargetCommand
        self.last_cmd_t = None

        cmd_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(TargetCommand, f'/{self.robot}/target_cmd', self._on_cmd, cmd_qos)
        self.create_subscription(PoseStamped, f'/{self.robot}/global/position', self._on_pose, 10)
        self.fb_pub = self.create_publisher(NavFeedback, f'/{self.robot}/nav_feedback', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.robot}/cmd_vel', 10)
        self.create_timer(0.1, self._control)   # your real control loop runs faster
        self.get_logger().info(
            f'rover_nav_ref[{self.robot}] — INTERFACE REFERENCE (routing/control = TODO teammate B)')

    def _on_pose(self, msg):
        self.pose = msg.pose
        # TODO(teammate B): use as control feedback (tracking error vs your planned path).

    def _on_cmd(self, msg):
        if msg.robot != self.robot:
            return
        self.cmd = msg
        self.last_cmd_t = self.get_clock().now().nanoseconds * 1e-9
        # TODO(teammate B): on a NEW cmd_id with mode GOTO, plan a route from your current
        # node to msg.target_node over your lane graph (reuse tools/path_editor/graph_route).

    def _control(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        cmd = self.cmd
        # DEADMAN — stale/absent command => STOP (safety; see contract §4)
        if self.last_cmd_t is None or (now - self.last_cmd_t) > self.deadman:
            self._stop_and_report('STOPPED' if self.last_cmd_t else 'IDLE')
            return
        # per-mode behavior (contract §2)
        if cmd.mode == 'GOTO':
            # TODO(teammate B): pure-pursuit track your route toward cmd.target_node using
            #   self.pose; publish Twist to cmd_vel; report TRACKING, then ARRIVED on arrival.
            self._report(cmd.cmd_id, 'TRACKING')
        elif cmd.mode == 'HOLD':
            self._stop_and_report('HOLDING', cmd.cmd_id)
        elif cmd.mode == 'STOP':
            self._stop_and_report('STOPPED', cmd.cmd_id)
        elif cmd.mode == 'RELEASE':
            self._stop_and_report('IDLE', cmd.cmd_id)

    def _stop_and_report(self, state, cmd_id=0):
        self.cmd_vel_pub.publish(Twist())   # zero velocity
        self._report(cmd_id, state)

    def _report(self, cmd_id, state):
        fb = NavFeedback()
        fb.header.stamp = self.get_clock().now().to_msg()
        fb.robot = self.robot
        fb.cmd_id = int(cmd_id)
        fb.state = state
        fb.current_node = -1   # TODO(teammate B): nearest lane node from self.pose
        fb.progress = 0.0
        self.fb_pub.publish(fb)


def main():
    rclpy.init()
    node = RoverNavRef()
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
