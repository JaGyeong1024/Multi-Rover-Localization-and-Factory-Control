#!/usr/bin/env python3
"""mock_rover_node — HOST TEST AID (stands in for a rover's nav). NOT rover code.

Closes the coordinator loop with no real robot or localization: subscribes the host
TargetCommand, plans a route over the shared lane graph (reuses tools/path_editor/
graph_route), integrates motion along it, and republishes the real contract topic
/{robot}/global/position + NavFeedback. Deadman: stale TargetCommand -> STOP.

The real rover nav (routing + cmd_vel control) lives on the rover side — see the
interface reference rover_nav_ref + docs/rover_interface_contract.md.
"""
import json
import math
import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from host_mission_interfaces.msg import NavFeedback, TargetCommand


def _import_graph_route(path_editor_dir):
    d = os.path.expanduser(path_editor_dir)
    if d not in sys.path:
        sys.path.insert(0, d)
    import graph_route  # noqa: E402
    return graph_route


class MockRover(Node):
    def __init__(self):
        super().__init__('mock_rover')
        self.robot = self.declare_parameter('robot', 'leader').value
        self.frame = self.declare_parameter('frame_id', 'uwb_global').value
        self.speed = float(self.declare_parameter('speed', 0.5).value)
        self.rate = float(self.declare_parameter('rate', 30.0).value)
        self.deadman = float(self.declare_parameter('deadman', 0.5).value)
        sx = float(self.declare_parameter('start_x', 0.0).value)
        sy = float(self.declare_parameter('start_y', 0.0).value)
        ped = self.declare_parameter(
            'path_editor_dir', os.path.expanduser('~/capstone/tools/path_editor')).value

        graph_file = self.declare_parameter('graph_file', 'lane_graph.json').value
        share = get_package_share_directory('host_mission')
        with open(os.path.join(share, 'config', graph_file)) as f:
            g = json.load(f)
        self._gr = _import_graph_route(ped)
        self.nodes = {int(n['id']): (float(n['x']), float(n['y'])) for n in g['nodes']}
        self.adj = {i: [] for i in self.nodes}
        for e in g['edges']:
            a, b = int(e['from']), int(e['to'])
            w = math.hypot(self.nodes[a][0] - self.nodes[b][0], self.nodes[a][1] - self.nodes[b][1])
            self.adj[a].append((b, w))
            if not e.get('directed', False):
                self.adj[b].append((a, w))

        self.x, self.y, self.yaw = sx, sy, 0.0
        self.route_pts = []
        self.cmd_id = -1
        self.mode = 'HOLD'
        self.stop_hold = True
        self.last_cmd_t = None
        self.state = 'IDLE'

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(TargetCommand, f'/{self.robot}/target_cmd', self._on_cmd, qos)
        self.pose_pub = self.create_publisher(PoseStamped, f'/{self.robot}/global/position', 10)
        self.fb_pub = self.create_publisher(NavFeedback, f'/{self.robot}/nav_feedback', 10)
        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info(f'mock_rover[{self.robot}] start=({sx:.2f},{sy:.2f})')

    def _nearest(self, x, y):
        return int(self._gr.nearest_node(self.nodes, x, y))

    def _on_cmd(self, msg):
        if msg.robot != self.robot:
            return
        self.last_cmd_t = self.get_clock().now().nanoseconds * 1e-9
        self.mode = msg.mode
        self.stop_hold = msg.stop_and_hold
        if msg.mode == 'GOTO' and msg.cmd_id != self.cmd_id:
            self.cmd_id = msg.cmd_id
            self._plan(int(msg.target_node))
        elif msg.mode in ('STOP', 'HOLD'):
            self.route_pts = []

    def _plan(self, target_node):
        src = self._nearest(self.x, self.y)
        route = self._gr.dijkstra(self.adj, src, target_node)
        if route is None:
            self.state = 'ERROR'
            self.route_pts = []
            self.get_logger().warn(f'no route {src} -> {target_node}')
            return
        self.route_pts = [self.nodes[n] for n in route]
        self.state = 'ROUTING'

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 1.0 / self.rate
        if self.last_cmd_t is not None and (now - self.last_cmd_t) > self.deadman:
            self.route_pts = []
            self.state = 'STOPPED'
        elif self.mode == 'STOP':
            self.route_pts = []
            self.state = 'STOPPED'
        elif self.mode == 'GOTO' and self.route_pts:
            self._advance(dt)
        elif self.mode == 'GOTO' and not self.route_pts:
            self.state = 'ARRIVED' if self.stop_hold else 'IDLE'
        self._publish()

    def _advance(self, dt):
        step = self.speed * dt
        while step > 0 and self.route_pts:
            tx, ty = self.route_pts[0]
            d = math.hypot(tx - self.x, ty - self.y)
            if d <= 1e-6:
                self.route_pts.pop(0)
                continue
            self.yaw = math.atan2(ty - self.y, tx - self.x)
            if d <= step:
                self.x, self.y = tx, ty
                step -= d
                self.route_pts.pop(0)
            else:
                self.x += (tx - self.x) / d * step
                self.y += (ty - self.y) / d * step
                step = 0.0
        self.state = 'TRACKING' if self.route_pts else ('ARRIVED' if self.stop_hold else 'IDLE')

    def _publish(self):
        p = PoseStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = self.frame
        p.pose.position.x = self.x
        p.pose.position.y = self.y
        p.pose.orientation.z = math.sin(self.yaw / 2.0)
        p.pose.orientation.w = math.cos(self.yaw / 2.0)
        self.pose_pub.publish(p)

        fb = NavFeedback()
        fb.header.stamp = p.header.stamp
        fb.robot = self.robot
        fb.cmd_id = max(self.cmd_id, 0)
        fb.state = self.state
        fb.current_node = self._nearest(self.x, self.y)
        fb.progress = 0.0
        self.fb_pub.publish(fb)


def main():
    rclpy.init()
    node = MockRover()
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
