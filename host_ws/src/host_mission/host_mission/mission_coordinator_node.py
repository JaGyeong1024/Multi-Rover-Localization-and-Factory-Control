#!/usr/bin/env python3
"""mission_coordinator_node — host-side fleet coordinator (domain 30).

Owns the mission: queue, capacity-N batch sequencing, per-robot TARGET dispatch. Sends
TargetCommand (target + mode, NOT dense waypoints — the rover owns routing/control).
Consumes /{robot}/global/position,
/{robot}/nav_feedback, /localization/diagnostics. Boundary: host = 조율 / rover = 플래닝·제어.

Republishes each robot's TargetCommand every tick so the rover deadman stays fresh
(header.stamp); cmd_id only bumps when the target changes.
"""
import json
import math
import os

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from host_mission.graph_map import GraphMap
from host_mission.health_gate import HealthGate
from host_mission.mission_queue import Item, MissionQueueModel
from host_mission.task_sequencer import MissionStateMachine
from host_mission.world_model import WorldModel
from host_mission_interfaces.msg import (MissionItem, MissionQueue, NavFeedback,
                                         PickPlaceCommand, PickPlaceStatus,
                                         RobotStatus, TargetCommand)


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MissionCoordinatorNode(Node):
    def __init__(self):
        super().__init__('mission_coordinator_node')
        self.frame = self.declare_parameter('frame_id', 'uwb_global').value
        tick_rate = float(self.declare_parameter('tick_rate', 5.0).value)
        self.start_grace = float(self.declare_parameter('start_grace', 2.0).value)
        self.robots = list(self.declare_parameter('robots', ['leader', 'follower']).value)
        self.picker = self.declare_parameter('picker', 'leader').value
        # health gating pauses the mission on localization LOST. Disable for the mock
        # (rover_nav_ref supplies pose directly; real /localization/diagnostics is irrelevant).
        self.health_gating = bool(self.declare_parameter('health_gating', True).value)
        ped = self.declare_parameter(
            'path_editor_dir', os.path.expanduser('~/capstone/tools/path_editor')).value

        graph_file = self.declare_parameter('graph_file', 'lane_graph.json').value
        scenario_file = self.declare_parameter('scenario_file', 'mission_scenario.yaml').value
        share = get_package_share_directory('host_mission')
        with open(os.path.join(share, 'config', graph_file)) as f:
            graph_dict = json.load(f)
        with open(os.path.join(share, 'config', scenario_file)) as f:
            scen = yaml.safe_load(f)

        self.graph = GraphMap(graph_dict, ped)
        self.parts = {p['id']: p for p in scen['parts']}
        self.position_tol = float(scen.get('position_tol', 0.15))
        self.queue = MissionQueueModel()
        self.seq = MissionStateMachine(
            self.queue, scen['standby_node'], scen['unloading_node'], int(scen['capacity']),
            pick_dwell=float(scen.get('pick_dwell', 2.0)),
            drop_dwell=float(scen.get('drop_dwell', 2.0)),
            phase_timeout=float(scen.get('phase_timeout', 90.0)),
            follower_home_node=scen.get('follower_home_node'))
        self.world = WorldModel(self.graph, self.position_tol)
        self.health = HealthGate()

        cmd_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
        self.cmd_pub = {r: self.create_publisher(TargetCommand, f'/{r}/target_cmd', cmd_qos)
                        for r in self.robots}
        self.status_pub = {r: self.create_publisher(RobotStatus, f'/{r}/status', 10)
                           for r in self.robots}
        # latched (transient_local) so late-joining dashboard/rovers get the current state,
        # and so the QoS matches the mission domain bridge.
        self.queue_pub = self.create_publisher(MissionQueue, '/mission/queue', cmd_qos)
        self.state_pub = self.create_publisher(String, '/mission/state', cmd_qos)
        self.pp_cmd_pub = self.create_publisher(
            PickPlaceCommand, f'/{self.picker}/pickplace_cmd', cmd_qos)
        self.create_subscription(PickPlaceStatus, f'/{self.picker}/pickplace_status',
                                 self._on_pickplace_status, 10)

        for r in self.robots:
            self.create_subscription(PoseStamped, f'/{r}/global/position',
                                     lambda m, rr=r: self._on_pose(m, rr), 10)
            self.create_subscription(NavFeedback, f'/{r}/nav_feedback',
                                     lambda m, rr=r: self._on_fb(m, rr), 10)
        self.create_subscription(DiagnosticArray, '/localization/diagnostics', self._on_diag, 10)
        self.create_subscription(String, '/mission/dispatch', self._on_dispatch, 10)
        self.create_subscription(String, '/mission/control', self._on_control, 10)

        self._cmd_id = {r: 0 for r in self.robots}
        self._last_target = {r: None for r in self.robots}
        self._fb_state = {r: '' for r in self.robots}
        self._fb_cmd = {r: -1 for r in self.robots}
        self._pending_since = None
        self._pp_cmd_id = 0
        self._pp_requested_id = ''
        self._pp_substate = ''
        self._arrived = {}
        self.create_timer(1.0 / tick_rate, self._tick)
        self.get_logger().info(
            f'mission_coordinator ready — robots={self.robots}, parts={list(self.parts)}')

    # --- callbacks ---
    def _on_pose(self, msg, robot):
        p = msg.pose.position
        self.world.update_pose(robot, p.x, p.y, _yaw(msg.pose.orientation))

    def _on_fb(self, msg, robot):
        self._fb_state[robot] = msg.state
        self._fb_cmd[robot] = msg.cmd_id

    def _on_diag(self, msg):
        self.health.update(msg)

    def _on_pickplace_status(self, msg):
        if msg.cmd_id != self._pp_cmd_id:
            return                         # stale (not the current pick request)
        self._pp_substate = msg.state
        if msg.state == 'DONE':
            self.seq.pick_complete(True)
        elif msg.state == 'FAILED':
            self.seq.pick_complete(False)

    def _on_dispatch(self, msg):
        token = msg.data.strip()
        name = token.split(':', 1)[1] if ':' in token else token
        p = self.parts.get(name)
        if p is None:
            self.get_logger().warn(f'dispatch: unknown part "{name}" (have {list(self.parts)})')
            return
        node = self.graph.nearest_node(p['x'], p['y'])
        item = Item(id=f'fetch_{name}', part_id=name, x=float(p['x']), y=float(p['y']),
                    lane_node=int(node))
        if self.queue.enqueue(item):
            self.get_logger().info(f'enqueue {item.id} -> lane_node {node}')

    def _on_control(self, msg):
        cmd = msg.data.strip()
        if cmd == 'abort_all':
            self.seq.abort()
            self.get_logger().warn('control: abort_all -> ABORTED')
        elif cmd == 'clear_queue':
            self.seq.abort()
            self.queue.clear()
            self.get_logger().warn('control: clear_queue -> queue cleared, ABORTED')
        elif cmd == 'pause':
            self.seq.pause()
            self.get_logger().warn('control: pause -> PAUSED')
        elif cmd == 'resume':
            self.seq.resume()
            self.get_logger().warn('control: resume -> RUNNING')

    # --- tick ---
    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        # start-grace: let multi-select dispatches accumulate into one batch
        if self.seq.state in ('IDLE', 'DONE', 'ABORTED'):
            if self.queue.pending():
                if self._pending_since is None:
                    self._pending_since = now
                elif now - self._pending_since >= self.start_grace:
                    for ev in self.seq.start(now):
                        self.get_logger().info(f'[mission] {ev[0]} {ev[1]}')
                    self._pending_since = None
            else:
                self._pending_since = None
        arrived = {}
        for r in self.robots:
            # position is ground truth; feedback ARRIVED is trusted only when it
            # acknowledges the CURRENT command (avoids stale-ARRIVED race on retarget)
            pos_ok = self.world.arrived(r, self.seq.targets.get(r))
            fb_ok = (self._fb_state.get(r) in ('ARRIVED', 'HOLDING')
                     and self._fb_cmd.get(r) == self._cmd_id.get(r))
            arrived[r] = pos_ok or fb_ok
        self._arrived = arrived
        health = {r: self.health.get(r) for r in self.robots} if self.health_gating else {}
        targets, modes, events = self.seq.tick(now, arrived, health)
        for ev in events:
            self.get_logger().info(f'[mission] {ev[0]} {ev[1]}')
        self._handle_pickplace()
        for r in self.robots:
            self._publish_cmd(r, targets.get(r), modes.get(r, 'GOTO'))
        self._publish_queue()
        self._publish_status()
        self._publish_state()

    def _publish_cmd(self, robot, node, mode='GOTO'):
        if node is None:
            return
        node = int(node)
        if self._last_target[robot] != node:
            self._cmd_id[robot] += 1
            self._last_target[robot] = node
        msg = TargetCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.robot = robot
        msg.cmd_id = self._cmd_id[robot]
        msg.mode = mode
        msg.target_node = node
        x, y = self.graph.xy(node)
        msg.target_coord.x = x
        msg.target_coord.y = y
        msg.stop_and_hold = True
        msg.speed_cap = 0.0
        self.cmd_pub[robot].publish(msg)

    def _publish_queue(self):
        m = MissionQueue()
        m.header.stamp = self.get_clock().now().to_msg()
        m.active_id = self.queue.active_id()
        for it in self.queue.items:
            mi = MissionItem()
            mi.id = it.id
            mi.part.id = it.part_id
            mi.part.coord.x = it.x
            mi.part.coord.y = it.y
            mi.part.state = it.part_state
            mi.lane_node = it.lane_node
            mi.x, mi.y = self.graph.xy(it.lane_node)
            mi.status = it.status
            m.items.append(mi)
        self.queue_pub.publish(m)

    def _publish_status(self):
        for r in self.robots:
            s = RobotStatus()
            s.name = r
            if r in self.world.pose:
                x, y, _yw = self.world.pose[r]
                s.pose.position.x = x
                s.pose.position.y = y
                s.lane_node = self.world.nearest_node(r)
            s.health = self.health.get(r)
            s.current_task = self.seq.active_id
            s.state = self.seq.robot_state(r, self._arrived.get(r, False))
            self.status_pub[r].publish(s)

    def _handle_pickplace(self):
        # on entering PICK_PLACE for a new item, request the pick from teammate A (leader)
        if self.seq.phase == 'PICKING' and self.seq.active_id \
                and self.seq.active_id != self._pp_requested_id:
            it = self.queue.by_id(self.seq.active_id)
            if it is None:
                return
            self._pp_requested_id = self.seq.active_id
            self._pp_cmd_id += 1
            self._pp_substate = ''
            m = PickPlaceCommand()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.frame
            m.robot = self.picker
            m.cmd_id = self._pp_cmd_id
            m.part_id = it.part_id
            m.part_coord.x = it.x
            m.part_coord.y = it.y
            m.target_robot = 'follower'
            self.pp_cmd_pub.publish(m)
            self.get_logger().info(f'[pickplace] request {it.part_id} (cmd {self._pp_cmd_id})')

    def _publish_state(self):
        st = self.seq.system_state()
        if st == 'PICK_PLACE' and self._pp_substate:
            st = f'PICK_PLACE:{self._pp_substate}'
        m = String()
        m.data = st
        self.state_pub.publish(m)


def main():
    rclpy.init()
    node = MissionCoordinatorNode()
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
