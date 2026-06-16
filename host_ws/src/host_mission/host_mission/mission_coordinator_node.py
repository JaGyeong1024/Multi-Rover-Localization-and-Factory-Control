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
from std_msgs.msg import Bool, Empty, String

from host_mission.graph_map import GraphMap
from host_mission.health_gate import HealthGate
from host_mission.mission_queue import Item, MissionQueueModel
from host_mission.task_sequencer import MissionStateMachine
from host_mission.world_model import WorldModel
from host_mission_interfaces.msg import (MissionItem, MissionQueue, NavFeedback,
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
        # 사림이형 doesn't report arrival in the new flow — host judges it from
        # /{robot}/global/position alone. Set True to also trust rover NavFeedback ARRIVED.
        self.use_nav_feedback = bool(self.declare_parameter('use_nav_feedback', False).value)
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
            follower_home_node=scen.get('follower_home_node'),
            order_cost=self.graph.route_cost,    # time-optimal pickup order on directed lanes
            cooldown=float(scen.get('pick_cooldown', 0.0)))  # per-shelf replenish cooldown (s)
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
        # picker PnP: Empty 'start' (latched) out, Bool 'working' in (host acts on falling edge)
        self.pnp_start_pub = self.create_publisher(Empty, '/pnp/start', cmd_qos)
        self.create_subscription(Bool, '/pnp/working', self._on_pnp_working, 10)

        for r in self.robots:
            self.create_subscription(PoseStamped, f'/{r}/global/position',
                                     lambda m, rr=r: self._on_pose(m, rr), 10)
            self.create_subscription(NavFeedback, f'/{r}/nav_feedback',
                                     lambda m, rr=r: self._on_fb(m, rr), 10)
        self.create_subscription(DiagnosticArray, '/localization/diagnostics', self._on_diag, 10)
        self.create_subscription(String, '/mission/dispatch', self._on_dispatch, 10)
        self.create_subscription(String, '/task/add', self._on_dispatch, 10)      # 관제 alias
        self.create_subscription(String, '/mission/control', self._on_control, 10)
        self.create_subscription(Empty, '/unload/done', self._on_unload_done, 10)
        # E-stop: latched safety line; coordinator forces STOP + pauses the FSM
        self.create_subscription(Bool, '/estop', self._on_estop, cmd_qos)

        self._cmd_id = {r: 0 for r in self.robots}
        self._last_target = {r: None for r in self.robots}
        self._last_mode = {r: None for r in self.robots}
        self._fb_state = {r: '' for r in self.robots}
        self._fb_cmd = {r: -1 for r in self.robots}
        self._pending_since = None
        self._task_seq = 0
        self._pnp_requested_id = ''
        self._pnp_episode = 0
        self._pnp_started_episode = -1
        self._pnp_working_prev = False
        self._estop = False
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

    def _on_pnp_working(self, msg):
        # picker's '작업중' flag. Act ONLY on a true->false edge that followed a true within
        # the CURRENT pnp episode — rejects a leftover true or a standalone false (stale guard).
        cur = bool(msg.data)
        if cur and not self._pnp_working_prev:
            self._pnp_started_episode = self._pnp_episode
        elif (not cur) and self._pnp_working_prev \
                and self._pnp_started_episode == self._pnp_episode:
            self.seq.pick_complete(True)
        self._pnp_working_prev = cur

    def _on_unload_done(self, _msg):
        # operator '하역완료' — accept only while the follower is actually at the unloading zone
        if self.seq.system_state() == 'UNLOADING':
            self.seq.unload_done()
            self.get_logger().info('[mission] unload_done (operator 하역완료)')
        else:
            self.get_logger().warn(f'unload_done ignored (state={self.seq.system_state()})')

    def _on_estop(self, msg):
        engaged = bool(msg.data)
        if engaged != self._estop:
            self.get_logger().warn(f'[ESTOP] {"ENGAGED" if engaged else "released"}')
        self._estop = engaged
        self.seq.estop(engaged)

    def _on_dispatch(self, msg):
        token = msg.data.strip()
        name = token.split(':', 1)[1] if ':' in token else token
        p = self.parts.get(name)
        if p is None:
            self.get_logger().warn(f'dispatch: unknown part "{name}" (have {list(self.parts)})')
            return
        ln = int(p.get('leader_node', self.graph.nearest_node(p['x'], p['y'])))
        fn = int(p.get('follower_node', ln))
        # each press = a distinct task instance (unique id) — lets the operator queue N of the
        # SAME part; the cooldown scheduler then spaces same-shelf picks into separate trips.
        self._task_seq += 1
        item = Item(id=f'fetch_{name}#{self._task_seq}', part_id=name,
                    x=float(p['x']), y=float(p['y']),
                    lane_node=ln, leader_node=ln, follower_node=fn)
        if self.queue.enqueue(item):
            self.get_logger().info(f'enqueue {item.id} -> leader {ln} / follower {fn}')

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
        elif cmd.startswith('cancel:'):
            item_id = cmd.split(':', 1)[1]
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.seq.cancel_item(item_id, now):
                self.get_logger().info(f'control: cancel {item_id}')
            else:
                self.get_logger().warn(f'control: cancel rejected ({item_id} — picking/returning/missing)')

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
            # host position estimate is the arrival source (사림이형 doesn't report arrival)
            pos_ok = self.world.arrived(r, self.seq.targets.get(r))
            if self.use_nav_feedback:
                fb_ok = (self._fb_state.get(r) in ('ARRIVED', 'HOLDING')
                         and self._fb_cmd.get(r) == self._cmd_id.get(r))
                arrived[r] = pos_ok or fb_ok
            else:
                arrived[r] = pos_ok
        self._arrived = arrived
        health = {r: self.health.get(r) for r in self.robots} if self.health_gating else {}
        targets, modes, events = self.seq.tick(now, arrived, health)
        for ev in events:
            self.get_logger().info(f'[mission] {ev[0]} {ev[1]}')
            if ev[0] == 'done':
                self.queue.purge_finished()   # auto-reset to a clean, dispatchable queue
        self._handle_pickplace()
        for r in self.robots:
            mode = 'STOP' if self._estop else modes.get(r, 'GOTO')
            self._publish_cmd(r, targets.get(r), mode)
        self._publish_queue()
        self._publish_status()
        self._publish_state()

    def _publish_cmd(self, robot, node, mode='GOTO'):
        if node is None:
            return
        node = int(node)
        # bump cmd_id when the target OR mode changes, so the rover re-plans after a
        # STOP -> GOTO (E-stop release / pause resume), not only on a new target node.
        if self._last_target[robot] != node or self._last_mode[robot] != mode:
            self._cmd_id[robot] += 1
            self._last_target[robot] = node
            self._last_mode[robot] = mode
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
        now = self.get_clock().now().nanoseconds * 1e-9
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
            # cooldown remaining for a still-queued part whose shelf was just picked (0 = ready)
            mi.ready_in = self.seq.shelf_ready_in(it.lane_node, now) if it.status == 'pending' else 0.0
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
        # on entering LOADING (PICKING) for a new item, tell the picker to start PnP
        if self.seq.phase == 'PICKING' and self.seq.active_id \
                and self.seq.active_id != self._pnp_requested_id:
            self._pnp_requested_id = self.seq.active_id
            self._pnp_episode += 1
            self._pnp_started_episode = -1
            self._pnp_working_prev = False
            self.pnp_start_pub.publish(Empty())
            self.get_logger().info(f'[pnp] start {self.seq.active_id} (episode {self._pnp_episode})')

    def _publish_state(self):
        m = String()
        m.data = self.seq.system_state()
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
