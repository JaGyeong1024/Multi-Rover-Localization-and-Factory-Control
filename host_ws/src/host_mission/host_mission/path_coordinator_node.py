#!/usr/bin/env python3
"""path_coordinator_node: the mission coordinator (``nav_mode: path``).

Thin ROS shell around ``MissionEngine``: callbacks copy topic data into the engine
(stamped with the host's own monotonic receive time), a timer ticks it, and the
resulting path commands / events / status are published.

Subscribes
  /{robot}/global/position        PoseStamped   pose of the tracked point (uwb_global)
  /{robot}/heading_aligned        Bool          latched; yaw is valid only while true
  /{robot}/localization_quality   String        latched; GOOD | DEGRADED | LOST
  /{robot}/nav_feedback           NavFeedback   IDLE|TRACKING|HOLDING|PARKED|STOPPED|ERROR
  /{robot}/safety_state           SafetyState
  /leader/pick_place_done         Bool          latched; true = pick succeeded
  /leader/work_state              String        latched; leader pick coordinator state
  /task/add                       String        "fetch:<part_id>"
  /mission/control                String        pause|resume|abandon|clear_queue|cancel:<id>
  /mission/teach_goal             String        "<robot>:<goal_name>" store the rover's pose
  /mission/obstacles              String        JSON [{"x","y","r"} circle | {"x","y",
                                                "width","height","yaw","pad"} rectangle]
  /mission/mapped_obstacles       String        latched, same JSON (obstacle_mapper_node)
  /unload/done                    Empty         operator "unload done"
  /estop                          Bool          latched
Publishes
  /{robot}/path_cmd               PathCommand   latched depth 1, re-published at 2 Hz
  /mission/events                 String        JSON {"kind":..,"detail":..}
  /mission/state, /mission/queue, /{robot}/status   (dashboard feed)
  /mission/goals                  String        latched JSON of the goal table
  /mission/offered                String        latched JSON list of the operator commands the
                                                host would accept now (abandon, align)"""
import json
import math
import os
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose2D, PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty, String

from host_mission import plan_jobs
from host_mission.goals import GoalStore
from host_mission.mission_engine import MissionEngine
from host_mission.path_commander import pick_goals_without_fixture
from host_mission.settings import load_scenario, load_settings
from host_mission_interfaces.msg import (MissionItem, MissionQueue, NavFeedback,
                                         PathCommand, RobotStatus, SafetyState)

_DEPTH = 10
_EVENT_DEPTH = 50
_WARN_EVENTS = ('halt', 'hold', 'plan_failed', 'paused', 'phase_timeout', 'pick_failed',
                'teach_rejected', 'dispatch_rejected', 'control_rejected', 'estop',
                'superseded', 'start_after_turn_timeout')


def _latched(depth):
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _shape(entry):
    if 'width' in entry or 'height' in entry:
        return (float(entry['x']), float(entry['y']), float(entry.get('pad', 0.0)),
                float(entry.get('width', 0.0)), float(entry.get('height', 0.0)),
                float(entry.get('yaw', 0.0)))
    return float(entry['x']), float(entry['y']), float(entry['r'])


def resolve_config(name):
    """Absolute path of a config file: as given when absolute, else share/config."""
    if os.path.isabs(name):
        return name
    return os.path.join(get_package_share_directory('host_mission'), 'config', name)


# Overlays merged over mission.yaml: the arena (bounds, fixtures) and the rover geometry.
SETTINGS_OVERLAYS = ('site/arena.yaml', 'hardware/rovers.yaml')


def load_node_settings(node):
    """(settings file name, MissionSettings) from the node's ``settings_file`` and
    ``settings_overlays`` parameters; an empty overlay list loads the one file alone."""
    name = node.declare_parameter('settings_file', 'mission.yaml').value
    overlays = node.declare_parameter('settings_overlays', list(SETTINGS_OVERLAYS)).value or []
    return name, load_settings(resolve_config(name),
                               [resolve_config(overlay) for overlay in overlays if overlay])


class PathCoordinatorNode(Node):
    def __init__(self):
        super().__init__('mission_coordinator_node')
        self.declare_parameter('nav_mode', 'path')
        self.frame = self.declare_parameter('frame_id', 'uwb_global').value
        _, settings = load_node_settings(self)
        scenario = load_scenario(resolve_config(
            self.declare_parameter('scenario_file', 'mission_scenario.yaml').value))
        goals = GoalStore.load(resolve_config(
            self.declare_parameter('goals_file', 'goals.yaml').value))
        # Paths are computed on a worker thread: a slow plan must not stop the tick (the
        # rovers' command keep-alive, the state the operator sees, a pause taking effect).
        self.engine = MissionEngine(settings, scenario, goals, epoch=int(time.time()),
                                    plan_runner=plan_jobs.worker())
        self._tick_period = 1.0 / settings.tick_rate
        self.robots = self.engine.robots
        self._published = {}        # robot -> cmd_id last published
        self._make_publishers()
        self._make_subscriptions(settings.pick.picker)
        self.create_timer(1.0 / settings.tick_rate, self._tick)
        self.create_timer(1.0 / settings.republish_rate, self._republish)
        self._publish_goals()
        self._warn_pick_goals_without_fixture(settings, scenario, goals)
        self.get_logger().info(
            f'path coordinator ready: robots={self.robots}, parts={sorted(scenario.parts)}, '
            f'goals={goals.names()}, epoch={self.engine.commander.epoch}')

    def _warn_pick_goals_without_fixture(self, settings, scenario, goals):
        """One start-up warning naming the pick goals no registered fixture faces."""
        names = pick_goals_without_fixture(settings, scenario, goals)
        if names:
            self.get_logger().warning(
                f'pick goals without a registered fixture within '
                f'{settings.robots[settings.pick.picker].front_exempt_reach} m ahead: {names} '
                f'-> the picker stops at the front stop before the shelf and the shelf spot '
                f'is mapped as an obstacle (no path). Measure the shelf rectangles into '
                f'`fixtures` of the settings file.')

    def _make_publishers(self):
        self.cmd_pub = {r: self.create_publisher(PathCommand, f'/{r}/path_cmd', _latched(1))
                        for r in self.robots}
        self.status_pub = {r: self.create_publisher(RobotStatus, f'/{r}/status', _DEPTH)
                           for r in self.robots}
        self.queue_pub = self.create_publisher(MissionQueue, '/mission/queue', _latched(_DEPTH))
        self.state_pub = self.create_publisher(String, '/mission/state', _latched(_DEPTH))
        self.goals_pub = self.create_publisher(String, '/mission/goals', _latched(1))
        self.offered_pub = self.create_publisher(String, '/mission/offered', _latched(1))
        self._offered = None            # last published list (publish on change)
        self._superseded = {}           # robot -> newer epoch it answered (reported once)
        self.event_pub = self.create_publisher(String, '/mission/events', _EVENT_DEPTH)

    def _make_subscriptions(self, picker):
        sub = self.create_subscription
        for r in self.robots:
            sub(PoseStamped, f'/{r}/global/position', lambda m, r=r: self._on_pose(m, r), _DEPTH)
            sub(Bool, f'/{r}/heading_aligned',
                lambda m, r=r: self.engine.world.update_heading_aligned(r, m.data), _latched(1))
            sub(String, f'/{r}/localization_quality',
                lambda m, r=r: self.engine.world.update_quality(r, m.data), _latched(1))
            sub(NavFeedback, f'/{r}/nav_feedback', lambda m, r=r: self._on_nav(m, r), _DEPTH)
            sub(SafetyState, f'/{r}/safety_state', lambda m, r=r: self._on_safety(m, r), _DEPTH)
        sub(Bool, f'/{picker}/pick_place_done',
            lambda m: self.engine.on_pick_done(m.data), _latched(1))
        sub(String, f'/{picker}/work_state',
            lambda m: self.engine.on_work_state(m.data), _latched(1))
        sub(String, '/task/add', self._on_dispatch, _DEPTH)
        sub(String, '/mission/control', self._on_control, _DEPTH)
        sub(String, '/mission/teach_goal', self._on_teach, _DEPTH)
        sub(String, '/mission/obstacles', self._on_obstacles, _DEPTH)
        sub(String, '/mission/mapped_obstacles', self._on_mapped_obstacles, _latched(1))
        sub(Empty, '/unload/done', lambda _m: self.engine.unload_done(), _DEPTH)
        sub(Bool, '/estop', lambda m: self.engine.estop(m.data, time.monotonic()),
            _latched(_DEPTH))

    def _on_pose(self, msg, robot):
        p = msg.pose.position
        self.engine.world.update_pose(robot, p.x, p.y, _yaw(msg.pose.orientation),
                                      time.monotonic())

    def _on_nav(self, msg, robot):
        self.engine.world.update_nav(robot, msg.state, msg.detail, msg.epoch, msg.cmd_id,
                                     time.monotonic())
        # A rover obeys the sender with the largest epoch (rover_drive core.state). One that
        # answers a larger epoch than ours took a command from a sender started after this host
        # (e.g. drive_check.py) and rejects ours until restart: say so once.
        if msg.epoch > self.engine.commander.epoch and self._superseded.get(robot) != msg.epoch:
            self._superseded[robot] = msg.epoch
            self._emit('superseded', robot)

    def _on_safety(self, msg, robot):
        self.engine.world.update_safety(robot, msg.state, msg.reasons, msg.speed_scale,
                                        msg.blocked_x, msg.blocked_y, time.monotonic())

    def _on_dispatch(self, msg):
        token = msg.data.strip()
        self.engine.dispatch(token.split(':', 1)[1] if ':' in token else token)

    def _on_control(self, msg):
        self.engine.control(msg.data.strip(), time.monotonic())

    def _on_teach(self, msg):
        robot, _sep, name = msg.data.strip().partition(':')
        if self.engine.teach_goal(robot.strip(), name.strip(), time.monotonic()):
            self._publish_goals()

    def _on_obstacles(self, msg):
        circles = self._parse_circles(msg)
        if circles is not None:
            self.engine.set_obstacles(circles)

    def _on_mapped_obstacles(self, msg):
        circles = self._parse_circles(msg)
        if circles is not None:
            self.engine.set_mapped_obstacles(circles)

    def _parse_circles(self, msg):
        """Obstacle shapes from the JSON (an entry with "width"/"height" is a rectangle grown by
        "pad"), or None with an event."""
        try:
            return [_shape(o) for o in json.loads(msg.data)]
        except (ValueError, KeyError, TypeError) as error:
            self._emit('obstacles_rejected', str(error))
            return None

    def _tick(self):
        started = time.monotonic()
        for kind, detail in self.engine.tick(started):
            self._emit(kind, detail)
        took = time.monotonic() - started
        if took > 2.0 * self._tick_period:      # the rovers time a silent host out
            self.get_logger().warn(f'[mission] slow tick: {took * 1000.0:.0f} ms',
                                   throttle_duration_sec=5.0)
        for robot, cmd in self.engine.commands().items():
            if self._published.get(robot) != cmd.cmd_id:
                self._publish_cmd(cmd)
        self._publish_queue()
        self._publish_status()
        self.state_pub.publish(String(data=self.engine.seq.system_state()))
        offered = self.engine.offered_controls()
        if offered != self._offered:
            self._offered = offered
            self.offered_pub.publish(String(data=json.dumps(offered)))

    def _republish(self):
        for cmd in self.engine.commands().values():
            self._publish_cmd(cmd)

    def _emit(self, kind, detail):
        # Two call sites: rclpy rejects a severity change on one call site.
        if kind in _WARN_EVENTS:
            self.get_logger().warn(f'[mission] {kind} {detail}')
        else:
            self.get_logger().info(f'[mission] {kind} {detail}')
        self.event_pub.publish(String(data=json.dumps({'kind': kind, 'detail': detail})))

    def _publish_cmd(self, cmd):
        msg = PathCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.robot, msg.epoch, msg.cmd_id, msg.mode = cmd.robot, cmd.epoch, cmd.cmd_id, cmd.mode
        msg.poses = [Pose2D(x=float(p.x), y=float(p.y), theta=float(p.theta)) for p in cmd.poses]
        msg.speed_cap = float(cmd.speed_cap)
        msg.goal_pos_tol = float(cmd.goal_pos_tol)
        msg.goal_yaw_tol = float(cmd.goal_yaw_tol)
        msg.pick_at_goal = bool(cmd.pick_at_goal)
        msg.lead_robot = cmd.lead_robot
        msg.gap = float(cmd.gap)
        msg.pivot_first = bool(cmd.pivot_first)
        msg.pivot_heading = float(cmd.pivot_heading)
        msg.fixture_ahead = bool(cmd.fixture_ahead)
        self.cmd_pub[cmd.robot].publish(msg)
        self._published[cmd.robot] = cmd.cmd_id

    def _publish_goals(self):
        self.goals_pub.publish(String(data=json.dumps(self.engine.world.goals.resolved())))

    def _publish_queue(self):
        now = time.monotonic()
        msg = MissionQueue()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.active_id = self.engine.queue.active_id()
        msg.items = [self._queue_item(it, now) for it in self.engine.queue.items]
        self.queue_pub.publish(msg)

    def _queue_item(self, it, now):
        item = MissionItem()
        item.id, item.status = it.id, it.status
        item.part.id, item.part.state = it.part_id, it.part_state
        item.part.coord.x, item.part.coord.y = float(it.x), float(it.y)
        item.lane_node = -1                      # path mode: stops are named goals
        goal = self.engine.world.goals.get(it.leader_node)
        if goal is not None:
            item.x, item.y, item.yaw = goal.x, goal.y, goal.theta
        if it.status == 'pending':
            item.ready_in = self.engine.seq.shelf_ready_in(it.leader_node, now)
        return item

    def _publish_status(self):
        at_target = self.engine.at_target()
        for robot in self.robots:
            state = self.engine.world.robot(robot)
            msg = RobotStatus()
            msg.name = robot
            if state.pose is not None:
                msg.pose.position.x, msg.pose.position.y, yaw = state.pose
                msg.pose.orientation.z = math.sin(yaw / 2.0)
                msg.pose.orientation.w = math.cos(yaw / 2.0)
            msg.lane_node = -1
            msg.health = state.quality
            msg.current_task = self.engine.seq.active_id
            msg.state = self.engine.seq.robot_state(robot, at_target[robot])
            self.status_pub[robot].publish(msg)
