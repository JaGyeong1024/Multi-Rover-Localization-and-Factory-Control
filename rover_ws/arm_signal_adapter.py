#!/usr/bin/env python3
"""arm_signal_adapter: maps the arm program's status to the host pick contract.

Runs on the leader in its ROS domain. It never moves the arm or the base; the only
command it can send is /mp_control/cancel (Bool true) in the Stay fallback below,
which stops mp_control's Stay retry loop without arm or gripper motion.

Subscribes
  /leader/nav_feedback      host_mission_interfaces/NavFeedback (rover_drive path_tracker)
  /mp_control/status        std_msgs/String, reliable + transient_local (arm program)
  /joint_states             sensor_msgs/JointState
  /leader/global/position   geometry_msgs/PoseStamped (selects the marker ID)
Publishes (reliable + transient_local depth 1, bridged to the host)
  /leader/work_state        String  NAVIGATING | PICKING | DONE | ERROR
  /leader/pick_place_done   Bool    true = part placed, false = arm reported an error
  /leader/pick_place_result String  diagnostic text (not bridged)

host_mission.pick_monitor expects PICKING while the leader HOLDS at a pick goal,
then pick_place_done, then DONE or ERROR, in that order.

Arm session (status substrings):
  start    'grasp sequence started', seen while HOLDING
  success  'cargo_loaded:'
  fallback 'handoff release complete', then the arm measured at Stay (|q - STAY| <=
           stay_tol, |dq| <= stay_vel_tol) for stay_hold_s, at least stay_fallback_s
           after the release (the arm program may retry Stay forever)
  error    'abort' / 'cancel requested' / 'start ignored after', or the status
           publisher gone for exit_grace_s. ERROR is published only once the arm is
           at Stay; until then PICKING is kept and an error is logged every 2 s.
A started session is tracked to its end even if the leader leaves HOLDING.

Auto-launch (parameter launch_cmd, empty = off): after hold_settle_s of HOLDING with
no session in this HOLDING window, runs bash -c "<launch_cmd>" once in its own
process group, with {stamp} replaced by a time stamp and {marker} by the shelf's
marker ID (0 if the leader's y < marker_split_y, else 1). The arm program is never
killed (stopping it mid-motion freezes the arm).

    python3 arm_signal_adapter.py [--ros-args -p launch_cmd:="..."]   (run_leader.sh start)
"""
import os
import signal
import subprocess
import time

import rclpy
from host_mission_interfaces.msg import NavFeedback
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

STAY = (0.104311, 0.027612, -0.001534, -1.638291)   # same as the bringup stay pose
START_KEY = 'grasp sequence started'
DONE_KEY = 'cargo_loaded:'
RELEASED_KEY = 'handoff release complete'
ERROR_KEYS = ('abort', 'cancel requested', 'start ignored after')


def _qos(depth, latched):
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=(DurabilityPolicy.TRANSIENT_LOCAL if latched
                                  else DurabilityPolicy.VOLATILE),
                      history=HistoryPolicy.KEEP_LAST, depth=depth)


class ArmSignalAdapter(Node):
    def __init__(self):
        super().__init__('arm_signal_adapter')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.picker = p('picker', 'leader')
        self.status_topic = p('status_topic', '/mp_control/status')
        self.launch_cmd = p('launch_cmd', '')
        self.hold_settle_s = float(p('hold_settle_s', 1.0))
        self.nav_timeout_s = float(p('nav_timeout_s', 2.0))
        self.stay_tol = float(p('stay_tol', 0.08))
        self.stay_vel_tol = float(p('stay_vel_tol', 0.03))
        self.stay_hold_s = float(p('stay_hold_s', 1.0))
        self.stay_fallback_s = float(p('stay_fallback_s', 12.0))
        self.cancel_on_fallback = bool(p('cancel_on_fallback', True))
        self.exit_grace_s = float(p('exit_grace_s', 3.0))
        # {marker} in launch_cmd: ArUco ID of the shelf the leader stands at, chosen by its y
        # (site: gear shelf -> ID 0 below the split, bolt shelf -> ID 1 above it)
        self.marker_split_y = float(p('marker_split_y', 1.75))
        self.pose_y = None
        from geometry_msgs.msg import PoseStamped
        self.create_subscription(PoseStamped, f'/{self.picker}/global/position',
                                 lambda m: setattr(self, 'pose_y', m.pose.position.y), 10)

        self.nav_state, self.nav_time = '', 0.0
        self.hold_since = None        # leader HOLDING since (monotonic)
        self.window_done = False      # an arm session already finished in this HOLDING window
        self.launched = False         # auto-launch already fired in this HOLDING window
        self.session = None           # dict while an arm session is tracked
        self.at_stay_since = None
        self.child = None
        self.work_state = None
        self._last_err_log = 0.0

        self.work_pub = self.create_publisher(String, f'/{self.picker}/work_state', _qos(1, True))
        self.done_pub = self.create_publisher(Bool, f'/{self.picker}/pick_place_done', _qos(1, True))
        self.result_pub = self.create_publisher(
            String, f'/{self.picker}/pick_place_result', _qos(1, True))
        self.cancel_pub = self.create_publisher(Bool, '/mp_control/cancel', 10)
        self.create_subscription(NavFeedback, f'/{self.picker}/nav_feedback', self._on_nav,
                                 _qos(10, False))
        self.create_subscription(String, self.status_topic, self._on_status, _qos(10, True))
        self.create_subscription(JointState, '/joint_states', self._on_joints,
                                 qos_profile_sensor_data)
        self.create_timer(0.1, self._tick)
        self._set_work('NAVIGATING')
        self.get_logger().info(
            f'arm_signal_adapter: domain {os.environ.get("ROS_DOMAIN_ID", "0")}, '
            f'auto-launch {"ON" if self.launch_cmd else "off"}')

    # ---------- inputs ----------
    def _on_nav(self, msg):
        self.nav_state, self.nav_time = str(msg.state).strip().upper(), time.monotonic()

    def _on_joints(self, msg):
        try:
            idx = [list(msg.name).index(f'joint{i}') for i in range(1, 5)]
            ok = all(abs(msg.position[j] - q) <= self.stay_tol
                     and (len(msg.velocity) <= j or abs(msg.velocity[j]) <= self.stay_vel_tol)
                     for j, q in zip(idx, STAY))
        except (ValueError, IndexError):
            return
        now = time.monotonic()
        if not ok:
            self.at_stay_since = None
        elif self.at_stay_since is None:
            self.at_stay_since = now

    def _on_status(self, msg):
        text, now = msg.data, time.monotonic()
        s = self.session
        if s is None:
            if START_KEY in text:
                if self._holding(now) and not self.window_done:
                    self.session = {'start': now, 'released': None, 'error': None,
                                    'gone_since': None}
                    self._set_work('PICKING')
                    self.get_logger().info('arm session started (grasp sequence started)')
                else:
                    self.get_logger().error(
                        'grasp sequence started while the leader is NOT holding at a pick goal '
                        '(or a session already finished here): NOT reported to the host')
            return
        low = text.lower()
        if DONE_KEY in text:
            self._finish(True, text)
        elif RELEASED_KEY in text and s['released'] is None:
            s['released'] = now
            self.get_logger().info('arm released the part; waiting for Stay')
        elif s['error'] is None and any(k in low for k in ERROR_KEYS):
            s['error'] = text
            self.get_logger().error(f'arm reported: {text}')

    # ---------- helpers ----------
    def _holding(self, now):
        return self.nav_state == 'HOLDING' and now - self.nav_time <= self.nav_timeout_s

    def _arm_at_stay(self, now):
        return self.at_stay_since is not None and now - self.at_stay_since >= self.stay_hold_s

    def _set_work(self, state):
        if state != self.work_state:
            self.work_state = state
            self.work_pub.publish(String(data=state))
            self.get_logger().info(f'/{self.picker}/work_state -> {state}')

    def _finish(self, ok, detail):
        self.done_pub.publish(Bool(data=bool(ok)))       # before the state change (PickMonitor)
        self.result_pub.publish(String(data=str(detail)))
        self._set_work('DONE' if ok else 'ERROR')
        self.get_logger().info(f'arm session finished ok={ok}: {detail}')
        self.session = None
        self.window_done = True

    def _launch(self):
        marker = '0' if (self.pose_y is None or self.pose_y < self.marker_split_y) else '1'
        cmd = self.launch_cmd.replace('{stamp}', time.strftime('%Y%m%d_%H%M%S')).replace('{marker}', marker)
        self.get_logger().info(f'leader y={self.pose_y} -> marker_id {marker}')
        self.child = subprocess.Popen(['bash', '-c', cmd], start_new_session=True)
        self.get_logger().warn(f'arm program launched pid={self.child.pid} (pgid same); '
                               f'stop: kill -INT -- -{self.child.pid}')

    # ---------- main loop ----------
    def _tick(self):
        now = time.monotonic()
        holding = self._holding(now)
        if holding and self.hold_since is None:
            self.hold_since = now
        elif not holding:
            self.hold_since = None

        s = self.session
        if s is None:
            if not holding:
                self.window_done = self.launched = False
                if self.work_state in ('DONE', 'ERROR', 'PICKING'):
                    self._set_work('NAVIGATING')
                return
            if self.work_state == 'NAVIGATING' and not self.window_done:
                self._set_work('PICKING')     # leader holds at a pick goal: the pick is on
            if (self.launch_cmd and not self.window_done and not self.launched
                    and self.child is None and now - self.hold_since >= self.hold_settle_s):
                self.launched = True
                self._launch()
            if self.child is not None and self.child.poll() is not None:
                self.get_logger().warn(f'arm program exited rc={self.child.returncode}')
                self.child = None
            return

        if self.child is not None and self.child.poll() is not None:
            self.child = None
        if self.count_publishers(self.status_topic) == 0:
            s['gone_since'] = s['gone_since'] or now
            if now - s['gone_since'] >= self.exit_grace_s and s['error'] is None:
                s['error'] = 'arm program exited without cargo_loaded'
        else:
            s['gone_since'] = None

        if (s['released'] is not None and now - s['released'] >= self.stay_fallback_s
                and self._arm_at_stay(now)):
            if self.cancel_on_fallback and self.count_publishers(self.status_topic) > 0:
                self.cancel_pub.publish(Bool(data=True))   # stop mp_control's Stay retry loop
            self._finish(True, 'released and arm measured at Stay (no cargo_loaded)')
            return
        if s['error'] is not None:
            if s['released'] is not None and self._arm_at_stay(now):
                self._finish(True, f'released, arm at Stay, then: {s["error"]}')
            elif self._arm_at_stay(now):
                self._finish(False, s['error'])
            elif now - self._last_err_log >= 2.0:
                self._last_err_log = now
                self.get_logger().error(
                    f'ARM NOT AT STAY after: {s["error"]} -- keeping PICKING; PAUSE the '
                    f'mission and recover the arm (host pick_dwell still runs)')


def main():
    rclpy.init()
    node = ArmSignalAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
