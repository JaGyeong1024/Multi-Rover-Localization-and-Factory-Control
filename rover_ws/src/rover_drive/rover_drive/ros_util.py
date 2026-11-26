"""Small ROS helpers shared by the two nodes."""
import dataclasses
import math
import signal
import time

import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

from .core import state as st

_STOP_REPEATS = 3        # zero Twists sent on shutdown (delivery is not guaranteed for one)
_STOP_INTERVAL_S = 0.02


def declare_config(node, cls):
    """Declare one ROS parameter per dataclass field and return the filled config."""
    cfg = cls()
    for field in dataclasses.fields(cls):
        setattr(cfg, field.name, _declare(node, field.name, getattr(cfg, field.name)))
    return cfg


def _declare(node, name: str, default):
    if not isinstance(default, tuple):
        return node.declare_parameter(name, default).value
    value = node.declare_parameter(name, Parameter.Type.DOUBLE_ARRAY).value
    return default if value is None else tuple(float(v) for v in value)


def reliable_qos(depth: int = 10) -> QoSProfile:
    qos = QoSProfile(depth=depth)
    qos.reliability = ReliabilityPolicy.RELIABLE
    return qos


def latched_qos(depth: int = 1) -> QoSProfile:
    """reliable + transient_local (late joiners get the last sample)."""
    qos = reliable_qos(depth)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


def command_qos() -> QoSProfile:
    """path_cmd subscription: reliable, depth 1, deliberately VOLATILE."""
    return reliable_qos(1)


def to_path_cmd(msg) -> st.PathCmd:
    """``host_mission_interfaces/PathCommand`` -> ROS-free command."""
    return st.PathCmd(
        epoch=int(msg.epoch), cmd_id=int(msg.cmd_id), mode=str(msg.mode).strip().upper(),
        poses=tuple((p.x, p.y, p.theta) for p in msg.poses),
        speed_cap=float(msg.speed_cap), goal_pos_tol=float(msg.goal_pos_tol),
        goal_yaw_tol=float(msg.goal_yaw_tol), pick_at_goal=bool(msg.pick_at_goal),
        lead_robot=str(msg.lead_robot).strip(), gap=float(msg.gap),
        pivot_first=bool(msg.pivot_first), pivot_heading=float(msg.pivot_heading),
        fixture_ahead=bool(msg.fixture_ahead))


def yaw_from_quaternion(q) -> float:
    """Yaw (CCW from +x, rad) of a geometry_msgs Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def now_s() -> float:
    """Receive-time clock. Freshness is judged on our own clock, not header stamps."""
    return time.monotonic()


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def run_node(node_factory) -> None:
    """Spin a node and send zero velocity before the context goes down."""
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = node_factory()
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for _ in range(_STOP_REPEATS):
            node.publish_stop()
            time.sleep(_STOP_INTERVAL_S)
        node.destroy_node()
        rclpy.shutdown()
