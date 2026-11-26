"""path_cmd delivery to a (re)started rover node: never the last latched command.

The domain bridge hands the host's last (transient_local) PathCommand to every late joiner, even
long after the host died. Only a command sent after the node started may count; the host re-sends
at 2 Hz. Runs the real nodes in an isolated ROS domain under a test robot name.
"""
import time

import pytest

rclpy = pytest.importorskip('rclpy')
interfaces = pytest.importorskip('host_mission_interfaces.msg')

from geometry_msgs.msg import Pose2D                       # noqa: E402
from rclpy.executors import SingleThreadedExecutor         # noqa: E402

from rover_drive.path_tracker_node import PathTrackerNode  # noqa: E402
from rover_drive.ros_util import latched_qos, reliable_qos  # noqa: E402
from rover_drive.safety_gate_node import SafetyGateNode    # noqa: E402

_DOMAIN = 222             # not a project domain
_ROBOT = 'qos_probe'
_DEADLINE_S = 5.0


@pytest.fixture
def ros():
    rclpy.init(domain_id=_DOMAIN, args=['--ros-args', '-p', f'robot:={_ROBOT}'])
    yield
    rclpy.shutdown()


class _Host:
    """Stands in for host + bridge: a latched path_cmd publisher, feedback listeners."""

    def __init__(self):
        self.node = rclpy.create_node('fake_host')
        self.pub = self.node.create_publisher(interfaces.PathCommand, f'/{_ROBOT}/path_cmd',
                                              latched_qos())
        self.feedback = self.gate = None
        self.node.create_subscription(interfaces.NavFeedback, f'/{_ROBOT}/nav_feedback',
                                      lambda m: setattr(self, 'feedback', m), reliable_qos())
        self.node.create_subscription(interfaces.SafetyState, f'/{_ROBOT}/safety_state',
                                      lambda m: setattr(self, 'gate', m), reliable_qos())

    def send(self):
        msg = interfaces.PathCommand(epoch=1, cmd_id=7, mode='FOLLOW')
        msg.poses = [Pose2D(x=0.0, y=0.0, theta=0.0), Pose2D(x=1.0, y=0.0, theta=0.0)]
        self.pub.publish(msg)


def _spin_until(executor, condition, timeout=_DEADLINE_S):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not condition():
        executor.spin_once(timeout_sec=0.05)
    return condition()


def test_restarted_nodes_await_new_command(ros):
    host = _Host()
    host.send()
    tracker, gate = PathTrackerNode(), SafetyGateNode()
    executor = SingleThreadedExecutor()
    for node in (host.node, tracker, gate):
        executor.add_node(node)
    # Both nodes matched the latched publisher and report: a replay would be in by now.
    assert _spin_until(executor, lambda: host.pub.get_subscription_count() == 2
                       and host.feedback is not None and host.gate is not None)
    _spin_until(executor, lambda: False, timeout=0.5)
    assert host.feedback.cmd_id == 0
    assert 'STALE_CMD' in host.gate.reasons
    host.send()
    assert _spin_until(executor, lambda: host.feedback.cmd_id == 7
                       and 'STALE_CMD' not in host.gate.reasons)
    for node in (host.node, tracker, gate):
        node.destroy_node()
