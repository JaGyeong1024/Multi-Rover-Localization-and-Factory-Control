"""PathCoordinatorNode._emit with a real rclpy node (isolated DDS domain).

Guards the crash "Logger severity cannot be changed between calls": rclpy rejects a
second severity on one logging call site, so _emit must keep info and warn apart.
"""
import json
import os

os.environ['ROS_DOMAIN_ID'] = '47'
os.environ['ROS_LOCALHOST_ONLY'] = '1'

import pytest

rclpy = pytest.importorskip('rclpy')

from host_mission import path_coordinator_node as pcn  # noqa: E402
from helpers import DATA  # noqa: E402

# Every kind the engine / sequencer / supervisor / commander can emit (grep for
# ``events.append((`` and ``Action(``); the warn ones come from _WARN_EVENTS.
_INFO_KINDS = ('enqueued', 'control', 'unload_done', 'obstacles', 'goal_saved', 'hold_cleared',
               'plan_recovered', 'path_issued', 'plan_waiting', 'resumed', 'loaded',
               'delivered', 'next_batch', 'done', 'add_obstacle', 'replan')


@pytest.fixture
def node():
    rclpy.init(args=[
        '--ros-args',
        '-p', f'settings_file:={os.path.join(DATA, "mission.yaml")}',
        '-p', 'settings_overlays:=[]',
        '-p', f'scenario_file:={os.path.join(DATA, "mission_scenario.yaml")}',
        '-p', f'goals_file:={os.path.join(DATA, "goals.yaml")}'])
    node = pcn.PathCoordinatorNode()
    node.engine.world.goals.path = None          # never rewrite the example goal file
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_emit_survives_any_event_order(node):
    sent = []
    node.event_pub.publish = lambda msg: sent.append(json.loads(msg.data))
    kinds = list(pcn._WARN_EVENTS) + list(_INFO_KINDS)
    # Alternate warn / info, twice: a severity change on one call site fails on the 2nd call.
    order = [k for pair in zip(pcn._WARN_EVENTS, _INFO_KINDS) for k in pair] + kinds
    for kind in order:
        node._emit(kind, 'detail')
    assert [e['kind'] for e in sent] == order
    assert set(pcn._WARN_EVENTS) <= set(kinds)
