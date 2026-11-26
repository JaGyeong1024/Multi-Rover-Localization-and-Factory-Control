"""Rover yaml files: every key is a parameter a launched node really declares.

ROS silently ignores a yaml key that no node declared, so a misspelled key turns into the
dataclass default (which is the follower's value). Reads the node sources and launch
files as text, so it runs without rclpy.
"""
import dataclasses
import os
import re

import pytest
import yaml

from rover_drive.core import config as config_module
from sim import load_params

_PKG = os.path.join(os.path.dirname(__file__), '..')
_ROBOTS = ('leader', 'follower')


def _read(*parts):
    with open(os.path.join(_PKG, *parts)) as f:
        return f.read()


def _yaml_params(robot):
    return load_params(robot)


def _declared_by(node):
    """Parameter names a node declares: the fields of the configs it passes to declare_config."""
    classes = re.findall(r'declare_config\(self, (\w+)\)', _read('rover_drive', node + '.py'))
    assert classes, f'{node} declares no config dataclass'
    return {f.name for name in classes
            for f in dataclasses.fields(getattr(config_module, name))}


def _launched_nodes(robot):
    return re.findall(r"drive_node\('(\w+)'", _read('launch', robot + '.launch.py'))


@pytest.mark.parametrize('robot', _ROBOTS)
def test_yaml_keys_declared_by_launched_node(robot):
    # The yaml has one '/**' section shared by all nodes of the robot; a key is real if at
    # least one of the robot's launched nodes declares it.
    nodes = _launched_nodes(robot)
    assert nodes
    declared = set().union(*(_declared_by(n) for n in nodes))
    unknown = sorted(set(_yaml_params(robot)) - declared)
    assert not unknown, f'{robot} yaml keys no launched node declares: {unknown}'


def test_leader_has_front_sensor_for_gate():
    # With neither a LiDAR nor a front distance topic the front stop is silently off.
    params = _yaml_params('leader')
    assert params['lidar_enabled'] is True or params['front_distance_topic'] != ''


_PRESET_DIR = os.path.join(_PKG, 'config', 'presets')
_PRESETS = sorted(f[:-len('.yaml')] for f in os.listdir(_PRESET_DIR) if f.endswith('.yaml'))


def _preset(name):
    text = _read('config', 'presets', name + '.yaml')
    return text, yaml.safe_load(text)['/**']['ros__parameters']


def test_shipped_presets_are_agreed_ones():
    assert _PRESETS == ['follower_front_path', 'follower_gap_bridge', 'follower_pivot_slow',
                        'follower_speed_025', 'leader_depth_scan_off', 'leader_face_align',
                        'leader_front_memory', 'leader_front_path', 'leader_pivot_slow',
                        'leader_runin_plain', 'leader_speed_015',
                        'leader_speed_020', 'leader_xtrack_off']


@pytest.mark.parametrize('name', _PRESETS)
def test_preset_keys_in_base_and_declared(name):
    robot = name.split('_', 1)[0]
    text, params = _preset(name)
    assert text.startswith('# Use when: ')
    assert params and 'robot' not in params
    base = _yaml_params(robot)
    assert set(params) - set(base) == set(), f'{name}: keys not in {robot}.yaml'
    declared = set().union(*(_declared_by(n) for n in _launched_nodes(robot)))
    assert set(params) - declared == set(), f'{name}: keys no launched node declares'
    for key, value in params.items():
        assert type(value) is type(base[key]), f'{name} {key}: {value!r}'
        assert value != base[key], f'{name} {key}: same as the base (changes nothing)'


@pytest.mark.parametrize('name', _PRESETS)
def test_preset_keys_in_startup_log(name):
    robot = name.split('_', 1)[0]
    for key in _preset(name)[1]:
        owners = [n for n in _launched_nodes(robot) if key in _declared_by(n)]
        assert owners
        for node in owners:
            source = _read('rover_drive', node + '.py')
            node_class = source[source.index('(Node):'):]
            init = node_class[node_class.index('def __init__'):]
            init = init[:init.index('\n    def ')]
            assert re.search(r'get_logger\(\)\.info\([^)]*\.' + key + r'\}', init, re.S), \
                f'{node} does not log {key} at start-up'
