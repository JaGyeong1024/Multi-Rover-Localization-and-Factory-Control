"""Mapper yaml files: every key is a parameter obstacle_mapper_node really declares.

ROS silently ignores a yaml key the node never declared, so a misspelled key becomes the
MapperParams / ScanSource default. Reads the node source as text (no rclpy needed).
"""
import dataclasses
import os
import re

import pytest
import yaml

from host_mission.obstacle_mapper import MapperParams, ScanSource
from host_mission.settings import merge_dicts

from helpers import FIELD_CONFIG, HARDWARE_CONFIG, PACKAGE, SITE_CONFIG

# Each entry is the parameter files the node is launched with (later wins).
_FILES = {'config': (os.path.join(FIELD_CONFIG, 'obstacle_mapper.yaml'),
                     os.path.join(SITE_CONFIG, 'obstacle_mapper.yaml'),
                     os.path.join(HARDWARE_CONFIG, 'sensor_mounts.yaml')),
          'scenarios': (os.path.join(PACKAGE, 'scenarios', 'data', 'obstacle_mapper.yaml'),)}
# The settings_file / settings_overlays parameters are declared in the shared loader.
_NODE_SOURCES = [os.path.join(PACKAGE, 'host_mission', name)
                 for name in ('obstacle_mapper_node.py', 'path_coordinator_node.py')]


def _node_source():
    text = ''
    for path in _NODE_SOURCES:
        with open(path) as f:
            text += f.read()
    return text


def _params(paths):
    merged = {}
    for path in paths:
        with open(path) as f:
            merged = merge_dicts(merged,
                                 yaml.safe_load(f)['obstacle_mapper_node']['ros__parameters'])
    return merged


def _flatten(params, prefix=''):
    """ROS flattens nested yaml maps into dotted parameter names."""
    flat = {}
    for key, value in params.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f'{prefix}{key}.'))
        else:
            flat[prefix + key] = value
    return flat


def _declared(params):
    """Every parameter name the node declares for this file's sources and robots."""
    source = _node_source()
    # literal names the node declares by hand: declare('x', ...), declare_parameter('x', ...)
    names = set(re.findall(r"declare(?:_parameter)?\('(\w+)'", source))
    names |= {f.name for f in dataclasses.fields(MapperParams)}
    for source_name in params.get('sources', ()):
        names |= {f'{source_name}.{f.name}' for f in dataclasses.fields(ScanSource)
                  if f.name != 'name'}
        names.add(f'{source_name}.scan_topic')
    names |= {f'robot_radius.{robot}' for robot in params.get('robots', ())}
    names |= {f'heading_accuracy.{params[s]["robot"]}' for s in params.get('sources', ())}
    for name in _switched_off_sources(params):     # documentation of a mount, see below
        names |= {f'{name}.{key}' for key in params[name]}
        names.add(f'heading_accuracy.{params[name]["robot"]}')
    return names


def _switched_off_sources(params):
    """Source blocks not listed in ``sources``: the node ignores them, but they stay as the
    single record of that sensor's mount (test_cross_config reads it). Accepted only when
    every key is a ScanSource field, so a misspelled key in such a block still fails."""
    fields = {f.name for f in dataclasses.fields(ScanSource)} | {'scan_topic'}
    return [name for name, value in params.items()
            if isinstance(value, dict) and name not in params.get('sources', ())
            and 'robot' in value and set(value) <= fields]


@pytest.mark.parametrize('where', _FILES)
def test_mapper_yaml_keys_declared_by_node(where):
    params = _params(_FILES[where])
    flat = set(_flatten(params))
    unknown = sorted(flat - _declared(params))
    assert not unknown, f'{list(_FILES[where])}: keys the node never declares: {unknown}'


def test_node_declares_hand_written_parameters():
    # Guards the regex above: these are declared by hand in the node and its settings loader.
    source = _node_source()
    declared = set(re.findall(r"declare(?:_parameter)?\('(\w+)'", source))
    assert {'output_topic', 'ignore_zones', 'sources', 'robots', 'settings_file',
            'settings_overlays'} <= declared
