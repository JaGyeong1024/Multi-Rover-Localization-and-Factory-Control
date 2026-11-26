"""Relations BETWEEN the host and rover config files, as the yaml comments state them.

Each rule lives in a comment in one file and depends on a number in another; nothing else
checks it, so a field edit on one side breaks the other silently. Reads the shipped files
(host_mission/config, rover_drive/config, config/) and the filter's source text; no ROS needed."""
import math
import os
import re

import pytest
import yaml

from host_mission.settings import merge_dicts

from helpers import FIELD_CONFIG, HARDWARE_CONFIG, PACKAGE, REPO, SITE_CONFIG, field_settings_raw

_REPO = REPO
_ROVER_CONFIG = os.path.join(_REPO, 'rover_ws', 'src', 'rover_drive', 'config')
_SE2NHC = os.path.join(_REPO, 'host_ws', 'src', 'host_localization', 'host_localization',
                       'fusion', 'se2nhc.py')


def _yaml(*parts):
    with open(os.path.join(*parts)) as f:
        return yaml.safe_load(f)


def _rover_params(name):
    """Drive parameters of a rover: its base file with the hardware file over it."""
    base = _yaml(_ROVER_CONFIG, name + '.yaml')['/**']['ros__parameters']
    hardware = _yaml(HARDWARE_CONFIG, f'rover_{name}.yaml')['/**']['ros__parameters']
    return merge_dicts(base, hardware)


@pytest.fixture(scope='module')
def mission():
    return field_settings_raw()


@pytest.fixture(scope='module')
def mapper():
    files = [(FIELD_CONFIG, 'obstacle_mapper.yaml'), (SITE_CONFIG, 'obstacle_mapper.yaml'),
             (HARDWARE_CONFIG, 'sensor_mounts.yaml')]
    merged = {}
    for folder, name in files:
        merged = merge_dicts(merged, _yaml(folder, name)['obstacle_mapper_node']['ros__parameters'])
    return merged


@pytest.fixture(scope='module', params=['leader', 'follower'])
def rover(request):
    return request.param, _rover_params(request.param)


@pytest.fixture(scope='module')
def rovers():
    return {name: _rover_params(name) for name in ('leader', 'follower')}


def _footprint(mission, robot):
    front, rear, half_width = mission['robots'][robot]['plan']['footprint']
    return front, rear, half_width


def _cog_vmin():
    with open(_SE2NHC) as f:
        match = re.search(r'cog_vmin=([0-9.]+)', f.read())
    assert match, 'cog_vmin default not found in se2nhc.py'
    return float(match.group(1))


def test_platoon_path_gap_covers_free_gap_offsets(mission, rovers):
    # mission.yaml platoon.gap: keep gap >= gap_free_target + follower point->front edge
    # + leader point->rear, so the fallback never comes closer than the LiDAR target.
    follower_front = _footprint(mission, 'follower')[0]
    leader_rear = _footprint(mission, 'leader')[1]
    need = rovers['follower']['gap_free_target'] + follower_front + leader_rear
    assert mission['platoon']['gap'] >= need


def test_alignment_creep_above_course_gate(rover):
    # rover yaml: align speed must stay above se2nhc cog_vmin or the heading never anchors.
    # The rover really creeps at min(align_speed, max_linear_speed) (controller.py).
    name, params = rover
    creep = min(params['align_speed'], params['max_linear_speed'])
    assert creep >= 1.2 * _cog_vmin(), f'{name}: creep {creep} m/s, gate {_cog_vmin()} m/s'


def test_gate_lets_the_alignment_creep_pass(rover):
    # gate: unaligned, only 0..align_pass_speed passes; below the creep it would block it.
    name, params = rover
    assert params['align_pass_speed'] >= min(params['align_speed'], params['max_linear_speed'])


def test_unseen_box_depth_is_mapper_depth(mission, mapper):
    # mission.yaml supervisor: "same value as the mapper's box_depth".
    assert mission['supervisor']['assumed_obstacle_depth'] == mapper['box_depth']


@pytest.mark.parametrize('robot', ['leader', 'follower'])
def test_mapper_exclusion_covers_footprint(mission, mapper, robot):
    # A scan must never map the other rover: the exclusion radius reaches the farthest
    # footprint corner from the tracked point.
    front, rear, half_width = _footprint(mission, robot)
    corner = math.hypot(max(front, rear), half_width)
    assert mapper['robot_radius'][robot] >= corner


def test_leader_pick_approach_covers_run_in(
        mission, rovers):
    # leader.yaml "pick approach": exempt only on the straight run-in and within
    # pick_approach_dist; the run-in length (host approach_len) + shelf distance at the goal
    # must exceed the stop distance (~lidar_stop_base), else the gate stops for the shelf
    # before the run-in. Reading: (1) the exempt zone covers the whole run-in:
    # pick_approach_dist >= approach_len; (2) with the shelf as close to the goal as
    # front_exempt_reach allows, the first run-in point already sees the shelf beyond the
    # stop distance (lidar_stop_base + lidar_to_front).
    leader = rovers['leader']
    robot = mission['robots']['leader']
    approach_len = robot['plan']['approach_len']
    stop = leader['lidar_stop_base'] + leader['lidar_to_front']
    assert leader['pick_approach_dist'] >= approach_len
    assert approach_len + robot['front_exempt_reach'] > stop


def _slow_start(params):
    """Sensor distance at which the gate starts limiting the rover's top speed: the d
    with limits.stopping_speed(d - lidar_to_front, base, decel, reaction) = max speed."""
    v = params['max_linear_speed']
    return (params['lidar_to_front'] + params['lidar_stop_base'] + v * params['lidar_reaction_s']
            + v * v / (2.0 * params['lidar_brake_decel']))


def test_leader_front_zone_starts_at_depth_camera(mission, mapper):
    # mission.yaml plan.front_zone[0] = the sensor ahead of the tracked point, whose
    # single source is obstacle_mapper.yaml (the sensor mounts).
    start = mission['robots']['leader']['plan']['front_zone'][0]
    assert start == mapper['leader_depth']['mount_x']


def test_front_zone_reaches_gate_slow_distance(mission, rover):
    # Shorter, and the host plans poses where the gate slows or stops the rover.
    name, params = rover
    _start, length, _half = mission['robots'][name]['plan']['front_zone']
    assert length >= _slow_start(params) + 0.05 - 1e-9, name


def test_front_zone_as_wide_as_gate_band(mission, rover):
    # Leader: depth band columns depth_col_min..max of a depth_hfov view, as wide as at the
    # zone's far end. Follower: the LiDAR corridor lidar_half_width. Both + 0.05 margin.
    name, params = rover
    _start, length, half_width = mission['robots'][name]['plan']['front_zone']
    if params['lidar_enabled']:
        band = params['lidar_half_width']
    else:
        edge = max(abs(params['depth_col_min'] - 0.5), abs(params['depth_col_max'] - 0.5))
        band = length * 2.0 * edge * math.tan(params['depth_hfov'] / 2.0)
    assert half_width >= band + 0.05 - 1e-9, name


def test_run_in_exemption_is_pick_approach(mission, rover):
    # The planner skips the zone exactly on the stretch the rover exempts (fixture_ahead).
    name, params = rover
    assert mission['robots'][name]['plan']['front_zone_runin'] == params['pick_approach_dist']


def test_leader_run_in_sees_shelf_beyond_zone(mission):
    # Pick goals face the shelf: the first pose of the run-in (approach_len before the goal)
    # must have the shelf beyond the zone, or the planner (like the gate) refuses the
    # approach before the exempt stretch begins; and the whole run-in must be exempt.
    plan = mission['robots']['leader']['plan']
    start, length, _half = plan['front_zone']
    data = os.path.join(PACKAGE, 'scenarios', 'data')
    goals = _yaml(data, 'goals.yaml')['goals']
    faces = {s['y']: s['x'] - s['width'] / 2.0 for s in _yaml(data, 'layouts.yaml')['shelves']}
    for name in ('pick_gear', 'pick_bolt'):
        goal = goals[name]
        assert goal['theta'] == 0.0 and goal['y'] in faces, name
        shelf_distance = faces[goal['y']] - goal['x']
        assert plan['approach_len'] + shelf_distance > start + length, name
    assert plan['approach_len'] <= plan['front_zone_runin']


def test_approach_timeout_exceeds_feedback_period(rover):
    # The gate drops approach_active when it is older than approach_timeout_s; the tracker
    # refreshes it with every feedback, so the timeout must span several periods.
    name, params = rover
    period = 1.0 / params['feedback_rate_hz']
    assert params['approach_timeout_s'] >= 2.0 * period, name


def test_goal_turn_ends_inside_tolerances(mission, rover):
    # path_commander._turn_at_goal re-enters a heading-only miss with a leading pivot at the
    # goal (rover within goal_pos_tol of it). The rover turns only within pivot_pos_tol of
    # the pivot's position, and ends the turn within pivot_tol, which must be inside
    # goal_yaw_tol or the goal is missed again.
    name, params = rover
    robot = mission['robots'][name]
    assert params['pivot_pos_tol'] >= robot['goal_pos_tol'], name
    assert params['pivot_tol'] < robot['goal_yaw_tol'], name
