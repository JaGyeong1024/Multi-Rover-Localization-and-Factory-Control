"""Sanity of the config files AS SHIPPED (the field values), without any simulation.

The logic tests read frozen copies (test/data, see helpers.py); this file is the only
one that reads config/. It encodes the relations that mission.yaml and the settings
code state as requirements, so a careless field edit turns red here and nowhere else.
"""
import math
import os

import pytest

from host_mission.goals import GoalStore
from host_mission.settings import load_scenario, load_settings

from helpers import FIELD_CONFIG, FIELD_OVERLAYS, SITE_CONFIG


@pytest.fixture(scope='module')
def settings():
    return load_settings(os.path.join(FIELD_CONFIG, 'mission.yaml'), FIELD_OVERLAYS)


@pytest.fixture(scope='module')
def scenario():
    return load_scenario(os.path.join(SITE_CONFIG, 'mission_scenario.yaml'))


@pytest.fixture(scope='module')
def store():
    return GoalStore.load(os.path.join(SITE_CONFIG, 'goals.yaml'))


def _inside(bounds, x, y, grow=0.0):
    return (bounds.xmin - grow <= x <= bounds.xmax + grow
            and bounds.ymin - grow <= y <= bounds.ymax + grow)


def _distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _part_goal_pairs(scenario, store):
    """(part id, leader goal pose, follower goal pose) for every part."""
    return [(part.id, store.get(part.leader_goal), store.get(part.follower_goal))
            for part in scenario.parts.values()]


def test_bounds_are_a_proper_rectangle(settings):
    bounds = settings.bounds
    assert bounds is not None
    assert bounds.xmin < bounds.xmax and bounds.ymin < bounds.ymax


def test_scenario_names_only_taught_goals(scenario, store):
    for part in scenario.parts.values():
        missing = [name for name in scenario.goal_names(part) if store.get(name) is None]
        assert not missing, f'{part.id}: goals not in goals.yaml (or broken): {missing}'


def test_every_goal_lies_inside_the_bounds(settings, store):
    poses = {name: store.get(name) for name in store.names()}
    outside = {name: (pose.x, pose.y) for name, pose in poses.items()
               if not _inside(settings.bounds, pose.x, pose.y)}
    assert not outside, f'goals outside bounds {settings.bounds}: {outside}'


def test_fixtures_are_shelves_near_arena(settings):
    for box in settings.fixtures:
        assert box.width > 0 and box.height > 0
        reach = math.hypot(box.width, box.height) / 2
        assert _inside(settings.bounds, box.x, box.y, grow=reach), f'fixture {box} off the arena'


def test_conflict_dist_below_load_spacing(settings, scenario, store):
    # mission.yaml: else the follower always waits until the leader has parked.
    for part, pick, load in _part_goal_pairs(scenario, store):
        assert settings.platoon.conflict_dist < _distance(pick, load), part


def test_pick_dwell_below_phase_timeout(scenario):
    # mission_scenario.yaml: the pick dwell must stay below phase_timeout.
    assert 0 < scenario.pick_dwell < scenario.phase_timeout
