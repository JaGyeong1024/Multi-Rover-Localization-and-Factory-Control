"""Goal store and world model."""
import math
import os

import pytest

from host_mission.goals import GoalPose, GoalStore, RelativeGoal
from host_mission.settings import settings_from_dict
from host_mission.world_model import WorldModel

import helpers


def test_goal_store_roundtrip(tmp_path):
    path = str(tmp_path / 'goals.yaml')
    store = GoalStore.load(path)                 # missing file -> empty store
    assert store.names() == []
    store.set('pick_gear', GoalPose(4.9, 2.9, 0.0))
    store.set('leader_standby', GoalPose(2.6, 3.3, math.pi / 2))
    store.save()
    again = GoalStore.load(path)
    assert again.names() == ['leader_standby', 'pick_gear']
    assert again.get('leader_standby') == GoalPose(2.6, 3.3, round(math.pi / 2, 4))
    assert again.get('nope') is None and 'pick_gear' in again


def test_goal_store_save_keeps_symlink(tmp_path):
    # symlink-install links install/.../config to src: a save must write THROUGH the link
    real = tmp_path / 'src_goals.yaml'
    real.write_text('goals: {}\n')
    link = tmp_path / 'install_goals.yaml'
    os.symlink(real, link)
    store = GoalStore.load(str(link))
    store.set('a', GoalPose(1.0, 2.0, 0.5))
    store.save()
    assert os.path.islink(link)
    assert GoalStore.load(str(real)).get('a') == GoalPose(1.0, 2.0, 0.5)


def test_goal_store_rejects_bad_entries(tmp_path):
    path = tmp_path / 'goals.yaml'
    path.write_text('goals:\n  broken: {x: 1.0, y: 2.0}\n')
    with pytest.raises(ValueError):
        GoalStore.load(str(path))
    with pytest.raises(ValueError):
        GoalStore().set('', GoalPose(0.0, 0.0, 0.0))


def test_example_config_is_consistent():
    scenario, store = helpers.scenario(), helpers.goals()
    for part in scenario.parts.values():
        assert all(name in store for name in scenario.goal_names(part))
    cfg = helpers.settings()
    assert cfg.robot_names == ['leader', 'follower']
    assert cfg.republish_rate == 2.0


def test_settings_reject_unknown_keys():
    with pytest.raises(ValueError):
        settings_from_dict({'robots': {'leader': {'sped_cap': 0.1}}})
    with pytest.raises(ValueError):
        settings_from_dict({'robots': {}})


def test_usable_pose_rejects_stale_unaligned():
    world = WorldModel(pose_timeout=0.5)
    assert world.usable_pose('leader', 1.0) == (None, 'NO_POSE')
    world.update_pose('leader', 1.0, 2.0, 0.3, now=1.0)
    assert world.usable_pose('leader', 1.2) == (None, 'NOT_ALIGNED')
    assert world.usable_pose('leader', 1.2, need_heading=False)[0] == (1.0, 2.0, 0.3)
    world.update_heading_aligned('leader', True)
    assert world.usable_pose('leader', 1.2) == ((1.0, 2.0, 0.3), '')
    assert world.usable_pose('leader', 1.6) == (None, 'STALE_POSE')
    assert world.pose_fresh('leader', 1.4) and not world.pose_fresh('leader', 1.6)


def test_world_keeps_feedback_safety_quality():
    world = WorldModel()
    world.update_quality('leader', 'degraded')
    world.update_nav('leader', 'holding', '', 1, 7, now=3.0)
    world.update_safety('leader', 'stopped', ['lidar_stop'], 0.0, 4.0, 3.0, now=3.1)
    world.update_safety('follower', 'OK', [], 1.0, math.nan, math.nan, now=3.1)
    leader = world.robot('leader')
    assert leader.quality == 'DEGRADED'
    assert (leader.nav.state, leader.nav.cmd_id, leader.nav.time) == ('HOLDING', 7, 3.0)
    assert leader.safety.reasons == ('LIDAR_STOP',) and leader.safety.blocked == (4.0, 3.0)
    assert world.robot('follower').safety.blocked is None    # NaN = nothing reported


def test_obstacles_manual_detected_expiry():
    world = WorldModel()
    assert world.set_manual_obstacles([(3.0, 3.0, 0.2)])
    assert not world.set_manual_obstacles([(3.0, 3.0, 0.2)])     # unchanged -> no re-plan
    version = world.obstacle_version
    assert world.add_detected_region([(4.2, 3.0, 0.3)], (4.0, 3.0), expires=10.0)
    assert not world.add_detected_region([(4.25, 3.0, 0.3)], (4.05, 3.0), expires=20.0)
    assert [o.source for o in world.obstacles()] == ['manual', 'lidar']
    assert world.obstacle_version == version + 1
    assert not world.expire_obstacles(15.0)
    assert world.expire_obstacles(20.0)
    assert [o.source for o in world.obstacles()] == ['manual']


def test_world_holds_goal_store():
    world = WorldModel(goals=helpers.goals())
    assert world.goals.get('leader_standby') == GoalPose(2.6, 3.3, 0.0)


def test_relative_goal_follows_base_survives_save(tmp_path):
    """Follower load pose = 0.55 m to the right of the leader's pick pose."""
    path = tmp_path / 'g.yaml'
    path.write_text('goals:\n  pick: {x: 4.9, y: 4.15, theta: 1.5707963}\n'
                    '  load: {relative_to: pick, dx: 0.0, dy: -0.55, dtheta: 0.0}\n')
    store = GoalStore.load(str(path))
    load = store.get('load')
    assert (round(load.x, 3), round(load.y, 3)) == (5.45, 4.15)
    # a cycle cannot be loaded (see below) but a store built in memory can hold one
    assert GoalStore({'loop': RelativeGoal('loop', 1.0, 0.0, 0.0)}).get('loop') is None
    store.set('pick', GoalPose(4.0, 3.0, 0.0))
    assert (round(store.get('load').x, 3), round(store.get('load').y, 3)) == (4.0, 2.45)
    store.save()
    assert GoalStore.load(str(path)).get('load') == store.get('load')


def _load_text(tmp_path, text):
    path = tmp_path / 'g.yaml'
    path.write_text(text)
    return GoalStore.load(str(path))


def test_relative_goal_unknown_base_rejected(tmp_path):
    # A silent None here would only show up as "goal not taught" at dispatch time.
    with pytest.raises(ValueError, match="'load'.*unknown goal 'pik'"):
        _load_text(tmp_path, 'goals:\n  pick: {x: 1.0, y: 1.0, theta: 0.0}\n'
                             '  load: {relative_to: pik, dx: 0.0}\n')


@pytest.mark.parametrize('text', [
    'goals:\n  loop: {relative_to: loop, dx: 1.0}\n',
    'goals:\n  a: {relative_to: b, dx: 1.0}\n  b: {relative_to: a, dx: 1.0}\n',
    'goals:\n  p: {x: 1.0, y: 1.0, theta: 0.0}\n  a: {relative_to: b, dx: 1.0}\n'
    '  b: {relative_to: c, dx: 1.0}\n  c: {relative_to: b, dx: 1.0}\n'])
def test_relative_goal_cycle_rejected(tmp_path, text):
    with pytest.raises(ValueError, match='cycle'):
        _load_text(tmp_path, text)


def test_relative_goal_chain_loads(tmp_path):
    store = _load_text(tmp_path, 'goals:\n  p: {x: 1.0, y: 1.0, theta: 0.0}\n'
                                 '  a: {relative_to: p, dx: 1.0}\n  b: {relative_to: a, dx: 1.0}\n')
    assert store.get('b') == GoalPose(3.0, 1.0, 0.0)


def test_dispatch_rejects_unresolved_goal():
    """`in` tests only the raw name; a goal made unresolvable at run time must be refused."""
    engine = helpers.engine()
    name, part = next(iter(engine.scenario.parts.items()))
    engine.world.goals.set(part.follower_goal, RelativeGoal('gone', 0.0, -0.55, 0.0))
    assert part.follower_goal in engine.world.goals
    assert engine.dispatch(name) is False
