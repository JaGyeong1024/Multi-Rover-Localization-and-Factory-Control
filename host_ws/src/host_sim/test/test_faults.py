"""Fault injection: each fault, timed expiry, and the JSON command interface."""
import json

import pytest
from conftest import make_world

from host_sim.core.commands import apply_command


def step(world, duration, command=None, dt=0.01):
    """Step the world; return the localization samples emitted for 'leader'."""
    samples = []
    for _ in range(int(round(duration / dt))):
        if command is not None:
            world.set_command('leader', *command)
        world.step(dt)
        sample = world.poll_pose('leader')
        if sample is not None:
            samples.append(sample)
    return samples


def test_pose_blackout_stops_resumes(world):
    assert len(step(world, 1.0)) > 20
    world.injector.pose_blackout('leader')
    assert step(world, 1.0) == []
    world.injector.pose_blackout('leader', enabled=False)
    assert len(step(world, 1.0)) > 20


def test_timed_fault_reverts_by_itself(world):
    step(world, 0.5)
    world.injector.pose_blackout('leader', duration=1.0)
    assert step(world, 0.9) == []
    assert len(step(world, 1.0)) > 20
    assert world.faults['leader'].pose_blackout is False


def test_single_sample_pose_jump(world):
    step(world, 0.5)
    world.injector.pose_jump('leader', 0.5, -0.3)
    xs = [(s.x, s.y) for s in step(world, 0.5)]
    assert xs[0] == pytest.approx((3.5, 2.7))
    assert all(xy == pytest.approx((3.0, 3.0)) for xy in xs[1:])


def test_sustained_pose_jump_until_cleared(world):
    world.injector.pose_jump('leader', 0.4, 0.0, sustained=True)
    assert all(s.x == pytest.approx(3.4) for s in step(world, 1.0))
    world.injector.clear('leader')
    assert step(world, 0.5)[-1].x == pytest.approx(3.0)


def test_quality_override(world):
    assert world.pose_status('leader') == (True, 'GOOD')
    world.injector.set_quality('leader', 'LOST')
    assert world.pose_status('leader')[1] == 'LOST'
    assert step(world, 0.5)[-1].quality == 'LOST'
    world.injector.set_quality('leader', None)
    assert world.pose_status('leader')[1] == 'GOOD'
    with pytest.raises(ValueError):
        world.injector.set_quality('leader', 'BAD')


def test_hold_unaligned_zeroes_the_heading():
    world = make_world(start=(3.0, 3.0, 1.0))
    world.injector.hold_unaligned('leader')
    sample = step(world, 0.5)[-1]
    assert not sample.heading_aligned and sample.theta == 0.0
    world.injector.hold_unaligned('leader', enabled=False)
    sample = step(world, 0.5)[-1]
    assert sample.heading_aligned and sample.theta == pytest.approx(1.0)


def test_blocked_command_path_times_out(world):
    step(world, 1.0, command=(0.2, 0.0))
    world.injector.block_commands('leader')
    step(world, 1.0, command=(0.2, 0.0))
    robot = world.robots['leader']
    assert robot.v == pytest.approx(0.0)
    assert robot.x == pytest.approx(3.0 + 0.2 * 1.5, abs=0.01)  # coasted for cmd_timeout
    world.injector.block_commands('leader', enabled=False)
    step(world, 0.2, command=(0.2, 0.0))
    assert robot.v == pytest.approx(0.2)


def test_stuck_rover_spins_wheels(world):
    world.injector.stuck('leader')
    step(world, 1.0, command=(0.2, 0.5))
    robot = world.robots['leader']
    assert (robot.x, robot.y, robot.theta) == (3.0, 3.0, 0.0)
    assert robot.v == pytest.approx(0.2)
    world.injector.stuck('leader', enabled=False)
    step(world, 1.0, command=(0.2, 0.0))
    assert robot.x == pytest.approx(3.2)


def test_dropped_obstacle_in_scan_expires():
    world = make_world(walls=False)
    name = world.injector.obstacle_ahead('leader', distance=0.6, radius=0.1, duration=1.0)
    assert world.scan('leader')[0] == pytest.approx(0.5)
    assert [o.name for o in world.obstacles] == [name]
    step(world, 1.1)
    assert world.obstacles == []


def test_clear_removes_every_fault(world):
    injector = world.injector
    injector.pose_blackout('leader')
    injector.stuck('leader')
    injector.block_commands('leader', duration=5.0)
    injector.clear()
    fault = world.faults['leader']
    assert not (fault.pose_blackout or fault.stuck or fault.cmd_blocked)


@pytest.mark.parametrize('text, check', [
    ('{"cmd":"fault","type":"pose_blackout","robot":"leader","duration":2.0}',
     lambda w: w.faults['leader'].pose_blackout),
    ('{"cmd":"fault","type":"pose_jump","robot":"leader","dx":0.5,"dy":0.1}',
     lambda w: w.faults['leader'].jump_once == (0.5, 0.1)),
    ('{"cmd":"fault","type":"pose_jump","robot":"leader","dx":0.5,"dy":0,"sustained":true}',
     lambda w: w.faults['leader'].jump_offset == (0.5, 0.0)),
    ('{"cmd":"fault","type":"quality","robot":"leader","quality":"DEGRADED"}',
     lambda w: w.faults['leader'].quality == 'DEGRADED'),
    ('{"cmd":"fault","type":"hold_unaligned","robot":"leader"}',
     lambda w: w.faults['leader'].hold_unaligned),
    ('{"cmd":"fault","type":"cmd_blocked","robot":"leader","enabled":true}',
     lambda w: w.faults['leader'].cmd_blocked),
    ('{"cmd":"fault","type":"stuck","robot":"leader"}',
     lambda w: w.faults['leader'].stuck),
    ('{"cmd":"fault","type":"camera_lost","robot":"leader","duration":2.0}',
     lambda w: w.faults['leader'].camera_lost),
    ('{"cmd":"fault","type":"obstacle_ahead","robot":"leader","distance":0.5,"radius":0.1}',
     lambda w: len(w.obstacles) == 1 and w.obstacles[0].x == pytest.approx(3.5)),
    ('{"cmd":"add_obstacle","type":"rect","x":4,"y":3,"width":0.4,"height":0.2,"name":"box"}',
     lambda w: w.obstacles[0].name == 'box'),
    ('{"cmd":"reset_pose","robot":"leader","x":2.0,"y":2.5,"theta":1.0}',
     lambda w: (w.robots['leader'].x, w.robots['leader'].theta) == (2.0, 1.0)),
])
def test_json_commands(world, text, check):
    assert apply_command(world, json.loads(text))
    assert check(world)


def test_json_obstacle_lifecycle_fault_clear(world):
    apply_command(world, {'cmd': 'add_obstacle', 'type': 'circle', 'x': 4.0, 'y': 3.0,
                          'radius': 0.1, 'name': 'post'})
    assert 'removed' in apply_command(world, {'cmd': 'remove_obstacle', 'name': 'post'})
    assert 'not found' in apply_command(world, {'cmd': 'remove_obstacle', 'name': 'post'})
    apply_command(world, {'cmd': 'fault', 'type': 'stuck', 'robot': 'leader'})
    apply_command(world, {'cmd': 'clear_faults'})
    assert not world.faults['leader'].stuck


@pytest.mark.parametrize('command', [
    {'cmd': 'explode'},
    {'cmd': 'fault', 'type': 'gremlin', 'robot': 'leader'},
    {'cmd': 'fault', 'type': 'stuck', 'robot': 'nobody'},
    {'cmd': 'add_obstacle', 'type': 'circle', 'x': 1.0},
    {'cmd': 'reset_pose', 'robot': 'leader'},
])
def test_bad_commands_are_rejected(world, command):
    with pytest.raises((ValueError, KeyError, TypeError)):
        apply_command(world, command)
