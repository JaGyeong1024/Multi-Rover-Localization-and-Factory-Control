"""FusedLocalization: the real fusion front-end on simulated camera / UWB / IMU in a minimal world."""
import math

import numpy as np
import pytest
from conftest import instant_params

from host_sim.core import Arena, FusedLocalization, FusedParams, LidarParams, Robot, World
from host_sim.core.util import wrap_angle

DT = 0.01
START = (3.0, 3.0, 0.0)


def fused_world(name='follower', seed=0, pose=START, **sections):
    """One rover with sensor-level localization (FusedParams sections as keyword dicts)."""
    world = World(Arena(walls_enabled=False), seed=seed)
    world.add_robot(name, Robot(instant_params(), *pose), FusedParams.from_dict(sections),
                    LidarParams(noise_std=0.0))
    return world


def run(world, duration, command=None, name=None):
    """Step the world; returns [(arrival time, PoseMeasurement, true tracked point)]."""
    name = name or next(iter(world.robots))
    out = []
    for _ in range(int(round(duration / DT))):
        if command is not None:
            world.set_command(name, *command)
        world.step(DT)
        pose = world.poll_pose(name)
        if pose is not None:
            out.append((world.t, pose, world.robots[name].tracked_point()))
    return out


def position_error(row):
    _, pose, truth = row
    return math.hypot(pose.x - truth[0], pose.y - truth[1])


def test_world_builds_sensor_model_fused():
    world = fused_world()
    assert isinstance(world.sensors['follower'], FusedLocalization)
    assert world.pose_status('follower') == (False, 'LOST')    # before any measurement


def test_parked_pose_on_truth_no_heading():
    world = fused_world()
    rows = run(world, 5.0)
    assert len(rows) == pytest.approx(5.0 * 30, abs=3)          # global_position_node rate
    last = rows[-1]
    assert position_error(last) < 0.01
    assert last[1].quality == 'GOOD'
    assert not last[1].heading_aligned and last[1].theta == 0.0
    assert world.pose_status('follower') == (False, 'GOOD')


@pytest.mark.parametrize('name', ['leader', 'follower'])
def test_prealigned_rover_has_heading_parked(name):
    yaw = 2.0
    rows = run(fused_world(name, pose=(3.0, 3.0, yaw), start={'aligned': True}), 1.0)
    assert rows[-1][1].heading_aligned
    assert abs(wrap_angle(rows[-1][1].theta - yaw)) < 0.02
    late = [r for r in rows if r[0] > 0.5]
    assert all(r[1].heading_aligned for r in late)


def test_prealignment_error_in_heading():
    rows = run(fused_world(start={'aligned': True, 'heading_error': 0.05}), 1.0)
    assert rows[-1][1].heading_aligned
    assert abs(wrap_angle(rows[-1][1].theta - START[2]) - 0.05) < 0.01


def test_pose_arrives_after_relay_delay():
    rows = run(fused_world(), 3.0)[10:]
    lags = np.array([t - pose.stamp for t, pose, _ in rows])    # stamp = fusion emit time
    assert lags.min() >= 0.03 - 1e-9                            # transport delay
    assert lags.max() <= 0.03 + 1.0 / 20.0 + DT                 # + IMU emit period + step


@pytest.mark.parametrize('name', ['leader', 'follower'])
def test_straight_drive_aligns_the_heading(name):
    world = fused_world(name, pose=(2.0, 3.0, 0.3))
    rows = run(world, 5.5, command=(0.2, 0.0))
    _, pose, truth = rows[-1]
    assert pose.heading_aligned
    assert abs(wrap_angle(pose.theta - truth[2])) < math.radians(5.0)
    assert position_error(rows[-1]) < 0.03


def test_short_dropout_keeps_quality_pose():
    world = fused_world()
    run(world, 3.0)
    world.injector.camera_lost('follower', duration=0.8)       # shorter than bev_timeout 1 s
    rows = run(world, 2.0)
    assert all(pose.quality == 'GOOD' for _, pose, _ in rows)
    assert np.diff([t for t, _, _ in rows]).max() < 0.05
    assert max(position_error(row) for row in rows) < 0.02


def test_dropout_without_uwb_stops_pose():
    world = fused_world(uwb={'enabled': False})
    run(world, 3.0)
    lost_at = world.t
    world.injector.camera_lost('follower', duration=5.0)
    rows = run(world, 8.0)
    during = [t - lost_at for t, _, _ in rows if t - lost_at < 5.0]
    # last pose: last fix + pose_timeout 1 s + stale_timeout 0.5 s + transport, no later
    assert 1.0 < max(during) < 1.0 + 0.5 + 0.03 + 0.1
    assert world.pose_status('follower')[1] == 'GOOD'
    after = [row for row in rows if row[0] - lost_at >= 5.0]
    assert after[0][0] - lost_at < 5.0 + 0.2                    # back within a few frames
    assert position_error(after[-1]) < 0.01


def test_quality_lost_without_fresh_source():
    world = fused_world(uwb={'enabled': False})
    run(world, 3.0)
    world.injector.camera_lost('follower')
    run(world, 2.0)
    assert world.pose_status('follower') == (False, 'LOST')
    assert run(world, 0.5) == []


def test_foreign_frame_uwb_ignored_parked():
    """Camera lost next to the leader, UWB in a frame 1.7 m away: the estimator must not
    slide onto the UWB frame or refuse the camera afterwards."""
    world = fused_world(uwb={'offset_x': 1.2, 'offset_y': 1.2})     # 1.70 m
    before = run(world, 5.0)[-1]
    world.injector.camera_lost('follower', duration=15.0)
    during = run(world, 15.0)
    after = run(world, 10.0)
    assert position_error(before) < 0.05
    assert max(position_error(row) for row in during) < 0.15    # UWB carries it in the camera frame
    assert not during[-1][1].heading_aligned                    # no heading from UWB noise
    assert position_error(after[-1]) < 0.01                     # camera used again
    assert after[-1][1].quality == 'GOOD'


def test_output_faults_as_in_ideal_mode():
    world = fused_world()
    run(world, 2.0)
    world.injector.pose_jump('follower', 0.5, 0.0, sustained=True)
    world.injector.set_quality('follower', 'DEGRADED')
    _, pose, truth = run(world, 0.2)[-1]
    assert pose.x - truth[0] == pytest.approx(0.5, abs=0.01)
    assert pose.quality == 'DEGRADED'
    world.injector.pose_blackout('follower')
    assert run(world, 0.5) == []


def test_same_seed_same_fused_output():
    def sequence(seed):
        return [(p.x, p.y, p.theta) for _, p, _ in run(fused_world(seed=seed), 2.0, (0.2, 0.0))]
    assert sequence(3) == sequence(3)
    assert sequence(3) != sequence(4)
