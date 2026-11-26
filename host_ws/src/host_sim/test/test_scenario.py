"""Scenario runner: closed-loop runs with a plugged-in controller."""
import math

import pytest

from conftest import fixed_frame_config
from host_sim.core import Circle, World
from host_sim.scenario import Goal, run_scenario

GOAL_X, GOAL_Y = 3.4, 3.3


def go_to_point(goal_x, goal_y, speed=0.15, gain=1.5, stop_radius=0.03):
    """Minimal test controller: steer toward a point, stop when close."""
    def controller(obs):
        if obs.pose is None:
            return 0.0, 0.0
        dx, dy = goal_x - obs.pose.x, goal_y - obs.pose.y
        if math.hypot(dx, dy) < stop_radius:
            return 0.0, 0.0
        error = math.atan2(dy, dx) - obs.pose.theta
        error = math.atan2(math.sin(error), math.cos(error))
        return speed * max(0.0, math.cos(error)), gain * error
    return controller


def default_world(seed=1):
    return World.from_config(fixed_frame_config(), seed=seed)


def test_follower_reaches_goal_no_collision():
    result = run_scenario(
        default_world(), {'follower': go_to_point(1.9, 4.2)}, duration=40.0,
        goals={'follower': Goal(1.9, 4.2, pos_tol=0.08)},
        reference_paths={'follower': [[1.9, 3.3], [1.9, 4.2]]})
    follower = result.robots['follower']
    assert follower.arrived and follower.arrival_time < 30.0
    assert follower.final_pos_error < 0.08
    assert follower.max_cross_track < 0.25
    assert follower.path_length > 0.8
    assert result.duration < 40.0          # stopped early on arrival
    assert not result.collided
    assert 0.0 < result.min_robot_gap < 0.5
    assert follower.trace.shape[1] == 4


def test_slipping_leader_also_reaches_goal():
    result = run_scenario(
        default_world(), {'leader': go_to_point(GOAL_X, GOAL_Y)}, duration=40.0,
        goals={'leader': Goal(GOAL_X, GOAL_Y, pos_tol=0.08)})
    assert result.robots['leader'].arrived
    assert not result.collided


def test_silent_controller_reaches_no_goal():
    result = run_scenario(default_world(), {'leader': lambda obs: None}, duration=2.0,
                          goals={'leader': Goal(GOAL_X, GOAL_Y)})
    leader = result.robots['leader']
    assert not leader.arrived and leader.arrival_time is None
    assert leader.path_length == 0.0
    assert leader.final_pos_error == pytest.approx(GOAL_X - 2.65, abs=1e-6)
    assert result.duration == pytest.approx(2.0)


def test_obstacle_on_path_is_collision():
    world = default_world()
    world.add_obstacle(Circle(3.1, 3.3, 0.1, name='post'))
    result = run_scenario(world, {'leader': go_to_point(GOAL_X, GOAL_Y)}, duration=20.0,
                          goals={'leader': Goal(GOAL_X, GOAL_Y)})
    assert result.collided
    assert result.collisions[0].other == 'post'
    assert result.min_obstacle_clearance < 0.0


def test_event_injects_a_fault_mid_run():
    def stick(world):
        world.injector.stuck('leader')
    result = run_scenario(default_world(), {'leader': go_to_point(GOAL_X, GOAL_Y)},
                          duration=10.0, goals={'leader': Goal(GOAL_X, GOAL_Y)},
                          events=[(1.0, stick)])
    leader = result.robots['leader']
    assert not leader.arrived
    assert 0.0 < leader.path_length < 0.2


def test_observation_contents():
    seen = []

    def controller(obs):
        seen.append(obs)
        return 0.0, 0.0
    run_scenario(default_world(), {'leader': controller}, duration=1.0)
    first, last = seen[0], seen[-1]
    assert len(seen) == 20                      # 20 Hz control for 1 s
    assert first.pose is None and first.scan is not None and first.scan_is_new
    assert last.robot == 'leader' and last.pose.quality == 'GOOD'
    assert set(last.poses) == {'leader', 'follower'}
    assert last.poses['follower'].x == pytest.approx(1.9, abs=0.1)
    assert last.scan.shape == last.scan_angles.shape == (360,)
    assert any(obs.pose_is_new for obs in seen)
    assert last.depth_scan.shape == last.depth_scan_angles.shape == (80,)
    assert sum(obs.depth_scan_is_new for obs in seen) == 10                 # 10 Hz for 1 s


def test_rover_without_lidar_or_depth():
    config = fixed_frame_config()
    config['robots']['leader']['lidar'] = None
    del config['robots']['leader']['depth_scan']
    world = World.from_config(config, seed=1)
    assert 'leader' not in world.lidars and not world.lidar_due('leader')
    seen = []
    run_scenario(world, {'leader': lambda obs: seen.append(obs) or (0.0, 0.0)}, duration=0.5)
    assert all(obs.scan is None and obs.scan_angles is None and not obs.scan_is_new
               and obs.depth_scan is None and obs.depth_scan_angles is None for obs in seen)


def test_heading_goal_is_checked():
    controller = go_to_point(GOAL_X, GOAL_Y)
    world = default_world()
    # With speed truncation this toy controller overshoots and pivots in place, sweeping
    # through every heading at rest.
    world.robots['leader'].params.linear_quantum = 0.0
    world.robots['leader'].params.hw_angular_sign = 1.0   # the toy controller does not apply the rover's angular_sign
    wrong = run_scenario(world, {'leader': controller}, duration=30.0,
                         goals={'leader': Goal(GOAL_X, GOAL_Y, theta=math.pi / 2, pos_tol=0.08)})
    assert not wrong.robots['leader'].arrived
    assert wrong.robots['leader'].final_yaw_error > 1.0
