"""Shared pytest helpers for the host_sim core tests."""
import copy

import pytest

from host_sim.config import load_config
from host_sim.core import (Arena, LidarParams, LocalizationParams, Robot,
                           RobotParams, World)


def instant_params(**overrides):
    """Base without lag or acceleration limit, so motion follows commands exactly."""
    values = dict(linear_time_constant=0.0, angular_time_constant=0.0,
                  max_linear_accel=1e9, max_angular_accel=1e9)
    values.update(overrides)
    return RobotParams(**values)


def drive(robot, v, w, duration, dt=0.01):
    """Hold a command for `duration` seconds (re-sent every step)."""
    for _ in range(int(round(duration / dt))):
        robot.set_command(v, w)
        robot.step(dt)


def make_world(seed=0, walls=True, loc=None, lidar=None, base=None, start=(3.0, 3.0, 0.0)):
    """One-rover world ('leader') with noise-free sensors unless overridden."""
    world = World(Arena(walls_enabled=walls), seed=seed)
    loc = loc or LocalizationParams(pos_noise_std=0.0, heading_noise_std=0.0)
    lidar = lidar or LidarParams(noise_std=0.0)
    world.add_robot('leader', Robot(base or instant_params(), *start), loc, lidar)
    return world


@pytest.fixture
def world():
    return make_world()


def fixed_frame_config():
    """Default world with a pinned arena and start poses (the old frame these tests were
    written in), so they do not follow the site layout edited in sim_default.yaml."""
    config = copy.deepcopy(load_config())
    config['arena'].update(x_min=1.24, x_max=6.03, y_min=1.97, y_max=4.66)
    config['robots']['leader']['start'] = [2.6, 3.3, 0.0]
    config['robots']['follower']['start'] = [1.9, 3.3, 0.0]
    return config
