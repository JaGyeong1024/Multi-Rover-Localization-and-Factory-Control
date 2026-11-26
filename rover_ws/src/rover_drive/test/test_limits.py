"""Speed limiters, the velocity ramp and omega quantisation."""
import dataclasses
import math

import numpy as np
import pytest

from rover_drive.core import limits as lim
from rover_drive.core.tracker import lookahead_distance

from sim import Rover, Sim, drive_config, latched, make_cmd, s_path


def test_strictest_picks_lowest_bound_reason():
    chosen = lim.strictest([lim.Limit(0.2, ''), lim.Limit(0.05, 'GAP'), lim.Limit(0.1, 'CURVE')])
    assert chosen == lim.Limit(0.05, 'GAP')


def test_curvature_limit_bounds_accel_omega():
    cfg = drive_config('follower')
    assert lim.curvature_limit(cfg, 0.0).speed == lim.NO_LIMIT
    v = lim.curvature_limit(cfg, -2.0).speed
    assert v * v * 2.0 <= cfg.max_lateral_accel + 1e-9
    assert lim.curvature_limit(cfg, 50.0).speed * 50.0 <= cfg.max_angular_speed + 1e-9


def test_goal_limit_decelerates_to_zero():
    cfg = drive_config('follower')
    speeds = [lim.goal_limit(cfg, d).speed for d in (1.0, 0.3, 0.1, 0.012)]
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[-1] == cfg.approach_speed
    assert lim.goal_limit(cfg, cfg.goal_stop_margin).speed == 0.0
    assert lim.goal_limit(cfg, -0.05).speed == 0.0


def test_arrival_limit_keeps_lookahead_in_path():
    cfg = dataclasses.replace(drive_config('leader'), max_linear_speed=0.15, runin_lookaheads=2.0)
    for remaining in (2.0, 0.6, 0.5, 0.4, 0.2, 0.0):
        speed = min(lim.arrival_limit(cfg, remaining).speed, cfg.max_linear_speed)
        lookahead = lookahead_distance(cfg, speed)
        assert lookahead * cfg.runin_lookaheads <= max(remaining, 1e-9) + 1e-9 \
            or lookahead == cfg.lookahead_min, remaining
    assert lim.arrival_limit(cfg, 0.6).speed == pytest.approx(0.15)
    assert lim.arrival_limit(cfg, 0.4).speed == pytest.approx(0.10)
    assert lim.arrival_limit(cfg, 0.0).speed == pytest.approx(0.10)


@pytest.mark.parametrize('robot', ['leader', 'follower'])
def test_arrival_limit_off_in_shipped_configs(robot):
    cfg = drive_config(robot)
    assert cfg.runin_lookaheads == 0.0
    assert lim.arrival_limit(cfg, 0.1).speed == lim.NO_LIMIT


def test_heading_limit_crawls_on_large_error():
    cfg = drive_config('follower')
    assert lim.heading_limit(cfg, 0.1, 0.2).speed == lim.NO_LIMIT
    assert lim.heading_limit(cfg, 2.5, 0.2).speed == pytest.approx(cfg.turn_speed)
    mid = lim.heading_limit(cfg, -0.8, 0.2).speed
    assert cfg.turn_speed < mid < 0.2


def test_quality_limit():
    cfg = drive_config('follower')
    assert lim.quality_limit(cfg, 'GOOD').speed == lim.NO_LIMIT
    assert lim.quality_limit(cfg, 'DEGRADED').speed == cfg.degraded_speed
    assert lim.quality_limit(cfg, 'LOST') == lim.Limit(0.0, 'QUALITY_LOW')
    assert lim.quality_limit(cfg, 'garbage').speed == 0.0


def test_lidar_limit_interface():
    cfg = drive_config('follower')
    assert lim.lidar_limit(cfg, None).speed == lim.NO_LIMIT
    assert lim.lidar_limit(cfg, cfg.lidar_stop_base - 0.01) == lim.Limit(0.0, 'LIDAR_STOP')
    slow = lim.lidar_limit(cfg, cfg.lidar_stop_base + 0.05)
    assert slow.reason == 'LIDAR_SLOW' and 0.0 < slow.speed < 0.2
    assert lim.lidar_limit(cfg, 3.0).speed > cfg.max_linear_speed


def test_stopping_speed_matches_stop_distance():
    v = lim.stopping_speed(0.5, base=0.2, decel=0.5, reaction_s=0.2)
    assert 0.2 + v * 0.2 + v * v / (2 * 0.5) == pytest.approx(0.5)


def test_ramp_respects_accel_decel_and_jerk():
    cfg = drive_config('follower')
    ramp, dt = lim.VelocityRamp(cfg), 0.05
    up = [ramp.step(0.2, 0.0, dt)[0] for _ in range(40)]
    accel = np.diff([0.0] + up) / dt
    assert accel.max() <= cfg.max_linear_accel + 1e-9
    assert np.diff(accel).max() <= cfg.max_linear_jerk * dt + 1e-9
    assert up[-1] == pytest.approx(0.2)
    down = [ramp.step(0.0, 0.0, dt)[0] for _ in range(20)]
    decel = np.diff([0.2] + down) / dt
    assert decel.min() >= -cfg.max_linear_decel - 1e-9
    assert decel[0] == pytest.approx(-cfg.max_linear_decel)
    assert down[-1] == 0.0


def test_ramp_limits_angular_accel_ties_omega():
    cfg = drive_config('follower')
    ramp, dt = lim.VelocityRamp(cfg), 0.05
    assert ramp.step(0.0, 5.0, dt) == (0.0, 0.0)
    ws = [ramp.step(0.2, 5.0, dt)[1] for _ in range(60)]
    assert np.diff([0.0] + ws).max() <= cfg.max_angular_accel * dt + 1e-9
    assert max(ws) <= cfg.max_angular_speed + 1e-9


def test_closed_loop_respects_limits():
    cfg = drive_config('follower')
    sim = Sim(cfg, make_cmd(s_path()), Rover())
    sim.run(60.0, until=latched)
    dt = 1.0 / cfg.control_rate_hz
    v = np.array([row[1] for row in sim.log])
    w = np.array([row[2] for row in sim.log])
    print(f'accel max {np.diff(v).max() / dt:.3f}, decel max {-np.diff(v).min() / dt:.3f} m/s^2, '
          f'alpha max {np.abs(np.diff(w)).max() / dt:.2f} rad/s^2')
    assert np.diff(v).max() / dt <= cfg.max_linear_accel + 1e-6
    assert np.diff(v).min() / dt >= -cfg.max_linear_decel - 1e-6
    assert np.abs(np.diff(w)).max() / dt <= cfg.max_angular_accel + 1e-6
    assert np.abs(w).max() <= cfg.max_angular_speed + 1e-9
    assert v.max() <= cfg.max_linear_speed + 1e-9


def test_omega_quantizer_steps_preserve_mean():
    quant = lim.OmegaQuantizer(0.1)
    out = [quant.apply(0.03) for _ in range(1000)]
    assert all(math.isclose(o / 0.1, round(o / 0.1), abs_tol=1e-9) for o in out)
    assert np.mean(out) == pytest.approx(0.03, abs=0.001)
    assert lim.OmegaQuantizer(0.0).apply(0.0314) == 0.0314


def test_omega_quantizer_no_dither_small_flip():
    """Cruising straight the request wanders by less than a step either side of 0; rounding sent
    +1 / -1 steps in turn. A step goes out only once a whole quantum is owed."""
    quant = lim.OmegaQuantizer(0.1)
    request = [0.06 if i % 2 == 0 else -0.06 for i in range(1000)]
    out = [quant.apply(w) for w in request]
    signs = [o > 0 for o in out if o != 0.0]
    assert sum(1 for a, b in zip(signs, signs[1:]) if a != b) == 0
    assert np.mean(out) == pytest.approx(np.mean(request), abs=0.001)


def test_leader_commands_multiples_of_quantum():
    cfg = drive_config('leader', omega_quantum=0.1, omega_quantize='round')
    sim = Sim(cfg, make_cmd(s_path()), Rover(omega_quantum=0.1))
    sim.run(20.0)
    assert all(math.isclose(row[2] / 0.1, round(row[2] / 0.1), abs_tol=1e-9) for row in sim.log)
