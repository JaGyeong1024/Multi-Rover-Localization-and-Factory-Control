"""Safety gate: freshness, E-stop, alignment, LiDAR, clamps and angular_sign."""
import math

import numpy as np
import pytest

from rover_drive.core.safety import LIMITED, OK, STOPPED, SafetyGate, front_distance

from sim import gate_config


def _gate(robot='follower', aligned=True, **overrides):
    """A gate with every input fresh at t = 0 (front sensor off unless asked for)."""
    overrides.setdefault('lidar_enabled', False)
    overrides.setdefault('lidar_resume_s', 0.0)
    overrides.setdefault('front_distance_topic', '')
    gate = SafetyGate(gate_config(robot, **overrides))
    gate.on_pose(1.0, 2.0, 0.0, 0.0)
    gate.on_cmd(0.0)
    gate.set_aligned(aligned)
    gate.on_input(0.15, 0.3, 0.0)
    return gate


def test_passes_a_valid_command_unchanged():
    out = _gate().step(0.1)
    assert (out.v, out.w, out.state, out.reasons, out.speed_scale) == (0.15, 0.3, OK, (), 1.0)
    assert math.isnan(out.blocked_x) and math.isnan(out.blocked_y)


def test_nothing_received_means_stopped():
    out = SafetyGate(gate_config('follower')).step(0.0)
    assert (out.v, out.w, out.state) == (0.0, 0.0, STOPPED)
    assert {'INPUT_TIMEOUT', 'STALE_POSE', 'STALE_CMD'} <= set(out.reasons)


@pytest.mark.parametrize('refresh,t,reason', [
    (('pose', 'cmd'), 0.6, 'INPUT_TIMEOUT'),
    (('input', 'cmd'), 0.6, 'STALE_POSE'),
    (('input', 'pose'), 1.6, 'STALE_CMD'),
])
def test_each_stale_input_stops_and_recovers(refresh, t, reason):
    gate = _gate()
    feeds = {'input': lambda now: gate.on_input(0.15, 0.3, now),
             'pose': lambda now: gate.on_pose(1.0, 2.0, 0.0, now),
             'cmd': gate.on_cmd}
    for name in refresh:
        feeds[name](t)
    out = gate.step(t)
    assert (out.v, out.w, out.state, out.reasons) == (0.0, 0.0, STOPPED, (reason,))
    assert out.speed_scale == 0.0
    for feed in feeds.values():
        feed(t + 0.1)
    assert gate.step(t + 0.1).state == OK


def test_command_just_inside_timeouts_passes():
    gate = _gate()
    gate.on_input(0.15, 0.3, 1.4)
    gate.on_pose(1.0, 2.0, 0.0, 1.4)
    assert gate.step(1.45).state == OK


def test_estop_stops_until_released():
    gate = _gate()
    gate.set_estop(True)
    out = gate.step(0.1)
    assert (out.v, out.w, out.state, out.reasons) == (0.0, 0.0, STOPPED, ('ESTOP',))
    gate.set_estop(False)
    assert gate.step(0.2).v == 0.15


def test_unaligned_allows_only_slow_straight():
    gate = _gate(aligned=False)
    blocked = gate.step(0.1)
    assert (blocked.v, blocked.w, blocked.state) == (0.0, 0.0, STOPPED)
    assert 'NOT_ALIGNED' in blocked.reasons
    gate.on_input(0.05, 0.0, 0.1)
    creeping = gate.step(0.1)
    assert (creeping.v, creeping.w, creeping.state) == (0.05, 0.0, LIMITED)
    assert creeping.reasons == ('NOT_ALIGNED',)
    gate.on_input(0.05, 0.3, 0.1)
    assert gate.step(0.1).state == STOPPED
    gate.on_input(-0.02, 0.0, 0.1)
    assert gate.step(0.1).v == 0.0


def test_unknown_alignment_treated_unaligned():
    gate = SafetyGate(gate_config('follower'))
    gate.on_pose(0.0, 0.0, 0.0, 0.0)
    gate.on_cmd(0.0)
    gate.on_input(0.15, 0.0, 0.0)
    assert 'NOT_ALIGNED' in gate.step(0.0).reasons


@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_angular_sign_applies_to_output_only(sign):
    out = _gate(angular_sign=sign).step(0.1)
    assert out.v == 0.15
    assert out.w == pytest.approx(sign * 0.3)


def test_speed_clamp_preserves_the_arc():
    gate = _gate()
    gate.on_input(0.4, 0.8, 0.0)
    out = gate.step(0.1)
    assert out.v == pytest.approx(0.20) and out.w == pytest.approx(0.4)
    assert out.state == LIMITED and out.reasons == ('SPEED_CLAMP',)
    assert out.speed_scale == pytest.approx(0.5)
    gate.on_input(0.0, 5.0, 0.1)
    assert gate.step(0.1).w == pytest.approx(1.0)
    gate.on_input(-0.1, 0.0, 0.1)
    assert gate.step(0.1).v == 0.0


def test_leader_config_mux_topic_own_limits():
    cfg = gate_config('leader')
    assert cfg.input_topic == '/cmd_vel_mux' and cfg.output_topic == '/cmd_vel'
    gate = _gate('leader')
    out = gate.step(0.1)
    assert out.v == pytest.approx(0.10)
    gate.on_input(-0.05, 0.0, 0.1)
    assert gate.step(0.1).v == pytest.approx(-0.05)


def test_lidar_disabled_ignores_the_scan():
    gate = _gate()
    gate.on_front_distance(0.05, 0.0)
    assert gate.step(0.1).state == OK


def test_lidar_stop_slow_and_clear_zones():
    gate = _gate(lidar_enabled=True)
    gate.on_front_distance(3.0, 0.0)
    assert gate.step(0.1).state == OK
    gate.on_front_distance(0.25, 0.1)
    slow = gate.step(0.1)
    assert slow.state == LIMITED and slow.reasons == ('LIDAR_SLOW',)
    assert 0.0 < slow.v < 0.15
    assert slow.w / slow.v == pytest.approx(0.3 / 0.15)
    assert slow.speed_scale == pytest.approx(slow.v / 0.15)
    gate.on_front_distance(0.15, 0.1)
    stop = gate.step(0.1)
    assert (stop.v, stop.w, stop.state, stop.reasons) == (0.0, 0.0, STOPPED, ('LIDAR_STOP',))
    assert (stop.blocked_x, stop.blocked_y) == pytest.approx((1.15, 2.0))


def test_lidar_stop_reported_with_zero_input():
    gate = _gate(lidar_enabled=True)
    gate.on_input(0.0, 0.0, 0.0)
    gate.on_front_distance(0.1, 0.0)
    assert gate.step(0.1).reasons == ('LIDAR_STOP',)


def test_faster_input_limited_from_farther():
    gate = _gate(lidar_enabled=True)
    gate.on_front_distance(0.25, 0.0)
    gate.on_input(0.05, 0.0, 0.0)
    assert gate.step(0.1).state == OK
    gate.on_input(0.20, 0.0, 0.0)
    assert gate.step(0.1).reasons == ('LIDAR_SLOW',)


def test_enabled_lidar_missing_scan_stops():
    gate = _gate(lidar_enabled=True)
    assert 'STALE_SCAN' in gate.step(0.1).reasons
    gate.on_front_distance(3.0, 0.1)
    assert gate.step(0.2).state == OK
    gate.on_input(0.15, 0.3, 0.8)
    gate.on_pose(1.0, 2.0, 0.0, 0.8)
    assert gate.step(0.8).reasons == ('STALE_SCAN',)


def test_front_distance_forward_corridor_only():
    scan = np.full(360, 5.0)
    scan[0], scan[90], scan[180] = 0.8, 0.2, 0.1
    assert front_distance(scan, 0.0, math.radians(1.0), 0.15, 0.05) == pytest.approx(0.8)
    scan[5] = 0.5
    assert front_distance(scan, 0.0, math.radians(1.0), 0.15, 0.05) == pytest.approx(
        0.5 * math.cos(math.radians(5)))
    scan[0], scan[5] = float('inf'), 0.01
    assert front_distance(scan, 0.0, math.radians(1.0), 0.15, 0.05) > 1.0
    assert front_distance([], 0.0, 0.01, 0.15, 0.05) == math.inf


def test_tracker_reasons_relayed_report_only():
    gate = _gate()
    gate.on_input(0.0, 0.0, 0.0)
    gate.on_tracker('STOPPED', 'OFF_PATH', 0.0)
    out = gate.step(0.1)
    assert (out.state, out.reasons) == (STOPPED, ('OFF_PATH',))
    gate.on_tracker('TRACKING', 'QUALITY_LOW', 0.1)
    gate.on_input(0.05, 0.0, 0.1)
    out = gate.step(0.1)
    assert (out.v, out.state, out.reasons) == (0.05, LIMITED, ('QUALITY_LOW',))
    gate.on_tracker('TRACKING', 'GAP', 0.1)
    assert gate.step(0.1).state == OK


@pytest.mark.parametrize('robot', ['follower', 'leader'])
@pytest.mark.parametrize('v_in,w_in', [(math.nan, 0.0), (0.1, math.nan), (math.inf, 0.0),
                                       (0.05, -math.inf)])
def test_non_finite_input_zeroed_with_reason(robot, v_in, w_in):
    """NaN compares false, so it passed the clamps: v = inf, w = NaN reached the base."""
    gate = _gate(robot, lidar_enabled=(robot == 'follower'),
                 front_distance_topic=gate_config(robot).front_distance_topic)
    gate.on_front_distance(math.inf, 0.0)
    gate.on_input(v_in, w_in, 0.0)
    out = gate.step(0.1)
    assert (out.v, out.w, out.state) == (0.0, 0.0, STOPPED)
    assert 'BAD_INPUT' in out.reasons and out.speed_scale == 0.0


def test_non_finite_pose_not_fresh():
    gate = _gate()
    gate.on_input(0.15, 0.3, 0.6)
    gate.on_cmd(0.6)
    gate.on_pose(math.nan, 2.0, 0.0, 0.6)
    assert gate.step(0.6).reasons == ('STALE_POSE',)
