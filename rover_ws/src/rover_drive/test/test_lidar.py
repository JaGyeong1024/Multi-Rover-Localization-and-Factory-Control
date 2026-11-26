"""Follower LiDAR stop: scan processing and the gate's reaction (YDLIDAR G6 format)."""
import math

import numpy as np
import pytest

from rover_drive.core.lidar import in_sectors, scan_front
from rover_drive.core.safety import LIMITED, OK, STOPPED, SafetyGate

from sim import gate_config

BEAMS = 1830
INC = 2.0 * math.pi / BEAMS
ANGLE_MIN = -math.pi
RANGE_MIN = 0.12
POSE = (2.0, 3.0, math.pi / 2.0)


def _cfg(**overrides):
    # Scans are painted in the body frame (angle 0 = ahead), i.e. an unrotated LiDAR mount.
    overrides.setdefault('lidar_yaw_offset', 0.0)
    return gate_config('follower', **overrides)


def _scan(fill=0.0):
    """Empty hall: every beam is 0.0 (no return) unless filled."""
    return np.full(BEAMS, fill)


def _beam(angle):
    return int(round((angle - ANGLE_MIN) / INC)) % BEAMS


def _paint(scan, angle, dist, half_width_beams=6):
    """A small object: a few neighbouring beams at about the same range."""
    centre = _beam(angle)
    for i in range(centre - half_width_beams, centre + half_width_beams + 1):
        scan[i % BEAMS] = dist
    return scan


def _front(scan, cfg=None, lead_xy=None):
    return scan_front(scan, ANGLE_MIN, INC, RANGE_MIN, cfg or _cfg(), lead_xy)


def _gate(**overrides):
    overrides.setdefault('lidar_resume_s', 0.0)
    gate = SafetyGate(_cfg(**overrides))
    gate.on_pose(*POSE, 0.0)
    gate.on_cmd(0.0)
    gate.set_aligned(True)
    gate.on_input(0.20, 0.0, 0.0)
    return gate


def _step(gate, scan, t=0.1):
    gate.on_scan(scan, ANGLE_MIN, INC, RANGE_MIN, t)
    return gate.step(t)


def test_no_return_beams_are_unknown():
    for no_return in (0.0, math.inf, math.nan, 0.05):
        front = _front(_scan(no_return))
        assert math.isnan(front.dist) and front.valid_beams == 0


def test_nearest_forward_valid_beam_wins():
    scan = _paint(_paint(_scan(4.0), 0.0, 0.9), math.radians(5), 0.6)
    front = _front(scan)
    assert front.dist == pytest.approx(0.6 * math.cos(math.radians(5)), abs=0.01)
    assert (front.x, front.y) == pytest.approx(
        (0.6 * math.cos(math.radians(5)), 0.6 * math.sin(math.radians(5))), abs=0.02)


def test_side_and_rear_returns_do_not_count():
    scan = _scan(4.0)
    for angle in (math.radians(90), math.radians(-90), math.pi - 0.01, math.radians(50)):
        _paint(scan, angle, 0.2)
    assert _front(scan).dist > 3.5


def test_zero_beams_do_not_hide_object():
    scan = _paint(_scan(0.0), 0.0, 0.5, half_width_beams=10)
    scan[_beam(0.0)] = 0.0
    assert _front(scan).dist == pytest.approx(0.5, abs=0.01)


def test_corridor_ignores_wide_objects():
    scan = _paint(_scan(0.0), math.radians(25), 1.0, half_width_beams=8)
    assert _front(scan).dist == math.inf
    assert _front(scan, _cfg(lidar_half_width=0.0)).dist == pytest.approx(
        math.cos(math.radians(25)), abs=0.02)


def test_yaw_offset_rotates_scan_to_body():
    scan = _paint(_scan(4.0), math.radians(90), 0.5)
    assert _front(scan).dist > 3.5
    turned = _cfg(lidar_yaw_offset=-math.pi / 2.0)
    assert _front(scan, turned).dist == pytest.approx(0.5, abs=0.01)


def test_ignore_sectors_drop_own_structure():
    scan = _paint(_scan(4.0), math.radians(10), 0.15)
    assert _front(scan).dist < 0.2
    cfg = _cfg(lidar_ignore_sectors=(math.radians(5), math.radians(15)))
    assert _front(scan, cfg).dist > 3.5


def test_ignore_sector_crossing_rear_empty_pair():
    angle = np.array([3.0, -3.0, 0.0, 1.0])
    assert list(in_sectors(angle, (2.9, -2.9))) == [True, True, False, False]
    assert not in_sectors(angle, (0.0, 0.0)).any()
    assert list(in_sectors(angle, (0.5, 1.5, -0.1, 0.1))) == [False, False, True, True]


def test_gate_slows_stops_speed_dependent():
    gate = _gate()
    assert _step(gate, _paint(_scan(4.0), 0.0, 2.0)).state == OK
    slow = _step(gate, _paint(_scan(4.0), 0.0, 0.25))
    assert slow.state == LIMITED and slow.reasons == ('LIDAR_SLOW',) and 0.0 < slow.v < 0.20
    gate.on_input(0.05, 0.0, 0.1)
    assert _step(gate, _paint(_scan(4.0), 0.0, 0.25)).state == OK
    stop = _step(gate, _paint(_scan(4.0), 0.0, 0.19))
    assert (stop.v, stop.state, stop.reasons) == (0.0, STOPPED, ('LIDAR_STOP',))


def test_stop_distance_from_front_edge():
    """lidar_to_front + lidar_stop_base: the clearance is kept ahead of the bumper."""
    cfg = _cfg()
    edge = cfg.lidar_to_front + cfg.lidar_stop_base
    assert _step(_gate(), _paint(_scan(4.0), 0.0, edge + 0.03)).state == LIMITED
    assert _step(_gate(), _paint(_scan(4.0), 0.0, edge - 0.01)).state == STOPPED
    longer = _gate(lidar_to_front=0.20)
    assert _step(longer, _paint(_scan(4.0), 0.0, edge + 0.03)).state == STOPPED


def test_blocked_position_is_global_point():
    """Rover at (2, 3) facing +y; object 0.18 m ahead and 0.05 m to its left."""
    angle = math.atan2(0.05, 0.18)
    out = _step(_gate(), _paint(_scan(4.0), angle, math.hypot(0.18, 0.05), half_width_beams=0))
    assert out.reasons == ('LIDAR_STOP',)
    assert (out.blocked_x, out.blocked_y) == pytest.approx((2.0 - 0.05, 3.0 + 0.18), abs=0.01)


def test_blocked_position_unknown_no_heading():
    gate = _gate()
    gate.set_aligned(False)
    out = _step(gate, _paint(_scan(4.0), 0.0, 0.15))
    assert 'LIDAR_STOP' in out.reasons and math.isnan(out.blocked_x)


def test_blind_scan_is_limited_not_stopped():
    """Wide hall: most beams are 0.0. Default reaction is a speed cap."""
    gate = _gate()
    out = _step(gate, _scan(0.0))
    cap = _cfg().lidar_blind_speed
    assert (out.state, out.reasons) == (LIMITED, ('LIDAR_BLIND',))
    assert out.v == pytest.approx(cap) and out.speed_scale == pytest.approx(cap / 0.20)
    gate.on_input(0.05, 0.0, 0.1)
    slow = _step(gate, _scan(0.0))
    assert (slow.v, slow.state, slow.reasons) == (0.05, LIMITED, ('LIDAR_BLIND',))


def test_blind_can_be_configured_to_stop():
    out = _step(_gate(lidar_blind_speed=0.0), _scan(0.0))
    assert (out.v, out.state, out.reasons) == (0.0, STOPPED, ('LIDAR_BLIND',))


def test_a_few_valid_beams_are_still_blind():
    few = _paint(_scan(0.0), 0.0, 2.0, half_width_beams=1)
    assert _step(_gate(), few).reasons == ('LIDAR_BLIND',)
    enough = _paint(_scan(0.0), 0.0, 2.0, half_width_beams=3)
    assert _step(_gate(), enough).state == OK


def test_vanished_close_object_stops():
    """Closer than range_min the G6 returns 0.0: that must not read as 'gone'."""
    gate = _gate()
    gate.on_input(0.05, 0.0, 0.0)
    assert _step(gate, _paint(_scan(0.0), 0.0, 0.30, half_width_beams=20)).state == OK
    out = _step(gate, _scan(0.0), 0.2)
    assert (out.v, out.state, out.reasons) == (0.0, STOPPED, ('LIDAR_STOP',))
    assert (out.blocked_x, out.blocked_y) == pytest.approx((2.0, 3.30), abs=0.03)
    assert _step(gate, _paint(_scan(0.0), 0.0, 2.0), 0.3).state == OK


def test_vanished_far_object_only_blind():
    gate = _gate()
    _step(gate, _paint(_scan(0.0), 0.0, 1.5))
    assert _step(gate, _scan(0.0), 0.2).reasons == ('LIDAR_BLIND',)


def test_stale_scan_stops():
    gate = _gate()
    assert 'STALE_SCAN' in gate.step(0.1).reasons
    _step(gate, _paint(_scan(4.0), 0.0, 2.0), 0.1)
    gate.on_input(0.2, 0.0, 0.7)
    gate.on_pose(*POSE, 0.7)
    assert gate.step(0.7).reasons == ('STALE_SCAN',)


def test_resume_needs_front_clear_for_a_while():
    """An object at the corridor edge flickers in and out of view: no stop/go."""
    gate = _gate(lidar_resume_s=1.0)
    clear, blocked = _paint(_scan(4.0), 0.0, 2.0), _paint(_scan(4.0), 0.0, 0.15)
    t = 0.1
    assert _step(gate, blocked, t).reasons == ('LIDAR_STOP',)
    while t < 1.0:
        t += 0.1
        gate.on_input(0.20, 0.0, t)
        gate.on_pose(*POSE, t)
        gate.on_cmd(t)
        out = _step(gate, clear, t)
        assert (out.v, out.reasons) == (0.0, ('LIDAR_STOP',))
        assert out.blocked_y == pytest.approx(3.15, abs=0.01)
    for t in (1.15, 1.25):
        gate.on_input(0.20, 0.0, t)
        gate.on_pose(*POSE, t)
        out = _step(gate, clear, t)
    assert (out.v, out.state) == (0.20, OK)


def _convoy_gate(lead_ahead=0.5, **overrides):
    gate = _gate(**overrides)
    gate.on_cmd(0.0, 'leader')
    gate.on_lead_pose(2.0, 3.0 + lead_ahead, 0.0)
    return gate


def test_lead_not_obstacle_in_gap_control():
    lead_scan = _paint(_scan(4.0), 0.0, 0.26, half_width_beams=40)
    assert _step(_gate(), lead_scan).state == LIMITED
    assert _step(_convoy_gate(), lead_scan).state == OK


def test_other_objects_stop_while_following():
    scan = _paint(_paint(_scan(4.0), 0.0, 0.40, half_width_beams=40), 0.0, 0.15)
    out = _step(_convoy_gate(), scan)
    assert out.reasons == ('LIDAR_STOP',)
    assert out.blocked_y == pytest.approx(3.15, abs=0.01)


@pytest.mark.parametrize('spoil', ['stale_lead', 'unaligned', 'disabled', 'cleared'])
def test_lead_excluded_only_if_position_trusted(spoil):
    rear_face = _paint(_scan(4.0), 0.0, 0.19, half_width_beams=40)
    # control: trusted lead is left out of the front stop (only the gap minimum slows)
    assert _step(_convoy_gate(0.4), rear_face).reasons == ('GAP_SLOW',)
    gate = _convoy_gate(0.4, lidar_exclude_lead=spoil != 'disabled')
    t = 0.1
    if spoil == 'stale_lead':
        t = 0.7
        gate.on_input(0.20, 0.0, t)
        gate.on_pose(*POSE, t)
    elif spoil == 'unaligned':
        gate.set_aligned(False)
        gate.on_input(0.05, 0.0, t)
    elif spoil == 'cleared':
        gate.on_cmd(t, '')
    assert 'LIDAR_STOP' in _step(gate, rear_face, t).reasons


def test_follower_yaml_enables_the_lidar():
    cfg = gate_config('follower')
    assert cfg.lidar_enabled and cfg.front_distance_topic == '' and cfg.scan_topic == '/scan'
    assert cfg.lidar_yaw_offset == -0.5236 and cfg.angular_sign == 1.0
