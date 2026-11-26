"""Free gap to the lead rover from the follower LiDAR (lidar.lead_gap), the gate's hard minimum
(GAP_SLOW / GAP_MIN) and the tracker's choice between GAP_LIDAR and the relayed-pose GAP."""
import math

import numpy as np
import pytest

from rover_drive.core.controller import DriveController
from rover_drive.core.lidar import lead_gap, scan_points
from rover_drive.core.safety import LIMITED, OK, STOPPED, SafetyGate

from sim import drive_config, gate_config, make_cmd, straight_path

BEAMS = 1830
INC = 2.0 * math.pi / BEAMS
ANGLE_MIN = -math.pi
RANGE_MIN = 0.12
POSE = (2.0, 3.0, math.pi / 2.0)
LEAD_AHEAD = 0.45
REAR = 0.30
LEAD_REAR = 0.15
LIDAR_TO_FRONT = gate_config('follower').lidar_to_front


def _cfg(**overrides):
    # Scans are painted in the body frame (angle 0 = ahead), i.e. an unrotated LiDAR mount.
    overrides.setdefault('lidar_yaw_offset', 0.0)
    return gate_config('follower', **overrides)


def _scan(fill=0.0):
    return np.full(BEAMS, fill)


def _paint(scan, angle, dist, half_width_beams=30):
    centre = int(round((angle - ANGLE_MIN) / INC)) % BEAMS
    for i in range(centre - half_width_beams, centre + half_width_beams + 1):
        scan[i % BEAMS] = dist
    return scan


def _x_min(dist, half_width_beams):
    """Nearest forward coordinate of a constant-range patch centred straight ahead."""
    return dist * math.cos(half_width_beams * INC)


def _measure(scan, lead_xy=(LEAD_AHEAD, 0.0), **overrides):
    cfg = _cfg(**overrides)
    return lead_gap(scan_points(scan, ANGLE_MIN, INC, RANGE_MIN, cfg), cfg, lead_xy)


def test_free_gap_is_nearest_lead_point_minus_front():
    gap = _measure(_paint(_scan(4.0), 0.0, REAR))
    assert gap.points == 61
    assert gap.gap == pytest.approx(_x_min(REAR, 30) - LIDAR_TO_FRONT, abs=1e-6)


def test_unnamed_or_unseen_lead_not_measured():
    assert math.isnan(_measure(_paint(_scan(4.0), 0.0, REAR), lead_xy=None).gap)
    empty = _measure(_scan(0.0))
    assert math.isnan(empty.gap) and empty.points == 0
    elsewhere = _measure(_paint(_scan(4.0), 0.0, 1.5))
    assert math.isnan(elsewhere.gap) and elsewhere.points == 0


def test_occluded_lead_needs_enough_points():
    min_points = _cfg().lidar_lead_min_points
    few = _measure(_paint(_scan(0.0), 0.0, REAR, half_width_beams=(min_points - 2) // 2))
    assert math.isnan(few.gap) and 0 < few.points < min_points
    enough = _measure(_paint(_scan(0.0), 0.0, REAR, half_width_beams=min_points // 2))
    assert enough.points >= min_points
    assert enough.gap == pytest.approx(REAR - LIDAR_TO_FRONT, abs=0.01)


def test_lead_circle_points_only_shrink_gap():
    scan = _paint(_paint(_scan(4.0), 0.0, REAR), math.radians(15), 0.22, half_width_beams=6)
    gap = _measure(scan)
    assert gap.gap < REAR - LIDAR_TO_FRONT - 0.05
    assert gap.gap == pytest.approx(0.22 * math.cos(math.radians(15)) - LIDAR_TO_FRONT,
                                    abs=0.01)


def test_points_behind_sensor_ignored():
    gap = _measure(_paint(_scan(0.0), math.pi, 0.2), lead_xy=(0.05, 0.0))
    assert math.isnan(gap.gap)


def _rear(free_gap):
    """Scan of the lead's rear face (61 beams) whose nearest point is at ``free_gap``."""
    rng = (free_gap + LIDAR_TO_FRONT) / math.cos(30 * INC)
    return _paint(_scan(4.0), 0.0, rng)


def _gate(**overrides):
    overrides.setdefault('lidar_resume_s', 0.0)
    gate = SafetyGate(_cfg(**overrides))
    gate.set_aligned(True)
    return gate


def _tick(gate, t, free_gap, v_in=0.20, scan=None, lead_fresh=True, lead='leader'):
    """Every input fresh at t; the lead stands ``free_gap`` ahead of my front edge."""
    gate.on_input(v_in, 0.0, t)
    gate.on_pose(*POSE, t)
    gate.on_cmd(t, lead)
    if lead_fresh:
        ahead = free_gap + LIDAR_TO_FRONT + LEAD_REAR
        gate.on_lead_pose(POSE[0], POSE[1] + ahead, t)
    gate.on_scan(_rear(free_gap) if scan is None else scan, ANGLE_MIN, INC, RANGE_MIN, t)
    return gate.step(t)


def test_gate_reports_gap_excludes_lead():
    gate = _gate()
    out = _tick(gate, 0.0, 0.20)
    assert gate.lead_gap == pytest.approx(0.20, abs=1e-6)
    assert (out.v, out.state, out.reasons) == (0.20, OK, ())


def test_gate_slows_then_stops_at_hard_min():
    cfg = _cfg()
    slow = _tick(_gate(), 0.0, 0.12)
    assert (slow.state, slow.reasons) == (LIMITED, ('GAP_SLOW',))
    assert 0.0 < slow.v < 0.20
    gate = _gate()
    assert _tick(gate, 0.0, 0.12, v_in=0.05).state == OK
    stop = _tick(gate, 0.1, cfg.gap_hard_min - 0.005, v_in=0.05)
    assert (stop.v, stop.state, stop.reasons) == (0.0, STOPPED, ('GAP_MIN',))
    assert math.isnan(stop.blocked_x)


def test_gap_min_holds_for_the_resume_time():
    gate = _gate(lidar_resume_s=1.0)
    assert _tick(gate, 0.0, 0.05).reasons == ('GAP_MIN',)
    for t in (0.5, 1.0):
        assert _tick(gate, t, 0.30, v_in=0.05).reasons == ('GAP_MIN',)
    assert _tick(gate, 1.2, 0.30, v_in=0.05).state == OK


def test_lead_lost_inside_min_keeps_stop():
    """Closer than range_min the LiDAR returns 0.0: the lead must not read as 'gone'."""
    gate = _gate()
    assert _tick(gate, 0.0, 0.06, v_in=0.05).reasons == ('GAP_MIN',)
    lost = _tick(gate, 0.1, 0.03, v_in=0.05, scan=_scan(4.0))
    assert math.isnan(gate.lead_gap) and lost.reasons == ('GAP_MIN',)
    far = _gate()
    _tick(far, 0.0, 0.40, v_in=0.05)
    assert 'GAP_MIN' not in _tick(far, 0.1, 0.40, v_in=0.05, scan=_scan(4.0)).reasons


def test_stale_lead_position_is_obstacle_again():
    """Nothing is left out without a fresh lead position: the front stop takes over."""
    gate = _gate()
    _tick(gate, 0.0, 0.25)
    out = _tick(gate, 0.7, 0.06, lead_fresh=False)
    assert math.isnan(gate.lead_gap)
    assert out.reasons == ('LIDAR_STOP',)


def test_lead_beside_gets_no_gap_stop():
    """Side by side (lead 0.55 m to the right): its forward distance is ~0, so no gap and no
    GAP_MIN; its points are an ordinary obstacle outside my lane."""
    gate = _gate()
    t, side = 0.0, 0.55
    gate.on_input(0.20, 0.0, t)
    gate.on_pose(*POSE, t)
    gate.on_cmd(t, 'leader')
    gate.on_lead_pose(POSE[0] - side, POSE[1], t)
    out = None
    for t in (0.0, 0.1):
        gate.on_scan(_paint(_scan(4.0), math.pi / 2.0, side - 0.18, 60), ANGLE_MIN, INC,
                     RANGE_MIN, t)
        out = gate.step(t)
    assert math.isnan(gate.lead_gap)
    assert (out.v, out.state, out.reasons) == (0.20, OK, ())


def test_other_objects_stop_while_gap_measured():
    scan = _paint(_rear(0.25), 0.0, 0.15, half_width_beams=3)
    gate = _gate()
    out = _tick(gate, 0.0, 0.25, scan=scan)
    assert out.reasons == ('LIDAR_STOP',)
    assert gate.lead_gap == pytest.approx(0.25, abs=1e-6)


def test_new_lead_forgets_the_old_gap():
    gate = _gate()
    _tick(gate, 0.0, 0.06, v_in=0.05)
    gate.on_cmd(0.1, 'other')
    assert math.isnan(gate.lead_gap)
    out = _tick(gate, 0.1, 0.40, v_in=0.05, scan=_scan(4.0), lead_fresh=False, lead='')
    assert out.state == OK


PATH_LEAD = 0.65
DT = 0.05


class _Follower:
    """Controller on a straight path, moving by its own output; the lead keeps its distance."""

    def __init__(self):
        self.ctrl = DriveController(drive_config('follower'))
        self.ctrl.set_heading_aligned(True)
        self.cmd = make_cmd(straight_path(6.0), lead_robot='leader', gap=0.6)
        self.x = 0.5


def _ctrl():
    return _Follower()


def _run(rover, t0, t1, free_gap=None, gap_age=0.0, lead_age=0.0):
    """Drive at 20 Hz from t0 to t1; returns the last output. ``free_gap`` is the gate's
    /lead_gap (None = nothing received)."""
    out, ctrl = None, rover.ctrl
    for t in np.arange(t0, t1, DT):
        ctrl.on_command(rover.cmd, t)
        ctrl.on_pose(rover.x, 0.0, 0.0, t)
        ctrl.on_lead_pose(rover.x + PATH_LEAD, 0.0, t - lead_age)
        if free_gap is not None:
            ctrl.on_lead_gap(free_gap, t - gap_age)
        out = ctrl.step(t)
        rover.x += out.v * DT
    return out


def test_tracker_uses_fresh_lidar_free_gap():
    cfg = drive_config('follower')
    out = _run(_ctrl(), 0.0, 3.0, free_gap=0.25)
    assert out.detail == 'GAP_LIDAR'
    assert out.v == pytest.approx(cfg.gap_gain * (0.25 - cfg.gap_free_target), abs=1e-6)


def test_lidar_gap_overrides_path_gap_both_ways():
    cfg = drive_config('follower')
    assert _run(_ctrl(), 0.0, 3.0, free_gap=0.21).v == pytest.approx(cfg.gap_gain * 0.01)
    assert _run(_ctrl(), 0.0, 3.0, free_gap=0.19).v == 0.0
    out = _run(_ctrl(), 0.0, 3.0, free_gap=0.60)
    assert out.detail not in ('GAP', 'GAP_LIDAR') and out.v == pytest.approx(0.20)


@pytest.mark.parametrize('free_gap,gap_age', [(None, 0.0), (math.nan, 0.0), (0.25, 0.35)])
def test_tracker_falls_back_to_the_path_gap(free_gap, gap_age):
    """Nothing received, 'not seen' (NaN) or older than lead_gap_timeout_s: relayed pose."""
    cfg = drive_config('follower')
    out = _run(_ctrl(), 0.0, 3.0, free_gap=free_gap, gap_age=gap_age)
    assert out.detail == 'GAP'
    assert out.v == pytest.approx(cfg.gap_gain * (PATH_LEAD - 0.6), abs=1e-6)


def test_gap_source_switch_keeps_moving():
    ctrl = _ctrl()
    assert _run(ctrl, 0.0, 2.0, free_gap=0.25).detail == 'GAP_LIDAR'
    out = _run(ctrl, 2.0, 3.0, free_gap=math.nan)
    assert out.detail == 'GAP' and out.v > 0.0
    out = _run(ctrl, 3.0, 4.0, free_gap=0.25)
    assert out.detail == 'GAP_LIDAR' and out.v > 0.0


def test_stale_lead_stops_despite_lidar_gap():
    out = _run(_ctrl(), 0.0, 1.0, free_gap=0.40, lead_age=0.6)
    assert (out.v, out.detail) == (0.0, 'STALE_LEAD')
