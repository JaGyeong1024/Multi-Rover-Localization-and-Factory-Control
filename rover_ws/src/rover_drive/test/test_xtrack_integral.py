"""Integral cross-track term: sign, clamp, distance basis, when it runs, when it resets."""
import pytest

from rover_drive.core.controller import DriveController
from rover_drive.core.tracker import CrossTrackIntegral

from sim import drive_config, make_cmd, straight_path

DT = 0.05
LEFT = 0.05


def _controller(gain=50.0, limit=1.5, aligned=True, **cmd_kw):
    # continuous omega: this is about the curvature term, not the base's angular steps
    cfg = drive_config('leader', xtrack_integral_gain=gain, xtrack_integral_limit=limit,
                       omega_quantum=0.0)
    ctrl = DriveController(cfg)
    ctrl.set_heading_aligned(aligned)
    ctrl.on_command(make_cmd(straight_path(3.0), **cmd_kw), 0.0)
    return ctrl


def _drive_in_place(ctrl, steps, y=LEFT, t0=0.0, x=0.5):
    """Feed a pose that does not move (an offset that persists) and step the controller."""
    out = None
    for i in range(1, steps + 1):
        t = t0 + i * DT
        ctrl.on_pose(x, y, 0.0, t)
        ctrl.on_command(ctrl._cmd, t)
        out = ctrl.step(t)
    return out


def _integral(ctrl):
    return ctrl._xtrack.curvature


def test_integral_pulls_back_toward_path():
    left, right = _controller(), _controller()
    _drive_in_place(left, 40, y=LEFT)
    _drive_in_place(right, 40, y=-LEFT)
    assert _integral(left) < 0.0 < _integral(right)
    assert _integral(left) == pytest.approx(-_integral(right))


def test_term_starts_zero_then_changes_command():
    with_term, without = _controller(50.0), _controller(0.0)
    first = _drive_in_place(with_term, 1)
    assert first.w == _drive_in_place(without, 1).w
    assert _drive_in_place(with_term, 40, t0=DT).w < _drive_in_place(without, 40, t0=DT).w


def test_zero_gain_adds_nothing():
    ctrl = _controller(gain=0.0)
    _drive_in_place(ctrl, 80)
    assert _integral(ctrl) == 0.0


def test_added_curvature_within_limit():
    ctrl = _controller(gain=50.0, limit=0.4)
    _drive_in_place(ctrl, 400, y=0.2)
    assert _integral(ctrl) == pytest.approx(-0.4)
    _drive_in_place(ctrl, 400, y=-0.2, t0=20.0)
    assert -0.4 <= _integral(ctrl) <= 0.4


def test_integral_by_distance_no_growth_standing():
    ctrl = _controller()
    ctrl.set_lidar_front(0.0)
    _drive_in_place(ctrl, 80)
    assert _integral(ctrl) == 0.0


def test_integrator_lateral_times_distance():
    integral = CrossTrackIntegral(drive_config('leader', xtrack_integral_gain=50.0))
    integral.update(0.02, -0.1, 0.5)
    assert integral.curvature == pytest.approx(-50.0 * 0.02 * 0.1 * 0.5)


def test_new_command_resets_keepalive_does_not():
    ctrl = _controller()
    _drive_in_place(ctrl, 40)
    before = _integral(ctrl)
    assert before < 0.0
    same = ctrl._cmd
    ctrl.on_command(same, 3.0)
    assert _integral(ctrl) == before
    ctrl.on_command(make_cmd(straight_path(3.0), cmd_id=2), 3.0)
    assert _integral(ctrl) == 0.0


def test_no_integration_during_pivot():
    ctrl = _controller(pivot_first=True, pivot_heading=1.5)
    out = _drive_in_place(ctrl, 10, x=0.0)
    assert out.v == 0.0 and out.w != 0.0
    assert _integral(ctrl) == 0.0


def test_no_integration_when_unaligned():
    ctrl = _controller(aligned=False)
    out = _drive_in_place(ctrl, 40)
    assert out.detail == 'ALIGNING' and out.v > 0.0
    assert _integral(ctrl) == 0.0


def test_no_integration_while_halted():
    ctrl = _controller()
    ctrl.set_gate('STOPPED', ('ESTOP',), 0.0, 0.0)
    for i in range(1, 41):
        ctrl.set_gate('STOPPED', ('ESTOP',), 0.0, i * DT)
        ctrl.on_pose(0.5, LEFT, 0.0, i * DT)
        ctrl.on_command(ctrl._cmd, i * DT)
        ctrl.step(i * DT)
    assert _integral(ctrl) == 0.0
