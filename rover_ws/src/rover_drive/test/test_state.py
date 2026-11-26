"""Command acceptance (epoch, cmd_id), state latching and HOLDING vs PARKED."""
import math

import pytest

from rover_drive.core import state as st
from rover_drive.core.controller import DriveController

from sim import Rover, Sim, drive_config, latched, make_cmd, straight_path


def _cmd(epoch, cmd_id):
    return st.PathCmd(epoch=epoch, cmd_id=cmd_id)


@pytest.mark.parametrize('current,incoming,expected', [
    (None, (100, 1), st.NEW),
    ((100, 5), (100, 5), st.KEEPALIVE),
    ((100, 5), (100, 6), st.NEW),
    ((100, 5), (100, 4), st.REJECT),
    ((100, 5), (101, 1), st.NEW),
    ((100, 5), (99, 9), st.REJECT),
])
def test_classify_command(current, incoming, expected):
    cur = _cmd(*current) if current else None
    assert st.classify_command(cur, _cmd(*incoming)) == expected


def test_idle_before_any_command():
    out = DriveController(drive_config('follower')).step(0.0)
    assert (out.state, out.v, out.w, out.cmd_id) == (st.IDLE, 0.0, 0.0, 0)


def test_keepalive_does_not_reload_the_path():
    sim = Sim(drive_config('follower'), make_cmd(straight_path(4.0), cmd_id=3), Rover())
    sim.run(5.0)
    progress = sim.out.progress
    # Same (epoch, cmd_id) with different content: only a keep-alive.
    sim.cmd = make_cmd(straight_path(1.0, y=2.0), cmd_id=3)
    sim.run(2.0)
    assert sim.out.state == st.TRACKING
    assert sim.out.progress > progress
    assert sim.out.cmd_id == 3


def test_keepalive_keeps_command_old_ids_dont():
    cfg = drive_config('follower')
    ctrl = DriveController(cfg)
    ctrl.set_heading_aligned(True)
    cmd = make_cmd(straight_path(4.0), cmd_id=3)
    assert ctrl.on_command(cmd, 0.0) == st.NEW
    ctrl.on_pose(0.0, 0.0, 0.0, 1.0)
    assert ctrl.on_command(cmd, 1.0) == st.KEEPALIVE
    assert ctrl.step(1.1).state == st.TRACKING
    ctrl.on_pose(0.0, 0.0, 0.0, 3.0)
    assert ctrl.on_command(make_cmd(straight_path(4.0), cmd_id=2), 3.0) == st.REJECT
    out = ctrl.step(3.0)
    assert (out.state, out.detail) == (st.STOPPED, 'STALE_CMD')


def test_higher_cmd_id_replaces_the_path():
    sim = Sim(drive_config('follower'), make_cmd(straight_path(4.0), cmd_id=1), Rover())
    sim.run(5.0)
    sim.cmd = make_cmd(straight_path(1.5), cmd_id=2)
    sim.run(30.0, until=latched)
    assert sim.out.state == st.PARKED and sim.out.cmd_id == 2
    assert sim.rover.tracked[0] == pytest.approx(1.5, abs=0.05)


def test_new_epoch_drops_path_despite_lower_id():
    sim = Sim(drive_config('follower'), make_cmd(straight_path(4.0), cmd_id=7, epoch=100),
              Rover())
    sim.run(4.0)
    sim.cmd = make_cmd((), cmd_id=1, epoch=200, mode='STOP')
    sim.run(2.0)
    assert (sim.out.state, sim.out.detail, sim.out.cmd_id) == (st.STOPPED, 'CMD_STOP', 1)
    sim.cmd = make_cmd(straight_path(4.0), cmd_id=9, epoch=100)
    sim.run(1.0)
    assert sim.out.cmd_id == 1 and sim.out.v == 0.0


@pytest.mark.parametrize('pick,expected', [(True, st.HOLDING), (False, st.PARKED)])
def test_holding_only_at_a_pick_goal(pick, expected):
    sim = Sim(drive_config('leader'),
              make_cmd(straight_path(1.0), goal_pos_tol=0.05, pick_at_goal=pick), Rover())
    sim.run(60.0, until=latched)
    assert sim.out.state == expected
    assert st.HOLDING not in sim.states() or pick
    assert sim.out.progress == pytest.approx(1.0, abs=0.02)


def test_arrival_latched_cleared_by_new_command():
    sim = Sim(drive_config('leader'), make_cmd(straight_path(1.0), pick_at_goal=True), Rover())
    sim.run(60.0, until=latched)
    sim.cmd_enabled = False
    sim.run(3.0)
    assert sim.out.state == st.HOLDING and sim.out.v == 0.0
    sim.cmd_enabled = True
    sim.cmd = make_cmd(straight_path(2.0), cmd_id=2)
    sim.run(2.0)
    assert sim.out.state == st.TRACKING


def test_empty_follow_path_is_an_error():
    ctrl = DriveController(drive_config('follower'))
    ctrl.on_command(st.PathCmd(epoch=1, cmd_id=1), 0.0)
    out = ctrl.step(0.0)
    assert (out.state, out.detail, out.v) == (st.ERROR, 'EMPTY_PATH', 0.0)


@pytest.mark.parametrize('fields,bad', [
    ({'speed_cap': math.nan}, True),
    ({'goal_pos_tol': math.inf}, True),
    ({'goal_yaw_tol': math.nan}, True),
    ({'pivot_first': True, 'pivot_heading': math.nan}, True),
    ({'pivot_first': False, 'pivot_heading': math.nan}, False),
    ({'lead_robot': 'leader', 'gap': math.nan}, True),
    ({'poses': ((0.0, 0.0, 0.0), (math.nan, 1.0, 0.0))}, True),
    ({'poses': ((0.0, 0.0, 0.0), (1.0, 0.0, math.inf))}, True),
])
def test_non_finite_command_is_an_error(fields, bad):
    fields = {'poses': tuple(straight_path(2.0)), **fields}
    ctrl = DriveController(drive_config('leader'))
    ctrl.set_heading_aligned(True)
    ctrl.on_command(st.PathCmd(epoch=1, cmd_id=1, **fields), 0.0)
    ctrl.on_pose(0.0, 0.0, 0.0, 0.0)
    out = ctrl.step(0.05)
    assert ((out.state, out.detail) == (st.ERROR, 'BAD_CMD')) == bad
    assert math.isfinite(out.v) and math.isfinite(out.w)


def test_drive_state_latching_rules():
    state = st.DriveState()
    state.track()
    state.stop('STALE_POSE')
    state.track()
    assert state.state == st.TRACKING
    state.stop('STUCK')
    state.track()
    assert (state.state, state.detail) == (st.STOPPED, 'STUCK')
    state.reset()
    state.fail('ALIGN_FAILED')
    state.arrive(True)
    assert state.state == st.ERROR
    state.reset()
    state.arrive(True)
    state.stop('STALE_CMD')
    assert state.state == st.HOLDING


def test_align_maneuver_distance_cap():
    align = st.AlignManeuver(0.6)
    assert align.update(False, 0.0, 0.0) == st.ALIGNING
    assert align.update(False, 0.5, 0.0) == st.ALIGNING
    assert align.update(True, 0.5, 0.0) == st.ALIGNED
    assert align.update(False, 1.0, 0.0) == st.ALIGNING
    assert align.update(False, 1.6, 0.0) == st.ALIGN_FAILED


def test_dataclass_defaults_match_follower_yaml():
    """config.py promises this; a node started without a params file must behave the same."""
    from rover_drive.core.config import DriveConfig, GateConfig
    from sim import gate_config
    assert drive_config('follower') == DriveConfig()
    assert gate_config('follower') == GateConfig()
