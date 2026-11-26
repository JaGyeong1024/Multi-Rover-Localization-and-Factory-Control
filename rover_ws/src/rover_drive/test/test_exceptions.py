"""Exception handling in the tracker: stop when unsure, resume by itself."""
import math

import pytest

from rover_drive.core import state as st
from rover_drive.core.controller import DriveController
from rover_drive.core.monitor import PoseJumpFilter, StuckMonitor, judge_arrival

from sim import Rover, Sensor, Sim, drive_config, latched, make_cmd, straight_path


def _sim(**kw):
    cfg = drive_config('follower')
    cmd = make_cmd(straight_path(4.0), goal_pos_tol=0.05, **kw.pop('cmd', {}))
    return Sim(cfg, cmd, kw.pop('rover', Rover()), **kw)


def _status(sim):
    return sim.out.state, sim.out.detail


def test_stale_pose_stops_then_resumes():
    sim = _sim()
    sim.run(4.0)
    assert sim.out.v > 0.15
    sim.pose_enabled = False
    sim.run(sim.cfg.pose_timeout_s + 0.2)
    assert sim.out.v == 0.0 and _status(sim) == (st.STOPPED, 'STALE_POSE')
    sim.pose_enabled = True
    sim.run(2.0)
    assert sim.out.state == st.TRACKING and sim.out.v > 0.1
    sim.run(60.0, until=latched)
    assert sim.out.state == st.PARKED


def test_no_pose_at_all_never_moves():
    sim = _sim()
    sim.pose_enabled = False
    sim.run(3.0)
    assert max(row[1] for row in sim.log) == 0.0
    assert _status(sim) == (st.STOPPED, 'STALE_POSE')


def test_stale_command_stops_then_resumes():
    sim = _sim()
    sim.run(4.0)
    sim.cmd_enabled = False
    sim.run(1.0)
    assert sim.out.v > 0.15
    sim.run(1.0)
    assert sim.out.v == 0.0 and _status(sim) == (st.STOPPED, 'STALE_CMD')
    sim.cmd_enabled = True
    sim.run(2.0)
    assert sim.out.state == st.TRACKING and sim.out.v > 0.1


def test_stop_mode_ramps_down_and_reports():
    sim = _sim()
    sim.run(4.0)
    sim.cmd = make_cmd((), cmd_id=2, mode='STOP')
    sim.ctrl.on_command(sim.cmd, sim.t)
    sim.run(0.2)
    assert 0.0 < sim.out.v < 0.2
    sim.run(1.0)
    assert sim.out.v == 0.0 and _status(sim) == (st.STOPPED, 'CMD_STOP')


def test_single_jump_sample_is_discarded():
    sim = _sim()
    sim.run(4.0)
    jumped = []

    def hook(t, pose):
        if not jumped:
            jumped.append(t)
            return pose[0] + 1.0, pose[1] + 1.0, pose[2]
        return pose

    sim.pose_hook = hook
    sim.run(3.0)
    assert jumped
    assert sim.out.state == st.TRACKING
    assert sim.max_lateral() < 0.03


def test_sustained_jump_stops_then_adopted():
    """Localization really moved 1 m sideways: stop (POSE_JUMP), re-acquire, then OFF_PATH."""
    sim = _sim()
    sim.run(4.0)
    sim.pose_hook = lambda t, pose: (pose[0], pose[1] + 1.0, pose[2])
    seen = set()
    sim.run(3.0, until=lambda s: seen.add(_status(s)) or latched(s))
    assert (st.STOPPED, 'POSE_JUMP') in seen
    assert _status(sim) == (st.STOPPED, 'OFF_PATH')
    assert sim.out.v == 0.0


@pytest.mark.parametrize('robot', ['follower', 'leader'])
@pytest.mark.parametrize('bad', [(math.nan, 0.0, 0.0), (0.0, math.inf, 0.0), (0.0, 0.0, math.nan)])
def test_non_finite_pose_sample_is_discarded(robot, bad):
    """The jump filter took a NaN first sample (progress NaN: no OFF_PATH, no goal stop) and
    never looked at yaw (NaN broke the leader's omega quantizer)."""
    ctrl = DriveController(drive_config(robot))
    ctrl.set_heading_aligned(True)
    ctrl.on_command(make_cmd(straight_path(2.0)), 0.0)
    assert not ctrl.on_pose(*bad, 0.0)
    out = ctrl.step(0.05)
    assert (out.v, out.w, out.state, out.detail) == (0.0, 0.0, st.STOPPED, 'STALE_POSE')
    assert ctrl.on_pose(0.0, 0.0, 0.0, 0.1)
    assert not ctrl.on_pose(*bad, 0.6)
    assert ctrl.step(0.7).detail == 'STALE_POSE'


def test_jump_filter_allows_reachable_motion():
    flt = PoseJumpFilter(drive_config('follower'))
    assert flt.accept(0.0, 0.0, 0.0)
    assert flt.accept(0.02, 0.0, 0.033)
    assert not flt.accept(0.8, 0.0, 0.066) and flt.rejecting
    assert flt.accept(0.03, 0.0, 0.1) and not flt.rejecting


def test_off_path_at_start_never_moves():
    sim = _sim(rover=Rover(x=1.0, y=0.4))
    sim.run(2.0)
    assert max(row[1] for row in sim.log) == 0.0
    assert _status(sim) == (st.STOPPED, 'OFF_PATH')


def test_off_path_latched_until_new_command():
    sim = _sim(rover=Rover(x=1.0, y=0.4))
    sim.run(1.0)
    sim.rover.cy = 0.0
    sim.run(1.0)
    assert _status(sim) == (st.STOPPED, 'OFF_PATH')
    sim.cmd = make_cmd(straight_path(4.0), cmd_id=2)
    sim.run(2.0)
    assert sim.out.state == st.TRACKING and sim.out.v > 0.1


def test_stuck_when_commands_give_no_motion():
    sim = _sim(rover=Rover(slip=0.0))
    sim.run(10.0, until=latched)
    assert _status(sim) == (st.STOPPED, 'STUCK')
    assert sim.out.v == 0.0
    assert sim.t < 4.0


def test_partial_slip_is_not_stuck():
    sim = _sim(rover=Rover(slip=0.6))
    sim.run(90.0, until=latched)
    assert sim.out.state == st.PARKED


def test_stuck_monitor_is_distance_based():
    mon = StuckMonitor(drive_config('follower'))
    assert not any(mon.update(0.2, 0.0, 0.0, 0.05) for _ in range(19))
    assert mon.update(0.2, 0.0, 0.0, 0.05)
    assert not any(mon.update(0.2, 0.01 * i, 0.0, 0.05) for i in range(40))


def test_gate_estop_tracker_waits():
    sim = _sim()
    sim.run(3.0)
    for _ in range(int(5.0 / 0.005)):
        sim.ctrl.set_gate('STOPPED', ['ESTOP'], 0.0, sim.t)
        sim.rover.command(0.0, 0.0)
        sim.tick()
    assert sim.out.v == 0.0 and _status(sim) == (st.STOPPED, 'ESTOP')
    sim.ctrl.set_gate('OK', [], 1.0, sim.t)
    sim.run(0.1)
    assert 0.0 < sim.out.v < 0.1
    sim.run(60.0, until=latched)
    assert sim.out.state == st.PARKED


def test_degraded_slows_lost_stops():
    sim = _sim()
    sim.run(3.0)
    sim.ctrl.set_quality('DEGRADED')
    sim.run(2.0)
    assert sim.out.v == pytest.approx(sim.cfg.degraded_speed)
    assert _status(sim) == (st.TRACKING, 'QUALITY_LOW')
    sim.ctrl.set_quality('LOST')
    sim.run(1.0)
    assert sim.out.v == 0.0 and _status(sim) == (st.STOPPED, 'QUALITY_LOW')
    sim.ctrl.set_quality('GOOD')
    sim.run(2.0)
    assert sim.out.state == st.TRACKING and sim.out.v > 0.15


def test_unaligned_creeps_then_tracks():
    sim = _sim(aligned=False)
    sim.run(4.0)
    moving = [row for row in sim.log if row[0] > 1.0]
    assert all(row[2] == 0.0 for row in moving)
    assert all(row[1] <= sim.cfg.align_speed + 1e-9 for row in moving)
    assert _status(sim) == (st.TRACKING, 'ALIGNING')
    sim.ctrl.set_heading_aligned(True)
    sim.run(60.0, until=latched)
    assert sim.out.state == st.PARKED


def test_no_heading_creep_not_off_path():
    """Align from the home spot: the path was planned from an assumed heading and the rover faces
    the other way. OFF_PATH during its creep must not stop it."""
    sim = _sim(aligned=False, rover=Rover(yaw=math.pi / 2))
    sim.run(4.0)
    assert sim.rover.tracked[1] > sim.cfg.off_path_max
    assert _status(sim) == (st.TRACKING, 'ALIGNING')
    sim.ctrl.set_heading_aligned(True)
    sim.run(0.5)
    assert _status(sim) == (st.STOPPED, 'OFF_PATH')


def test_alignment_failure_at_distance_cap():
    sim = _sim(aligned=False)
    sim.run(60.0, until=latched)
    assert _status(sim) == (st.ERROR, 'ALIGN_FAILED')
    assert sim.rover.tracked[0] == pytest.approx(sim.cfg.align_max_dist, abs=0.05)
    sim.ctrl.set_heading_aligned(True)
    sim.run(1.0)
    assert sim.out.state == st.ERROR and sim.out.v == 0.0


def test_judge_arrival_tolerances():
    goal = (1.0, 1.0, 0.0)
    assert judge_arrival((1.03, 1.0, 0.5), goal, 0.05, 0.0, True).ok
    assert not judge_arrival((1.06, 1.0, 0.0), goal, 0.05, 0.0, True).ok
    assert judge_arrival((1.0, 1.0, 0.1), goal, 0.05, 0.15, True).ok
    assert not judge_arrival((1.0, 1.0, 0.2), goal, 0.05, 0.15, True).ok
    assert not judge_arrival((1.0, 1.0, 0.0), goal, 0.05, 0.15, False).ok
    assert judge_arrival((1.0, 1.0, math.pi - 0.05), (1.0, 1.0, -math.pi + 0.05),
                         0.05, 0.15, True).ok


def test_goal_missed_outside_tolerance():
    """Stops at the end of the path, but the heading is wrong: no arrival is declared."""
    poses = straight_path(2.0)
    poses[-1] = (poses[-1][0], poses[-1][1], math.radians(60))
    sim = Sim(drive_config('follower'),
              make_cmd(poses, goal_pos_tol=0.05, goal_yaw_tol=0.1, pick_at_goal=True), Rover())
    sim.run(60.0, until=latched)
    assert _status(sim) == (st.STOPPED, 'GOAL_MISSED')
    assert sim.goal_error()[0] < 0.05
    assert st.HOLDING not in sim.states()


def test_noisy_pose_keeps_run_exception_free():
    sim = _sim(sensor=Sensor(delay=0.10, pos_noise=0.01))
    sim.run(60.0, until=latched)
    assert sim.out.state == st.PARKED
    after_first_pose = {row[3] for row in sim.log if row[0] > 0.5}
    assert after_first_pose <= {st.TRACKING, st.PARKED}
