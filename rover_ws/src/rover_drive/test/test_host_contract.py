"""What the host (path_commander / arrival / supervisor) relies on from the rover."""
import math

import pytest

from rover_drive.core import state as st
from rover_drive.core.controller import DriveController
from rover_drive.core.safety import LIMITED, OK, STOPPED, SafetyGate

from sim import Rover, Sensor, Sim, drive_config, gate_config, latched, make_cmd, straight_path

EXACT = dict(delay=0.0, pos_noise=0.0, yaw_noise=0.0)


def _status(sim):
    return sim.out.state, sim.out.detail


def _standing(x, y, yaw, cmd, aligned=True, robot='follower'):
    """Controller of a rover standing at a pose that has just received ``cmd``."""
    ctrl = DriveController(drive_config(robot))
    ctrl.set_heading_aligned(aligned)
    ctrl.on_pose(x, y, yaw, 0.0)
    ctrl.on_command(cmd, 0.0)
    return ctrl


def _stand(ctrl, pose, t0=0.0, duration=0.5):
    """Keep feeding the same pose (30 Hz) and stepping (20 Hz); last output."""
    out, t = None, t0
    while t < t0 + duration:
        t += 0.05
        ctrl.on_pose(*pose, t)
        out = ctrl.step(t)
    return out


def test_feedback_carries_current_epoch():
    ctrl = DriveController(drive_config('follower'))
    assert ctrl.step(0.0).epoch == 0
    ctrl.on_command(make_cmd(straight_path(1.0), cmd_id=4, epoch=1700000000), 0.0)
    out = ctrl.step(0.05)
    assert (out.epoch, out.cmd_id) == (1700000000, 4)
    ctrl.on_command(make_cmd(straight_path(1.0), cmd_id=1, epoch=1700000500), 0.1)
    out = ctrl.step(0.1)
    assert (out.epoch, out.cmd_id) == (1700000500, 1)


def test_off_path_reported_stopped_with_reason():
    sim = Sim(drive_config('follower'), make_cmd(straight_path(3.0)), Rover(y=0.5))
    sim.run(2.0)
    assert _status(sim) == (st.STOPPED, 'OFF_PATH')
    assert st.ERROR not in sim.states()


def test_stuck_reported_stopped_with_reason():
    sim = Sim(drive_config('follower'), make_cmd(straight_path(3.0)), Rover(slip=0.0))
    sim.run(30.0, until=latched)
    assert _status(sim) == (st.STOPPED, 'STUCK')


def test_goal_missed_reported_stopped_reason():
    poses = straight_path(1.0)
    poses[-1] = (poses[-1][0], poses[-1][1], math.radians(60))
    sim = Sim(drive_config('follower'), make_cmd(poses, goal_yaw_tol=0.1), Rover())
    sim.run(60.0, until=latched)
    assert _status(sim) == (st.STOPPED, 'GOAL_MISSED')


def test_latched_stop_cleared_by_new_cmd_id():
    assert set(st.LATCHED_STOPS) == {'OFF_PATH', 'STUCK', 'GOAL_MISSED'}
    sim = Sim(drive_config('follower'), make_cmd(straight_path(3.0)), Rover(y=0.5))
    sim.run(3.0)
    assert _status(sim) == (st.STOPPED, 'OFF_PATH')
    sim.cmd = make_cmd(straight_path(3.0, y=0.5), cmd_id=2)
    sim.run(1.0)
    assert sim.out.state == st.TRACKING
    sim.run(60.0, until=latched)
    assert sim.out.state == st.PARKED


def test_new_cmd_id_projects_on_new_path():
    """The rover is 2 m along the old path; the new path passes it at its own s = 0.5."""
    sim = Sim(drive_config('follower'), make_cmd(straight_path(4.0)), Rover(), Sensor(**EXACT))
    sim.run(60.0, until=lambda s: s.rover.tracked[0] >= 2.0)
    x = sim.rover.tracked[0]
    new_path = [(x - 0.5 + px, py, yaw) for px, py, yaw in straight_path(2.0)]
    sim.cmd = make_cmd(new_path, cmd_id=2)
    sim.run(0.6)
    assert sim.out.cmd_id == 2 and sim.out.state == st.TRACKING
    assert sim.out.progress == pytest.approx(0.5 / 2.0, abs=0.08)
    sim.run(60.0, until=latched)
    assert sim.out.state == st.PARKED
    assert sim.goal_error()[0] < 0.05


def test_arrival_judged_on_settle_window():
    """One outlier sample (jump, white noise) at the moment of the verdict is not a miss."""
    goal = (1.0, 2.0, 0.0)
    ctrl = _standing(1.01, 2.0, 0.0, make_cmd([goal], goal_pos_tol=0.05, goal_yaw_tol=0.10))
    out, t = ctrl.step(0.05), 0.05
    while out.state == st.TRACKING:
        t += 0.05
        outlier = t + 0.05 >= ctrl._cfg.arrival_settle_s
        ctrl.on_pose(1.08 if outlier else 1.01, 2.0, 0.12 if outlier else 0.0, t)
        out = ctrl.step(t)
    assert out.state == st.PARKED


def test_reentry_to_parked_goal_is_driven():
    """The host re-enters (whole path, same goal) when its pose says the rover is off the goal;
    keeping the old verdict would answer PARKED at once without moving."""
    goal = (1.0, 2.0, 0.0)
    ctrl = _standing(1.03, 2.0, 0.0, make_cmd([goal], goal_pos_tol=0.05))
    assert _stand(ctrl, (1.03, 2.0, 0.0)).state == st.PARKED
    loop = [(1.03 + 0.05 * i, 2.0, 0.0) for i in range(20)] + [goal]
    ctrl.on_command(make_cmd(loop, cmd_id=2, goal_pos_tol=0.05), 0.6)
    ctrl.on_pose(1.03, 2.0, 0.0, 0.62)
    out = ctrl.step(0.65)
    assert (out.state, out.cmd_id) == (st.TRACKING, 2) and out.v > 0.0


@pytest.mark.parametrize('pick,expected', [(True, st.HOLDING), (False, st.PARKED)])
def test_one_point_path_in_tolerance_arrives(pick, expected):
    cmd = make_cmd([(1.0, 2.0, 0.0)], goal_pos_tol=0.05, goal_yaw_tol=0.15, pick_at_goal=pick)
    ctrl = _standing(1.03, 2.0, 0.05, cmd)
    first = ctrl.step(0.05)
    assert (first.state, first.detail, first.v, first.w) == (st.TRACKING, 'GOAL', 0.0, 0.0)
    out = _stand(ctrl, (1.03, 2.0, 0.05), 0.05)
    assert (out.state, out.v, out.w, out.progress) == (expected, 0.0, 0.0, 1.0)


@pytest.mark.parametrize('pose', [(1.20, 2.0, 0.0),
                                  (1.60, 2.0, 0.0),
                                  (1.00, 2.0, 0.5)])
def test_one_point_path_outside_is_goal_missed(pose):
    cmd = make_cmd([(1.0, 2.0, 0.0)], goal_pos_tol=0.05, goal_yaw_tol=0.15, pick_at_goal=True)
    ctrl = _standing(*pose, cmd)
    out = _stand(ctrl, pose)
    assert (out.state, out.detail, out.v, out.w) == (st.STOPPED, 'GOAL_MISSED', 0.0, 0.0)
    ctrl.on_pose(1.0, 2.0, 0.0, 0.6)
    assert ctrl.step(0.6).state == st.STOPPED


def test_flag_change_switches_parked_to_holding():
    """Host re-sends [goal] with pick_at_goal to a rover that already arrived."""
    poses = straight_path(1.0)
    sim = Sim(drive_config('leader'), make_cmd(poses, goal_pos_tol=0.05), Rover())
    sim.run(60.0, until=latched)
    assert sim.out.state == st.PARKED
    sim.cmd = make_cmd(poses[-1:], cmd_id=2, goal_pos_tol=0.05, pick_at_goal=True)
    sim.run(0.6)
    assert (sim.out.state, sim.out.cmd_id) == (st.HOLDING, 2)
    assert all(row[1] == 0.0 for row in sim.log if row[0] > sim.t - 0.5)


def test_unaligned_follow_creeps_until_aligned():
    """Tracker creeps straight and slowly; the gate lets exactly that through."""
    cfg = drive_config('follower')
    sim = Sim(cfg, make_cmd(straight_path(2.0)), Rover(), aligned=False)
    gate = SafetyGate(gate_config('follower', lidar_enabled=False))
    gate.set_aligned(False)
    passed = []
    for _ in range(int(3.0 / 0.005)):
        sim.tick()
        gate.on_pose(*sim.rover.tracked, sim.t)
        gate.on_cmd(sim.t)
        gate.on_input(sim.out.v, sim.out.w, sim.t)
        passed.append(gate.step(sim.t))
    moving = [g for g in passed if g.v > 0.0]
    assert moving and all(g.state == LIMITED and g.reasons == ('NOT_ALIGNED',) for g in moving)
    assert all(g.w == 0.0 and g.v <= cfg.align_speed + 1e-9 for g in moving)
    assert _status(sim) == (st.TRACKING, 'ALIGNING')
    assert 0.05 < sim.rover.tracked[0] < cfg.align_max_dist


def test_unaligned_follow_fails_at_distance_cap():
    sim = Sim(drive_config('follower'), make_cmd(straight_path(2.0)), Rover(), aligned=False)
    sim.run(60.0, until=latched)
    assert _status(sim) == (st.ERROR, 'ALIGN_FAILED')
    assert sim.rover.tracked[0] == pytest.approx(sim.cfg.align_max_dist, abs=0.05)


def test_holding_gate_passes_pick_while_republished():
    """Leader pick: the pick code drives the base through the mux for a long time."""
    sim = Sim(drive_config('leader'), make_cmd(straight_path(0.5), pick_at_goal=True), Rover())
    sim.run(60.0, until=latched)
    assert sim.out.state == st.HOLDING
    gate = SafetyGate(gate_config('leader', front_distance_topic=''))
    gate.set_aligned(True)
    t_end, next_cmd = sim.t + 20.0, sim.t
    while sim.t < t_end:
        sim.tick()
        assert sim.out.state == st.HOLDING and sim.out.v == 0.0
        gate.on_pose(*sim.rover.tracked, sim.t)
        if sim.t >= next_cmd:
            next_cmd += 0.5
            gate.on_cmd(sim.t)
        gate.on_tracker(sim.out.state, sim.out.detail, sim.t)
        gate.on_input(0.03, 0.0, sim.t)
        out = gate.step(sim.t)
        assert (out.v, out.state) == (0.03, OK)


def test_gate_blocks_pick_when_host_silent():
    gate = SafetyGate(gate_config('leader', front_distance_topic=''))
    gate.set_aligned(True)
    gate.on_cmd(0.0)
    for t in (0.5, 1.0, 1.4, 1.6):
        gate.on_pose(0.0, 0.0, 0.0, t)
        gate.on_input(0.03, 0.0, t)
    out = gate.step(1.6)
    assert (out.v, out.state, out.reasons) == (0.0, STOPPED, ('STALE_CMD',))


def test_flag_change_keeps_arrival_with_noise():
    """PARKED -> HOLDING re-send of the same goal is not judged again: the rover has not
    moved, and a noisy sample outside the tolerance must not become GOAL_MISSED."""
    goal = (1.0, 2.0, 0.0)
    parked = make_cmd([goal], goal_pos_tol=0.05, goal_yaw_tol=0.15, pick_at_goal=False)
    ctrl = _standing(1.03, 2.0, 0.0, parked)
    assert _stand(ctrl, (1.03, 2.0, 0.0)).state == st.PARKED
    ctrl.on_pose(1.07, 2.0, 0.0, 0.6)
    ctrl.on_command(make_cmd([goal], cmd_id=2, goal_pos_tol=0.05, goal_yaw_tol=0.15,
                             pick_at_goal=True), 0.6)
    assert ctrl.step(0.65).state == st.HOLDING
    other = make_cmd([(1.2, 2.0, 0.0)], cmd_id=3, goal_pos_tol=0.05, goal_yaw_tol=0.15)
    ctrl.on_command(other, 0.7)
    out = _stand(ctrl, (1.03, 2.0, 0.0), 0.7)
    assert (out.state, out.detail) == (st.STOPPED, 'GOAL_MISSED')
