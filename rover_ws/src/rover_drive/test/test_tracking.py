"""Closed-loop tracking on an S path with 30 Hz delayed, noisy pose input."""
import math

import pytest

from rover_drive.core import state as st
from rover_drive.core.limits import NO_LIMIT, heading_limit
from rover_drive.core.tracker import lookahead_distance, steer
from rover_drive.core.path import Path

from sim import Rover, Sensor, Sim, drive_config, latched, make_cmd, s_path, straight_path

MAX_LATERAL = 0.05
GOAL_POS_TOL = 0.05
GOAL_YAW_TOL = 0.15

CASES = [
    # robot, speed_cap, pose delay, plant omega quantum, ref offset
    ('follower', 0.0, 0.05, 0.0, 0.0),
    ('follower', 0.0, 0.10, 0.0, 0.0),
    ('follower', 0.0, 0.10, 0.0, 0.05),
    ('leader', 0.0, 0.05, 0.0, 0.0),
    ('leader', 0.0, 0.10, 0.0, 0.0),
    ('leader', 0.08, 0.10, 0.0, 0.0),
    ('leader', 0.0, 0.10, 0.1, 0.0),
]


def _run_s_path(robot, cap, delay, quantum, offset, seed=0):
    cfg = drive_config(robot, omega_quantum=quantum, ref_offset_x=offset)
    cmd = make_cmd(s_path(), speed_cap=cap, goal_pos_tol=GOAL_POS_TOL,
                   goal_yaw_tol=GOAL_YAW_TOL, pick_at_goal=True)
    sim = Sim(cfg, cmd, Rover(ref_offset=offset, omega_quantum=quantum),
              Sensor(delay=delay, seed=seed))
    sim.run(120.0, until=latched)
    return sim


@pytest.mark.parametrize('robot,cap,delay,quantum,offset', CASES)
def test_s_path_is_followed_and_goal_reached(robot, cap, delay, quantum, offset):
    sim = _run_s_path(robot, cap, delay, quantum, offset)
    pos_err, yaw_err = sim.goal_error()
    print(f'{robot} cap={cap} delay={delay} q={quantum} off={offset}: '
          f'lateral max {sim.max_lateral():.3f} m, goal {pos_err:.3f} m / {yaw_err:.3f} rad, '
          f't={sim.t:.1f} s')
    assert sim.out.state == st.HOLDING
    assert sim.max_lateral() < MAX_LATERAL
    assert pos_err < GOAL_POS_TOL
    assert yaw_err < GOAL_YAW_TOL
    v_max = cap or sim.cfg.max_linear_speed
    assert max(row[1] for row in sim.log) <= v_max + 1e-9


@pytest.mark.parametrize('seed', range(5))
def test_tracking_stable_across_noise(seed):
    sim = _run_s_path('follower', 0.0, 0.10, 0.0, 0.0, seed=seed)
    assert sim.out.state == st.HOLDING
    assert sim.max_lateral() < MAX_LATERAL


@pytest.mark.parametrize('seed', range(3))
def test_follower_meets_heading_after_tight_turn(seed):
    """Minimum-radius turn (0.30 m) into the 0.40 m run-in at full speed: final heading error
    stays below the 5.7 deg tolerance."""
    cmd = make_cmd(s_path(radius=0.3, lead_in=0.4), goal_pos_tol=0.05, goal_yaw_tol=0.10)
    sim = Sim(drive_config('follower'), cmd, Rover(),
              Sensor(delay=0.08, pos_noise=0.003, yaw_noise=0.005, seed=seed))
    sim.run(60.0, until=latched)
    pos_err, yaw_err = sim.goal_error()
    assert sim.out.state == st.PARKED
    assert pos_err < 0.02 and yaw_err < math.radians(2.5)


def _pivots(sim):
    """Rows with omega but no forward motion (rotation in place)."""
    return [row for row in sim.log if row[1] < 1e-9 and abs(row[2]) > 1e-9]


def test_no_spin_while_tracking_or_at_goal():
    """Without a leading pivot omega only comes with forward motion, also when the last two path
    poses share a position with another heading."""
    sim = _run_s_path('follower', 0.0, 0.08, 0.0, 0.0)
    assert not _pivots(sim)
    poses = straight_path(1.0)
    turn_at_goal = poses + [(poses[-1][0], poses[-1][1], math.pi / 2)]
    sim = Sim(drive_config('leader'), make_cmd(turn_at_goal, goal_pos_tol=GOAL_POS_TOL),
              Rover(), Sensor(delay=0.08, seed=1))
    sim.run(40.0, until=latched)
    assert sim.out.state == st.PARKED and not _pivots(sim)


@pytest.mark.parametrize('robot,quantum', [('follower', 0.0), ('leader', 0.0697)])
def test_leading_pivot_turns_then_tracks(robot, quantum):
    """pivot_first: rotate in place (v = 0) to pivot_heading, then track the path without further
    rotation in place. The leader base truncates omega to 0.0697 rad/s steps."""
    rest = [(0.0, 0.05 * i, math.pi / 2) for i in range(0, 21)]
    sim = Sim(drive_config(robot), make_cmd(rest, goal_pos_tol=GOAL_POS_TOL,
                                            goal_yaw_tol=0.10, pivot_first=True,
                                            pivot_heading=math.pi / 2),
              Rover(omega_quantum=quantum, omega_truncate=quantum > 0.0),
              Sensor(delay=0.08, seed=2))
    sim.run(60.0, until=latched)
    pivots = _pivots(sim)
    assert pivots and pivots[0][4] == 'PIVOT'
    moving = next(row[0] for row in sim.log if row[1] > 1e-9)
    assert all(row[0] < moving for row in pivots)
    assert sim.out.state == st.PARKED
    pos_err, yaw_err = sim.goal_error()
    assert pos_err < GOAL_POS_TOL and yaw_err < 0.10


@pytest.mark.parametrize('robot,quantum', [('follower', 0.0), ('leader', 0.0697)])
@pytest.mark.parametrize('seed', [1, 2, 3])
def test_half_turn_pivot_keeps_direction(robot, quantum, seed):
    """Back to the home pose after the alignment creep (facing away): the wrapped heading error
    is +-pi under pose noise; the turn must not follow its sign."""
    rest = [(-0.05 * i, 0.0, math.pi) for i in range(0, 11)]
    sim = Sim(drive_config(robot), make_cmd(rest, goal_pos_tol=GOAL_POS_TOL,
                                            goal_yaw_tol=0.10, pivot_first=True,
                                            pivot_heading=math.pi),
              Rover(omega_quantum=quantum, omega_truncate=quantum > 0.0),
              Sensor(delay=0.08, seed=seed))
    sim.run(90.0, until=latched)
    rates = [row[2] for row in _pivots(sim) if abs(row[2]) > 1e-9]
    assert rates and (min(rates) > 0.0 or max(rates) < 0.0)
    assert sim.out.state == st.PARKED
    pos_err, yaw_err = sim.goal_error()
    assert pos_err < GOAL_POS_TOL and yaw_err < 0.10


@pytest.mark.parametrize('robot,quantum', [('follower', 0.0), ('leader', 0.0697)])
def test_pivot_on_zero_length_path_parks(robot, quantum):
    """Re-entry of a goal missed in heading only: the goal alone with a leading pivot. The rover
    turns in place, never drives, and is judged PARKED inside goal_yaw_tol."""
    cmd = make_cmd([(0.0, 0.0, 0.0)], goal_pos_tol=GOAL_POS_TOL, goal_yaw_tol=0.10,
                   pivot_first=True, pivot_heading=0.0)
    sim = Sim(drive_config(robot), cmd,
              Rover(x=0.02, y=-0.015, yaw=0.148, omega_quantum=quantum,
                    omega_truncate=quantum > 0.0),
              Sensor(delay=0.08, seed=5))
    sim.run(30.0, until=latched)
    assert _pivots(sim) and all(row[1] < 1e-9 for row in sim.log)
    assert sim.out.state == st.PARKED
    assert abs(sim.goal_error()[1]) < 0.10


def test_pivot_skipped_away_from_position():
    """A path re-sent after the rover drove on must not make it turn back: the pivot runs only
    when standing at its position."""
    rest = [(0.0, 0.05 * i, math.pi / 2) for i in range(0, 21)]
    sim = Sim(drive_config('follower'), make_cmd(rest, goal_pos_tol=GOAL_POS_TOL,
                                                 pivot_first=True,
                                                 pivot_heading=math.pi / 2),
              Rover(y=0.3, yaw=math.pi / 2), Sensor(delay=0.08, seed=3))
    sim.run(40.0, until=latched)
    assert sim.out.state == st.PARKED and not _pivots(sim)


def _pivot_sim(robot, heading, rover, delay=0.08, seed=4):
    """Stand at the origin facing +x; pivot to ``heading``, then 0.5 m along it."""
    rest = [(0.05 * i * math.cos(heading), 0.05 * i * math.sin(heading), heading)
            for i in range(11)]
    cmd = make_cmd(rest, goal_pos_tol=GOAL_POS_TOL, pivot_first=True, pivot_heading=heading)
    return Sim(drive_config(robot), cmd, rover, Sensor(delay=delay, seed=seed))


@pytest.mark.parametrize('robot', ['follower', 'leader'])
@pytest.mark.parametrize('fault', ['tracks_do_not_turn', 'reversed_sign', 'heading_frozen'])
def test_pivot_without_heading_change_is_stuck(robot, fault):
    """A pivot must end on its own (IMU lost, or hunting with a reversed angular sign), not only
    after pivot_timeout_s (30 s) of blind spinning."""
    quantum = drive_config(robot).omega_quantum
    rover = Rover(omega_quantum=quantum, omega_truncate=quantum > 0.0,
                  turn_ratio={'tracks_do_not_turn': 0.0, 'reversed_sign': -1.0}.get(fault, 1.0))
    sim = _pivot_sim(robot, 2.5, rover)
    if fault == 'heading_frozen':
        sim.pose_hook = lambda t, pose: (pose[0], pose[1], 0.0)
    sim.run(60.0, until=latched)
    assert (sim.out.state, sim.out.detail) == (st.STOPPED, 'STUCK')
    assert sim.t < 6.0
    assert abs(sim.rover.yaw) < 1.0


def test_pivot_held_by_the_gate_is_not_stuck():
    """While the gate zeroes the output (speed_scale 0) the base does not turn, so that rotation
    does not count as commanded."""
    sim = _pivot_sim('follower', 2.5, Rover(turn_ratio=0.0))
    while sim.t < 5.0:
        sim.ctrl.set_gate('STOPPED', ('GAP_MIN',), 0.0, sim.t)
        sim.tick()
    sim.rover.turn_ratio = 1.0
    sim.run(60.0, until=latched)
    assert sim.out.state == st.PARKED


@pytest.mark.parametrize('delay', [0.15, 0.4])
@pytest.mark.parametrize('heading', [0.3, -math.pi + 0.05])
@pytest.mark.parametrize('seed', range(2))
def test_slow_truncated_pivot_not_stuck(delay, heading, seed):
    """Leader at its worst: half the commanded yaw rate executed, omega cut to whole steps,
    0.15-0.4 s pose delay. The pivot completes and the path is driven."""
    rover = Rover(omega_quantum=0.0697, omega_truncate=True, turn_ratio=0.5)
    sim = _pivot_sim('leader', heading, rover, delay=delay, seed=seed)
    sim.run(90.0, until=latched)
    assert 'STUCK' not in {row[4] for row in sim.log}
    assert sim.out.state == st.PARKED and _pivots(sim)


def test_large_heading_error_recovers_no_spin():
    """Start facing away from the path: slow tight arc, then normal tracking."""
    cfg = drive_config('follower')
    cmd = make_cmd(straight_path(3.0), goal_pos_tol=GOAL_POS_TOL)
    sim = Sim(cfg, cmd, Rover(yaw=math.radians(150)))
    sim.run(4.0)
    turning = [row for row in sim.log if 0.5 < row[0] < 1.5]
    assert all(row[1] <= cfg.turn_speed + 0.02 for row in turning)
    assert all(row[1] > 0.0 for row in turning)
    sim.run(60.0, until=latched)
    assert sim.out.state == st.PARKED
    assert sim.max_lateral() < cfg.off_path_max
    assert sim.max_lateral(t_from=sim.t - 5.0) < 0.03


def test_lookahead_scales_with_speed():
    cfg = drive_config('follower')
    assert lookahead_distance(cfg, 0.0) == cfg.lookahead_min
    assert lookahead_distance(cfg, 0.3) == pytest.approx(cfg.lookahead_time_s * 0.3)
    assert lookahead_distance(cfg, 5.0) == cfg.lookahead_max


def test_lookahead_follows_path_around_corner():
    """Just before a left corner the tracker already steers left (REP-103: +)."""
    cfg = drive_config('follower')
    corner = Path([(0, 0), (1, 0), (1, 1)])
    before = steer(cfg, corner, 0.90, 0.90, 0.0, 0.0, speed=0.2)
    far = steer(cfg, corner, 0.30, 0.30, 0.0, 0.0, speed=0.2)
    assert before.curvature > 0.5
    assert far.curvature == pytest.approx(0.0, abs=1e-9)


def test_steering_sign_is_rep103():
    cfg = drive_config('follower')
    line = Path([(0, 0), (3, 0)])
    assert steer(cfg, line, 1.0, 1.0, -0.1, 0.0, 0.2).curvature > 0
    assert steer(cfg, line, 1.0, 1.0, 0.1, 0.0, 0.2).curvature < 0


def test_ref_offset_keeps_lookahead():
    """Ld is measured from the rotation centre (the steered point), not the tracked point."""
    line = Path([(0, 0), (3, 0)])
    cx, cy, yaw = 1.0, -0.05, math.radians(10)
    results = []
    for offset in (0.0, 0.05):
        cfg = drive_config('leader', ref_offset_x=offset, runin_heading_gain=0.0)
        x, y = cx + offset * math.cos(yaw), cy + offset * math.sin(yaw)
        s = line.project(x, y).s
        results.append(steer(cfg, line, s, x, y, yaw, 0.1).curvature)
    # Projections of C and P differ by offset * cos(yaw); the targets coincide up to ~0.8 mm.
    assert results[1] == pytest.approx(results[0], rel=0.02)


def test_runin_heading_term_turns_to_heading():
    """On the final straight a body heading off the line heading is corrected even with the
    rotation centre on the line; off the run-in nothing changes."""
    line = Path([(0, 0), (1, 0)])
    plain = drive_config('leader', runin_heading_gain=0.0)
    damped = drive_config('leader', runin_heading_gain=10.0)
    yaw = math.radians(5)
    k_plain = steer(plain, line, 0.5, 0.5, 0.0, yaw, 0.1).curvature
    k_damped = steer(damped, line, 0.5, 0.5, 0.0, yaw, 0.1).curvature
    assert k_damped == pytest.approx(k_plain - 10.0 * math.sin(yaw))
    corner = Path([(0, 0), (0.5, 0), (0.5, 0.5)])
    assert steer(damped, corner, 0.2, 0.2, 0.0, yaw, 0.1).curvature == pytest.approx(
        steer(plain, corner, 0.2, 0.2, 0.0, yaw, 0.1).curvature)


def test_target_behind_gives_full_turn():
    cfg = drive_config('follower')
    line = Path([(0, 0), (3, 0)])
    behind = steer(cfg, line, 1.0, 1.0, 0.0, math.radians(179), 0.0)
    assert abs(behind.curvature) > 5.0


def test_restart_resumes_from_position():
    """A fresh tracker (node restart) gets the latched command mid-path and carries on."""
    cfg = drive_config('follower')
    cmd = make_cmd(s_path(), goal_pos_tol=GOAL_POS_TOL)
    first = Sim(cfg, cmd, Rover())
    first.run(6.0)
    rover = first.rover
    rover.command(0.0, 0.0)
    second = Sim(cfg, cmd, rover)
    second.run(1.0)
    assert second.out.progress > 0.3 and second.out.state == st.TRACKING
    second.run(60.0, until=latched)
    assert second.out.state == st.PARKED
    assert second.max_lateral() < MAX_LATERAL


ARC_RADIUS = 0.25


def _arc_path(radius=ARC_RADIUS, n=90):
    """Left half circle from (0, 0) heading +x."""
    return Path([(radius * math.sin(a), radius * (1.0 - math.cos(a)))
                 for a in (math.pi * i / n for i in range(n + 1))])


@pytest.mark.parametrize('robot', ['leader', 'follower'])
def test_heading_error_zero_on_tight_arc(robot):
    """The lookahead bearing a curve itself causes is not a heading error: a rover on a 0.25 m
    arc facing along it is not slowed by the HEADING limit. A real error still is."""
    cfg = drive_config(robot)
    path = _arc_path()
    phi = 0.8
    x, y = ARC_RADIUS * math.sin(phi), ARC_RADIUS * (1.0 - math.cos(phi))
    s = path.project(x, y).s
    steering = steer(cfg, path, s, x, y, path.tangent(s), cfg.max_linear_speed)
    assert abs(steering.heading_error) < 0.1
    assert steering.curvature > 0.0
    assert heading_limit(cfg, steering.heading_error, cfg.max_linear_speed).speed == NO_LIMIT
    off = steer(cfg, path, s, x, y, path.tangent(s) - 0.6, cfg.max_linear_speed)
    assert off.heading_error == pytest.approx(0.6, abs=0.1)
