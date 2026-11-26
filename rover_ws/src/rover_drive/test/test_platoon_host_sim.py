"""Platooning closed loop against ``host_sim``: LiDAR free gap, fallback, hard minimum.

Leader and follower each run tracker + gate (``host_loop.RoverStack``); the follower's gate hands
the LiDAR free gap to its tracker. The true free gap comes from the simulated footprints (circles),
so ``lidar_to_front`` is the simulated follower's radius (LiDAR at its centre).
Needs ``host_ws/src/host_sim`` on PYTHONPATH (skipped otherwise).
"""
import math

import pytest

pytest.importorskip('host_sim')

from host_sim.scenario import run_scenario           # noqa: E402

from host_loop import RoverStack, make_world         # noqa: E402
from sim import drive_config, gate_config, make_cmd  # noqa: E402

Y = 3.3
LEADER_X0, LEADER_GOAL = 2.65, 5.6     # tracked point
FOLLOWER_X0 = 1.9
HOST_GAP = 0.60                        # PathCommand.gap, centre to centre


def _line(x0, x1):
    n = int(round((x1 - x0) / 0.05))
    return [(x0 + i * 0.05, Y, 0.0) for i in range(n + 1)]


class Platoon:
    """One closed-loop run; ``rows`` = (t, true free gap, follower cmd v, state, detail, gate)."""

    def __init__(self, lead_speed=0.08, host_gap=HOST_GAP, leader_start=None,
                 lidar_gap=True, **follower_gate):
        start = {'start': leader_start} if leader_start else {}
        self.world = make_world(1, leader={'omega_quantum': 0.0, 'omega_deadband': 0.0,
                                           **start})
        self.leader = None
        if leader_start is None:
            self.leader = RoverStack(
                drive_config('leader', ref_offset_x=0.05, omega_quantum=0.0),
                gate_config('leader'),
                make_cmd(_line(LEADER_X0, LEADER_GOAL), speed_cap=lead_speed))
        radius = self.world.robots['follower'].params.radius
        self.follower = RoverStack(
            drive_config('follower'),
            gate_config('follower', lidar_to_front=radius, **follower_gate),
            make_cmd(_line(FOLLOWER_X0, LEADER_GOAL), lead_robot='leader', gap=host_gap))
        self.follower.lead_gap_link = lidar_gap
        self.rows = []

    def _follower(self, obs):
        cmd = self.follower(obs)
        lead, me = self.world.robots['leader'], self.world.robots['follower']
        free = (math.hypot(lead.x - me.x, lead.y - me.y) - lead.params.radius
                - me.params.radius)
        _, state, detail, _, reasons = self.follower.log[-1][:5]
        self.rows.append((obs.t, free, cmd[0], state, detail, reasons))
        return cmd

    def run(self, duration, leader_stop_at=None):
        controllers = {'follower': self._follower}
        if self.leader is not None:
            controllers['leader'] = self.leader
        events = []
        if leader_stop_at is not None:
            events.append((leader_stop_at, lambda w: self.leader.gate.set_estop(True)))
        self.result = run_scenario(self.world, controllers, duration=duration, events=events,
                                   stop_on_arrival=False)
        return self

    def window(self, t0, t1=math.inf):
        return [r for r in self.rows if t0 <= r[0] < t1]


def _changes(values):
    return sum(1 for a, b in zip(values, values[1:]) if a != b)


def _gate_reasons(rows):
    return {reason for row in rows for reason in row[5]}


@pytest.fixture(scope='module')
def steady():
    return Platoon(lead_speed=0.08).run(48.0)


def test_steady_follow_holds_gap_no_stop_go(steady):
    cfg = drive_config('follower')
    rows = steady.window(6.0, 36.0)
    free = [r[1] for r in rows]
    stop_go = _changes([r[2] > 0.0 for r in rows])
    toggles = _changes([(r[3], r[4]) for r in rows])
    mean = sum(free) / len(free)
    print(f'steady 30 s: free gap mean {mean:.3f} min {min(free):.3f} m, '
          f'cmd stop/go {stop_go}, state/detail toggles {toggles}')
    assert {r[4] for r in rows} == {'GAP_LIDAR'}
    assert stop_go == 0 and toggles == 0
    # P law: free gap = target + lead speed / gain; the min over noisy beams reads a bit short.
    assert mean == pytest.approx(cfg.gap_free_target + 0.08 / cfg.gap_gain, abs=0.03)
    assert min(free) > cfg.gap_free_target
    assert not steady.result.collided


def test_follower_parks_at_target_gap(steady):
    cfg, gate = drive_config('follower'), gate_config('follower')
    tail = steady.window(46.0)
    print(f'parked: free gap {tail[-1][1]:.3f} m')
    assert all(r[2] == 0.0 for r in tail)
    assert cfg.gap_free_target <= tail[-1][1] <= cfg.gap_free_target + gate.lidar_to_front
    assert steady.result.min_robot_gap > gate.gap_hard_min


@pytest.mark.parametrize('lead_speed', [0.08, 0.10])
def test_sudden_stop_keeps_gap_above_hard_min(lead_speed):
    run = Platoon(lead_speed=lead_speed).run(25.0, leader_stop_at=15.0)
    before = run.window(14.9, 15.0)[-1]
    after = run.window(15.0)
    low = min(r[1] for r in after)
    print(f'leader {lead_speed} m/s stops dead: free gap {before[1]:.3f} -> min {low:.3f} m '
          f'(follower at {before[2]:.3f} m/s)')
    assert before[2] > 0.05
    assert low > gate_config('follower').gap_hard_min + 0.10
    assert 'GAP_MIN' not in _gate_reasons(after)
    assert all(r[2] == 0.0 for r in run.window(20.0)) and not run.result.collided


def test_fast_approach_to_standing_leader_stops():
    run = Platoon(leader_start=[4.0, Y, 0.0]).run(25.0)
    peak = max(r[2] for r in run.rows)
    final = run.rows[-1][1]
    print(f'standing leader: peak {peak:.3f} m/s, final free gap {final:.3f}, '
          f'min {run.result.min_robot_gap:.3f} m')
    assert peak > 0.19
    assert drive_config('follower').gap_free_target <= final < 0.26
    assert run.rows[-1][4] == 'GAP_LIDAR' and not run.result.collided


def test_lidar_gap_lost_falls_back_to_path_gap():
    run = Platoon(lidar_gap=False).run(30.0)
    rows = run.window(6.0)
    print(f'path gap fallback: min free gap {min(r[1] for r in rows):.3f} m')
    assert 'GAP' in {r[4] for r in rows} and 'GAP_LIDAR' not in {r[4] for r in rows}
    assert min(r[1] for r in rows) > drive_config('follower').gap_free_target
    assert not run.result.collided


@pytest.mark.parametrize('hard_min', [None, 0.08])
def test_gate_alone_keeps_hard_min_on_tracker_fail(hard_min):
    """Tracker without the LiDAR gap and a host gap far too small (0.30 centre to centre).

    The lead is left out of the front stop, so without the hard minimum nothing
    in the gate protects it (control case: collision).
    """
    gate = {} if hard_min is not None else {'gap_hard_min': -10.0}
    run = Platoon(lead_speed=0.10, host_gap=0.30, lidar_gap=False, **gate)
    run.run(20.0, leader_stop_at=10.0)
    low = min(r[1] for r in run.rows)
    print(f'gap_hard_min={hard_min}: min free gap {low:.3f} m, collided {run.result.collided}')
    if hard_min is None:
        assert run.result.collided
        return
    assert not run.result.collided
    assert low > hard_min
    assert 'GAP_MIN' in _gate_reasons(run.rows)
