"""Gap control: follower (0.20 m/s) behind a slower leader (0.08 m/s) on one path."""
import numpy as np
import pytest

from rover_drive.core import state as st
from rover_drive.core.limits import GapLimiter

from sim import Rover, Sensor, Sim, drive_config, make_cmd, straight_path

GAP = 0.6
LEAD_START = 0.8
LEAD_SPEED = 0.08
LEAD_GOAL = 3.5


class Convoy:
    """Leader and follower sims in lockstep; the follower hears the leader at 30 Hz."""

    def __init__(self, seed=0):
        full = straight_path(6.0)
        lead_poses = [p for p in full if LEAD_START - 1e-9 <= p[0] <= LEAD_GOAL + 1e-9]
        self.leader = Sim(drive_config('leader'), make_cmd(lead_poses, speed_cap=LEAD_SPEED),
                          Rover(x=LEAD_START), Sensor(seed=seed + 1))
        self.follower = Sim(drive_config('follower'),
                            make_cmd(full, lead_robot='leader', gap=GAP),
                            Rover(), Sensor(seed=seed + 2))
        self.link = Sensor(seed=seed + 3)
        self.link_enabled = True
        self.rows = []

    def tick(self):
        self.link.sample(self.leader.t, self.leader.rover.tracked)
        for x, y, _ in self.link.deliver(self.leader.t):
            if self.link_enabled:
                self.follower.ctrl.on_lead_pose(x, y, self.follower.t)
        self.leader.tick()
        self.follower.tick()
        out = self.follower.out
        gap = self.leader.rover.tracked[0] - self.follower.rover.tracked[0]
        self.rows.append((self.follower.t, gap, out.v, out.state, out.detail))

    def run(self, duration):
        end = self.follower.t + duration
        while self.follower.t < end:
            self.tick()

    def window(self, t0, t1):
        return [r for r in self.rows if t0 <= r[0] < t1]


def _stop_go_transitions(rows):
    moving = [r[2] > 0.0 for r in rows]
    return sum(1 for a, b in zip(moving, moving[1:]) if a != b)


@pytest.fixture(scope='module')
def convoy():
    c = Convoy()
    c.run(60.0)
    return c


def test_gap_is_held_while_the_leader_moves(convoy):
    steady = convoy.window(12.0, 30.0)
    gap_err = np.array([r[1] for r in steady]) - GAP
    v = np.array([r[2] for r in steady])
    print(f'steady gap error mean {gap_err.mean():.3f} m, std {gap_err.std():.3f} m, '
          f'min {gap_err.min():.3f} m; follower v mean {v.mean():.3f} std {v.std():.3f}')
    # Proportional law: steady error = lead speed / gap_gain = 0.08 / 1.5.
    assert gap_err.mean() == pytest.approx(LEAD_SPEED / convoy.follower.cfg.gap_gain, abs=0.02)
    assert gap_err.std() < 0.02
    assert gap_err.min() > 0.0
    assert v.mean() == pytest.approx(LEAD_SPEED, abs=0.01)


def test_no_stop_go_chatter_while_following(convoy):
    steady = convoy.window(12.0, 30.0)
    assert _stop_go_transitions(steady) == 0
    assert min(r[2] for r in steady) > 0.03
    assert all(r[3] == st.TRACKING for r in steady)


def test_follower_stops_when_leader_stops(convoy):
    assert convoy.leader.out.state == st.PARKED
    tail = convoy.window(50.0, 60.0)
    assert all(r[2] == 0.0 for r in tail)
    final_gap = tail[-1][1]
    print(f'final gap {final_gap:.3f} m (wanted {GAP})')
    assert GAP - 0.01 <= final_gap <= GAP + 0.06
    assert convoy.follower.out.state == st.TRACKING and convoy.follower.out.detail == 'GAP'
    assert _stop_go_transitions(convoy.rows) <= 2


def test_follower_stops_on_stale_lead_position():
    c = Convoy()
    c.run(15.0)
    c.link_enabled = False
    c.run(2.0)
    assert c.follower.out.v == 0.0
    assert (c.follower.out.state, c.follower.out.detail) == (st.STOPPED, 'STALE_LEAD')
    c.link_enabled = True
    c.run(3.0)
    assert c.follower.out.state == st.TRACKING and c.follower.out.v > 0.0


def test_follower_waits_without_lead_position():
    c = Convoy()
    c.link_enabled = False
    c.run(3.0)
    assert max(r[2] for r in c.rows) == 0.0
    assert c.follower.out.detail == 'STALE_LEAD'


def test_gap_limiter_hysteresis():
    cfg = drive_config('follower')
    gap = GapLimiter(cfg)
    assert gap.limit(0.10).speed == pytest.approx(0.15)
    assert gap.limit(0.005).speed == pytest.approx(0.0075)
    assert gap.limit(0.0).speed == 0.0
    assert gap.limit(cfg.gap_resume_margin - 0.01).speed == 0.0
    assert gap.limit(cfg.gap_resume_margin + 0.01).speed > 0.0
    assert gap.limit(0.03).speed > 0.0
    assert gap.limit(-0.10).speed == 0.0


JOIN_X = 3.2
LEAD_SIDE = 0.6
BRANCH_X = 3.0


def _polyline(points, spacing=0.05):
    """Poses along straight legs through ``points``, ``spacing`` apart."""
    poses = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length, yaw = np.hypot(x1 - x0, y1 - y0), np.arctan2(y1 - y0, x1 - x0)
        for d in np.arange(0.0, length, spacing):
            poses.append((x0 + d * np.cos(yaw), y0 + d * np.sin(yaw), yaw))
    poses.append((*points[-1], yaw))
    return poses


class Merge(Convoy):
    """The leader drives its own path and the follower's path only shares part of it."""

    def __init__(self, lead_points, lead_yaw, follower_goal=6.0, seed=0):
        self.leader = Sim(drive_config('leader'),
                          make_cmd(_polyline(lead_points), speed_cap=LEAD_SPEED),
                          Rover(x=lead_points[0][0], y=lead_points[0][1], yaw=lead_yaw),
                          Sensor(seed=seed + 1))
        self.follower = Sim(drive_config('follower'),
                            make_cmd(straight_path(follower_goal), lead_robot='leader', gap=GAP),
                            Rover(), Sensor(seed=seed + 2))
        self.link = Sensor(seed=seed + 3)
        self.link_enabled = True
        self.rows = []


def test_follower_waits_for_leader_to_join_path():
    """The follower's stretch to the shared path is shorter and it is faster: it must still reach
    the shared path behind the leader."""
    start = (0.8, LEAD_SIDE)
    merge = Merge([start, (JOIN_X, 0.0), (4.5, 0.0)], lead_yaw=np.arctan2(-LEAD_SIDE, JOIN_X - 0.8))
    merge.run(45.0)
    assert merge.leader.rover.tracked[0] > JOIN_X + 0.5, 'leader should be on the shared path by now'
    moving = [r for r in merge.rows if r[0] > 2.0]
    ahead = min(r[1] for r in moving)
    print(f'smallest lead along the follower path {ahead:.3f} m (wanted about {GAP})')
    assert ahead >= GAP - 0.10
    assert max(r[2] for r in merge.rows) > 0.05, 'follower should follow, not just wait'


def test_follower_reaches_goal_after_leader_left():
    """Leader ahead on my path, then it branches off to its own goal and parks beside my
    path: only the range to it is kept from then on (it is no longer in my way)."""
    merge = Merge([(LEAD_START, 0.0), (BRANCH_X, 0.0), (3.8, -0.9)], lead_yaw=0.0,
                  follower_goal=4.5)
    merge.run(90.0)
    assert merge.leader.out.state == st.PARKED
    print(f'follower ends at x={merge.follower.rover.tracked[0]:.3f} state={merge.follower.out.state}')
    assert merge.follower.out.state == st.PARKED
    assert min(np.hypot(r[1], 0.0) for r in merge.rows) > 0.0
