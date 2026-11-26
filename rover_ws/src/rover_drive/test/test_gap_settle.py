"""Gap control behind a lead that keeps driving: settle, do not stop and go.

Controller only (no plant lag): the follower moves by its own output on a straight path,
the lead is a point moving at a set speed. The LiDAR free gap is reported at 10 Hz,
the lead's relayed position at every step. free gap = path gap - OFFSET (both definitions
consistent unless a test makes the relayed position wrong).
"""
import math

import numpy as np
import pytest

from rover_drive.core.controller import DriveController
from rover_drive.core.limits import GapLimiter

from sim import drive_config, make_cmd, straight_path

DT = 0.05
SCAN_PERIOD = 0.1
HOST_GAP = 0.6
OFFSET = HOST_GAP - drive_config('follower').gap_free_target


class Convoy:
    def __init__(self, lead_speed=0.10, lead_x0=1.0, x0=0.0):
        self.cfg = drive_config('follower')
        self.ctrl = DriveController(self.cfg)
        self.ctrl.set_heading_aligned(True)
        self.cmd = make_cmd(straight_path(30.0), lead_robot='leader', gap=HOST_GAP)
        self.lead_speed, self.lead_x, self.x, self.t = lead_speed, lead_x0, x0, 0.0
        self.rows = []
        self._next_scan = 0.0
        self.scan_index = 0

    def free_gap(self):
        return self.lead_x - self.x - OFFSET

    def step(self, measured=lambda conv, gap: gap, relayed_offset=0.0):
        """One 20 Hz step. ``measured(conv, true gap)``: what the gate reports (NaN = not seen)."""
        t, ctrl = self.t, self.ctrl
        ctrl.on_command(self.cmd, t)
        ctrl.on_pose(self.x, 0.0, 0.0, t)
        ctrl.on_lead_pose(self.lead_x + relayed_offset, 0.0, t)
        if t + 1e-9 >= self._next_scan:
            self._next_scan += SCAN_PERIOD
            ctrl.on_lead_gap(measured(self, self.free_gap()), t)
            self.scan_index += 1
        out = ctrl.step(t)
        self.rows.append((t, self.free_gap(), out.v, out.detail))
        self.x += out.v * DT
        self.lead_x += self.lead_speed * DT
        self.t += DT
        return out

    def run(self, duration, **kw):
        end = self.t + duration
        while self.t < end - 1e-9:
            self.step(**kw)
        return self

    def window(self, t0, t1=math.inf):
        return [r for r in self.rows if t0 <= r[0] < t1]


def _stops(rows):
    return sum(1 for a, b in zip(rows, rows[1:]) if a[2] > 0.0 and b[2] <= 0.0)


def test_follower_tracks_steady_lead_no_standing():
    """Constant time gap: settles lead speed / gap_gain behind the wanted gap, never stands."""
    conv = Convoy(lead_speed=0.10).run(40.0)
    tail = conv.window(25.0)
    err = np.array([r[1] for r in tail]) - conv.cfg.gap_free_target
    v = np.array([r[2] for r in tail])
    print(f'gap error mean {err.mean():.4f} m, speed mean {v.mean():.4f} m/s')
    assert err.mean() == pytest.approx(0.10 / conv.cfg.gap_gain, abs=0.005)
    assert v.mean() == pytest.approx(0.10, abs=0.003)
    assert min(r[2] for r in conv.window(5.0)) > 0.0


def test_scan_missing_lead_keeps_lidar_gap():
    """The lead at the sector edge is seen in every other scan, and the path gap reads 0.15 m
    short (lead projected onto a curve). The last measured free gap stays in use."""
    def flicker(conv, gap):
        return gap if conv.scan_index % 2 == 0 else math.nan
    conv = Convoy(lead_speed=0.10).run(40.0, measured=flicker, relayed_offset=-0.15)
    rows = conv.window(10.0)
    print(f"details {sorted({r[3] for r in rows})}, stops {_stops(rows)}")
    assert {r[3] for r in rows if r[3]} <= {'GAP_LIDAR'}
    assert _stops(rows) == 0


def test_gap_dip_slows_but_does_not_stop():
    """The lead turning ahead: its nearest point reads 6 cm closer for a second."""
    def dip(conv, gap):
        return gap - 0.06 if 30.0 <= conv.t < 31.0 else gap
    conv = Convoy(lead_speed=0.10).run(40.0, measured=dip)
    rows = conv.window(29.0, 35.0)
    print(f'lowest speed in the dip {min(r[2] for r in rows):.3f} m/s')
    assert min(r[2] for r in rows) > 0.0
    assert min(r[2] for r in conv.window(30.5, 31.0)) < 0.08


@pytest.mark.parametrize('stop_at', [20.0, 20.07, 30.0])
def test_lead_dead_stop_keeps_stop_gap(stop_at):
    conv = Convoy(lead_speed=0.10)
    conv.run(stop_at)
    conv.lead_speed = 0.0
    conv.run(20.0)
    after = conv.window(stop_at)
    low = min(r[1] for r in after)
    print(f'stop at {stop_at}: lowest free gap {low:.3f} m (stop gap {conv.cfg.gap_hard_min})')
    assert low > conv.cfg.gap_hard_min + 0.03
    tail = conv.window(stop_at + 10.0)
    # P law: the last centimetre closes exponentially; the rover creeps toward the wanted gap, never crosses it
    assert max(r[2] for r in tail) < 0.005
    assert max(r[1] for r in tail) - min(r[1] for r in tail) < 0.005
    assert min(r[1] for r in tail) >= conv.cfg.gap_free_target - 1e-9


def test_follower_restarts_when_lead_drives_on():
    conv = Convoy(lead_speed=0.10)
    conv.run(20.0)
    conv.lead_speed = 0.0
    conv.run(15.0)
    conv.lead_speed = 0.10
    conv.run(30.0)
    tail = conv.window(55.0)
    assert np.mean([r[2] for r in tail]) == pytest.approx(0.10, abs=0.005)
    assert _stops(conv.window(36.0)) == 0


def test_gap_limiter_zero_inside_wanted_gap():
    cfg = drive_config('follower')
    gap = GapLimiter(cfg)
    margin = cfg.gap_free_target - cfg.gap_hard_min
    assert gap.limit(0.02).speed == pytest.approx(cfg.gap_gain * 0.02)
    assert gap.limit(0.0).speed == 0.0
    assert gap.limit(-margin).speed == 0.0
    assert gap.limit(0.02).speed == 0.0
