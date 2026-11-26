"""Bumpless switch between the two gap sources (LiDAR free gap <-> path gap).

The free gap (body to body, ``gap_free_target``) and the path gap (centre to centre along the
path, ``PathCommand.gap``) define spacing differently. While the LiDAR sees the lead the controller
remembers d = path error - LiDAR error; without it uses path error - d (GAP_BRIDGED). No sighting
yet, or trust ended: plain path gap. Only with gap_bridge: true; default keeps the plain path gap.

Straight path: centre to centre = free gap + OFF (body offsets); with OFF = 0.285 m the two errors
differ by -0.115 m at the operating point.
"""
import math

import numpy as np
import pytest

from rover_drive.core.controller import DriveController

from sim import drive_config, make_cmd, straight_path

DT = 0.05
GAP = 0.60
OFF = 0.285
V_LEAD = 0.092
CFG = drive_config('follower', gap_bridge=True)
E_OP = V_LEAD / CFG.gap_gain
FREE_OP = CFG.gap_free_target + E_OP
STOPPED = 1e-3


class _Platoon:
    """Follower driven by its own output behind a lead with a scripted speed."""

    def __init__(self, free_gap=FREE_OP, cmd_id=1, cfg=CFG):
        self.ctrl = DriveController(cfg)
        self.ctrl.set_heading_aligned(True)
        self.cmd = make_cmd(straight_path(8.0), cmd_id=cmd_id, lead_robot='leader', gap=GAP)
        self.xf = 0.5
        self.xl = self.xf + free_gap + OFF
        self.min_free = math.inf
        self.log = []
        self.t = 0.0

    @property
    def free(self):
        return self.xl - self.xf - OFF

    def run(self, t1, v_lead=V_LEAD, seen=True, gap_age=0.0, lead_age=0.0, scan_every=2):
        """20 Hz control, LiDAR scans at 10 Hz (``seen`` False = NaN, lead not seen)."""
        ctrl, out = self.ctrl, None
        for k, t in enumerate(np.arange(self.t, t1 - 1e-9, DT)):
            ctrl.on_command(self.cmd, t)
            ctrl.on_pose(self.xf, 0.0, 0.0, t)
            ctrl.on_lead_pose(self.xl, 0.0, t - lead_age)
            if seen is not None and k % scan_every == 0:
                ctrl.on_lead_gap(self.free if seen else math.nan, t - gap_age)
            out = ctrl.step(t)
            self.xf += out.v * DT
            self.xl += v_lead * DT
            self.min_free = min(self.min_free, self.free)
            self.log.append((t, out.v, out.detail, self.free, self.xf))
        self.t = t1
        return out


def _settled(cfg=CFG):
    """Follower settled behind the cruising lead on the LiDAR law."""
    p = _Platoon(cfg=cfg)
    out = p.run(10.0)
    assert out.detail == 'GAP_LIDAR'
    assert p.free == pytest.approx(FREE_OP, abs=0.01)
    return p


def test_operating_point_reads_too_close():
    """Premise of the fix: the plain path error is negative where the LiDAR one is +0.06."""
    assert FREE_OP + OFF - GAP < -0.04


def test_lidar_dropout_does_not_stop_follower():
    p = _settled()
    p.log.clear()
    p.run(14.0, seen=False)
    blind = [r for r in p.log if r[0] >= 10.4]
    assert {r[2] for r in blind} == {'GAP_BRIDGED'}
    assert min(r[1] for r in p.log) > 0.5 * V_LEAD
    assert p.free == pytest.approx(FREE_OP, abs=0.01)
    out = p.run(18.0, seen=True)
    assert out.detail == 'GAP_LIDAR'
    assert min(r[1] for r in p.log) > 0.5 * V_LEAD


def _lead_stops(seen):
    p = _settled()
    p.run(11.0, seen=seen)
    out = p.run(30.0, v_lead=0.0, seen=seen)
    assert out.v < STOPPED
    return p


def test_unseen_lead_stop_keeps_lidar_law_gap():
    """The lead stops after the LiDAR lost it; compare with the same run on the LiDAR law."""
    lidar = _lead_stops(seen=True)
    blind = _lead_stops(seen=False)
    assert lidar.log[-1][2] == 'GAP_LIDAR' and blind.log[-1][2] == 'GAP_BRIDGED'
    assert blind.min_free >= lidar.min_free - 1e-3
    assert blind.min_free > CFG.gap_hard_min
    assert blind.min_free < GAP - OFF - 0.05


def test_no_lidar_keeps_conservative_path_gap():
    """Never seen (no message, NaN only, or always older than lead_gap_timeout_s)."""
    for seen, age in ((None, 0.0), (False, 0.0), (True, 0.35)):
        p = _Platoon(free_gap=GAP - OFF)
        out = p.run(15.0, seen=seen, gap_age=age)
        assert out.detail == 'GAP'
        assert p.xl - p.xf == pytest.approx(GAP + E_OP, abs=2e-3)
        assert out.v == pytest.approx(V_LEAD, abs=1e-3)
        out = p.run(30.0, v_lead=0.0, seen=seen, gap_age=age)
        assert out.v < STOPPED and out.detail == 'GAP'
        assert GAP - 0.03 <= p.xl - p.xf <= GAP + 1e-6


def test_trust_ends_after_one_gap_of_travel():
    p = _settled()
    p.log.clear()
    x0 = p.xf
    p.run(25.0, seen=False)
    first_plain = next(r for r in p.log if r[2] == 'GAP')
    assert first_plain[4] - x0 == pytest.approx(GAP, abs=0.05)
    after = [r for r in p.log if r[0] >= first_plain[0]]
    assert {r[2] for r in after} == {'GAP'}
    assert min(r[1] for r in after) < STOPPED
    assert all(r[2] == 'GAP_BRIDGED' for r in p.log if r[0] >= 10.4 and r[4] - x0 < GAP - 0.05)


def test_reissue_drops_bridge_until_lidar_sees():
    p = _settled()
    p.run(11.0, seen=False)
    assert p.log[-1][2] == 'GAP_BRIDGED'
    p.cmd = make_cmd(straight_path(8.0), cmd_id=2, lead_robot='leader', gap=GAP)
    out = p.run(12.0, seen=False)
    assert out.detail == 'GAP' and out.v < E_OP * CFG.gap_gain
    out = p.run(13.0, seen=True)
    assert out.detail == 'GAP_LIDAR'
    out = p.run(16.0, seen=False)
    assert out.detail == 'GAP_BRIDGED' and out.v > 0.0


def test_new_lead_drops_the_bridge():
    p = _settled()
    p.cmd = make_cmd(straight_path(8.0), cmd_id=2, lead_robot='other', gap=GAP)
    out = p.run(12.0, seen=False)
    assert out.detail == 'GAP'


def test_stale_lead_stops_and_drops_bridge():
    p = _settled()
    p.run(11.0, seen=False)
    out = p.run(12.0, v_lead=0.0, seen=False, lead_age=0.6)
    assert (out.v, out.detail) == (0.0, 'STALE_LEAD')
    out = p.run(16.0, v_lead=0.0, seen=False)
    assert out.v < STOPPED and out.detail == 'GAP'


def test_default_off_keeps_flip_stop_in_dropout():
    """gap_bridge false (default): plain GAP."""
    base = drive_config('follower')
    assert base.gap_bridge is False
    p = _settled(base)
    p.log.clear()
    p.run(14.0, seen=False)
    blind = [r for r in p.log if r[0] >= 10.4]
    assert {r[2] for r in blind} == {'GAP'}
    assert min(r[1] for r in blind) < STOPPED
    stand = next(r for r in blind if r[1] < STOPPED)
    centre = stand[3] + OFF
    assert centre <= GAP + 1e-6
    # it resumes only once the lead has pulled away to cmd.gap + gap_resume_margin
    moving = [r for r in blind if r[0] > stand[0] and r[1] > STOPPED]
    assert moving and moving[0][3] + OFF >= GAP + CFG.gap_resume_margin - 0.01
