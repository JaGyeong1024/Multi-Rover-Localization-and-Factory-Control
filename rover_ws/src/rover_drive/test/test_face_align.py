"""Squaring up to the fixture face (face_align): the face fit, the gyro yaw and the
controller's FACE_ALIGN phase on synthetic depth scans of a flat face.

Sign convention under test: a rover turned left (CCW) of square by alpha measures
phi = +alpha and turns by -alpha.
"""
import math

import numpy as np
import pytest

from rover_drive.core import state as st
from rover_drive.core.controller import (FACE_ALIGN, FACE_ALIGN_FAILED, FACE_ALIGNED,
                                         DriveController)
from rover_drive.core.depth_front import DepthScan
from rover_drive.core.face_align import (BAD_FIT, FEW_POINTS, GYRO_TIMEOUT_S, TOO_SKEWED,
                                         WEAK_FIT, GyroYaw, RememberedFace, fit_face)

from sim import SIM_DT, Rover, drive_config, make_cmd, straight_path

CFG = drive_config('leader', face_align=True)
TOL = CFG.face_align_tol_rad
HFOV = 1.019
BEAMS = 80
CAMERA_X = 0.045
LEADER_QUANTUM = 0.0697


class Face:
    """A flat face: centre (x, y), inward normal at world yaw ``yaw`` (the way a square
    rover faces), ``half`` m to either side."""

    def __init__(self, x, y=0.0, yaw=0.0, half=0.5):
        self.x, self.y, self.yaw, self.half = x, y, yaw, half

    def scan(self, pose, camera_min=0.30, noise=0.0, rng=None):
        """Depth scan of the face from a camera on the body x axis of ``pose`` (x, y, yaw):
        horizontal ranges, 0.0 where the beam misses the face or the depth (along the
        optical axis) is below ``camera_min``."""
        x, y, yaw = pose
        cx, cy = x + CAMERA_X * math.cos(yaw), y + CAMERA_X * math.sin(yaw)
        inc = HFOV / BEAMS
        angles = -HFOV / 2.0 + inc / 2.0 + inc * np.arange(BEAMS)
        nx, ny = math.cos(self.yaw), math.sin(self.yaw)
        ux, uy = np.cos(yaw + angles), np.sin(yaw + angles)
        facing = nx * ux + ny * uy
        with np.errstate(divide='ignore', invalid='ignore'):
            r = (nx * (self.x - cx) + ny * (self.y - cy)) / facing
        along = -ny * (cx + r * ux - self.x) + nx * (cy + r * uy - self.y)
        hit = (facing > 0.0) & (r > 0.0) & (np.abs(along) <= self.half) \
            & (r * np.cos(angles) >= camera_min)
        if noise:
            r = r + rng.normal(0.0, noise, r.shape)
        return DepthScan(np.where(hit, r, 0.0), float(angles[0]), inc, 0.30, 6.0)


def _scan_at(alpha, dist=0.5, **kw):
    """Rover at the origin turned ``alpha`` from square to a face ``dist`` ahead."""
    return Face(dist + CAMERA_X).scan((0.0, 0.0, alpha), **kw)


@pytest.mark.parametrize('alpha', [-0.4, -0.15, -0.03, 0.0, 0.03, 0.15, 0.4])
def test_fit_measures_yaw_from_face_normal(alpha):
    fit = fit_face(_scan_at(alpha), CFG)
    assert fit.ok, fit
    assert fit.phi == pytest.approx(alpha, abs=1e-6)
    assert fit.rms < 1e-6


def test_fit_with_noise_within_tolerance():
    rng = np.random.default_rng(3)
    for alpha in (-0.2, 0.0, 0.2):
        fits = [fit_face(_scan_at(alpha, noise=0.003, rng=rng), CFG) for _ in range(50)]
        assert all(f.ok for f in fits)
        assert max(abs(f.phi - alpha) for f in fits) < TOL / 2.0


def _with(scan, ranges):
    return DepthScan(ranges, scan.angle_min, scan.angle_increment, scan.range_min,
                     scan.range_max)


def test_fit_rejects_outliers_before_face():
    scan = _scan_at(0.1)
    ranges = scan.ranges.copy()
    ranges[38:41] -= 0.12
    fit = fit_face(_with(scan, ranges), CFG)
    assert fit.ok and fit.phi == pytest.approx(0.1, abs=0.005)


def test_fit_ignores_beams_without_a_return():
    scan = _scan_at(-0.1)
    ranges = scan.ranges.copy()
    ranges[::3] = 0.0
    fit = fit_face(_with(scan, ranges), CFG)
    assert fit.ok and fit.phi == pytest.approx(-0.1, abs=1e-6)


def test_too_few_points_is_no_measurement():
    assert fit_face(_scan_at(0.0, dist=0.2), CFG).reason == FEW_POINTS
    assert fit_face(_scan_at(0.0, dist=1.2), CFG).reason == FEW_POINTS
    narrow = Face(0.5 + CAMERA_X, half=0.012).scan((0.0, 0.0, 0.0))
    assert fit_face(narrow, CFG).reason == FEW_POINTS
    assert fit_face(_with(narrow, np.zeros(BEAMS)), CFG).reason == FEW_POINTS


def test_noisy_points_are_no_measurement():
    rng = np.random.default_rng(1)
    fit = fit_face(_scan_at(0.1, noise=0.04, rng=rng), CFG)
    assert fit.reason in (BAD_FIT, WEAK_FIT)


def test_adjacent_objects_are_not_one_face():
    """Two boxes 0.15 m apart must not be fitted as one 33 deg face (rms 17 mm)."""
    right = Face(0.70, y=-0.15, yaw=0.6, half=0.06).scan((0.0, 0.0, 0.0))
    left = Face(0.90, y=0.08, half=0.04).scan((0.0, 0.0, 0.0))
    fit = fit_face(_with(right, np.maximum(right.ranges, left.ranges)), CFG)
    assert fit.reason == BAD_FIT


def test_angle_above_max_untrusted():
    assert fit_face(_scan_at(0.65), CFG).reason == TOO_SKEWED
    assert fit_face(_scan_at(-0.65), CFG).reason == TOO_SKEWED


def test_gyro_yaw_integrates_on_stamps():
    gyro = GyroYaw()
    for i in range(16):
        gyro.add(0.2, 100.0 + 0.1 * i, 0.1 * i)
    assert gyro.yaw == pytest.approx(0.3)
    assert gyro.fresh(1.5) and not gyro.fresh(1.5 + GYRO_TIMEOUT_S + 0.01)
    assert gyro.gaps == 0


def test_gyro_gap_voids_remembered_fit():
    gyro = GyroYaw()
    gyro.add(0.0, 0.0, 0.0)
    memory = RememberedFace(fit_face(_scan_at(0.1), CFG), gyro.yaw, gyro.gaps, 0.0)
    gyro.add(0.5, 0.1, 0.1)
    assert memory.carried(gyro, 0.1, 15.0).phi == pytest.approx(0.1 + 0.025)
    gyro.add(0.5, 2.0, 2.0)
    assert gyro.yaw == pytest.approx(0.025) and gyro.gaps == 1
    assert memory.carried(gyro, 2.0, 15.0) is None
    fresh = RememberedFace(memory.fit, gyro.yaw, gyro.gaps, 2.0)
    assert fresh.carried(gyro, 18.0, 15.0) is None


class Rig:
    """Leader in front of a face: pose (with a heading bias: the global heading is off),
    gyro and depth scans into a DriveController, the command re-sent at 2 Hz."""

    def __init__(self, cmd, face, start, cfg=CFG, heading_bias=0.0, camera_min=0.30,
                 gyro_sign=1.0, gyro=True):
        self.cmd, self.face, self.cfg = cmd, face, cfg
        self.rover = Rover(*start, omega_quantum=LEADER_QUANTUM, omega_truncate=True,
                           turn_ratio=1.4)
        self.ctrl = DriveController(cfg)
        self.ctrl.set_heading_aligned(True)
        self.heading_bias, self.camera_min = heading_bias, camera_min
        self.gyro_sign, self.gyro = gyro_sign, gyro
        self.t, self.log, self.notes = 0.0, [], []
        self._due = {'cmd': 0.0, 'pose': 0.0, 'gyro': 0.0, 'scan': 0.0, 'ctrl': 0.0}

    def _every(self, name, period):
        if self.t + 1e-9 < self._due[name]:
            return False
        self._due[name] += period
        return True

    def tick(self):
        x, y, yaw = self.rover.tracked
        if self._every('cmd', 0.5):
            self.ctrl.on_command(self.cmd, self.t)
        if self._every('pose', 1.0 / 30.0):
            self.ctrl.on_pose(x, y, yaw + self.heading_bias, self.t)
        if self._every('gyro', 1.0 / 15.0) and self.gyro:
            self.ctrl.on_gyro(self.gyro_sign * self.rover.w, 1000.0 + self.t, self.t)
        if self._every('scan', 0.1):
            self.ctrl.on_depth_scan(self.face.scan((x, y, yaw), self.camera_min), self.t)
        if self._every('ctrl', 1.0 / self.cfg.control_rate_hz):
            out = self.ctrl.step(self.t)
            self.rover.command(out.v, out.w)
            self.log.append((self.t, out))
            self.notes += self.ctrl.take_notes()
        self.rover.advance(SIM_DT)
        self.t += SIM_DT

    def run(self, duration):
        end = self.t + duration
        while self.t < end:
            self.tick()
            out = self.log[-1][1] if self.log else None
            if out is not None and out.state in (st.HOLDING, st.PARKED, st.ERROR):
                return out
        return self.log[-1][1]

    def aligning(self):
        return [out for _, out in self.log if out.detail == FACE_ALIGN]

    def turned(self):
        return any(out.w != 0.0 for _, out in self.log)


GOAL = (2.0, 0.0, 0.0)


def _at_goal(alpha, dist=0.5, **kw):
    """Standing on the goal turned ``alpha`` from square to a face ``dist`` ahead of the
    camera; the global heading says it faces the goal heading (bias -alpha)."""
    cmd = make_cmd([GOAL], goal_pos_tol=0.05, goal_yaw_tol=0.10, pick_at_goal=True,
                   fixture_ahead=True)
    kw.setdefault('heading_bias', -alpha)
    return Rig(cmd, Face(GOAL[0] + CAMERA_X + dist), (GOAL[0], GOAL[1], alpha), **kw)


@pytest.mark.parametrize('alpha', [0.30, -0.30, 0.08, -0.08])
def test_squares_up_by_face_not_heading(alpha):
    rig = _at_goal(alpha)
    out = rig.run(15.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGNED)
    assert abs(rig.rover.yaw) <= TOL
    assert rig.aligning() and all(o.v == 0.0 for o in rig.aligning())
    assert all(o.approach_active for o in rig.aligning())
    turns = [n for n in rig.notes if 'turning' in n]
    assert 1 <= len(turns) <= 3 and f'{math.degrees(-alpha):+.1f} deg' in turns[0]
    assert any(n.startswith(FACE_ALIGNED) for n in rig.notes)


def test_already_square_arrives_no_turn():
    rig = _at_goal(0.01)
    out = rig.run(5.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGNED)
    assert not rig.turned()


def test_no_face_fails_without_turning():
    rig = _at_goal(0.2, dist=0.2)
    out = rig.run(5.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGN_FAILED)
    assert not rig.turned()
    assert any(FEW_POINTS in n for n in rig.notes)


def test_untrusted_angle_fails_no_turn():
    rig = _at_goal(0.65)
    out = rig.run(5.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGN_FAILED)
    assert not rig.turned()


def test_no_gyro_no_turn():
    rig = _at_goal(0.2, gyro=False)
    out = rig.run(5.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGN_FAILED)
    assert not rig.turned()


def test_reversed_gyro_stops_turn():
    rig = _at_goal(0.2, gyro_sign=-1.0)
    out = rig.run(10.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGN_FAILED)
    assert abs(rig.rover.yaw - 0.2) < 0.06
    assert any('wrong way' in n for n in rig.notes)


def test_lost_heading_holds_until_timeout():
    rig = _at_goal(0.2)
    rig.run(1.0)
    assert rig.log[-1][1].detail == FACE_ALIGN
    rig.ctrl.set_heading_aligned(False)
    out = rig.run(CFG.face_align_timeout_s + 1.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGN_FAILED)
    assert any('timeout' in n for n in rig.notes)


@pytest.mark.parametrize('cfg, fixture', [(drive_config('leader'), True), (CFG, False)])
def test_arrival_unchanged_without_align_fixture(cfg, fixture):
    cmd = make_cmd([GOAL], goal_pos_tol=0.05, goal_yaw_tol=0.10, pick_at_goal=True,
                   fixture_ahead=fixture)
    rig = Rig(cmd, Face(GOAL[0] + 0.5), (GOAL[0], GOAL[1], 0.2), cfg=cfg, heading_bias=-0.2)
    out = rig.run(3.0)
    assert (out.state, out.detail) == (st.HOLDING, '')
    assert out.detail == '' and not rig.aligning() and not rig.turned()
    assert rig.log[-1][0] < cfg.arrival_settle_s + 0.2


def _run_in(memory_s, face_yaw=0.2):
    """Straight 1 m run-in to a face skewed by ``face_yaw``; at the goal the face is 0.25 m
    from the camera, below its 0.45 m minimum: only the run-in saw it."""
    cfg = drive_config('leader', face_align=True, face_align_memory_s=memory_s)
    goal_x = 1.0
    face = Face(goal_x + CAMERA_X + 0.25, yaw=face_yaw)
    cmd = make_cmd(straight_path(goal_x), goal_pos_tol=0.05, goal_yaw_tol=0.10,
                   pick_at_goal=True, fixture_ahead=True)
    return Rig(cmd, face, (0.0, 0.0, 0.0), cfg=cfg, camera_min=0.45)


def test_close_face_uses_gyro_carried_fit():
    rig = _run_in(15.0)
    out = rig.run(40.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGNED)
    assert rig.rover.yaw == pytest.approx(0.2, abs=TOL)
    assert any('run-in fit' in n for n in rig.notes)


def test_close_face_without_memory_fails():
    rig = _run_in(0.0)
    out = rig.run(40.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGN_FAILED)
    assert not any(o.w != 0.0 for o in rig.aligning())


def test_a_new_command_ends_the_squaring_up():
    rig = _at_goal(0.3)
    rig.run(1.5)
    assert rig.log[-1][1].detail == FACE_ALIGN
    rig.cmd = make_cmd(straight_path(1.0, y=0.5), cmd_id=2)
    rig.run(1.0)
    out = rig.log[-1][1]
    assert out.cmd_id == 2 and out.detail != FACE_ALIGN and not out.approach_active


def test_flag_resend_keeps_outcome():
    """The host re-sends [goal] (PARKED -> HOLDING): the verdict and its detail stay."""
    rig = _at_goal(0.2, dist=0.2)
    rig.cmd = make_cmd([GOAL], goal_pos_tol=0.05, goal_yaw_tol=0.10, fixture_ahead=True)
    out = rig.run(5.0)
    assert (out.state, out.detail) == (st.PARKED, FACE_ALIGN_FAILED)
    rig.cmd = make_cmd([GOAL], cmd_id=2, goal_pos_tol=0.05, goal_yaw_tol=0.10,
                       pick_at_goal=True, fixture_ahead=True)
    for _ in range(int(0.6 / SIM_DT)):
        rig.tick()
    out = rig.log[-1][1]
    assert (out.state, out.detail, out.cmd_id) == (st.HOLDING, FACE_ALIGN_FAILED, 2)


def test_run_in_fit_survives_same_goal_only():
    """The host may send a new command to the same goal while the rover squares up (e.g. a
    flag change): the run-in's fit of that goal's face is kept, for another goal dropped."""
    rig = _run_in(15.0)
    while not rig.aligning():
        rig.tick()
    goal = rig.cmd.poses[-1]
    rig.cmd = make_cmd([goal], cmd_id=2, goal_pos_tol=0.05, goal_yaw_tol=0.10,
                       pick_at_goal=True, fixture_ahead=True)
    out = rig.run(15.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGNED)
    assert rig.rover.yaw == pytest.approx(0.2, abs=TOL)
    other = _run_in(15.0)
    while not other.aligning():
        other.tick()
    other.cmd = make_cmd([(goal[0] + 0.002, goal[1], goal[2])], cmd_id=2, goal_pos_tol=0.05,
                         goal_yaw_tol=0.10, pick_at_goal=True, fixture_ahead=True)
    out = other.run(15.0)
    assert (out.state, out.detail) == (st.HOLDING, FACE_ALIGN_FAILED)
