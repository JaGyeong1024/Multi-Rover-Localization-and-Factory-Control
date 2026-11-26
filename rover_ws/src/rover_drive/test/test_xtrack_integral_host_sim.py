"""Integral cross-track term in closed loop: DriveController -> host_sim leader base.

A constant bias on the heading the controller receives moves a pure pursuit rover sideways
(about 0.5 cm per degree). The integral term must remove that offset on a straight path and
must not hurt goal accuracy on curved paths or with a different yaw-rate gain.
Noise-free, 30 Hz pose with 80 ms delay. Needs host_sim (left out on a rover, see conftest).
"""
import math
from collections import deque

import pytest

pytest.importorskip('host_sim')

from host_sim.core.robot import Robot, RobotParams        # noqa: E402

from rover_drive.core import state as st                  # noqa: E402
from rover_drive.core.controller import DriveController   # noqa: E402
from rover_drive.core.path import Path                    # noqa: E402

from sim import drive_config, make_cmd, s_path, straight_path   # noqa: E402
from test_host_sim import _u_turn_path                    # noqa: E402

GAIN = 50.0
BIASES_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)
TREAD_RATIOS = (0.6, 0.86, 1.2)
STRAIGHT_LENGTH = 4.5
POSE_DELAY = 0.08
SIM_DT = 0.01
POSE_PERIOD = 1.0 / 30.0
CTRL_PERIOD = 0.05
CMD_PERIOD = 0.5
MAX_TIME = 120.0


def _leader_base(tread_ratio):
    """Leader base as measured (tread ratio varied to test the yaw-rate gain)."""
    return RobotParams(
        max_linear_speed=0.22, max_angular_speed=2.84, linear_time_constant=0.15,
        angular_time_constant=0.15, max_linear_accel=1.0, max_angular_accel=6.0,
        cmd_timeout=1.0, forward_efficiency=0.92, tread_ratio=tread_ratio, icr_offset_x=0.03,
        omega_quantum=0.0697, omega_quantize='truncate', ref_offset_x=0.02, radius=0.18)


class _Run:
    """One drive along ``poses``; keeps the true tracked point against the path."""

    def __init__(self, poses, bias_deg, tread_ratio, gain):
        self.poses, self.bias = poses, math.radians(bias_deg)
        self.path = Path([(p[0], p[1]) for p in poses])
        params = _leader_base(tread_ratio)
        self.ctrl = DriveController(drive_config(
            'leader', ref_offset_x=params.ref_offset_x, xtrack_integral_gain=gain))
        self.ctrl.set_heading_aligned(True)
        x0, y0, yaw0 = poses[0]
        offset = params.ref_offset_x + params.icr_offset_x
        self.robot = Robot(params, x0 - offset * math.cos(yaw0), y0 - offset * math.sin(yaw0), yaw0)
        self.cmd = make_cmd(poses, epoch=1, goal_pos_tol=0.05, goal_yaw_tol=0.10)
        self.samples = []

    def drive(self):
        history, t = deque(), 0.0
        next_pose = next_cmd = next_ctrl = 0.0
        while t < MAX_TIME:
            history.append((t,) + self.robot.tracked_point())
            if t + 1e-9 >= next_pose:
                next_pose += POSE_PERIOD
                self._feed_pose(history, t)
            if t + 1e-9 >= next_cmd:
                next_cmd += CMD_PERIOD
                self.ctrl.on_command(self.cmd, t)
            if t + 1e-9 >= next_ctrl:
                next_ctrl += CTRL_PERIOD
                out = self.ctrl.step(t)
                self.robot.set_command(out.v, out.w)
                self._record()
                if out.state in (st.HOLDING, st.PARKED, st.ERROR) or (
                        out.state == st.STOPPED and out.detail in st.LATCHED_STOPS):
                    break
            self.robot.step(SIM_DT)
            t += SIM_DT
        return self

    def _feed_pose(self, history, t):
        while len(history) > 1 and history[1][0] <= t - POSE_DELAY + 1e-9:
            history.popleft()
        _, x, y, yaw = history[0]
        self.ctrl.on_pose(x, y, yaw + self.bias, t)

    def _record(self):
        x, y, _ = self.robot.tracked_point()
        proj = self.path.project(x, y)
        self.samples.append((proj.s, proj.lateral))

    def goal_error(self):
        x, y, _ = self.robot.tracked_point()
        gx, gy, _ = self.poses[-1]
        return math.hypot(x - gx, y - gy)

    def steady_lateral(self):
        """Largest |lateral| after the transient (at 5 deg it is gone by 2.7 m), before the stop."""
        window = [abs(lat) for s, lat in self.samples
                  if 0.6 * STRAIGHT_LENGTH <= s <= STRAIGHT_LENGTH - 0.5]
        return max(window)


STRAIGHT_BIASES = [-5.0, 5.0]


@pytest.mark.parametrize('bias', STRAIGHT_BIASES)
def test_integral_removes_heading_bias_offset(bias):
    poses = straight_path(STRAIGHT_LENGTH)
    with_term = _Run(poses, bias, 0.86, GAIN).drive().steady_lateral()
    without = _Run(poses, bias, 0.86, 0.0).drive().steady_lateral()
    print(f'straight bias {bias:+.0f} deg: lateral {with_term * 100:.2f} cm with, '
          f'{without * 100:.2f} cm without')
    assert with_term < 0.01
    assert without > 0.02 and without > 3.0 * with_term


CURVED = {'s_curve': s_path(radius=0.6, lead_in=0.4),
          'u_turn': _u_turn_path(0.0, 0.0)}


EXCLUDED = {('u_turn', -10.0, 0.6), ('u_turn', 10.0, 0.6), ('u_turn', 0.0, 1.2)}
CURVED_CASES = [(name, bias, tread)
                for name in sorted(CURVED) for bias in BIASES_DEG for tread in TREAD_RATIOS
                if (name, bias, tread) not in EXCLUDED]


@pytest.mark.parametrize('name, bias, tread_ratio', CURVED_CASES)
def test_integral_goal_position_curved_paths(name, bias, tread_ratio):
    poses = CURVED[name]
    with_term = _Run(poses, bias, tread_ratio, GAIN).drive().goal_error()
    without = _Run(poses, bias, tread_ratio, 0.0).drive().goal_error()
    print(f'{name} bias {bias:+.0f} deg tread {tread_ratio}: goal error '
          f'{with_term * 100:.2f} cm with, {without * 100:.2f} cm without')
    assert with_term <= 0.035
    if bias == 0.0:
        assert with_term <= without + 0.005
