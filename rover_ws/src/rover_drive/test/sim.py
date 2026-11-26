"""Test helpers: YAML config loading, test paths and a small kinematic simulation.

The plant is a unicycle about the rotation centre. The controller sees the
*tracked point* (ref_offset_x ahead of the centre) at 30 Hz, delayed and noisy,
exactly like ``/{robot}/global/position``.
"""
import dataclasses
import math
import os
from collections import deque

import numpy as np
import yaml

from rover_drive.core import state as st
from rover_drive.core.config import DriveConfig, GateConfig
from rover_drive.core.controller import DriveController
from rover_drive.core.path import Path

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')
# Hardware keys live in the repository's config/hardware (next to rover_ws/).
_HARDWARE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'config',
                             'hardware')
SIM_DT = 0.005
POSE_PERIOD = 1.0 / 30.0
CMD_PERIOD = 0.5


def load_params(robot: str) -> dict:
    """Shipped parameters of a rover: config/<robot>.yaml plus the hardware file over it."""
    params = {}
    for folder, name in ((_CONFIG_DIR, robot), (_HARDWARE_DIR, 'rover_' + robot)):
        with open(os.path.join(folder, name + '.yaml')) as f:
            params.update(yaml.safe_load(f)['/**']['ros__parameters'])
    return params


def load_config(cls, robot: str, **overrides):
    """Build a config dataclass from the shipped parameters of a rover."""
    params = load_params(robot)
    names = {f.name for f in dataclasses.fields(cls)}
    values = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in params.items() if k in names}
    values.update(overrides)
    return cls(**values)


def drive_config(robot: str, **overrides) -> DriveConfig:
    return load_config(DriveConfig, robot, **overrides)


def gate_config(robot: str, **overrides) -> GateConfig:
    return load_config(GateConfig, robot, **overrides)


def straight_path(length: float, spacing: float = 0.05, y: float = 0.0) -> list:
    n = int(round(length / spacing))
    return [(i * spacing, y, 0.0) for i in range(n + 1)]


def s_path(radius: float = 0.6, lead_in: float = 0.5, spacing: float = 0.05) -> list:
    """Straight, left quarter turn, right quarter turn, straight run-in to the goal."""
    pts, x, y, yaw = [(0.0, 0.0)], 0.0, 0.0, 0.0
    arc = radius * math.pi / 2.0
    for seg_len, kappa in ((lead_in, 0.0), (arc, 1.0 / radius), (arc, -1.0 / radius),
                           (lead_in, 0.0)):
        for _ in range(int(round(seg_len / spacing))):
            yaw_mid = yaw + 0.5 * kappa * spacing
            x += spacing * math.cos(yaw_mid)
            y += spacing * math.sin(yaw_mid)
            yaw += kappa * spacing
            pts.append((x, y))
    path = Path(pts)
    return [(px, py, path.tangent(i * spacing)) for i, (px, py) in enumerate(pts)]


def make_cmd(poses, cmd_id: int = 1, epoch: int = 100, **kw) -> st.PathCmd:
    return st.PathCmd(epoch=epoch, cmd_id=cmd_id, poses=tuple(poses), **kw)


def quantize_omega(w: float, quantum: float, truncate: bool = False) -> float:
    """What a base with a coarse omega resolution executes for a commanded omega.

    ``truncate`` = cut toward zero (leader hardware: ``int(x * 100)``), else round.
    """
    if quantum <= 0.0:
        return w
    return (math.trunc(w / quantum) if truncate else round(w / quantum)) * quantum


class Rover:
    """Unicycle plant with optional slip, omega quantization and a tracked-point offset.

    ``turn_ratio`` = executed / commanded yaw rate after the quantization (skid-steer
    turning efficiency; 0 = the tracks do not turn the body, < 0 = reversed hardware).
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, ref_offset=0.0, slip=1.0, omega_quantum=0.0,
                 omega_truncate=False, turn_ratio=1.0):
        self.cx = x - ref_offset * math.cos(yaw)
        self.cy = y - ref_offset * math.sin(yaw)
        self.yaw = yaw
        self.ref_offset, self.slip, self.omega_quantum = ref_offset, slip, omega_quantum
        self.omega_truncate, self.turn_ratio = omega_truncate, turn_ratio
        self.v = self.w = 0.0

    def command(self, v: float, w: float) -> None:
        self.v = v * self.slip
        self.w = self.turn_ratio * quantize_omega(w, self.omega_quantum, self.omega_truncate)

    def advance(self, dt: float) -> None:
        self.cx += self.v * math.cos(self.yaw) * dt
        self.cy += self.v * math.sin(self.yaw) * dt
        self.yaw += self.w * dt

    @property
    def tracked(self) -> tuple:
        return (self.cx + self.ref_offset * math.cos(self.yaw),
                self.cy + self.ref_offset * math.sin(self.yaw), self.yaw)


class Sensor:
    """30 Hz pose stream with a transport delay and Gaussian noise."""

    def __init__(self, delay=0.08, pos_noise=0.01, yaw_noise=0.02, seed=0):
        self.delay, self.pos_noise, self.yaw_noise = delay, pos_noise, yaw_noise
        self._rng = np.random.default_rng(seed)
        self._queue = deque()
        self._next = 0.0

    def sample(self, t: float, pose: tuple) -> None:
        if t + 1e-9 < self._next:
            return
        self._next += POSE_PERIOD
        x, y, yaw = pose
        noisy = (x + self._rng.normal(0.0, self.pos_noise),
                 y + self._rng.normal(0.0, self.pos_noise),
                 yaw + self._rng.normal(0.0, self.yaw_noise))
        self._queue.append((t + self.delay, noisy))

    def deliver(self, t: float) -> list:
        out = []
        while self._queue and self._queue[0][0] <= t + 1e-9:
            out.append(self._queue.popleft()[1])
        return out


class Sim:
    """One rover + controller closed loop. Hooks let a test inject faults."""

    def __init__(self, cfg: DriveConfig, cmd: st.PathCmd, rover: Rover, sensor: Sensor = None,
                 aligned: bool = True):
        self.cfg, self.cmd, self.rover = cfg, cmd, rover
        self.sensor = sensor or Sensor()
        self.ctrl = DriveController(cfg)
        self.ctrl.set_heading_aligned(aligned)
        self.path = Path([(p[0], p[1]) for p in cmd.poses]) if cmd.poses else None
        self.t = 0.0
        self.out = None
        self.log = []
        self.pose_enabled = True
        self.cmd_enabled = True
        self.pose_hook = None
        self._next_ctrl = self._next_cmd = 0.0

    def _feed(self) -> None:
        self.sensor.sample(self.t, self.rover.tracked)
        for pose in self.sensor.deliver(self.t):
            if self.pose_hook:
                pose = self.pose_hook(self.t, pose)
            if self.pose_enabled:
                self.ctrl.on_pose(*pose, self.t)
        if self.t + 1e-9 >= self._next_cmd:
            self._next_cmd += CMD_PERIOD
            if self.cmd_enabled:
                self.ctrl.on_command(self.cmd, self.t)

    def _control(self) -> None:
        if self.t + 1e-9 < self._next_ctrl:
            return
        self._next_ctrl += 1.0 / self.cfg.control_rate_hz
        self.out = self.ctrl.step(self.t)
        self.rover.command(self.out.v, self.out.w)
        lateral = self.path.project(*self.rover.tracked[:2]).lateral if self.path else 0.0
        self.log.append((self.t, self.out.v, self.out.w, self.out.state, self.out.detail, lateral))

    def tick(self) -> None:
        self._feed()
        self._control()
        self.rover.advance(SIM_DT)
        self.t += SIM_DT

    def run(self, duration: float, until=None) -> None:
        """Run for ``duration`` seconds, or until ``until(sim)`` is true."""
        end = self.t + duration
        while self.t < end:
            self.tick()
            if until and self.out and until(self):
                return

    def goal_error(self) -> tuple:
        gx, gy, gyaw = self.cmd.poses[-1]
        x, y, yaw = self.rover.tracked
        yaw_err = math.atan2(math.sin(yaw - gyaw), math.cos(yaw - gyaw))
        return math.hypot(x - gx, y - gy), abs(yaw_err)

    def max_lateral(self, t_from: float = 0.0) -> float:
        return max(abs(row[5]) for row in self.log if row[0] >= t_from)

    def states(self) -> list:
        return [row[3] for row in self.log]


def latched(sim: Sim) -> bool:
    return sim.out.state in (st.HOLDING, st.PARKED, st.ERROR) or (
        sim.out.state == st.STOPPED and sim.out.detail in st.LATCHED_STOPS)
