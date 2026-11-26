"""Tracker + safety gate as one ``host_sim.scenario`` controller callback.

host_sim plays the base, localization, LiDAR and the leader's depth camera; the host is replaced
by a fixed path command re-sent at 2 Hz.
"""
import copy
import math

import numpy as np
from host_sim.config import load_config
from host_sim.core import World

from rover_drive.core.config import DepthFrontConfig
from rover_drive.core.controller import DriveController
from rover_drive.core.safety import SafetyGate

from sim import CMD_PERIOD, load_config as rover_config, quantize_omega

LEADER_QUANTUM = 0.0697


def make_world(seed=1, **robot_overrides):
    """Default host_sim world with per-rover overrides, e.g. leader={'start': [...]}.
    Keys other than ``start`` go into the rover's ``base`` section."""
    config = copy.deepcopy(load_config())
    # Pinned to the frame the tests were written in (independent of sim_default.yaml's site layout).
    config['arena'].update(x_min=1.24, x_max=6.03, y_min=1.97, y_max=4.66)
    config['robots']['leader']['start'] = [2.6, 3.3, 0.0]
    config['robots']['follower']['start'] = [1.9, 3.3, 0.0]
    for robot, values in robot_overrides.items():
        entry = config['robots'][robot]
        for key, value in values.items():
            if key == 'start':
                entry['start'] = value
            else:
                entry['base'][key] = value
    return World.from_config(config, seed=seed)


def depth_band_distance(ranges, angles, cfg):
    """Depth-camera scan beams -> the depth node's front distance (``depth_front.front_distance``).

    A beam at bearing b (CCW positive) is column u = cx - fx tan(b), i.e. width fraction
    1/2 - tan(b) / (2 tan(depth_hfov / 2)) for an ideal pinhole. The band rule uses depth along the
    optical axis Z = range cos(b); a beam without return (0.0) is a pixel without depth. Beams
    count as equal shares of the band (columns per degree differ ~5 % inside +-12.6 deg).
    """
    angles, ranges = np.asarray(angles, dtype=float), np.asarray(ranges, dtype=float)
    col = 0.5 - np.tan(angles) / (2.0 * math.tan(cfg.depth_hfov / 2.0))
    band = (col >= cfg.depth_col_min) & (col <= cfg.depth_col_max)
    depth = ranges[band] * np.cos(angles[band])
    valid = depth[np.isfinite(depth) & (depth >= cfg.depth_valid_min_m)
                  & (depth <= cfg.depth_valid_max_m)]
    if valid.size == 0 or valid.size < cfg.depth_min_valid_ratio * band.sum():
        return math.nan
    return float(np.percentile(valid, cfg.depth_percentile))


def shift_path(poses, x, y):
    """Translate a test path so that it starts at (x, y)."""
    return [(px + x, py + y, yaw) for px, py, yaw in poses]


class RoverStack:
    """DriveController -> SafetyGate, fed from host_sim observations."""

    def __init__(self, drive_cfg, gate_cfg, cmd, truncate_quantum=0.0, range_min=0.12):
        self.ctrl = DriveController(drive_cfg)
        self.gate = SafetyGate(gate_cfg)
        self.gate_cfg, self.cmd = gate_cfg, cmd
        # leader: /{robot}/front_distance comes from the depth camera via depth_front_node
        self.depth_cfg = (rover_config(DepthFrontConfig, gate_cfg.robot)
                          if gate_cfg.front_distance_topic else None)
        self.front = math.nan
        self.truncate_quantum = truncate_quantum
        self.range_min = range_min
        self.lead_gap_link = True
        self.log = []
        self._next_cmd = 0.0

    def __call__(self, obs):
        t = obs.t
        self._feed_pose(obs, t)
        self._feed_command(t)
        self._feed_front(obs, t)
        out = self.ctrl.step(t)
        self.gate.on_input(out.v, out.w, t)
        self.gate.on_tracker(out.state, out.detail, t)
        self.gate.on_approach(out.approach_active, t)
        gated = self.gate.step(t)
        self.ctrl.set_gate(gated.state, gated.reasons, gated.speed_scale, t)
        self.log.append((t, out.state, out.detail, gated.state, gated.reasons,
                         gated.blocked_x, gated.blocked_y))
        return gated.v, quantize_omega(gated.w, self.truncate_quantum, truncate=True)

    def _feed_pose(self, obs, t):
        if not obs.pose_is_new:
            return
        pose = obs.pose
        self.ctrl.set_heading_aligned(pose.heading_aligned)
        self.ctrl.set_quality(pose.quality)
        self.ctrl.on_pose(pose.x, pose.y, pose.theta, t)
        self.gate.set_aligned(pose.heading_aligned)
        self.gate.on_pose(pose.x, pose.y, pose.theta, t)
        lead = obs.poses.get(self.cmd.lead_robot)
        if lead is not None:
            self.ctrl.on_lead_pose(lead.x, lead.y, t)
            self.gate.on_lead_pose(lead.x, lead.y, t)

    def _feed_command(self, t):
        if t + 1e-9 < self._next_cmd:
            return
        self._next_cmd += CMD_PERIOD
        self.ctrl.on_command(self.cmd, t)
        self.gate.on_cmd(t, self.cmd.lead_robot, self.cmd)

    def _feed_front(self, obs, t):
        """LiDAR scan -> gate, or (leader) the depth scan reduced as depth_front_node does."""
        if self.depth_cfg is not None:
            if obs.depth_scan_is_new:
                self.front = depth_band_distance(obs.depth_scan, obs.depth_scan_angles,
                                                 self.depth_cfg)
                self.gate.on_front_distance(self.front, t)
                if self.gate_cfg.front_memory:
                    angles = obs.depth_scan_angles
                    self.gate.on_depth_scan(obs.depth_scan, float(angles[0]),
                                            float(angles[1] - angles[0]),
                                            self.depth_cfg.depth_valid_min_m, t)
            return
        if not obs.scan_is_new:
            return
        angles = obs.scan_angles
        self.gate.on_scan(obs.scan, float(angles[0]), float(angles[1] - angles[0]),
                          self.range_min, t)
        if self.gate.lead_name and self.lead_gap_link:
            self.ctrl.on_lead_gap(self.gate.lead_gap, t)

    def states(self):
        return [row[1] for row in self.log]

    def gate_reasons(self):
        return {reason for row in self.log for reason in row[4]}
