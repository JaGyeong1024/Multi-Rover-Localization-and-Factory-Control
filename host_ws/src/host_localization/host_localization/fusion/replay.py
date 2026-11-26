"""Replay recorded sensor events through RobotFusion the way fusion_node drives it.

Input: an npz fixture of ``<robot>_<kind>`` arrays (see COLUMNS), times [s] from bag start.
Each measurement calls its handler, then ``output(rx, rx)``.
"""
import math
from typing import NamedTuple

import numpy as np

from .frontend import Gating, RobotFusion
from .se2nhc import HOFF
from .tracker import KW, OBB_HEADING, WHEEL_RADIUS, WHEEL_STILL, ZERO_MOTION

# Fixture columns; NaN = not available.
COLUMNS = {
    'bev': ('rx', 'stamp', 'x', 'y', 'box_yaw'),
    'uwb': ('rx', 'stamp', 'x', 'y'),
    'uwb_corrected': ('rx', 'stamp', 'x', 'y'),
    'qf': ('rx', 'value'),
    'imu': ('rx', 'stamp', 'yaw', 'wz'),     # optional: ax, ay, az, wx, wy
    'pose': ('rx', 'stamp', 'x', 'y', 'yaw'),
    'global_position': ('rx', 'stamp', 'x', 'y', 'yaw'),
    'wheels': ('rx', 'stamp', 'v_left', 'v_right'),
    'wheel_pos': ('rx', 'stamp', 'left', 'right'),   # optional: joint_states wheel positions [rad]
    'quality': ('rx',),                       # the text is stored as <robot>_quality_text
}

# Event kinds in tie-break order.
_QF, _IMU, _UWB, _BEV, _WHEEL = range(5)


class ReplayResult(NamedTuple):
    """Per output step (where the node would publish /<robot>/pose)."""

    rx: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray       # [rad], NaN while the node publishes the identity orientation
    aligned: np.ndarray       # bool, heading anchored after this step
    quality: np.ndarray       # GOOD | DEGRADED | LOST at the step's receive time
    bev_rx: np.ndarray        # receive time of every BEV fix of the fixture
    bev_used: np.ndarray      # bool per BEV fix: its position was used by the filter
    sigma: np.ndarray         # [m] position 1-sigma (worst axis) after the step
    still: np.ndarray         # bool, wheel encoders say standing still at the step


def load_fixture(path):
    """Load a fixture (no pickle)."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def column(fixture, robot, kind, name):
    """One column of ``<robot>_<kind>``."""
    return fixture[f'{robot}_{kind}'][:, COLUMNS[kind].index(name)]


def _events(fixture, robot):
    """(rx, kind, row index) of every filter input in receive order."""
    parts = []
    for kind, key in ((_QF, 'qf'), (_IMU, 'imu'), (_UWB, 'uwb_corrected'), (_BEV, 'bev'), (_WHEEL, 'wheel_pos')):
        if kind == _WHEEL and f'{robot}_wheel_pos' not in fixture:
            continue                          # optional
        rx = column(fixture, robot, key, 'rx')
        parts.append(np.column_stack([rx, np.full(len(rx), kind), np.arange(len(rx))]))
    events = np.concatenate(parts)
    return events[np.argsort(events[:, 0], kind='stable')]


def _feed(fusion, kind, row, rx):
    """fusion_node's handlers: t = min(stamp, now), then the RobotFusion call."""
    if kind == _QF:
        fusion.on_qf(row[1])
        return None
    t = min(row[1], rx)
    if kind == _WHEEL:
        fusion.on_wheels(t, row[2], row[3])
        return None
    if kind == _IMU:
        if len(row) >= 9:                     # with accel and roll/pitch rates
            fusion.on_imu(t, row[2], row[3], row[4:7], math.hypot(row[7], row[8]))
        else:
            fusion.on_imu(t, row[2], row[3])
    elif kind == _UWB:
        fusion.on_uwb(t, rx, row[2], row[3])
    else:
        obb_yaw = None if math.isnan(row[4]) else row[4]
        return fusion.on_bev(t, rx, row[2], row[3], obb_yaw)
    return None


def make_fusion(robot, gating=Gating(), zero_motion=None, wheel_still=None):
    """RobotFusion configured like fusion_node's defaults (None = per-robot default)."""
    return RobotFusion(dict(KW.get(robot, KW['follower'])), use_obb=OBB_HEADING.get(robot, False),
                       gating=gating,
                       wheel_still=WHEEL_STILL.get(robot, False) if wheel_still is None else wheel_still,
                       zero_motion=ZERO_MOTION.get(robot, False) if zero_motion is None else zero_motion,
                       wheel_radius=WHEEL_RADIUS.get(robot))


def seed_recorded_heading(fusion, fixture, robot, sigma=0.1):
    """Anchor the heading from the recorded pose at the first IMU sample (sigma [rad]); uses EKFSE2 internals."""
    imu, pose = fixture[f'{robot}_imu'], fixture[f'{robot}_pose']
    heading = column(fixture, robot, 'pose', 'yaw')
    if not len(imu) or not len(pose):
        return
    # First recorded heading at/after the first IMU sample (NaN = not aligned yet: do not seed).
    valid = np.nonzero((pose[:, 0] >= imu[0, 0]) & ~np.isnan(heading))[0]
    if not len(valid):
        return
    k = int(valid[0])
    j = int(np.searchsorted(imu[:, 0], pose[k, 0]).clip(0, len(imu) - 1))
    est = fusion.est
    est._reanchor((heading[k] - imu[j, 2] + math.pi) % (2 * math.pi) - math.pi)
    est.P[HOFF, HOFF] = sigma ** 2
    est._cog_seen = True


def replay_robot(fixture, robot, fusion=None, seed_heading=True):
    """Replay one robot's inputs in receive order; seed_heading starts from the recorded anchor."""
    fusion = fusion or make_fusion(robot)
    if seed_heading:
        seed_recorded_heading(fusion, fixture, robot)
    tables = {_QF: fixture[f'{robot}_qf'], _IMU: fixture[f'{robot}_imu'],
              _UWB: fixture[f'{robot}_uwb_corrected'], _BEV: fixture[f'{robot}_bev']}
    if f'{robot}_wheel_pos' in fixture:
        tables[_WHEEL] = fixture[f'{robot}_wheel_pos']
    steps, bev_used = [], {}
    for rx, kind, idx in _events(fixture, robot):
        kind, idx = int(kind), int(idx)
        used = _feed(fusion, kind, tables[kind][idx], rx)
        if kind == _BEV:
            bev_used[idx] = used
        sample = None if kind in (_QF, _WHEEL) else fusion.output(rx, rx)
        if sample is not None:
            heading = math.nan if sample.heading is None else sample.heading
            steps.append((rx, sample.x, sample.y, heading, fusion.aligned, fusion.quality(rx),
                          fusion.est.position_sigma, fusion.trk.wheels_still))
    bev = tables[_BEV]
    flags = np.array([bev_used[i] for i in range(len(bev))], dtype=bool)
    rx, x, y, heading, aligned, quality, sigma, still = (np.array(col) for col in zip(*steps))
    return ReplayResult(rx, x, y, heading, aligned, quality, bev[:, 0].copy(), flags, sigma, still)
