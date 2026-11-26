"""Front memory: near depth-scan points keep stopping the leader after they entered the camera's
blind zone, while they are in the forward corridor.

The camera is a noise-free 80-beam scan (FOV 1.019 rad, range_min 0.30, blind zone 0.6 m: nearer
faces return 0.0). The band distance is the face's distance while visible, else the far wall (3 m).
"""
import math

import numpy as np
import pytest

from rover_drive.core.front_memory import CELL_M, CLEAR_MARGIN_M, FrontMemory
from rover_drive.core.limits import stopping_speed
from rover_drive.core.safety import LIMITED, OK, SafetyGate

from sim import gate_config

FOV, BEAMS = 1.019, 80
INC = FOV / BEAMS
ANGLE_MIN = -FOV / 2.0 + INC / 2.0
ANGLES = ANGLE_MIN + INC * np.arange(BEAMS)
MSG_RANGE_MIN = 0.30
BLIND = 0.60
WALL = 3.0
DT = 0.1
V_IN = 0.15


def _scan(pose, face=None, blind=BLIND):
    """Ranges seen from ``pose`` (x, y, yaw); ``face`` = (x, y_lo, y_hi), a global face
    perpendicular to +x. Background: a wall WALL ahead of the camera."""
    px, py, yaw = pose
    a = ANGLES + yaw
    ranges = np.full(BEAMS, np.inf)
    cos = np.cos(a)
    with np.errstate(divide='ignore'):
        wall = np.where(cos > 0.0, WALL / np.maximum(cos, 1e-9), np.inf)
        ranges = wall
        if face is not None:
            fx, lo, hi = face
            t = np.where(cos > 0.0, (fx - px) / np.maximum(cos, 1e-9), np.inf)
            y_hit = py + t * np.sin(a)
            hit = (t > 0.0) & (y_hit >= lo) & (y_hit <= hi)
            ranges = np.where(hit, t, ranges)
    return np.where((ranges < blind) | ~np.isfinite(ranges), 0.0, ranges)


def _band(pose, face):
    """Band distance (depth node) for the face straight ahead: its distance or the far wall."""
    if face is None:
        return WALL
    dist = face[0] - pose[0]
    return dist if dist >= BLIND else WALL


def _gate(memory, **overrides):
    overrides.setdefault('lidar_resume_s', 0.0)
    overrides.setdefault('max_linear_speed', V_IN)
    return SafetyGate(gate_config('leader', front_memory=memory, **overrides))


def _feed(gate, pose, t, face, aligned=True, v_in=V_IN, w_in=0.0, scan=True, pose_ok=True,
          reported=None):
    """``reported`` = the pose the localization reports instead of the true one."""
    if pose_ok:
        gate.on_pose(*(reported or pose), t)
    gate.set_aligned(aligned)
    gate.on_cmd(t)
    gate.on_input(v_in, w_in, t)
    gate.on_front_distance(_band(pose, face), t)
    if scan:
        gate.on_depth_scan(_scan(pose, face), ANGLE_MIN, INC, MSG_RANGE_MIN, t)
    return gate.step(t)


def _drive(gate, face, start_dist=0.9, steps=80, **kwargs):
    """Drive straight along +x toward ``face``; the base executes the gate's output one tick
    late (reaction). Rows (distance to the face, gate output)."""
    x, rows, v = face[0] - start_dist, [], 0.0
    for i in range(steps):
        out = _feed(gate, (x, 0.0, 0.0), i * DT, face, **kwargs)
        rows.append((face[0] - x, out))
        x, v = x + v * DT, out.v
    return rows


# Partly covering box: the left 0.16 m of the 0.18 m half corridor.
PARTIAL = (3.0, 0.02, 0.60)


def test_partial_box_stops_with_memory():
    cfg = gate_config('leader')
    rows = _drive(_gate(True), PARTIAL)

    def speed(dist):
        return min(V_IN, stopping_speed(dist, cfg.lidar_stop_base, cfg.lidar_brake_decel,
                                        cfg.lidar_reaction_s))
    for dist, out in rows:
        assert out.v <= speed(dist + CELL_M) + 1e-9
        if 'LIDAR_STOP' not in out.reasons:
            assert out.v >= speed(dist - CELL_M) - 1e-9
    assert any(0.0 < o.v < V_IN for _, o in rows)
    stopped = [(d, o) for d, o in rows if 'LIDAR_STOP' in o.reasons]
    assert stopped
    dist, out = stopped[0]
    assert cfg.lidar_stop_base - CELL_M <= dist <= cfg.lidar_stop_base + CELL_M
    assert dist < BLIND
    assert rows[-1][0] >= cfg.lidar_stop_base - CELL_M
    assert PARTIAL[1] - CELL_M <= out.blocked_y <= cfg.lidar_half_width + CELL_M
    assert out.blocked_x == pytest.approx(PARTIAL[0], abs=CELL_M)


def test_partial_box_ignored_without_memory():
    rows = _drive(_gate(False), PARTIAL)
    assert not any('LIDAR_STOP' in o.reasons for _, o in rows)
    assert rows[-1][0] < 0.0


def test_remembered_point_off_corridor_no_stop():
    cfg = gate_config('leader')
    beside = (3.0, cfg.lidar_half_width + CELL_M, 0.60)
    rows = _drive(_gate(True), beside)
    assert all(o.v == V_IN and o.state == OK for _, o in rows)


def _remember_box(gate, at=0.55):
    """A box seen at ``at`` straight ahead (camera blind zone set below it)."""
    face = (at, -0.05, 0.05)
    gate.on_pose(0.0, 0.0, 0.0, 0.0)
    gate.set_aligned(True)
    gate.on_cmd(0.0)
    gate.on_input(V_IN, 0.0, 0.0)
    gate.on_front_distance(WALL, 0.0)
    gate.on_depth_scan(_scan((0.0, 0.0, 0.0), face, blind=MSG_RANGE_MIN), ANGLE_MIN, INC,
                       MSG_RANGE_MIN, 0.0)
    return face


def _step_with(gate, t, ranges, pose=(0.0, 0.0, 0.0)):
    gate.on_pose(*pose, t)
    gate.on_cmd(t)
    gate.on_input(V_IN, 0.0, t)
    gate.on_front_distance(WALL, t)
    gate.on_depth_scan(ranges, ANGLE_MIN, INC, MSG_RANGE_MIN, t)
    return gate.step(t)


def test_point_forgotten_only_when_beyond():
    gate = _gate(True)
    _remember_box(gate)
    limited = gate.step(0.0)
    assert limited.state == LIMITED and limited.v < V_IN
    no_return = np.zeros(BEAMS)
    assert _step_with(gate, 0.1, no_return).v == limited.v
    just_behind = _scan((0.0, 0.0, 0.0), (0.55 + CLEAR_MARGIN_M / 2.0, -0.05, 0.05),
                        blind=MSG_RANGE_MIN)
    assert _step_with(gate, 0.2, just_behind).v < V_IN
    assert _step_with(gate, 0.3, _scan((0.0, 0.0, 0.0))).v == V_IN
    assert _step_with(gate, 0.4, no_return).v == V_IN


def test_blind_or_out_of_view_point_kept():
    memory = FrontMemory(0.70)
    memory.update(_scan((0.0, 0.0, 0.0), (0.5, -0.05, 0.05), blind=0.3), ANGLE_MIN, INC,
                  MSG_RANGE_MIN)
    before = len(memory)
    assert before
    far = _scan((0.0, 0.0, 0.0), blind=0.3)
    memory.move(0.35, 0.0, 0.0)
    memory.update(far, ANGLE_MIN, INC, MSG_RANGE_MIN)
    assert len(memory) == before
    memory.move(-0.35, 0.0, 0.0)
    memory.move(0.0, 0.0, 1.2)
    memory.update(far, ANGLE_MIN, INC, MSG_RANGE_MIN)
    assert len(memory) == before
    memory.move(0.0, 0.0, -1.2)
    memory.update(far, ANGLE_MIN, INC, MSG_RANGE_MIN)
    assert len(memory) == 0


def test_moving_the_memory_follows_the_rover():
    memory = FrontMemory(0.70)
    memory.update(_scan((0.0, 0.0, 0.0), (0.5, -0.01, 0.01), blind=0.3), ANGLE_MIN, INC,
                  MSG_RANGE_MIN)
    memory.move(0.1, 0.0, math.pi / 2)
    x, y = memory.points()[0]
    assert (x, y) == pytest.approx((0.0, -0.4), abs=0.011)


def test_turned_away_forgets_points_behind():
    gate = _gate(True)
    _remember_box(gate)
    assert gate.step(0.0).v < V_IN
    no_return = np.zeros(BEAMS)
    assert _step_with(gate, 0.1, no_return, pose=(0.0, 0.0, math.pi / 2)).v == V_IN
    assert _step_with(gate, 0.2, no_return, pose=(0.0, 0.0, math.pi)).v == V_IN
    assert _step_with(gate, 0.3, no_return).v == V_IN


def _outputs(rows):
    return [(d, tuple(o)) for d, o in rows]


def _same(a, b):
    return len(a) == len(b) and all(
        da == db and all(x == y or (math.isnan(x) and math.isnan(y)) if isinstance(x, float)
                         else x == y for x, y in zip(oa, ob))
        for (da, oa), (db, ob) in zip(a, b))


def test_unaligned_memory_stops_by_dead_reckoning():
    # Cold start: heading not aligned, reported pose meaningless; only slow straight commands pass.
    cfg = gate_config('leader')
    rows = _drive(_gate(True), PARTIAL, aligned=False, v_in=0.10, reported=(5.0, 5.0, 0.0))
    # Dead reckoning follows the passed command, so the remembered distance closes on the stop line;
    # within a memory cell of it the gate stops (LIDAR_SLOW would never be re-planned).
    dist, out = rows[-1]
    assert out.v == 0.0 and dist < BLIND
    assert cfg.lidar_stop_base - CELL_M <= dist <= cfg.lidar_stop_base + 2 * CELL_M
    assert all(d >= cfg.lidar_stop_base - CELL_M for d, _ in rows)
    assert 'LIDAR_STOP' in out.reasons
    assert math.isnan(out.blocked_x)
    without = _drive(_gate(False), PARTIAL, aligned=False, v_in=0.10, reported=(5.0, 5.0, 0.0))
    assert not any('LIDAR_STOP' in o.reasons for _, o in without)
    assert without[-1][0] < cfg.lidar_stop_base - 0.1


def test_option_off_ignores_the_depth_scan():
    fed = _outputs(_drive(_gate(False), PARTIAL))
    not_fed = _outputs(_drive(_gate(False), PARTIAL, scan=False))
    assert _same(fed, not_fed)


def test_option_off_in_shipped_configs():
    assert gate_config('leader').front_memory is False
    assert gate_config('follower').front_memory is False
