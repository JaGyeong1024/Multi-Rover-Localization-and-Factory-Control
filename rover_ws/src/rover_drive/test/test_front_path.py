"""Path stop zone (``front_path``): the front stop looks along the current path command.

The rover stands at ``POSE`` (an arbitrary heading, so frame mistakes show); paths and
objects are written in its body frame (x forward, y left) and placed with it. The follower
cases use a G6-like scan, a far room (4 m) plus the objects painted in; the leader cases use
the band distance and the depth scan of front memory.
"""
import dataclasses
import math

import numpy as np
import pytest

from rover_drive.core.corridor import XTRACK_TOL, PathCorridor
from rover_drive.core.limits import stopping_speed
from rover_drive.core.safety import LIMITED, OK, STOPPED, SafetyGate

from sim import gate_config, make_cmd

BEAMS = 1830
INC = 2.0 * math.pi / BEAMS
ANGLE_MIN = -math.pi
RANGE_MIN = 0.12
ROOM = 4.0
POSE = (1.5, 2.5, 0.4)
DT = 0.1


def _follower_cfg(**overrides):
    # Scans are painted in the body frame (angle 0 = ahead), i.e. an unrotated LiDAR mount.
    overrides.setdefault('lidar_yaw_offset', 0.0)
    return gate_config('follower', **overrides)


def _glob(bx, by, pose=POSE):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pose[0] + c * bx - s * by, pose[1] + s * bx + c * by


def _poses(body_xy, pose=POSE):
    """Body-frame path points -> global path poses (theta along the path, goal keeps the
    last direction)."""
    pts = [_glob(x, y, pose) for x, y in body_xy]
    yaws = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(pts, pts[1:])]
    yaws.append(yaws[-1] if yaws else pose[2])
    return [(x, y, yaw) for (x, y), yaw in zip(pts, yaws)]


def _straight(length, start=0.0, spacing=0.05):
    n = int(round((length - start) / spacing))
    return [(start + i * spacing, 0.0) for i in range(n + 1)]


def _arc(radius, length, left=True, spacing=0.05, then=0.0):
    """Arc from the rover along its heading, then ``then`` metres straight on."""
    sign = 1.0 if left else -1.0
    pts = [(radius * math.sin(a), sign * radius * (1.0 - math.cos(a)))
           for a in (i * spacing / radius for i in range(int(round(length / spacing)) + 1))]
    end = length / radius
    for i in range(1, int(round(then / spacing)) + 1):
        pts.append((pts[-1][0] + spacing * math.cos(end), pts[-1][1] + sign * spacing
                    * math.sin(end)))
    return pts


def _corner(first, second=1.0, spacing=0.05):
    """``first`` metres ahead, then a quarter turn to the left and ``second`` metres on."""
    pts = [(min(i * spacing, first), 0.0) for i in range(int(math.ceil(first / spacing)) + 1)]
    pts += [(first, i * spacing) for i in range(1, int(round(second / spacing)) + 1)]
    return pts


def _cmd(body_xy, cmd_id=1, pose=POSE, **kw):
    return make_cmd(_poses(body_xy, pose), cmd_id=cmd_id, **kw)


def _beam(angle):
    return int(round((angle - ANGLE_MIN) / INC)) % BEAMS


def _scan(points=(), room=ROOM):
    """The room with single-beam objects at body-frame ``points``."""
    scan = np.full(BEAMS, room)
    for x, y in points:
        scan[_beam(math.atan2(y, x))] = math.hypot(x, y)
    return scan


def _measured(x, y):
    """Where the scan puts a point painted at (x, y): on its beam's centre."""
    a = ANGLE_MIN + INC * _beam(math.atan2(y, x))
    r = math.hypot(x, y)
    return r * math.cos(a), r * math.sin(a)


def _follower(front_path=True, **overrides):
    overrides.setdefault('lidar_resume_s', 0.0)
    return SafetyGate(_follower_cfg(front_path=front_path, **overrides))


def _step(gate, scan, cmd, t, pose=POSE, v=0.20, w=0.0, aligned=True, pose_age=0.0):
    gate.on_pose(*pose, t - pose_age)
    gate.set_aligned(aligned)
    gate.on_cmd(t, '', cmd)
    gate.on_input(v, w, t)
    gate.on_scan(scan, ANGLE_MIN, INC, RANGE_MIN, t)
    return gate.step(t)


def _same(a, b):
    assert (a.state, a.reasons) == (b.state, b.reasons)
    for x, y in zip(a[:2] + a[4:], b[:2] + b[4:]):
        assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, abs=1e-9)


def _limit(v_allowed, w=0.0, v_in=0.20):
    return pytest.approx((v_allowed, w * v_allowed / v_in), abs=1e-6)


def _random_scan(rng, half_width, half_angle):
    """Room at a random range, a few small objects ahead, sometimes too few front returns
    (blind). Returns inside the box but outside the front sector are left out: those count
    on the path only (test_points_beside_corners_count)."""
    scan = np.full(BEAMS, rng.uniform(1.0, 2.5))
    for _ in range(rng.integers(0, 5)):
        x, y = rng.uniform(0.02, 0.6), rng.uniform(-0.3, 0.3)
        centre = _beam(math.atan2(y, x))
        scan[centre - rng.integers(0, 6):centre + rng.integers(1, 7)] = math.hypot(x, y)
    angle = ANGLE_MIN + INC * np.arange(BEAMS)
    bx, by = scan * np.cos(angle), scan * np.sin(angle)
    scan[(bx > 0.0) & (np.abs(by) <= half_width) & (np.abs(angle) > half_angle)] = 0.0
    if rng.random() < 0.25:
        front = np.flatnonzero(np.abs(angle) <= half_angle)
        scan[rng.choice(front, size=front.size - int(rng.integers(0, 5)), replace=False)] = 0.0
    return scan


def test_straight_path_equals_box_output():
    cfg = _follower_cfg()
    rng = np.random.default_rng(7)
    for _ in range(150):
        pose = (rng.uniform(1.0, 5.0), rng.uniform(1.0, 5.0), rng.uniform(-math.pi, math.pi))
        cmd = _cmd(_straight(3.0, start=-rng.uniform(0.0, 0.5)), pose=pose)
        path_gate, box_gate = SafetyGate(_follower_cfg(front_path=True)), \
            SafetyGate(cfg)
        for k in range(6):
            scan, t = _random_scan(rng, cfg.lidar_half_width, cfg.lidar_front_half_angle), k * DT
            v, w = rng.uniform(0.0, 0.25), rng.uniform(-0.5, 0.5)
            _same(_step(path_gate, scan, cmd, t, pose, v, w),
                  _step(box_gate, scan, cmd, t, pose, v, w))


def test_points_beside_corners_count():
    # Inside the rover's width but outside the +-34 deg sector: the straight box never saw
    # it, the path corridor does (the front corner drives into it).
    cmd, scan = _cmd(_straight(2.0)), _scan([(0.10, 0.13)])
    assert _step(_follower(front_path=False), scan, cmd, 0.0).state == OK
    out = _step(_follower(), scan, cmd, 0.0)
    assert out.state == STOPPED and 'LIDAR_STOP' in out.reasons


def test_curve_point_limits_at_path_distance():
    # Corner 0.10 ahead: the point (0.10, 0.17) is 0.27 m along the path, outside the box
    # (|y| > 0.15) and the sector.
    cfg, cmd, point = _follower_cfg(), _cmd(_corner(0.10)), (0.10, 0.17)
    assert _step(_follower(front_path=False), _scan([point]), cmd, 0.0).state == OK
    out = _step(_follower(), _scan([point]), cmd, 0.0)
    mx, my = _measured(*point)
    along = 0.10 + my
    assert out.reasons == ('LIDAR_SLOW',)
    assert (out.v, out.w) == _limit(stopping_speed(along - cfg.lidar_to_front, cfg.lidar_stop_base,
                                                   cfg.lidar_brake_decel, cfg.lidar_reaction_s))


def test_curve_point_stops_and_reports():
    cmd, point = _cmd(_corner(0.02)), (0.02, 0.17)
    assert _step(_follower(front_path=False), _scan([point]), cmd, 0.0).state == OK
    out = _step(_follower(), _scan([point]), cmd, 0.0)
    assert out.state == STOPPED and out.reasons == ('LIDAR_STOP',)
    assert (out.blocked_x, out.blocked_y) == pytest.approx(_glob(*_measured(*point)), abs=1e-6)


def test_ahead_but_off_curve_not_counted():
    # Left arc of the planner's tightest radius: (0.25, -0.10) is in the straight box
    # (LIDAR_SLOW there) and 0.17 m outside the arc (half width 0.15).
    cmd, scan = _cmd(_arc(0.30, 0.45, then=1.0)), _scan([(0.25, -0.10)])
    assert _step(_follower(front_path=False), scan, cmd, 0.0).reasons == ('LIDAR_SLOW',)
    out = _step(_follower(), scan, cmd, 0.0)
    assert out.state == OK and (out.v, out.w) == (0.20, 0.0)


def test_corner_outer_side_counts():
    # Left corner 0.25 ahead; (0.30, -0.05) is past the first leg and before the second:
    # reached when the rover is at the corner (0.25 along), not later.
    cfg, cmd, point = _follower_cfg(), _cmd(_corner(0.25)), (0.30, -0.05)
    out = _step(_follower(), _scan([point]), cmd, 0.0)
    assert out.reasons == ('LIDAR_SLOW',)
    assert (out.v, out.w) == _limit(stopping_speed(0.25 - cfg.lidar_to_front, cfg.lidar_stop_base,
                                                   cfg.lidar_brake_decel, cfg.lidar_reaction_s))


def test_fixture_beyond_goal_no_stop():
    # Straight run-in of 0.40 m; the shelf face 0.10 beyond the front edge at the goal.
    # No approach exemption: front_pick_exempt off and approach_active never sent.
    cfg = _follower_cfg()
    face = 0.40 + cfg.lidar_to_front + 0.10
    cmd = _cmd(_straight(0.40))
    shelf = [(face, y) for y in np.linspace(-0.3, 0.3, 31)]
    seen = {True: set(), False: set()}
    for front_path in (True, False):
        gate = _follower(front_path=front_path, front_pick_exempt=False)
        for k, travelled in enumerate(np.linspace(0.0, 0.40, 21)):
            pose = (*_glob(travelled, 0.0), POSE[2])
            body = [(x - travelled, y) for x, y in shelf]
            out = _step(gate, _scan(body), cmd, k * DT, pose)
            seen[front_path] |= set(out.reasons)
    assert not seen[True] & {'LIDAR_STOP', 'LIDAR_SLOW'}
    assert 'LIDAR_STOP' in seen[False]


def test_front_edge_point_at_goal_counts():
    cfg = _follower_cfg()
    cmd = _cmd(_straight(0.10))
    edge = 0.10 + cfg.lidar_to_front - 0.01
    out = _step(_follower(), _scan([(edge, 0.0)]), cmd, 0.0)
    assert out.state == STOPPED
    beyond = _step(_follower(), _scan([(edge + 0.03, 0.0)]), cmd, 0.0)
    assert beyond.state == OK


@pytest.mark.parametrize('pick_exempt', [True, False])
def test_approach_exemption_needs_pick_exempt(pick_exempt):
    # An object on the run-in itself (not beyond the goal) during approach_active.
    cmd, scan = _cmd(_straight(0.40)), _scan([(0.15, 0.0)])
    gate = _follower(front_pick_exempt=pick_exempt)
    gate.on_approach(True, 0.0)
    out = _step(gate, scan, cmd, 0.0)
    assert (out.state == OK) is pick_exempt


def _pivot_cmd(body_xy):
    return dataclasses.replace(_cmd(body_xy), pivot_first=True, pivot_heading=POSE[2])


FALLBACKS = {
    # name: (pose, aligned, command or None, pose age) - the straight box must decide
    'not aligned': (POSE, False, 'arc', 0.0),
    'no command': (POSE, True, None, 0.0),
    'stop command': (POSE, True, 'stop', 0.0),
    'stale pose': (POSE, True, 'arc', 1.0),
    'off the path': ((*_glob(0.0, -(XTRACK_TOL + 0.03)), POSE[2]), True, 'arc', 0.0),
    'heading off the path': ((POSE[0], POSE[1], POSE[2] - 0.5), True, 'arc', 0.0),
    'leading pivot': ((POSE[0], POSE[1], POSE[2] - math.pi / 2.0), True, 'pivot', 0.0),
}


# Close point in the straight box (LIDAR_STOP), 0.17 m beside the arc; unaligned input passes the creep.
FALLBACK_SCAN = [(0.20, -0.13)]
FALLBACK_ARC = _arc(0.30, 0.45, then=1.0)
FALLBACK_V = 0.12


def test_fallback_cases_where_zones_differ():
    cmd, scan = _cmd(FALLBACK_ARC), _scan(FALLBACK_SCAN)
    assert _step(_follower(), scan, cmd, 0.0, v=FALLBACK_V).state == OK
    assert _step(_follower(front_path=False), scan, cmd, 0.0, v=FALLBACK_V).reasons == \
        ('LIDAR_STOP',)


@pytest.mark.parametrize('case', sorted(FALLBACKS))
def test_no_path_equals_box_output(case):
    pose, aligned, kind, pose_age = FALLBACKS[case]
    cmd = {'arc': _cmd(FALLBACK_ARC), 'stop': _cmd(FALLBACK_ARC, mode='STOP'),
           'pivot': _pivot_cmd(FALLBACK_ARC), None: None}[kind]
    scan = _scan(FALLBACK_SCAN)
    path_gate, box_gate = _follower(), _follower(front_path=False)
    for k in range(4):
        t = 1.0 + k * DT
        _same(*(_step(g, scan, cmd, t, pose, FALLBACK_V, aligned=aligned, pose_age=pose_age)
                for g in (path_gate, box_gate)))


def test_stale_or_rejected_command_no_zone():
    # Only commands the tracker accepts keep the zone fresh: an old host's (rejected) ones
    # keep STALE_CMD away but not the stale path.
    cmd, old, scan = _cmd(FALLBACK_ARC, cmd_id=3), _cmd(FALLBACK_ARC, cmd_id=2), \
        _scan(FALLBACK_SCAN)
    path_gate, box_gate = _follower(), _follower(front_path=False)
    for g in (path_gate, box_gate):
        _step(g, scan, cmd, 0.0, v=FALLBACK_V)
    t = 2.0
    _same(*(_step(g, scan, old, t, v=FALLBACK_V) for g in (path_gate, box_gate)))


def test_jumped_pose_falls_back_to_box():
    # One sample 0.3 m ahead moves the (monotonic) progress; the rover is then behind it,
    # farther from the path than the tolerance: never a zone that starts ahead of the rover.
    cmd, scan = _cmd(_straight(2.0)), _scan([(0.25, -0.10), (0.15, 0.0)])
    path_gate, box_gate = _follower(), _follower(front_path=False)
    for gate in (path_gate, box_gate):
        _step(gate, scan, cmd, 0.0)
        gate.on_pose(*_glob(0.30, 0.0), POSE[2], 0.05)
    _same(_step(path_gate, scan, cmd, 0.1), _step(box_gate, scan, cmd, 0.1))


def test_gap_to_first_pose_is_in_zone():
    # Path starts 0.08 ahead (planned from where the rover stood a moment ago); the point
    # beside the front corner lies between the rover and that first pose.
    cmd = _cmd(_straight(2.0, start=0.08))
    out = _step(_follower(), _scan([(0.06, 0.12)]), cmd, 0.0)
    assert out.state == STOPPED


def test_zone_follows_tracker_acceptance():
    arc = _arc(0.30, 0.45, then=1.0)
    straight = _straight(2.0)
    scan = _scan([(0.25, -0.10)])
    gate = _follower()
    assert _step(gate, scan, _cmd(arc, cmd_id=5), 0.0).state == OK
    old = _cmd(straight, cmd_id=4)
    assert _step(gate, scan, old, 0.1).state == OK
    stale_epoch = _cmd(straight, cmd_id=9, epoch=99)
    assert _step(gate, scan, stale_epoch, 0.2).state == OK
    assert _step(gate, scan, _cmd(straight, cmd_id=6), 0.3).reasons == ('LIDAR_SLOW',)
    assert _step(gate, scan, _cmd(arc, cmd_id=1, epoch=101), 0.4).state == OK


def test_unfollowed_command_gives_no_zone():
    scan = _scan([(0.25, -0.10)])
    for bad in (_cmd(_arc(0.30, 0.45), mode='STOP'), make_cmd(()),
                _cmd([(0.0, 0.0), (math.nan, 0.0)])):
        path_gate, box_gate = _follower(), _follower(front_path=False)
        _same(_step(path_gate, scan, bad, 0.0), _step(box_gate, scan, bad, 0.0))


def test_measurement_counts_for_its_command():
    # Scan evaluated on the arc (point off it); then a straight path arrives before the
    # next scan. Until a scan is evaluated on the new path, the straight zone decides.
    arc, straight, scan = _cmd(FALLBACK_ARC, cmd_id=1), _cmd(_straight(2.0), cmd_id=2), \
        _scan([(0.25, -0.10)])
    gate, box_gate = _follower(), _follower(front_path=False)
    assert _step(gate, scan, arc, 0.0).state == OK
    _step(box_gate, scan, arc, 0.0)
    for g in (gate, box_gate):
        g.on_cmd(0.05, '', straight)
    _same(gate.step(0.05), box_gate.step(0.05))


def test_stop_does_not_cycle_on_zero_input():
    cfg = _follower_cfg()
    gate = _follower(lidar_resume_s=cfg.lidar_resume_s)
    cmd, blocked, clear = _cmd(_corner(0.02)), _scan([(0.02, 0.17)]), _scan()
    assert _step(gate, blocked, cmd, 0.0).reasons == ('LIDAR_STOP',)
    for k in range(1, 30):
        out = _step(gate, blocked, cmd, k * DT, v=0.0)
        assert out.state == STOPPED and 'LIDAR_STOP' in out.reasons
    last_stop = 29 * DT
    for t in np.arange(last_stop + DT, last_stop + cfg.lidar_resume_s - DT / 2.0, DT):
        assert 'LIDAR_STOP' in _step(gate, clear, cmd, t).reasons
    assert _step(gate, clear, cmd, last_stop + cfg.lidar_resume_s + DT).state == OK


def test_vanished_close_point_on_path_stops():
    cfg = _follower_cfg()
    cmd, near = _cmd(_straight(2.0)), (0.30, 0.0)
    assert cfg.front_lost_stops and near[0] < cfg.front_lost_near_m
    gate = _follower()
    assert _step(gate, _scan([near]), cmd, 0.0).state == OK
    blind = _scan(room=0.0)
    assert _step(gate, blind, cmd, 0.1).reasons == ('LIDAR_STOP',)
    # A new path that turns away before the point: it is no longer in the zone.
    away = _cmd([(0.0, 0.0), (0.0, -0.05)] + [(0.0, -0.05 * i) for i in range(2, 30)],
                cmd_id=2)
    turned = (POSE[0], POSE[1], POSE[2] - math.pi / 2.0)
    out = _step(gate, blind, away, 0.2, turned)
    assert 'LIDAR_STOP' not in out.reasons and 'LIDAR_BLIND' in out.reasons


def test_vanished_close_point_beside_sector_stops():
    # On the corner path, (0.10, 0.18) is 0.28 m along (inside front_lost_near_m) but
    # outside the box and the sector: the straight rule would let it go.
    cmd, blind = _cmd(_corner(0.10)), _scan(room=0.0)
    path_gate, box_gate = _follower(), _follower(front_path=False)
    for gate in (path_gate, box_gate):
        assert _step(gate, _scan([(0.10, 0.18)]), cmd, 0.0).state == OK
    assert _step(path_gate, blind, cmd, 0.1).reasons == ('LIDAR_STOP',)
    assert _step(box_gate, blind, cmd, 0.1).reasons == ('LIDAR_BLIND',)


FOV, DEPTH_BEAMS = 1.019, 80
DEPTH_INC = FOV / DEPTH_BEAMS
DEPTH_MIN = -FOV / 2.0 + DEPTH_INC / 2.0
DEPTH_ANGLES = DEPTH_MIN + DEPTH_INC * np.arange(DEPTH_BEAMS)


def _leader(front_path=True, **overrides):
    overrides.setdefault('lidar_resume_s', 0.0)
    return SafetyGate(gate_config('leader', front_path=front_path, front_memory=True,
                                  **overrides))


def _depth_scan(points):
    """Depth scan with returns only at body-frame ``points`` (each on its nearest beam)."""
    ranges = np.zeros(DEPTH_BEAMS)
    for x, y in points:
        ranges[int(np.argmin(np.abs(DEPTH_ANGLES - math.atan2(y, x))))] = math.hypot(x, y)
    return ranges


def _on_beam(beam, rng):
    a = DEPTH_ANGLES[beam]
    return rng * math.cos(a), rng * math.sin(a)


def _leader_step(gate, cmd, band, t, depth_points=(), v=0.10, pose=POSE):
    gate.on_pose(*pose, t)
    gate.set_aligned(True)
    gate.on_cmd(t, '', cmd)
    gate.on_input(v, 0.0, t)
    gate.on_front_distance(band, t)
    gate.on_depth_scan(_depth_scan(depth_points), DEPTH_MIN, DEPTH_INC, 0.30, t)
    return gate.step(t)


def test_leader_band_on_the_path_stops():
    # Left arc R 0.6: the band at 0.45 m overlaps the corridor.
    cmd = _cmd(_arc(0.60, 0.9, then=1.0))
    out = _leader_step(_leader(), cmd, 0.45, 0.0)
    assert out.state == STOPPED and out.reasons == ('LIDAR_STOP',)
    bx, by = _to_body(out.blocked_x, out.blocked_y)
    assert bx == pytest.approx(0.45) and abs(by) <= 0.45 * math.tan(0.22) + 1e-6
    assert abs(math.hypot(bx, 0.60 - by) - 0.60) <= gate_config('leader').lidar_half_width


def test_leader_band_beside_curve_no_stop():
    # Left arc R 0.3: at 0.45 m the band (+-0.10 m) is outside the corridor; the straight
    # zone stops for it.
    cmd = _cmd(_arc(0.30, 0.47, then=1.0))
    assert _leader_step(_leader(front_path=False), cmd, 0.45, 0.0).state == STOPPED
    assert _leader_step(_leader(), cmd, 0.45, 0.0).state == OK


def test_leader_remembered_point_on_path_stops():
    cmd = _cmd(_arc(0.60, 0.9, then=1.0))
    on_path = _on_beam(66, 0.44)
    beside = _on_beam(20, 0.42)
    assert abs(math.hypot(on_path[0], 0.60 - on_path[1]) - 0.60) < 0.05
    assert abs(math.hypot(beside[0], 0.60 - beside[1]) - 0.60) > 0.20
    stopped = _leader_step(_leader(), cmd, ROOM, 0.0, [on_path])
    assert stopped.reasons == ('LIDAR_STOP',)
    assert (stopped.blocked_x, stopped.blocked_y) == pytest.approx(_glob(*on_path), abs=1e-6)
    assert _leader_step(_leader(), cmd, ROOM, 0.0, [beside]).state == OK
    assert _leader_step(_leader(front_path=False), cmd, ROOM, 0.0, [beside]).state == STOPPED


def test_leader_band_lost_beside_path_no_stop():
    # The other rover parked beside the way home: last seen 0.66 m ahead in the band, then
    # inside the camera's minimum range (NaN). Straight: "vanished while close" holds the
    # stop; on a path curving away it never was on the path.
    cmd = _cmd(_arc(0.30, 0.47, then=1.0))
    box_gate, path_gate = _leader(front_path=False), _leader()
    for gate in (box_gate, path_gate):
        _leader_step(gate, cmd, 0.66, 0.0)
    assert _leader_step(box_gate, cmd, math.nan, 0.1).reasons == ('LIDAR_STOP',)
    out = _leader_step(path_gate, cmd, math.nan, 0.1)
    assert out.state == LIMITED and out.reasons == ('LIDAR_BLIND',)


def test_leader_band_lost_on_the_path_stops():
    cmd = _cmd(_straight(2.0))
    gate = _leader()
    assert _leader_step(gate, cmd, 0.66, 0.0).state == OK
    assert _leader_step(gate, cmd, math.nan, 0.1).reasons == ('LIDAR_STOP',)


def test_leader_shelf_beyond_pick_no_stop():
    # Pick run-in 0.40 m, shelf 0.30 m beyond the camera at the goal (lidar_to_front 0).
    cmd = _cmd(_straight(0.40))
    path_gate, box_gate = _leader(front_pick_exempt=False), _leader(front_path=False)
    box_states = set()
    for k, travelled in enumerate(np.linspace(0.0, 0.40, 9)):
        pose = (*_glob(travelled, 0.0), POSE[2])
        out = _leader_step(path_gate, cmd, 0.70 - travelled, k * DT, pose=pose)
        assert out.state == OK, travelled
        box_states.add(_leader_step(box_gate, cmd, 0.70 - travelled, k * DT, pose=pose).state)
    assert STOPPED in box_states


def test_remembered_points_match_fresh_eval():
    # Drive the S path at 0.10 m/s past fixed clutter: depth scans for the first second
    # only, so remembered points also come into reach while no scan refreshes the memory.
    from sim import s_path
    rng = np.random.default_rng(3)
    body = [(p[0], p[1]) for p in s_path(radius=0.6, lead_in=0.5)]
    cmd = _cmd(body)
    path = PathCorridor(cmd.poses, 0.18, 0.0, 0.5, 0.05)._path
    clutter = []
    for _ in range(60):
        x, y = path.point(rng.uniform(0.2, path.length))
        clutter.append((x + rng.uniform(-0.35, 0.35), y + rng.uniform(-0.35, 0.35)))
    kept, fresh = _leader(), _leader()
    for k in range(int(3.0 / 0.02)):
        t = k * 0.02
        s = 0.10 + 0.10 * t
        x, y = path.point(s)
        pose = (x, y, path.tangent(s))
        outs = []
        for gate in (kept, fresh):
            gate.on_pose(*pose, t)
            gate.set_aligned(True)
            gate.on_cmd(t, '', cmd)
            gate.on_input(0.10, 0.0, t)
            if k % 5 == 0:
                gate.on_front_distance(ROOM, t)
                if t < 1.0:
                    seen = [_to_body(cx, cy, pose) for cx, cy in clutter]
                    seen = [(bx, by) for bx, by in seen
                            if bx > 0.0 and 0.3 <= math.hypot(bx, by)
                            and abs(math.atan2(by, bx)) < FOV / 2.0]
                    gate.on_depth_scan(_depth_scan(seen), DEPTH_MIN, DEPTH_INC, 0.30, t)
            if gate is fresh:
                gate._memory_cover = None
            outs.append(gate.step(t))
        _same(*outs)
    assert len(kept._memory) > 0


def test_far_detour_point_stops_on_arrival():
    # Remembered 0.6 m straight ahead, but the path first detours 0.4 m to the right: it is
    # 1.4 m along the path, out of reach when seen. No depth scan after that one.
    body = [(0.0, 0.0), (0.1, 0.0), (0.1, -0.4), (0.6, -0.4), (0.6, 0.0), (1.2, 0.0)]
    cmd = _cmd(body)
    path = PathCorridor(cmd.poses, 0.18, 0.0, 0.5, 0.05)._path
    kept, fresh = _leader(), _leader()
    reasons = set()
    for k in range(130):
        t = k * DT
        s = 0.10 + min(0.01 * k, 1.2)
        x, y = path.point(s)
        outs = []
        for gate in (kept, fresh):
            gate.on_pose(x, y, path.tangent(s), t)
            gate.set_aligned(True)
            gate.on_cmd(t, '', cmd)
            gate.on_input(0.10, 0.0, t)
            gate.on_front_distance(ROOM, t)
            if k == 0:
                gate.on_depth_scan(_depth_scan([(0.6, 0.0)]), DEPTH_MIN, DEPTH_INC, 0.30, t)
            if gate is fresh:
                gate._memory_cover = None
            outs.append(gate.step(t))
        _same(*outs)
        reasons |= set(outs[0].reasons)
    assert 'LIDAR_STOP' in reasons


def _to_body(x, y, pose=POSE):
    dx, dy = x - pose[0], y - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return c * dx + s * dy, -s * dx + c * dy


def test_corridor_ends_flat_after_short_piece():
    # Last piece 0.01 long: the disc of the vertex before it must not reach past the end.
    corridor = PathCorridor([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.51, 0.0, 0.0)],
                            half_width=0.15, front_offset=0.0, window=0.5, tie_margin=0.05)
    corridor.update(0.0, 0.0)
    pose = (0.0, 0.0, 0.0)
    assert corridor.nearest(pose, [0.505], [0.10], 1.0)[0] == pytest.approx(0.505)
    assert corridor.nearest(pose, [0.52], [0.0], 1.0) is None
    assert corridor.nearest(pose, [-0.02], [0.05], 1.0) is None


def test_corridor_reach_cuts_the_far_part():
    corridor = PathCorridor([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], 0.15, 0.0, 0.5, 0.05)
    corridor.update(0.0, 0.0)
    assert corridor.nearest((0.0, 0.0, 0.0), [0.3, 1.0], [0.0, 0.0], 0.5) == \
        pytest.approx((0.3, 0))
    assert corridor.nearest((0.0, 0.0, 0.0), [1.0], [0.0], 0.5) is None
