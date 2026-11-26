"""Leader front distance from synthetic depth images, and the gate's use of it."""
import math

import numpy as np
import pytest

from rover_drive.core import state as st
from rover_drive.core.config import DepthFrontConfig
from rover_drive.core.controller import DriveController
from rover_drive.core.depth_front import (Intrinsics, decode_depth, depth_scan, front_distance,
                                          intrinsics_from_camera_info, intrinsics_from_fov)
from rover_drive.core.path import Path
from rover_drive.core.safety import LIMITED, OK, STOPPED, SafetyGate

from sim import drive_config, gate_config, load_config, make_cmd, s_path, straight_path

H, W = 480, 640


def _cfg(**overrides):
    return load_config(DepthFrontConfig, 'leader', **overrides)


def _image_mm(fill_mm=3000):
    return np.full((H, W), fill_mm, dtype=np.uint16)


def _band_rows(cfg):
    return int(cfg.depth_row_min * H), int(math.ceil(cfg.depth_row_max * H))


def _distance(image, encoding='16UC1', cfg=None, **decode_kw):
    depth = decode_depth(image.tobytes(), image.shape[0], image.shape[1], encoding, **decode_kw)
    return front_distance(depth, cfg or _cfg())


def test_decode_16uc1_mm_32fc1_m():
    mm = np.array([[0, 470], [1500, 65535]], dtype=np.uint16)
    assert decode_depth(mm.tobytes(), 2, 2, '16UC1') == pytest.approx(
        np.array([[0.0, 0.47], [1.5, 65.535]]))
    m = np.array([[0.0, 0.47], [1.5, np.nan]], dtype=np.float32)
    out = decode_depth(m.tobytes(), 2, 2, '32FC1')
    assert out[0, 1] == pytest.approx(0.47) and math.isnan(out[1, 1])


def test_decode_padding_endian_array_buffers():
    import array
    mm = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    padded = np.zeros((2, 6), dtype=np.uint8)
    padded[:, :4] = mm.view(np.uint8).reshape(2, 4)
    assert decode_depth(padded.tobytes(), 2, 2, '16UC1', step=6)[1, 1] == pytest.approx(0.4)
    big = mm.astype('>u2')
    assert decode_depth(big.tobytes(), 2, 2, '16UC1', is_bigendian=True)[1, 0] == pytest.approx(
        0.3)
    as_msg_data = array.array('B', mm.tobytes())
    assert decode_depth(as_msg_data, 2, 2, '16UC1')[0, 1] == pytest.approx(0.2)


def test_decode_rejects_bad_encoding_short():
    with pytest.raises(ValueError):
        decode_depth(b'\x00' * 12, 2, 2, 'rgb8')
    with pytest.raises(ValueError):
        decode_depth(b'\x00' * 6, 2, 2, '16UC1')


def test_flat_wall_distance_both_encodings():
    assert _distance(_image_mm(2000)) == pytest.approx(2.0)
    metres = np.full((H, W), 2.0, dtype=np.float32)
    assert _distance(metres, '32FC1') == pytest.approx(2.0)


def test_band_object_found_by_low_percentile():
    image = _image_mm(3000)
    r0, r1 = _band_rows(_cfg())
    image[r0:r1, 300:340] = 800
    assert _distance(image) == pytest.approx(0.8)


def test_few_noisy_pixels_ignored():
    image = _image_mm(3000)
    r0, _ = _band_rows(_cfg())
    image[r0 + 5, 300:310] = 500
    assert _distance(image) == pytest.approx(3.0)


def test_floor_and_sides_outside_band_ignored():
    image = _image_mm(3000)
    cfg = _cfg()
    _, r1 = _band_rows(cfg)
    image[r1 + 5:, :] = 600
    image[:, :int(cfg.depth_col_min * W) - 5] = 500
    assert _distance(image) == pytest.approx(3.0)
    assert _distance(image, cfg=_cfg(depth_row_max=0.9)) == pytest.approx(0.6)


def test_invalid_pixels_skipped_few_give_nan():
    image = _image_mm(2500)
    r0, r1 = _band_rows(_cfg())
    image[r0:r1, 192:320] = 0
    assert _distance(image) == pytest.approx(2.5)
    image[r0:r1, 192:400] = 0
    assert math.isnan(_distance(image))
    assert math.isnan(_distance(_image_mm(0)))
    assert math.isnan(_distance(_image_mm(200)))
    assert math.isnan(_distance(_image_mm(9000)))
    nan_image = np.full((H, W), np.nan, dtype=np.float32)
    assert math.isnan(_distance(nan_image, '32FC1'))


def _scan_rows(cfg):
    return int(cfg.depth_scan_row_min * H), int(math.ceil(cfg.depth_scan_row_max * H))


def _beam(scan, bearing):
    return int(round((bearing - scan.angle_min) / scan.angle_increment))


def test_flat_wall_depth_over_cos_bearing():
    """A wall 2 m ahead (Z = 2 everywhere): range = Z / cos(bearing), symmetric FOV."""
    cfg = _cfg()
    intr = intrinsics_from_fov(W, cfg.depth_hfov)
    scan = depth_scan(decode_depth(_image_mm(2000).tobytes(), H, W, '16UC1'), intr, cfg)
    assert len(scan.ranges) == cfg.depth_scan_beams
    half = cfg.depth_hfov / 2.0
    assert scan.angle_min == pytest.approx(-half + scan.angle_increment / 2.0, abs=1e-3)
    assert scan.angle_max == pytest.approx(half - scan.angle_increment / 2.0, abs=1e-3)
    bearings = scan.angle_min + scan.angle_increment * np.arange(len(scan.ranges))
    assert scan.ranges == pytest.approx(2.0 / np.cos(bearings), abs=0.02)
    assert scan.ranges[len(scan.ranges) // 2] == pytest.approx(2.0, abs=0.005)
    assert scan.range_min == cfg.depth_valid_min_m
    assert scan.range_max == pytest.approx(cfg.depth_valid_max_m / math.cos(half), rel=0.01)


def test_edge_column_bearing_from_intrinsics():
    """Off-centre principal point: column u is at bearing -atan((u - cx) / fx)."""
    cfg = _cfg()
    intr = Intrinsics(fx=570.0, cx=300.0, width=W)
    image = _image_mm(0)
    r0, r1 = _scan_rows(cfg)
    image[r0:r1, 600:610] = 1500
    image[r0:r1, 20:28] = 1000
    scan = depth_scan(decode_depth(image.tobytes(), H, W, '16UC1'), intr, cfg)
    for u, z in ((604.5, 1.5), (23.5, 1.0)):
        bearing = -math.atan((u - intr.cx) / intr.fx)
        assert scan.ranges[_beam(scan, bearing)] == pytest.approx(z / math.cos(bearing),
                                                                  rel=0.01)
    hit_beams = [_beam(scan, -math.atan((u - intr.cx) / intr.fx)) for u in (604.5, 23.5)]
    assert hit_beams[0] < 10
    for beam in np.flatnonzero(scan.ranges):
        assert min(abs(beam - b) for b in hit_beams) <= 1
    assert scan.angle_min == pytest.approx(-math.atan((W - 0.5 - 300) / 570)
                                           + scan.angle_increment / 2.0)


def test_invalid_columns_zero_floor_ignored():
    cfg = _cfg()
    intr = intrinsics_from_fov(W, cfg.depth_hfov)
    image = _image_mm(3000)
    r0, r1 = _scan_rows(cfg)
    image[:, 300:340] = 0
    image[r0:r1, 0:40] = 200
    image[r0:r1, 600:] = 9000
    image[r1 + 2:, :] = 700
    scan = depth_scan(decode_depth(image.tobytes(), H, W, '16UC1'), intr, cfg)
    centre = len(scan.ranges) // 2
    assert scan.ranges[centre - 1:centre + 1] == pytest.approx([0.0, 0.0])
    assert scan.ranges[0] == 0.0 and scan.ranges[-1] == 0.0
    valid = scan.ranges[scan.ranges > 0.0]
    assert valid.min() > 2.99


def test_intrinsics_scale_reject_uncalibrated():
    k = (570.0, 0.0, 319.5, 0.0, 570.0, 239.5, 0.0, 0.0, 1.0)
    assert intrinsics_from_camera_info(k, 640, 640) == Intrinsics(570.0, 319.5, 640)
    half = intrinsics_from_camera_info(k, 640, 320)
    assert (half.fx, half.cx) == pytest.approx((285.0, 159.5))
    assert intrinsics_from_camera_info((0.0,) * 9, 640, 640) is None
    fov = intrinsics_from_fov(640, math.radians(60.0))
    assert math.degrees(2.0 * math.atan(320.0 / fov.fx)) == pytest.approx(60.0)


def _gate(**overrides):
    overrides.setdefault('lidar_resume_s', 0.0)
    gate = SafetyGate(gate_config('leader', **overrides))
    gate.on_pose(1.0, 2.0, 0.0, 0.0)
    gate.on_cmd(0.0)
    gate.set_aligned(True)
    gate.on_input(0.10, 0.0, 0.0)
    return gate


def _step(gate, dist, t=0.1):
    gate.on_front_distance(dist, t)
    return gate.step(t)


def test_leader_gate_uses_front_distance_topic():
    cfg = gate_config('leader')
    assert cfg.front_distance_topic == '/leader/front_distance' and not cfg.lidar_enabled
    gate = _gate()
    assert 'STALE_SCAN' in gate.step(0.1).reasons
    assert _step(gate, 2.0).state == OK
    slow = _step(gate, 0.51)
    assert slow.reasons == ('LIDAR_SLOW',) and 0.0 < slow.v < 0.10
    stop = _step(gate, 0.49)
    assert (stop.v, stop.state, stop.reasons) == (0.0, STOPPED, ('LIDAR_STOP',))
    assert (stop.blocked_x, stop.blocked_y) == pytest.approx((1.49, 2.0))


def test_unknown_distance_slows_down():
    cap = gate_config('leader').lidar_blind_speed
    gate = _gate(max_linear_speed=3.0 * cap)
    gate.on_input(3.0 * cap, 0.0, 0.0)
    out = _step(gate, math.nan)
    assert (out.state, out.reasons) == (LIMITED, ('LIDAR_BLIND',))
    assert out.v == pytest.approx(cap)


def test_lost_measurement_stops_until_back():
    """Depth is unreliable under ~0.47 m: 'was close, now NaN' means very close."""
    gate = _gate(lidar_stop_base=0.30)
    assert _step(gate, 0.60).state in (OK, LIMITED)
    lost = _step(gate, math.nan, 0.2)
    assert (lost.v, lost.state, lost.reasons) == (0.0, STOPPED, ('LIDAR_STOP',))
    assert lost.blocked_x == pytest.approx(1.60)
    assert _step(gate, math.nan, 0.3).state == STOPPED
    assert _step(gate, 1.5, 0.4).state == OK


def test_lost_object_no_stop_after_turning_away():
    """'Was close, now NaN' means 'very close' only while that object can still be ahead,
    e.g. not after a pivot away from the shelf."""
    gate = _gate()
    gate.on_approach(True, 0.0)
    assert _step(gate, 0.60).state == OK
    gate.on_approach(False, 0.2)
    gate.on_pose(1.0, 2.0, math.pi / 2, 0.2)
    assert _step(gate, math.nan, 0.2).reasons == ('LIDAR_BLIND',)
    gate.on_pose(1.0, 2.0, 0.1, 0.3)
    assert _step(gate, math.nan, 0.3).reasons == ('LIDAR_STOP',)


def test_lost_object_no_stop_after_sideways():
    """Other rover parked beside the way home: seen 0.65 m ahead at the band edge on a curve, then
    out of view. 0.25 m off the axis, it is not in the way, so the stop must end."""
    gate = _gate()
    assert _step(gate, 0.65).state in (OK, LIMITED)
    gate.on_pose(1.0, 2.0, 0.40, 0.2)
    assert _step(gate, math.nan, 0.2).reasons == ('LIDAR_BLIND',)
    gate.on_pose(1.0, 2.0, 0.20, 0.3)
    assert _step(gate, math.nan, 0.3).reasons == ('LIDAR_STOP',)


def test_lost_object_blocked_point_stays():
    """A reported point carried round with the rover's heading lands beside a known object and
    must not be matched to it."""
    gate = _gate(lidar_stop_base=0.30)
    _step(gate, 0.60)
    gate.on_pose(1.0, 2.0, 0.25, 0.2)
    lost = _step(gate, math.nan, 0.2)
    assert lost.reasons == ('LIDAR_STOP',)
    assert (lost.blocked_x, lost.blocked_y) == pytest.approx((1.60, 2.0))


def test_lost_approach_rule_switchable_off():
    gate = _gate(lidar_stop_base=0.30, front_lost_stops=False)
    _step(gate, 0.60)
    assert _step(gate, math.nan, 0.2).reasons == ('LIDAR_BLIND',)


def test_front_stop_off_on_pick_approach():
    gate = _gate()
    gate.on_approach(True, 0.0)
    for dist in (0.30, math.nan):
        out = _step(gate, dist)
        assert (out.v, out.state, out.reasons) == (0.10, OK, ())


def test_pick_exemption_needs_fresh_signal():
    gate = _gate()
    gate.on_approach(True, 0.0)
    for t in (0.5, 1.0, 1.2):
        gate.on_pose(1.0, 2.0, 0.0, t)
        gate.on_cmd(t)
        gate.on_input(0.10, 0.0, t)
    assert _step(gate, 0.30, 1.2).reasons == ('LIDAR_STOP',)
    gate.on_approach(False, 1.2)
    assert _step(gate, 0.30, 1.2).reasons == ('LIDAR_STOP',)
    disabled = _gate(front_pick_exempt=False)
    disabled.on_approach(True, 0.0)
    assert _step(disabled, 0.30).reasons == ('LIDAR_STOP',)


def _tracker(cmd, x, duration=0.05, y=0.0, yaw=0.0):
    """Tracker standing at (x, y, yaw) for ``duration``; its last output."""
    ctrl = DriveController(drive_config('leader'))
    ctrl.set_heading_aligned(True)
    ctrl.on_command(cmd, 0.0)
    out, t = None, 0.0
    while t < duration - 1e-9:
        ctrl.on_pose(x, y, yaw, t)
        t += 0.05
        out = ctrl.step(t)
    return out


def _approach_at(cmd, remaining, yaw_error=0.0):
    """approach_active standing on the path ``remaining`` before its end."""
    path = Path([p[:2] for p in cmd.poses])
    s = path.length - remaining
    x, y = path.point(s)
    return _tracker(cmd, x, y=y, yaw=path.tangent(s) + yaw_error).approach_active


def test_approach_exemption_straight_run_in_only():
    """Distance alone (0.6 m) must not exempt the end of the curve before the 0.40 m run-in."""
    cmd = make_cmd(s_path(radius=0.6, lead_in=0.4), pick_at_goal=True)
    assert not _approach_at(cmd, 0.55)
    assert _approach_at(cmd, 0.38)
    assert _approach_at(cmd, 0.30)
    assert not _approach_at(cmd, 0.30, yaw_error=0.5)
    short = make_cmd([(0.0, 0.0, 0.0), (0.3, 0.0, 0.0)], pick_at_goal=True)
    assert _approach_at(short, 0.2)
    at_goal = make_cmd([(2.0, 0.0, 0.0)], pick_at_goal=True)
    assert not _tracker(at_goal, 2.0, duration=0.5, yaw=0.5).approach_active


def test_tracker_reports_approach_near_pick():
    pick = make_cmd(straight_path(2.0), pick_at_goal=True)
    assert not _tracker(pick, 1.0).approach_active
    assert _tracker(pick, 1.45).approach_active
    assert not _tracker(make_cmd(straight_path(2.0)), 1.45).approach_active
    assert not DriveController(drive_config('leader')).step(0.0).approach_active


def test_fixture_run_in_exempt_without_pick():
    """A goal marked fixture_ahead exempts the final run-in even before pick_at_goal."""
    poses = straight_path(2.0)
    marked = make_cmd(poses, fixture_ahead=True)
    assert _tracker(marked, 1.45).approach_active
    assert not _tracker(marked, 1.0).approach_active
    assert not _tracker(make_cmd(poses), 1.45).approach_active


def test_approach_active_while_holding_at_pick():
    """During the pick itself the arm side drives the base toward the shelf."""
    out = _tracker(make_cmd([(2.0, 0.0, 0.0)], pick_at_goal=True), 2.0, duration=0.5)
    assert out.state == st.HOLDING and out.approach_active
    far = _tracker(make_cmd([(2.0, 0.0, 0.0)], pick_at_goal=True), 0.5, duration=0.5)
    assert far.state == st.STOPPED and not far.approach_active
