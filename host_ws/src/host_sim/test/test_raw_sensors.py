"""Sensor-level models: ceiling camera, UWB tag, IMU (rate, stamp, noise, dropouts, seeds)."""
import math

import numpy as np
import pytest

from host_sim.core.raw_sensors import (CameraParams, CeilingCamera, Imu, ImuParams,
                                       Ticker, UwbParams, UwbTag)

DT = 0.005
ROI = (0.0, 10.0, 0.0, 10.0)
POINT = (3.0, 2.0, 0.4)          # tracked point x, y, heading


def _camera(seed=0, box_axis=False, **params):
    return CeilingCamera(CameraParams(**params), ROI, box_axis, np.random.default_rng(seed))


def _uwb(seed=0, **params):
    return UwbTag(UwbParams(**params), np.random.default_rng(seed))


def _imu(seed=0, **params):
    return Imu(ImuParams(**params), np.random.default_rng(seed))


def _run(capture, duration, dt=DT):
    """Every non-None capture(t) over the duration."""
    out = []
    for i in range(int(round(duration / dt))):
        m = capture(i * dt)
        if m is not None:
            out.append(m)
    return out


def _xy(fixes):
    return np.array([[m.x, m.y] for m in fixes])


def test_ticker_sampled_and_timer_clocks():
    sampled = Ticker(20.0)
    assert sum(sampled.due(i * 0.001) for i in range(10000)) == pytest.approx(200, abs=1)
    timer = Ticker(10.0)
    assert timer.ticks_until(0.0) == [0.0]
    assert timer.ticks_until(0.35) == pytest.approx([0.1, 0.2, 0.3])
    assert timer.ticks_until(0.35) == []


# ---------- ceiling camera ----------

def test_camera_rate_and_latency_stamp():
    cam = _camera()
    fixes = _run(lambda t: cam.capture(t, POINT), 10.0)
    assert len(fixes) == pytest.approx(210, abs=2)
    assert all(m.deliver_at - m.stamp == pytest.approx(0.055) for m in fixes)
    assert np.diff([m.stamp for m in fixes]).mean() == pytest.approx(1.0 / 21.0, rel=0.01)


def test_camera_jitter_std():
    cam = _camera(seed=1)
    xy = _xy(_run(lambda t: cam.capture(t, POINT), 200.0))
    assert len(xy) > 4000
    assert xy.mean(axis=0) == pytest.approx(POINT[:2], abs=2e-4)
    assert xy.std(axis=0) == pytest.approx([0.002, 0.002], rel=0.06)


def test_camera_slow_bias_is_time_correlated():
    cam = _camera(seed=2, jitter_std=0.0, bias_std=0.03, bias_tau=2.0)
    x = _xy(_run(lambda t: cam.capture(t, POINT), 400.0))[:, 0]
    assert x.std() == pytest.approx(0.03, rel=0.25)
    assert np.corrcoef(x[:-1], x[1:])[0, 1] > 0.95


def test_camera_drops_point_outside_roi():
    def fixes(point):
        cam = CeilingCamera(CameraParams(), (0.0, 3.5, 0.0, 3.0), False, np.random.default_rng(0))
        return _run(lambda t: cam.capture(t, point), 1.0)
    assert fixes((3.6, 2.0, 0.0)) == []
    assert fixes((3.4, 2.0, 0.0))


def test_camera_drops_adjacent_rover():
    near = [(POINT[0] + 0.39, POINT[1])]
    far = [(POINT[0] + 0.41, POINT[1])]
    assert _run(lambda t: _camera().capture(t, POINT, near), 1.0) == []
    cam = _camera()
    assert _run(lambda t: cam.capture(t, POINT, far), 1.0)


def test_camera_drops_while_reported_lost():
    cam = _camera()
    assert _run(lambda t: cam.capture(t, POINT, lost=True), 1.0) == []


def test_camera_box_axis_is_heading_mod_period():
    cam = _camera(seed=3, box_axis=True)
    axes = np.array([m.axis for m in _run(lambda t: cam.capture(t, POINT), 100.0)])
    assert axes.min() >= -math.pi / 2 and axes.max() < math.pi / 2     # perception range
    period = math.pi / 2
    folded = (axes - POINT[2] + period / 2) % period - period / 2
    assert folded.mean() == pytest.approx(0.0, abs=0.003)
    assert folded.std() == pytest.approx(math.radians(2.4), rel=0.08)
    assert len(np.unique(np.round((axes - folded - POINT[2]) / period))) > 1   # both sides


def test_camera_without_box_axis_sends_none():
    cam = _camera(box_axis=False)
    assert all(m.axis is None for m in _run(lambda t: cam.capture(t, POINT), 1.0))


def test_camera_same_seed_same_sequence():
    def sequence(seed):
        cam = _camera(seed=seed, box_axis=True, bias_std=0.01)
        return [tuple(m) for m in _run(lambda t: cam.capture(t, POINT), 2.0)]
    assert sequence(4) == sequence(4)
    assert sequence(4) != sequence(5)


# ---------- UWB ----------

def test_uwb_rate_noise_and_quality_factor():
    uwb = _uwb(seed=1)
    fixes = _run(lambda t: uwb.capture(t, POINT), 300.0)
    assert len(fixes) == pytest.approx(3000, abs=2)
    xy = _xy(fixes)
    assert xy.mean(axis=0) == pytest.approx(POINT[:2], abs=0.003)
    assert xy.std(axis=0) == pytest.approx([0.04, 0.04], rel=0.05)
    assert all(m.qf == 80.0 and m.deliver_at == m.stamp for m in fixes)


def test_uwb_frame_offset():
    uwb = _uwb(noise_std=0.0, offset_x=1.27, offset_y=1.23)
    fix = _run(lambda t: uwb.capture(t, POINT), 0.2)[0]
    assert (fix.x, fix.y) == pytest.approx((POINT[0] + 1.27, POINT[1] + 1.23))


def test_uwb_disabled_sends_nothing():
    uwb = _uwb(enabled=False)
    assert _run(lambda t: uwb.capture(t, POINT), 2.0) == []


def test_uwb_same_seed_same_sequence():
    def sequence(seed):
        uwb = _uwb(seed=seed, bias_std=0.05)
        return [tuple(m) for m in _run(lambda t: uwb.capture(t, POINT), 2.0)]
    assert sequence(6) == sequence(6)
    assert sequence(6) != sequence(7)


# ---------- IMU ----------

def _turning(imu, rate=0.3, duration=10.0):
    """Samples while the heading turns at a constant rate from 0.4 rad."""
    return _run(lambda t: imu.capture(t, 0.4 + rate * t, rate), duration)


def test_imu_zero_offset_constant_in_run():
    imu = _imu(seed=1)
    samples = _turning(imu)
    assert len(samples) == pytest.approx(200, abs=1)
    offsets = [math.remainder(s.yaw - (0.4 + 0.3 * s.stamp), 2 * math.pi) for s in samples]
    assert np.ptp(offsets) < 1e-9
    assert offsets[0] == pytest.approx(math.remainder(imu.zero, 2 * math.pi))
    assert all(s.wz == pytest.approx(0.3) for s in samples)


def test_imu_zero_offset_differs_by_seed():
    zeros = {round(_imu(seed=seed).zero, 6) for seed in range(5)}
    assert len(zeros) == 5


def test_imu_scale_factor_bias_and_drift():
    imu = _imu(seed=2, yaw_rate_scale=1.1, gyro_bias=0.01)
    samples = _turning(imu, rate=0.2, duration=5.0)
    turned = math.remainder(samples[-1].yaw - samples[0].yaw, 2 * math.pi)
    true_turn = 0.2 * (samples[-1].stamp - samples[0].stamp)
    assert turned == pytest.approx(1.1 * true_turn, rel=1e-6)
    assert samples[-1].wz == pytest.approx(1.1 * 0.2 + 0.01)
    drifting = _imu(seed=3, yaw_drift_std=0.01)
    parked = _run(lambda t: drifting.capture(t, 0.4, 0.0), 100.0)
    assert abs(math.remainder(parked[-1].yaw - parked[0].yaw, 2 * math.pi)) > 1e-4


def test_imu_gyro_noise_std():
    imu = _imu(seed=4, gyro_noise_std=0.02)
    wz = np.array([s.wz for s in _run(lambda t: imu.capture(t, 0.4, 0.0), 200.0)])
    assert wz.std() == pytest.approx(0.02, rel=0.06)


def test_imu_same_seed_same_sequence():
    def sequence(seed):
        return [tuple(s) for s in _turning(_imu(seed=seed, yaw_drift_std=0.01), duration=2.0)]
    assert sequence(8) == sequence(8)
    assert sequence(8) != sequence(9)
