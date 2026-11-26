"""Zero-motion update (ZUPT on still wheels) and the estimator-based localization quality.

Field case: the follower stood beside the leader while the arm reached over it; the
camera lost it for 8.6 s and then flickered. The camera-age rule said LOST (mission paused three
times) although the wheels never turned, and through the gap the constant-velocity filter let the
position follow UWB. These tests pin the pure pieces: the quality rule, the zero-velocity update,
the camera / UWB overruling the wheels, and the blind-travel bound.
"""
import math

import numpy as np
import pytest

from host_localization.fusion import quality as Q
from host_localization.fusion.frontend import Gating, RobotFusion
from host_localization.fusion.se2nhc import EKFSE2, PX, PY, VX, VY
from host_localization.fusion.tracker import KW, RobotTracker, WHEEL_RADIUS

DT = 1.0 / 21.0


def _state(**kw):
    base = dict(in_camera_frame=True, position_age=0.05, camera_age=0.05, sigma=0.01,
                wheels_still=False, blind_travel=0.0, camera_refused=0)
    base.update(kw)
    return Q.EstimatorState(**base)


# ── quality.classify_estimator ───────────────────────────────────────────────
@pytest.mark.parametrize('kw, expected', [
    (dict(), Q.GOOD),
    (dict(in_camera_frame=False), Q.LOST),                          # UWB frame only
    (dict(position_age=None), Q.LOST),                              # never a position
    (dict(position_age=1.5, camera_age=1.5, wheels_still=True), Q.LOST),   # no source at all
    (dict(camera_age=0.2, wheels_still=False), Q.GOOD),             # short blind drive, sigma small
    (dict(camera_age=0.4, wheels_still=False), Q.DEGRADED),         # blind drive: crawl ...
    (dict(camera_age=0.6, wheels_still=False), Q.LOST),             # ... then stop
    (dict(camera_age=30.0, wheels_still=True), Q.GOOD),             # standing: no time bound
    (dict(blind_travel=0.05, camera_age=0.1), Q.DEGRADED),          # travelled blind: crawl
    (dict(blind_travel=0.12, wheels_still=True, camera_age=5.0), Q.LOST),  # travelled blind: stop
    (dict(blind_travel=None), Q.GOOD),                              # no wheel data: time bound only
    (dict(sigma=0.04), Q.DEGRADED),
    (dict(sigma=0.07), Q.LOST),
    (dict(camera_refused=5), Q.DEGRADED),                            # camera sees it elsewhere
])
def test_classify_estimator(kw, expected):
    assert Q.classify_estimator(_state(**kw)) == expected


def test_classify_limits_are_parameters():
    tight = Q.EstimatorLimits(sigma_good=0.005)
    assert Q.classify_estimator(_state(sigma=0.01), tight) == Q.DEGRADED


# ── EKFSE2: position_sigma and the zero-velocity update ─────────────────────
def test_position_sigma_is_the_worst_axis():
    est = EKFSE2()
    est.P[PX, PX], est.P[PY, PY], est.P[PX, PY] = 0.02 ** 2, 0.01 ** 2, 0.0
    assert est.position_sigma == pytest.approx(0.02)
    est.P[PX, PX] = est.P[PY, PY] = 1e-4
    est.P[PX, PY] = est.P[PY, PX] = 0.5e-4
    assert est.position_sigma == pytest.approx(math.sqrt(1.5e-4))


def _parked(zero_velocity, n=21 * 8, dt=DT):
    est = EKFSE2(**KW['follower'])
    est.update_bev(1.0, 1.0)
    for _ in range(40):                       # converge on the camera
        est.predict(DT)
        est.update_bev(1.0, 1.0)
    est.x[VX] = 0.05                          # a residual velocity estimate when the camera goes
    est.zero_velocity = zero_velocity
    for _ in range(n):
        est.predict(dt)
    return est


def test_zero_velocity_holds_parked_position():
    held, free = _parked(True), _parked(False)
    assert math.hypot(held.x[PX] - 1.0, held.x[PY] - 1.0) < 0.01     # velocity correlation, then still
    assert math.hypot(free.x[PX] - 1.0, free.x[PY] - 1.0) > 0.1
    assert held.position_sigma < 0.015 < free.position_sigma


def test_zero_velocity_info_rate_independent():
    slow = _parked(True, n=8 * 10, dt=0.1)
    fast = _parked(True, n=8 * 50, dt=0.02)
    # 5x the predict rate; a per-event noise would give sqrt(5) = 2.2x the information
    assert slow.position_sigma == pytest.approx(fast.position_sigma, rel=0.25)


# ── RobotTracker: wheel evidence, overruled by the camera / UWB ─────────────
def _tracker(zero_motion=True, wheel_still=False):
    trk = RobotTracker(dict(KW['follower']), zero_motion=zero_motion, wheel_still=wheel_still,
                       wheel_radius=WHEEL_RADIUS['follower'])
    return trk


def _stand(trk, t0, t1, x=1.0, y=1.0, camera=True, uwb=None, wheel=0.0):
    """Feed a standing rover from t0 to t1: wheels fixed, camera at (x, y), UWB at uwb (or none)."""
    t = t0
    k = 0
    while t < t1:
        trk.on_wheels(t, wheel, wheel)
        trk.on_imu(t, 0.0, 0.0)
        if camera:
            trk.on_bev(t, x, y)
        if uwb is not None and k % 2 == 0:
            trk.on_uwb(t, *uwb)
        t += DT
        k += 1
    return t


def test_zero_motion_needs_still_wheels():
    trk = _tracker()
    _stand(trk, 0.0, 2.0)
    assert trk.wheels_still and trk.est.zero_velocity
    off = _tracker(zero_motion=False)
    _stand(off, 0.0, 2.0)
    assert off.wheels_still and not off.est.zero_velocity and not off.est.stationary


def test_wheel_still_zero_motion_separate():
    trk = _tracker(zero_motion=True, wheel_still=False)
    _stand(trk, 0.0, 2.0)
    assert trk.est.zero_velocity and not trk.est.stationary      # heading offset not frozen


def test_standing_gap_biased_uwb_keeps_position():
    """8.6 s without the camera, UWB 0.15 m off (arm over the tag): the held estimate stays."""
    held, free = _tracker(True), _tracker(False)
    for trk in (held, free):
        t = _stand(trk, 0.0, 10.0, uwb=(1.0, 1.0))
        _stand(trk, t, t + 8.6, camera=False, uwb=(1.15, 1.0))
    assert abs(held.est.pose[0] - 1.0) < 0.01
    assert abs(free.est.pose[0] - 1.0) > abs(held.est.pose[0] - 1.0)
    assert held.zm_releases == 0


def test_camera_box_off_hold_unused():
    """Loading: arm + part merge both rovers into one box 0.11-0.20 m away. Wheels still
    -> the hold wins, the fixes are counted as disputes."""
    trk = _tracker()
    t = _stand(trk, 0.0, 5.0)
    _stand(trk, t, t + 8.0, x=1.15)
    assert abs(trk.est.pose[0] - 1.0) < 0.01
    assert trk.camera_disputes > 100
    assert trk.zm_releases == 0


def test_lasting_dispute_no_handover_standing():
    """A pick lasts 37-48 s with the camera box shifted by the arm: the hold keeps its anchor."""
    trk = _tracker()
    t = _stand(trk, 0.0, 5.0)
    _stand(trk, t, t + 45.0, x=1.12)
    assert abs(trk.est.pose[0] - 1.0) < 0.01 and trk.zm_releases == 0


def test_slow_arm_drift_keeps_hold():
    """The camera crept 0.13-0.18 m during a pick; a gate around the moving
    estimate let it through fix by fix. The anchor does not move."""
    trk = _tracker()
    t = _stand(trk, 0.0, 5.0)
    for k in range(int(20.0 / DT)):
        tt = t + k * DT
        trk.on_wheels(tt, 0.0, 0.0)
        trk.on_imu(tt, 0.0, 0.0)
        trk.on_bev(tt, 1.0 + 0.15 * k * DT / 20.0, 1.0)     # 0 -> 0.15 m in 20 s
    assert abs(trk.est.pose[0] - 1.0) < 0.035


_G = 9.81
_QUIET = (0.0, 0.0, _G)
_TILTED = (_G * math.sin(math.radians(35.0)), 0.0, _G * math.cos(math.radians(35.0)))


def _drag(trk, t, spikes, dist=0.26, dur=2.0):
    """Wheels still; the camera box walks dist [m] in dur [s]; the first `spikes` IMU samples read
    the base tilted 35 deg / 0.16 rad/s, the rest quiet."""
    for k in range(int(dur / DT)):
        tt = t + k * DT
        trk.on_wheels(tt, 0.0, 0.0)
        hit = k < spikes
        trk.on_imu(tt, 0.0, 0.0, accel=_TILTED if hit else _QUIET, rp_rate=0.16 if hit else 0.0)
        trk.on_bev(tt, 1.0 + dist * k * DT / dur, 1.0)
    return t + dur


def _quiet_stand(trk, t0, t1):
    t = t0
    while t < t1:
        trk.on_wheels(t, 0.0, 0.0)
        trk.on_imu(t, 0.0, 0.0, accel=_QUIET, rp_rate=0.0)
        trk.on_bev(t, 1.0, 1.0)
        t += DT
    return t


def test_one_sample_impact_keeps_hold():
    """The arm set the gear down on the standing follower; ONE IMU sample
    read 35 deg tilt and 0.14 rad/s (the samples around it 1.4 deg, 0). Taken as a hand move, the
    hold was off for ZM_IMU_QUIET_S while the arm's boxes walked the camera fix 0.26 m to the
    leader, and the hold re-anchored there."""
    trk = _tracker()
    t = _quiet_stand(trk, 0.0, 5.0)
    _drag(trk, t, spikes=1)
    assert abs(trk.est.pose[0] - 1.0) < 0.035 and trk.imu_releases == 0


def test_lasting_base_motion_ends_hold():
    """A hand move lasts: several consecutive disturbed samples hand the position to the camera."""
    trk = _tracker()
    t = _quiet_stand(trk, 0.0, 5.0)
    _drag(trk, t, spikes=int(1.0 / DT))
    assert trk.imu_releases >= 1 and trk.est.pose[0] > 1.15


def test_arm_working_standing_rover_holds():
    """The gear set down on the standing follower. With the arm at work the
    camera fix walks 0.30 m, the IMU reads lasting base motion and UWB sits 1.1 m off; the held
    position stands. When the arm is done and the wheels turn, the camera takes over again."""
    trk = _tracker()
    t = _quiet_stand(trk, 0.0, 5.0)
    trk.arm_busy = True
    for k in range(int(3.0 / DT)):
        tt = t + k * DT
        trk.on_wheels(tt, 0.0, 0.0)
        trk.on_imu(tt, 0.0, 0.0, accel=_TILTED, rp_rate=0.16)
        trk.on_bev(tt, 1.0 + 0.30 * min(1.0, k * DT / 1.5), 1.0)
        if k % 2 == 0:
            trk.on_uwb(tt, 2.1, 1.9)
    t += 3.0
    assert abs(trk.est.pose[0] - 1.0) < 0.01 and abs(trk.est.pose[1] - 1.0) < 0.01
    assert trk.imu_releases == 0 and trk.zm_releases == 0
    trk.arm_busy = False
    wheel = 0.0
    for k in range(int(3.0 / DT)):      # drives 0.15 m/s east: the camera follows the rover
        tt = t + k * DT
        wheel += 0.15 * DT / WHEEL_RADIUS['follower']
        trk.on_wheels(tt, wheel, wheel)
        trk.on_imu(tt, 0.0, 0.0, accel=_QUIET, rp_rate=0.0)
        trk.on_bev(tt, 1.0 + 0.15 * k * DT, 1.0)
    assert abs(trk.est.pose[0] - (1.0 + 0.15 * 3.0)) < 0.06


def test_arm_does_not_freeze_driving_rover():
    trk = _tracker()
    _quiet_stand(trk, 0.0, 5.0)
    trk.arm_busy = True
    t, wheel = 5.0, 0.0
    for k in range(int(3.0 / DT)):
        tt = t + k * DT
        wheel += 0.15 * DT / WHEEL_RADIUS['follower']
        trk.on_wheels(tt, wheel, wheel)
        trk.on_imu(tt, 0.0, 0.0, accel=_QUIET, rp_rate=0.0)
        trk.on_bev(tt, 1.0 + 0.15 * k * DT, 1.0)
    assert abs(trk.est.pose[0] - (1.0 + 0.15 * 3.0)) < 0.06


def test_small_camera_offsets_are_still_used():
    trk = _tracker()
    t = _stand(trk, 0.0, 5.0)
    _stand(trk, t, t + 3.0, x=1.02)                     # within ZM_DISPUTE_M of the anchor: used
    assert trk.camera_disputes == 0
    assert trk.est.pose[0] > 1.005


def test_uncertain_hold_corrected_not_disputed():
    """A hold that starts on a bad first fix (P still wide) converges to the camera."""
    trk = _tracker()
    trk.on_wheels(0.0, 0.0, 0.0)
    trk.on_bev(0.0, 1.12, 1.0)                          # first fix 12 cm off, P = p0
    _stand(trk, DT, 5.0)
    assert abs(trk.est.pose[0] - 1.0) < 0.01
    assert trk.camera_disputes == 0


def test_pushed_rover_camera_takes_over():
    trk = _tracker()
    t = _stand(trk, 0.0, 5.0)
    t = _stand(trk, t, t + 2.0, x=1.15)                 # pushed 15 cm, wheels never turned
    assert abs(trk.est.pose[0] - 1.0) < 0.01
    _stand(trk, t, t + 0.5, x=1.15, wheel=0.0)
    trk.on_wheels(t + 0.5, 0.2, 0.2)                    # it starts to drive: hold off, P opened
    _stand(trk, t + 0.5, t + 1.0, x=1.15, wheel=0.2)
    assert abs(trk.est.pose[0] - 1.15) < 0.02


def test_far_uwb_in_gap_releases_hold():
    trk = _tracker()
    t = _stand(trk, 0.0, 10.0, uwb=(1.0, 1.0))
    _stand(trk, t, t + 3.0, camera=False, uwb=(1.6, 1.0))    # carried away 0.6 m in the gap
    assert trk.zm_releases == 1


def test_blind_travel_counts_wheel_travel():
    trk = _tracker()
    trk.on_wheels(0.0, 0.0, 0.0)
    trk.on_bev(0.0, 1.0, 1.0)
    trk.on_wheels(1.0, 5.0, 5.0)                      # 5 rad on both wheels
    assert trk.blind_travel == pytest.approx(5.0 * WHEEL_RADIUS['follower'])
    trk.on_bev(1.0, 1.16, 1.0)
    assert trk.blind_travel == 0.0
    assert RobotTracker(dict(KW['follower'])).blind_travel is None   # no radius, no wheels


# ── RobotFusion: quality mode ────────────────────────────────────────────────
def _fusion(mode, zero_motion):
    return RobotFusion(dict(KW['follower']), gating=Gating(quality_mode=mode),
                       zero_motion=zero_motion, wheel_radius=WHEEL_RADIUS['follower'])


def _qualities(f, t0, t1, camera, wheel_rate=0.0, x0=1.0, v=0.0):
    """Drive f from t0 to t1 (straight in +x at v, wheels turning at wheel_rate rad/s);
    returns the quality at every step."""
    out = []
    t = t0
    k = 0
    while t < t1:
        x = x0 + v * (t - t0)
        f.on_wheels(t, wheel_rate * t, wheel_rate * t)
        f.on_imu(t, 0.0, 0.0)
        if camera:
            f.on_bev(t, t, x, 1.0)
        if k % 2 == 0:
            f.on_uwb(t, t, x, 1.0)
        f.output(t, t)
        out.append(f.quality(t))
        t += DT
        k += 1
    return out


def test_default_mode_is_the_camera_age_rule():
    assert Gating().quality_mode == Q.CAMERA_AGE
    f = _fusion(Q.CAMERA_AGE, False)
    _qualities(f, 0.0, 5.0, camera=True)
    assert _qualities(f, 5.0, 7.0, camera=False)[-1] == Q.LOST


def test_estimator_standing_gap_stays_good():
    f = _fusion(Q.ESTIMATOR, True)
    _qualities(f, 0.0, 5.0, camera=True)
    assert set(_qualities(f, 5.0, 13.6, camera=False)) == {Q.GOOD}


def test_estimator_blind_lost_within_bound():
    f = _fusion(Q.ESTIMATOR, True)
    _qualities(f, 0.0, 5.0, camera=True, wheel_rate=3.0, v=0.1)
    q = _qualities(f, 5.0, 8.0, camera=False, wheel_rate=3.0, x0=1.5, v=0.1)
    first = q.index(Q.LOST) * DT
    assert first <= Q.EstimatorLimits().blind_time_max + DT
    assert q[-1] == Q.LOST


# ── camera_age_hold: today's rule, GOOD only for a held standing rover ───────
@pytest.mark.parametrize('uwb_age, bev_age, held, sigma, expected', [
    (0.1, 0.1, False, 0.01, Q.GOOD),            # camera fresh: as before
    (0.1, 2.0, False, 0.01, Q.LOST),            # not held: the camera-age rule
    (0.1, 2.0, True, 0.01, Q.GOOD),             # held, sigma small: GOOD through the gap
    (0.1, 2.0, True, 0.05, Q.LOST),             # held but sigma beyond sigma_good
    (2.0, 2.0, True, 0.01, Q.LOST),             # no position source at all
])
def test_classify_held(uwb_age, bev_age, held, sigma, expected):
    assert Q.classify_held(uwb_age, bev_age, held, sigma, 0.03) == expected


def test_classify_unheld_is_camera_age_rule():
    for u in (None, 0.1, 2.0):
        for b in (None, 0.1, 0.9, 1.1, 5.0):
            assert Q.classify_held(u, b, False, 0.0, 0.03) == Q.classify(u, b)


def _hold_fusion():
    return RobotFusion(dict(KW['follower']), gating=Gating(quality_mode=Q.CAMERA_AGE_HOLD),
                       zero_motion=True, wheel_still=True, wheel_radius=WHEEL_RADIUS['follower'])


def test_hold_mode_standing_gap_stays_good():
    f = _hold_fusion()
    _qualities(f, 0.0, 5.0, camera=True)
    assert set(_qualities(f, 5.0, 13.6, camera=False)) == {Q.GOOD}


def test_hold_mode_moving_camera_age_rule():
    f, ref = _hold_fusion(), _fusion(Q.CAMERA_AGE, False)
    for g in (f, ref):
        _qualities(g, 0.0, 5.0, camera=True, wheel_rate=3.0, v=0.1)
    q = _qualities(f, 5.0, 7.0, camera=False, wheel_rate=3.0, x0=1.5, v=0.1)
    assert q == _qualities(ref, 5.0, 7.0, camera=False, wheel_rate=3.0, x0=1.5, v=0.1)
    assert Q.LOST in q


def test_hold_mode_no_stop_go_after_blind():
    """Held GOOD in a gap, starts to move (LOST), stops again: stays LOST until the camera is back."""
    f = _hold_fusion()
    t = 0.0
    _qualities(f, t, 5.0, camera=True)
    _qualities(f, 5.0, 8.0, camera=False)                       # held, GOOD
    q_move = _qualities(f, 8.0, 8.6, camera=False, wheel_rate=1.0)
    assert q_move[-1] == Q.LOST
    for k in range(int(2.0 / DT)):                              # stands again, wheels fixed
        tt = 8.6 + k * DT
        f.on_wheels(tt, 8.6, 8.6)
        f.on_imu(tt, 0.0, 0.0)
        if k % 2 == 0:
            f.on_uwb(tt, tt, 1.0, 1.0)
        f.output(tt, tt)
    assert f.quality(tt) == Q.LOST


def test_hold_mode_dispute_stays_held():
    f = _hold_fusion()
    t = 0.0
    q = []
    while t < 50.0:
        x = 1.0 if t < 5.0 else 1.12                             # arm-shifted / merged box from 5 s on
        f.on_wheels(t, 0.0, 0.0)
        f.on_imu(t, 0.0, 0.0)
        f.on_bev(t, t, x, 1.0)
        f.output(t, t)
        q.append((t, f.quality(t), f.est.pose[0]))
        t += DT
    assert {qq for _, qq, _ in q} == {Q.GOOD}
    assert all(abs(x - 1.0) < 0.01 for _, _, x in q)



# ── IMU: a base moved by hand counts as motion for the hold ──────────────────
G = (0.0, 0.0, 9.81)


def _stand_imu(trk, t0, t1, x=1.0, accel=G, rp=0.0, wz=0.0, camera=True):
    t = t0
    while t < t1:
        trk.on_wheels(t, 0.0, 0.0)
        trk.on_imu(t, 0.0, wz, accel, rp)
        if camera:
            trk.on_bev(t, x, 1.0)
        t += DT
    return t


def test_quiet_imu_holds_against_shifted_box():
    trk = _tracker()
    t = _stand_imu(trk, 0.0, 5.0)
    _stand_imu(trk, t, t + 20.0, x=1.12, accel=(0.05, 0.0, 9.81))   # 0.3 deg: arm on the base
    assert abs(trk.est.pose[0] - 1.0) < 0.01 and trk.imu_releases == 0


def test_lifted_base_followed_and_reheld():
    trk = _tracker()
    t = _stand_imu(trk, 0.0, 5.0)
    tilted = (9.81 * math.sin(math.radians(8.0)), 0.0, 9.81 * math.cos(math.radians(8.0)))
    t = _stand_imu(trk, t, t + 1.0, x=1.25, accel=tilted, rp=0.4)  # lifted, set down 25 cm away
    t = _stand_imu(trk, t, t + 4.0, x=1.25)                         # quiet again
    assert trk.imu_releases == 1
    assert abs(trk.est.pose[0] - 1.25) < 0.02
    _stand_imu(trk, t, t + 10.0, x=1.37)                            # then an arm-shifted box
    assert abs(trk.est.pose[0] - 1.25) < 0.02                       # the new anchor holds


@pytest.mark.parametrize('kw', [dict(wz=0.3), dict(rp=0.2)])
def test_imu_rates_alone_are_base_motion(kw):
    trk = _tracker()
    t = _stand_imu(trk, 0.0, 5.0)
    _stand_imu(trk, t, t + 0.5, x=1.1, **kw)
    assert trk.imu_releases == 1 and not trk.est.zero_velocity
