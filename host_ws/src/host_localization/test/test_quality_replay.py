"""Field recording (follower) replayed through RobotFusion: the camera-age quality rule vs the
estimator rule with zero-motion.

Fixture data/pick_follower_gaps.npz: follower inputs 1955-2130 s after the bag
start, wheel positions included.
The follower stands at (4.442, 0.630) with its wheels still until ~2084 s while a person beside it
hides it from the camera in gaps of 11.2, 8.3, 11.5 and 23.3 s; a box 0.27 m away (the person) is
detected at ~2065 s. Then it drives home (~2084-2124 s) with the camera.

With the camera-age rule the quality is LOST in every gap, the position slides with UWB, and
after the 23 s gap the wide gate takes the person's box (0.3 m off).
"""
from pathlib import Path

import numpy as np
import pytest

from host_localization.fusion import quality as Q
from host_localization.fusion.frontend import Gating
from host_localization.fusion.replay import load_fixture, make_fusion, replay_robot

FIXTURE = Path(__file__).parent / 'data' / 'pick_follower_gaps.npz'
STAND = (1995.0, 2084.0)            # wheels still, camera gaps inside
TRUTH = (4.442, 0.630)              # camera median while it stood
DRIVE_GAP = (2100.0, 2102.0)        # driving home at ~0.15 m/s with the camera: mask 2 s


def _new_fusion():
    return make_fusion('follower', Gating(quality_mode=Q.ESTIMATOR), zero_motion=True, wheel_still=True)


def _mask(fx, a, b):
    out = dict(fx)
    bev = fx['follower_bev']
    out['follower_bev'] = bev[(bev[:, 0] < a) | (bev[:, 0] >= b)]
    return out


def _span(r, a, b):
    return (r.rx >= a) & (r.rx < b)


def _lost_episodes(q):
    lost = q == Q.LOST
    return int(lost[0]) + int((lost[1:] & ~lost[:-1]).sum())


@pytest.fixture(scope='module')
def fixture():
    return load_fixture(FIXTURE)


@pytest.fixture(scope='module')
def old(fixture):
    return replay_robot(fixture, 'follower')


@pytest.fixture(scope='module')
def new(fixture):
    return replay_robot(fixture, 'follower', fusion=_new_fusion())


def test_old_rule_reproduces_site_failure(old):
    m = _span(old, *STAND)
    dev = np.hypot(old.x[m] - TRUTH[0], old.y[m] - TRUTH[1])
    assert _lost_episodes(old.quality[m]) >= 4
    assert dev.max() > 0.2                       # took the person's box after the long gap


def test_standing_rover_keeps_position_quality(new):
    m = _span(new, *STAND)
    dev = np.hypot(new.x[m] - TRUTH[0], new.y[m] - TRUTH[1])
    assert set(new.quality[m]) == {Q.GOOD}
    assert dev.max() < 0.02
    assert new.still[m].all()


def test_blind_driving_loses_quality_early(fixture):
    r = replay_robot(_mask(fixture, *DRIVE_GAP), 'follower', fusion=_new_fusion())
    ref = replay_robot(fixture, 'follower', fusion=_new_fusion())
    m = _span(r, *DRIVE_GAP)
    err = np.hypot(r.x[m] - np.interp(r.rx[m], ref.rx, ref.x), r.y[m] - np.interp(r.rx[m], ref.rx, ref.y))
    q = r.quality[m]
    first_lost = r.rx[m][q == Q.LOST][0] - DRIVE_GAP[0]
    assert first_lost <= Q.EstimatorLimits().blind_time_max
    assert err[q == Q.GOOD].max() < 0.05
    assert err[q != Q.LOST].max() < 0.10


def test_defaults_keep_camera_age_rule(fixture, old):
    """Default make_fusion = the node defaults: wheel data in the fixture changes nothing."""
    no_wheels = {k: v for k, v in fixture.items() if k != 'follower_wheel_pos'}
    ref = replay_robot(no_wheels, 'follower')
    np.testing.assert_array_equal(ref.x, old.x)
    np.testing.assert_array_equal(ref.quality, old.quality)


def _hold_fusion():
    return make_fusion('follower', Gating(quality_mode=Q.CAMERA_AGE_HOLD), zero_motion=True, wheel_still=True)


@pytest.fixture(scope='module')
def hold(fixture):
    return replay_robot(fixture, 'follower', fusion=_hold_fusion())


def test_hold_mode_standing_rover_stays_good(hold):
    m = _span(hold, *STAND)
    dev = np.hypot(hold.x[m] - TRUTH[0], hold.y[m] - TRUTH[1])
    assert set(hold.quality[m]) == {Q.GOOD}
    assert dev.max() < 0.02


def test_hold_mode_old_rule_when_moving(old, hold):
    common, i_old, i_hold = np.intersect1d(old.rx, hold.rx, return_indices=True)
    moving = ~hold.still[i_hold]
    assert moving.sum() > 500                         # the drive home
    np.testing.assert_array_equal(old.quality[i_old][moving], hold.quality[i_hold][moving])


# ── leader arm moving while the wheels stand ──────
# Arm joints move with hands off (IMU quiet); the camera box centre wanders up to 0.15 m.
ARM_FIXTURE = Path(__file__).parent / 'data' / 'lap_leader_arm_pick.npz'
ARM_PICK_T = 6611.6
# ── leader lifted, turned and set down 0.25 m away by hand, wheels still ──
HAND_FIXTURE = Path(__file__).parent / 'data' / 'pick_leader_hand_move.npz'
HAND_T0, HAND_T1 = 2424.0, 2431.0       # the move (camera), then it stands


@pytest.fixture(scope='module')
def arm_fixture():
    return load_fixture(ARM_FIXTURE)


@pytest.fixture(scope='module')
def hand_fixture():
    return load_fixture(HAND_FIXTURE)


def _held_leader():
    return make_fusion('leader', Gating(quality_mode=Q.CAMERA_AGE_HOLD), zero_motion=True, wheel_still=True)


def _dev_after(r, fx, t):
    bev = fx['leader_bev']
    pre = np.median(bev[(bev[:, 0] > t - 6) & (bev[:, 0] < t - 1), 2:4], axis=0)
    m = r.rx > t
    return np.hypot(r.x[m] - pre[0], r.y[m] - pre[1]), r.quality[m]


def test_old_filter_follows_the_arm(arm_fixture):
    dev, _ = _dev_after(replay_robot(arm_fixture, 'leader'), arm_fixture, ARM_PICK_T)
    assert dev.max() > 0.1


def test_held_leader_stays_through_pick(arm_fixture):
    dev, q = _dev_after(replay_robot(arm_fixture, 'leader', fusion=_held_leader()), arm_fixture, ARM_PICK_T)
    assert dev.max() < 0.03
    assert set(q) == {Q.GOOD}


def _error_to_new_place(r, fx, t):
    bev = fx['leader_bev']
    new = np.median(bev[(bev[:, 0] > HAND_T1 + 2) & (bev[:, 0] < HAND_T1 + 6), 2:4], axis=0)
    m = r.rx > t
    return np.hypot(r.x[m] - new[0], r.y[m] - new[1])


def test_hand_moved_leader_reacquired(hand_fixture):
    """The IMU (tilt, rates) says the base was moved: the hold lets the camera take it."""
    f = _held_leader()
    r = replay_robot(hand_fixture, 'leader', fusion=f)
    assert _error_to_new_place(r, hand_fixture, HAND_T1 + 3.0).max() < 0.03
    assert f.trk.imu_releases >= 1


def test_no_accelerometer_hold_keeps_place(hand_fixture):
    """Wheel evidence alone cannot tell a hand move from an arm-shifted box."""
    fx = dict(hand_fixture)
    fx['leader_imu'] = fx['leader_imu'][:, :4]
    r = replay_robot(fx, 'leader', fusion=_held_leader())
    assert _error_to_new_place(r, fx, HAND_T1 + 3.0).min() > 0.15
