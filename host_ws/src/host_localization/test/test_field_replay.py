"""Field recording (follower driven next to the leader) replayed through RobotFusion.

Fixture: data/motion_follower.npz, extracted from a host bag
(times are bag-relative, origin = bag start).

The replay starts with the heading anchor the live node already had (seed_recorded_heading);
without it a cold start only anchors at ~68 s and the heading differs by a constant ~43 deg.

Two kinds of checks:
  - replay fidelity: up to the camera gap the replay must equal what the live node published
    (same inputs, same rules there), which validates the replay itself;
  - required behaviour from the gap on, judged against the camera detections. The node that made
    the recording failed here: its position slid 1.77 m onto the UWB coordinates while the wheels
    stood still, its heading turned by 38 deg, and none of the 8285 returning camera fixes was used.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from host_localization.fusion.replay import column, load_fixture, replay_robot

FIXTURE = Path(__file__).parent / 'data' / 'motion_follower.npz'
MATCH_WINDOW = 0.02     # [s] a replay step counts as the recorded one when this close in receive time


@pytest.fixture(scope='module')
def fixture():
    return load_fixture(FIXTURE)


@pytest.fixture(scope='module')
def replay(fixture):
    return replay_robot(fixture, 'follower')


@pytest.fixture(scope='module')
def camera_gap(fixture):
    """(last detection before, first detection after) of the longest follower camera gap."""
    rx = column(fixture, 'follower', 'bev', 'rx')
    i = int(np.argmax(np.diff(rx)))
    return rx[i], rx[i + 1]


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def nearest_step(replay, rx):
    """Index of the replay step nearest in time to each rx."""
    i = np.searchsorted(replay.rx, rx).clip(1, len(replay.rx) - 1)
    return np.where(np.abs(replay.rx[i - 1] - rx) < np.abs(replay.rx[i] - rx), i - 1, i)


def heading_offset(replay, imu, t0, t1):
    """Median of (fused heading - IMU yaw) [deg] over the replay steps in [t0, t1)."""
    m = (replay.rx >= t0) & (replay.rx < t1) & ~np.isnan(replay.heading)
    imu_yaw = np.interp(replay.rx[m], imu[:, 0], np.unwrap(imu[:, 2]))
    return float(np.degrees(np.median(wrap(replay.heading[m] - imu_yaw))))


def test_camera_gap_matches_field_notes(camera_gap):
    # measured from the first topic message, which is 1.0 s after the bag start used here
    assert camera_gap == pytest.approx((67.4, 93.6), abs=0.2)


def recorded_before(fixture, replay, t_end):
    """(replay step index, mask) of the recorded poses the replay can be compared with before t_end."""
    rx = fixture['follower_pose'][:, 0]
    j = nearest_step(replay, rx)
    return j, (np.abs(replay.rx[j] - rx) < MATCH_WINDOW) & (rx < t_end)


def camera_course_legs(fixture, min_len=0.30, max_time=4.0):
    """(t0, t1, course) of camera-track legs at least min_len long driven within max_time."""
    rx = column(fixture, 'follower', 'bev', 'rx')
    x, y = column(fixture, 'follower', 'bev', 'x'), column(fixture, 'follower', 'bev', 'y')
    legs, k = [], 0
    while k < len(rx) - 1:
        ahead = np.flatnonzero(np.hypot(x[k:] - x[k], y[k:] - y[k]) >= min_len)
        j = k + int(ahead[0]) if len(ahead) else None
        if j is None or rx[j] - rx[k] > max_time:
            k += 1
            continue
        legs.append((rx[k], rx[j], math.atan2(y[j] - y[k], x[j] - x[k])))
        k = j
    return legs


def test_replay_position_up_to_gap(fixture, replay, camera_gap):
    j, compared = recorded_before(fixture, replay, camera_gap[0])
    err = np.hypot(replay.x[j] - column(fixture, 'follower', 'pose', 'x'),
                   replay.y[j] - column(fixture, 'follower', 'pose', 'y'))[compared]
    assert compared.sum() > 2500
    assert np.percentile(err, 99) < 0.01
    assert err.max() < 0.03


def test_replay_heading_up_to_gap(fixture, replay, camera_gap):
    imu = fixture['follower_imu']
    j, compared = recorded_before(fixture, replay, camera_gap[0])
    # before the first IMU sample (38.7 s) the bag has no yaw to replay
    compared &= fixture['follower_pose'][:, 0] > imu[0, 0] + 5.0
    recorded = column(fixture, 'follower', 'pose', 'yaw')
    assert (np.isnan(replay.heading[j][compared]) == np.isnan(recorded[compared])).all()
    both = compared & ~np.isnan(recorded)
    err = np.degrees(np.abs(wrap(replay.heading[j][both] - recorded[both])))
    assert both.sum() > 500
    # same rules as the recorded node up to the gap, except that course aiding now needs a
    # fresh camera fix (p99 difference 1.7 deg)
    assert np.percentile(err, 99) < 2.5
    assert err.max() < 4.0


def test_position_in_camera_frame_in_gap(fixture, replay, camera_gap):
    lost, back = camera_gap
    bev = fixture['follower_bev']
    rx = column(fixture, 'follower', 'bev', 'rx')
    xy = np.column_stack([column(fixture, 'follower', 'bev', 'x'), column(fixture, 'follower', 'bev', 'y')])
    last_seen, seen_again = xy[rx <= lost][-1], xy[rx >= back][0]
    assert np.linalg.norm(seen_again - last_seen) < 0.10            # the rover rolled 8 cm, then stood
    gap = (replay.rx > lost) & (replay.rx < back)
    fused = np.column_stack([replay.x[gap], replay.y[gap]])
    # UWB alone would carry it for 26 s and slide 1.77 m; the fusion stays within 0.15 m of
    # where the camera finds it again
    assert np.linalg.norm(fused - seen_again, axis=1).max() < 0.25
    assert np.linalg.norm(fused[-1] - seen_again) < 0.12
    assert len(bev) > 9000


def test_heading_holds_while_camera_blind(fixture, replay, camera_gap):
    lost, back = camera_gap
    imu = fixture['follower_imu']
    before = heading_offset(replay, imu, lost - 6.0, lost)
    after = heading_offset(replay, imu, back, back + 4.0)
    assert abs(after - before) < 5.0                                # recorded node: 38 deg; now 1.5 deg


def test_returning_camera_fixes_followed(fixture, replay, camera_gap):
    lost, back = camera_gap
    after = replay.bev_rx > back
    assert after.sum() > 8000 and replay.bev_used[after].all()      # recorded node: 0 of 8285
    rx = column(fixture, 'follower', 'bev', 'rx')
    err = np.hypot(np.interp(rx, replay.rx, replay.x) - column(fixture, 'follower', 'bev', 'x'),
                   np.interp(rx, replay.rx, replay.y) - column(fixture, 'follower', 'bev', 'y'))[rx > back]
    assert np.percentile(err, 99) < 0.02 and err.max() < 0.04       # measured 10 mm / 24 mm


def test_heading_matches_camera_course(fixture, replay, camera_gap):
    legs = [leg for leg in camera_course_legs(fixture) if leg[0] > camera_gap[1]]
    assert len(legs) >= 2                                           # the 0.75 m straight at ~433 s
    for t0, t1, course in legs:
        on_leg = (replay.rx >= t0) & (replay.rx <= t1) & ~np.isnan(replay.heading)
        heading = replay.heading[on_leg]
        mean = math.atan2(np.sin(heading).mean(), np.cos(heading).mean())
        assert abs(math.degrees(wrap(mean - course))) < 8.0
