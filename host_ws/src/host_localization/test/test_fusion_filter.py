"""Filter-level regression tests (pure numpy, no ROS): pytest test/test_fusion_filter.py

Covers the monotonic filter clock, the localization quality rule, the short-reverse rule
(must not flip a confirmed heading) and the box-axis heading aid.
"""
import math

import numpy as np
import pytest

from host_localization.fusion import quality
from host_localization.fusion.se2nhc import HOFF, wrap
from host_localization.fusion.tracker import KW, RobotTracker

DT = 0.05            # 20 Hz IMU and 20 Hz BEV, like the recorded bag
BEV_LATENCY = 0.035  # BEV stamp is this much older than the IMU stamp that arrived before it
H_OFF = 2.0          # true heading - imu_yaw [rad] (follower IMU has a large fixed offset)


class RewindingTracker(RobotTracker):
    """Pre-fix clock: a late stamp rewinds filter time (kept to prove the test catches it)."""

    def advance(self, t):
        if self.last_t is not None and t > self.last_t:
            self.est.predict(t - self.last_t)
        self.last_t = t
        return 0.0


def drive(trk, speed, heading, duration, start=(0.0, 0.0), t0=0.0, obb=False, rng=None):
    """Straight segment: IMU then a late-stamped BEV fix every DT. speed<0 = reversing."""
    rng = rng or np.random.default_rng(0)
    x0, y0 = start
    for k in range(int(duration / DT)):
        t = t0 + k * DT
        trk.on_imu(t, heading - H_OFF, 0.0)
        tb = t - BEV_LATENCY
        d = speed * (tb - t0)
        nx, ny = rng.normal(0.0, 0.002, 2)
        axis = heading + rng.normal(0.0, math.radians(2.0)) if obb else None
        trk.on_bev(tb, x0 + d * math.cos(heading) + nx, y0 + d * math.sin(heading) + ny, axis)
        trk.step_cog()
    d = speed * duration
    return (x0 + d * math.cos(heading), y0 + d * math.sin(heading)), t0 + duration


def heading_err(trk, heading):
    return abs(wrap(trk.est.heading() - heading))


@pytest.mark.parametrize('cls,lo,hi', [(RobotTracker, 0.93, 1.07), (RewindingTracker, 0.45, 0.75)])
def test_speed_scale_with_interleaved_stamps(cls, lo, hi):
    """With late BEV stamps the fused speed must match the true speed (old clock: ~0.6x)."""
    trk = cls(dict(KW['follower']))
    drive(trk, 0.05, 0.3, 20.0)
    assert lo < trk.est.speed / 0.05 < hi


def test_filter_clock_never_goes_back():
    trk = RobotTracker(dict(KW['follower']))
    trk.on_imu(10.0, 0.0, 0.0)
    assert trk.advance(9.9) == pytest.approx(0.1)
    assert trk.last_t == 10.0


def test_quality_rule():
    assert quality.classify(0.1, 0.1) == quality.GOOD
    assert quality.classify(0.1, 1.5) == quality.LOST          # UWB only: stop
    assert quality.classify(0.1, None) == quality.LOST
    assert quality.classify(0.1, 1.5, uwb_only=quality.DEGRADED) == quality.DEGRADED
    assert quality.classify(None, 0.1) == quality.GOOD         # camera only: full trust
    assert quality.classify(1.5, 1.5) == quality.LOST
    assert quality.classify(None, None) == quality.LOST
    assert quality.position_age(None, 0.4) == 0.4
    assert quality.position_age(None, None) is None


def test_short_reverse_keeps_confirmed_heading():
    """1 m of reversing after 3 m of confirmed forward travel must not flip the heading."""
    trk = RobotTracker(dict(KW['follower']))
    pos, t = drive(trk, 0.2, 0.5, 15.0)
    assert trk.est.heading_valid and heading_err(trk, 0.5) < math.radians(10)
    drive(trk, -0.2, 0.5, 5.0, start=pos, t0=t)
    assert heading_err(trk, 0.5) < math.radians(10)


def test_wrong_bootstrap_still_recovers():
    """A heading anchored 180 deg wrong (bootstrap while reversing) flips back on forward travel."""
    trk = RobotTracker(dict(KW['follower']))
    pos, t = drive(trk, -0.2, 0.5, 4.0)            # anchors on the tail direction
    assert trk.est.heading_valid and heading_err(trk, 0.5) > math.radians(150)
    drive(trk, 0.2, 0.5, 8.0, start=pos, t0=t)
    assert heading_err(trk, 0.5) < math.radians(10)


def test_box_axis_holds_heading_while_stopped():
    """Parked for 10 min the heading sigma stays small with the box axis, grows without."""
    sig = {}
    for use_obb in (False, True):
        trk = RobotTracker(dict(KW['leader']), use_obb=use_obb)
        pos, t = drive(trk, 0.2, 0.5, 15.0, obb=use_obb)
        drive(trk, 0.0, 0.5, 600.0, start=pos, t0=t, obb=use_obb)
        sig[use_obb] = math.degrees(math.sqrt(trk.est.P[HOFF, HOFF]))
        if use_obb:
            assert heading_err(trk, 0.5) < math.radians(5)
    assert sig[False] > 60.0 and sig[True] < 5.0


def test_box_axis_ignored_until_anchored():
    """The box axis is ambiguous (mod 90 deg): it must not create a heading on its own."""
    trk = RobotTracker(dict(KW['follower']), use_obb=True)
    drive(trk, 0.0, 0.5, 5.0, obb=True)
    assert not trk.est.heading_valid
    assert trk.est.x[HOFF] == 0.0


def test_box_axis_wrong_branch_recovers():
    """Heading forced 90 deg off with the box axis locked on it: COG hops it back."""
    trk = RobotTracker(dict(KW['follower']), use_obb=True)
    pos, t = drive(trk, 0.2, 0.5, 10.0, obb=True)
    trk.est._reanchor(trk.est.x[HOFF] + math.pi / 2)
    trk.est.P[HOFF, HOFF] = math.radians(3.0) ** 2
    drive(trk, 0.2, 0.5, 10.0, start=pos, t0=t, obb=True)
    assert heading_err(trk, 0.5) < math.radians(10)


def steer(trk, speed, omega, duration, pose, t0, lever=0.0):
    """Unicycle motion from pose (x, y, heading) of the turn centre; the camera sees the point
    ``lever`` ahead of it. IMU exact (yaw = heading - H_OFF). Returns (pose, time, worst error)."""
    x, y, h = pose
    worst = 0.0
    for k in range(int(round(duration / DT))):
        x += speed * DT * math.cos(h)
        y += speed * DT * math.sin(h)
        h = wrap(h + omega * DT)
        t = t0 + (k + 1) * DT
        trk.on_imu(t, wrap(h - H_OFF), omega)
        trk.on_bev(t, x + lever * math.cos(h), y + lever * math.sin(h))
        trk.step_cog()
        if trk.est.heading_valid:
            worst = max(worst, heading_err(trk, h))
    return (x, y, h), t0 + duration, worst


@pytest.mark.parametrize('radius', [0.37, 0.6, 1.2, 3.0])
def test_arc_course_does_not_bias_heading(radius):
    """The chord of a 0.15 m segment points where the rover headed half a segment ago: 12 deg
    behind the current heading on a 0.37 m arc, 3.6 deg on a 1.2 m one (leader, 0.10 m/s, IMU
    exact)."""
    trk = RobotTracker(dict(KW['leader']))
    pose, t, _ = steer(trk, 0.10, 0.0, 30.0, (0.0, 0.0, 0.5), 0.0)
    assert heading_err(trk, 0.5) < math.radians(1.5)
    pose, t, on_arc = steer(trk, 0.10, 0.10 / radius, 2.0 * radius / 0.10, pose, t)   # 115 deg of arc
    assert on_arc < math.radians(1.5)


def test_tracked_point_ahead_on_tight_arc():
    """A point 0.05 m ahead of the turn centre moves atan(0.05 / R) outward of the heading
    (7.7 deg at R 0.37): neither the course nor the no-side-slip constraint may steer the
    heading on such an arc."""
    trk = RobotTracker(dict(KW['leader']))
    pose, t, _ = steer(trk, 0.10, 0.0, 30.0, (0.0, 0.0, 0.5), 0.0, lever=0.05)
    pose, t, on_arc = steer(trk, 0.10, 0.10 / 0.37, 7.4, pose, t, lever=0.05)
    assert on_arc < math.radians(1.5)


def test_pivot_segment_gives_no_course():
    """In-place turn with the tracked point 0.05 m ahead of the turn centre: the point moves
    0.10 m sideways, and the segment that ends on the drive-off points nowhere near the heading."""
    trk = RobotTracker(dict(KW['leader']))
    pose, t, _ = steer(trk, 0.10, 0.0, 30.0, (0.0, 0.0, 0.0), 0.0, lever=0.05)
    pose, t, _ = steer(trk, 0.0, 0.0, 3.0, pose, t, lever=0.05)
    pose, t, turning = steer(trk, 0.0, 0.35, math.pi / 0.35, pose, t, lever=0.05)
    pose, t, leaving = steer(trk, 0.10, 0.0, 6.0, pose, t, lever=0.05)
    assert max(turning, leaving) < math.radians(2.0)


def test_standing_time_keeps_first_course_fast():
    """A rover parked for 6 s and then creeping at 0.085 m/s: the first 0.15 m segment took
    7.8 s from where it was opened, 0.019 m/s, and was refused as too slow; the heading came
    one segment (0.15 m, 2-3 s of creep) later. The segment's speed is that of its moving part."""
    trk = RobotTracker(dict(KW['follower']))
    pos, t = drive(trk, 0.0, 0.7, 6.0)
    travelled = 0.0
    while not trk.est.heading_valid and travelled < 1.0:
        pos, t = drive(trk, 0.085, 0.7, 0.25, start=pos, t0=t)
        travelled += 0.085 * 0.25
    assert travelled < 3 * 0.15 + 0.05          # three agreeing courses, none lost
    assert heading_err(trk, 0.7) < math.radians(5.0)
