"""RobotFusion output gating (no ROS): pose gate, alignment flag, COG suppression, quality."""
import pytest

from host_localization.fusion import quality
from host_localization.fusion.frontend import Gating, RobotFusion
from host_localization.fusion.tracker import KW

DT = 0.05       # [s] 20 Hz camera + IMU
SPEED = 0.20    # [m/s] above the course gate (cog_vmin)


def _drive_straight(fusion, t0, seconds, x0=1.0, imu_yaw=0.6):
    """Camera fixes along +x at SPEED, IMU with an arbitrary zero. Returns (end time, outputs)."""
    outputs = []
    steps = int(round(seconds / DT))
    for i in range(1, steps + 1):
        t = t0 + i * DT
        fusion.on_imu(t, imu_yaw, 0.0)
        fusion.on_bev(t, t, x0 + SPEED * (t - t0), 2.0)
        outputs.append(fusion.output(t, t))
    return t0 + steps * DT, outputs


def test_no_output_before_first_fix():
    fusion = RobotFusion(dict(KW['follower']))
    fusion.on_imu(0.0, 0.0, 0.0)
    assert fusion.output(0.0, 0.0) is None


def test_heading_withheld_until_anchored():
    fusion = RobotFusion(dict(KW['follower']))
    _, outputs = _drive_straight(fusion, 0.0, 6.0)
    assert outputs[0].heading is None and not outputs[0] is None
    assert fusion.aligned
    assert outputs[-1].heading == pytest.approx(0.0, abs=0.15)   # course +x, not the IMU zero 0.6


def test_pose_stops_without_fresh_source():
    gating = Gating(pose_timeout=1.0)
    fusion = RobotFusion(dict(KW['follower']), gating=gating)
    t, _ = _drive_straight(fusion, 0.0, 2.0)
    fusion.on_imu(t + 0.9, 0.6, 0.0)
    assert fusion.output(t + 0.9, t + 0.9) is not None
    fusion.on_imu(t + 1.1, 0.6, 0.0)
    assert fusion.output(t + 1.1, t + 1.1) is None        # IMU alone is not a position
    assert fusion.quality(t + 1.1) == quality.LOST


def test_quality_follows_the_freshest_source():
    fusion = RobotFusion(dict(KW['follower']))
    assert fusion.quality(0.0) == quality.LOST
    fusion.on_bev(0.0, 0.0, 1.0, 2.0)
    assert fusion.quality(0.5) == quality.GOOD
    fusion.on_uwb(1.4, 1.4, 1.0, 2.0)
    assert fusion.quality(1.5) == quality.LOST            # camera stale, UWB only (default rule)
    crawl = RobotFusion(dict(KW['follower']), gating=Gating(uwb_only_quality=quality.DEGRADED))
    crawl.on_uwb(0.0, 0.0, 1.0, 2.0)
    assert crawl.quality(0.5) == quality.DEGRADED


def test_course_suppressed_after_alignment():
    """A straight reverse right after alignment (course = heading + 180 deg) must not flip it."""
    def heading_after_reverse(cog_suppress_s):
        fusion = RobotFusion(dict(KW['follower']), gating=Gating(cog_suppress_s=cog_suppress_s))
        t, x = 0.0, 1.0
        while not fusion.aligned:                 # forward until the heading anchors
            t += DT
            x += SPEED * DT
            fusion.on_imu(t, 0.6, 0.0)
            fusion.on_bev(t, t, x, 2.0)
            fusion.output(t, t)
        sample = None
        for _ in range(int(4.0 / DT)):            # reverse 0.8 m inside the 5 s window
            t += DT
            x -= SPEED * DT
            fusion.on_imu(t, 0.6, 0.0)
            fusion.on_bev(t, t, x, 2.0)
            sample = fusion.output(t, t)
        return sample.heading

    assert abs(heading_after_reverse(5.0)) < 0.3            # kept: window covers the reverse
    assert abs(abs(heading_after_reverse(0.0)) - 3.1416) < 0.3   # no window: flipped by the reverse
