"""UWB and camera in different frames (seen on site), synthetic and without ROS.

On site the UWB coordinates were about 1.7 m away from the camera coordinates. When the camera
lost the rover for 26 s the fused position slid onto the UWB coordinates, the fake motion was
read as a course (heading wrong by 35 deg) and the returning camera was rejected for good.
Required behaviour, whatever the offset between the two frames:
  - the camera defines the frame; UWB carries the position through a camera gap *in that frame*;
  - a heading correction from the driven course needs motion the camera saw;
  - UWB still identifies the rover (a camera fix of the other rover is refused);
  - a fresh, steady camera track that agrees with UWB is never refused for long.
"""
import math

import numpy as np
import pytest

from host_localization.fusion.frontend import RobotFusion
from host_localization.fusion.tracker import KW

DT = 0.05                       # [s] 20 Hz camera and IMU; UWB every second step (10 Hz)
SPEED = 0.20                    # [m/s]
SITE_OFFSET = (1.27, 1.23)      # [m] UWB - camera
UWB_NOISE = 0.04                # [m] white, per axis
IMU_ZERO = 0.6                  # [rad] arbitrary IMU zero: heading = imu_yaw + offset


class Rig:
    """Feeds one RobotFusion from a scripted true pose; keeps the last output and BEV verdict."""

    def __init__(self, robot='follower', offset=SITE_OFFSET, seed=0):
        self.fusion = RobotFusion(dict(KW[robot]))
        self.rng = np.random.default_rng(seed)
        self.offset = offset
        self.t, self.k = 0.0, 0
        self.x, self.y, self.heading = 1.0, 2.0, 0.0
        self.out = None
        self.bev_used = []

    def step(self, speed=0.0, camera=True, uwb=True, camera_at=None, uwb_offset=None):
        """One 50 ms step. camera_at overrides where the camera reports the rover (ID switch)."""
        self.t += DT
        self.k += 1
        self.x += speed * DT * math.cos(self.heading)
        self.y += speed * DT * math.sin(self.heading)
        t = self.t
        self.fusion.on_imu(t, self.heading - IMU_ZERO, 0.0)
        if camera:
            cx, cy = camera_at or (self.x, self.y)
            jx, jy = self.rng.normal(0.0, 0.002, 2)
            self.bev_used.append(self.fusion.on_bev(t, t, cx + jx, cy + jy))
        if uwb and self.k % 2 == 0:
            ox, oy = uwb_offset or self.offset
            nx, ny = self.rng.normal(0.0, UWB_NOISE, 2)
            self.fusion.on_uwb(t, t, self.x + ox + nx, self.y + oy + ny)
        self.out = self.fusion.output(t, t)
        return self.out

    def run(self, seconds, **kw):
        for _ in range(int(round(seconds / DT))):
            self.step(**kw)

    def position_error(self):
        return math.hypot(self.out.x - self.x, self.out.y - self.y)

    def heading_error_deg(self):
        d = self.out.heading - self.heading
        return abs(math.degrees(math.atan2(math.sin(d), math.cos(d))))


def driven_and_parked(**kw):
    rig = Rig(**kw)
    rig.run(8.0, speed=SPEED)       # 1.6 m straight: heading anchored
    rig.run(3.0)                    # parked, camera and UWB present
    assert rig.fusion.aligned and rig.heading_error_deg() < 5.0 and rig.position_error() < 0.02
    return rig


def test_camera_gap_keeps_camera_frame():
    rig = driven_and_parked()
    worst = 0.0
    for _ in range(int(26.0 / DT)):             # the site gap: camera lost, UWB keeps coming
        rig.step(camera=False)
        worst = max(worst, rig.position_error())
    assert worst < 0.10, 'UWB carries the position in the camera frame (site: slid 1.7 m)'
    assert rig.heading_error_deg() < 5.0, 'no course correction without camera-seen motion (site: 35 deg)'
    rig.bev_used.clear()
    rig.run(3.0)
    assert np.mean(rig.bev_used) > 0.9, 'returning camera is used again (site: refused for good)'
    assert rig.position_error() < 0.02


def test_uwb_noise_never_turns_heading():
    rig = driven_and_parked(offset=(0.0, 0.0))
    for _ in range(int(120.0 / DT)):
        rig.step(camera=False)
    assert rig.heading_error_deg() < 5.0


def test_other_rover_camera_fix_refused():
    """ID switch: for 3 s the camera reports this rover where the other one stands (0.55 m away)."""
    rig = driven_and_parked()
    other = (rig.x, rig.y + 0.55)
    rig.bev_used.clear()
    worst = 0.0
    for _ in range(int(3.0 / DT)):
        rig.step(camera_at=other)
        worst = max(worst, rig.position_error())
    assert not any(rig.bev_used)
    assert worst < 0.10


def test_no_pose_before_camera_placement():
    rig = Rig()
    rig.run(2.0, camera=False)                  # UWB first: its coordinates are not the frame
    assert rig.out is None, 'a pose 1.7 m off in another frame is worse than none'
    rig.step()                                  # first camera fix
    assert rig.position_error() < 0.03, 'the camera defines the frame'
    rig.run(1.0)
    assert rig.position_error() < 0.03


def test_late_uwb_does_not_pull_position():
    rig = Rig()
    worst = 0.0
    rig.run(2.0, uwb=False)                     # camera only
    for _ in range(int(2.0 / DT)):              # UWB joins, 1.7 m away
        rig.step()
        worst = max(worst, rig.position_error())
    assert worst < 0.03


def test_camera_track_agreeing_uwb_reacquired():
    """Rover carried 1 m during a camera gap to a place where the UWB error differs by 0.25 m:
    the returning camera disagrees with the prediction, but it is steady and UWB confirms the
    rover is there -> used again within 2 s."""
    rig = driven_and_parked()
    rig.run(2.0, camera=False)
    rig.x += 1.0                                # carried while nobody looked
    shifted = (SITE_OFFSET[0] + 0.25, SITE_OFFSET[1])
    rig.run(5.0, camera=False, uwb_offset=shifted)
    rig.bev_used.clear()
    rig.run(2.0, uwb_offset=shifted)
    assert rig.bev_used[-1] and rig.position_error() < 0.03


@pytest.mark.parametrize('robot', ['leader', 'follower'])
def test_uwb_error_field_stays_on_camera(robot):
    """UWB error changing 0.15 m per metre of travel must not show in the fused position."""
    rig = Rig(robot=robot)
    worst = 0.0
    for _ in range(int(15.0 / DT)):
        drift = 0.15 * (rig.x - 1.0)
        rig.step(speed=SPEED, uwb_offset=(SITE_OFFSET[0] + drift, SITE_OFFSET[1] - drift))
        if rig.t > 1.0:
            worst = max(worst, rig.position_error())
    assert worst < 0.03
    assert rig.heading_error_deg() < 5.0
