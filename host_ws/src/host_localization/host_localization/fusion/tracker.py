"""Per-robot tracker: EKFSE2 + monotonic filter clock + COG trigger + wheel-hold logic (no rclpy).

A measurement stamped not newer than the filter time is applied without predict.
"""
import math
from collections import deque

import numpy as np

from .se2nhc import EKFSE2

COG_MIN_MOVE = 0.15     # [m] min displacement between COG samples
COG_STILL = 0.03        # [m] within this of a segment's start the rover has not left yet
CAMERA_FRESH_S = 0.3    # [s] camera "sees the motion" while its last used fix is this recent
UWB_FRESH_S = 1.0       # [s] a UWB fix this recent can vouch for a camera fix
WHEEL_STILL_S = 0.5     # [s] still = no wheel turned more than WHEEL_STILL_RAD [rad] this long
WHEEL_STILL_RAD = 0.05
WHEEL_FRESH_S = 0.3     # [s] older wheel data counts as not stationary

# Per-robot EKF parameters.
KW = {
    'leader':   dict(q_vel=0.02, q_hoff=5e-3, r_nhc=0.10 ** 2, uwb_sigma0=0.12, r_bev=0.03 ** 2, use_wheel=False),
    'follower': dict(q_vel=0.02, q_hoff=2e-4, r_nhc=0.045 ** 2, uwb_sigma0=0.20, r_bev=0.03 ** 2, use_wheel=False),
}

# Use the camera box axis as a heading aid.
OBB_HEADING = {'leader': False, 'follower': True}

# Freeze the heading offset while the wheels are still.
WHEEL_STILL = {'leader': True, 'follower': False}

# Zero-velocity update while the wheels are still.
ZERO_MOTION = {'leader': False, 'follower': False}
# Wheel radius [m/rad]: travel since the last camera fix (blind-travel bound).
WHEEL_RADIUS = {'leader': 0.037, 'follower': 0.032}
# Position hold: a camera fix farther than ZM_DISPUTE_M [m] from the hold anchor is counted, not
# used; the anchor is set once sigma is below ZM_ANCHOR_SIGMA [m]; offsets beyond ZM_DISPUTE_MAX_M
# [m] are another object and do not open P.
ZM_DISPUTE_M = 0.03
ZM_ANCHOR_SIGMA = 0.02
ZM_DISPUTE_MAX_M = 0.25
# IMU says the base was moved by hand: tilt > ZM_TILT_DEG, roll/pitch rate > ZM_RP_RATE or yaw rate
# > ZM_WZ_RATE for ZM_IMU_MIN_SAMPLES consecutive samples (one is an impact); counts as moving
# until ZM_IMU_QUIET_S [s] after the last such sample.
ZM_TILT_DEG = 3.0
ZM_RP_RATE = 0.08
ZM_WZ_RATE = 0.08
ZM_IMU_QUIET_S = 2.0
ZM_IMU_MIN_SAMPLES = 2
# The hold ends after ZM_UWB_RELEASE_N consecutive offset-corrected UWB fixes farther than
# ZM_UWB_RELEASE_M [m] without camera confirmation (rover carried away).
ZM_UWB_RELEASE_M = 0.35
ZM_UWB_RELEASE_N = 10


class RobotTracker:
    """One robot: EKF, filter clock and the course-over-ground (COG) reference point."""

    def __init__(self, kw, cog_min_move=COG_MIN_MOVE, use_obb=False, wheel_still=False,
                 zero_motion=False, wheel_radius=None):
        self.use_obb = bool(use_obb)
        self.wheel_still = bool(wheel_still)    # still wheels freeze the heading offset
        self.zero_motion = bool(zero_motion)    # still wheels hold the position (ZUPT)
        self.wheel_radius = wheel_radius        # [m/rad] or None: no travel from the wheels
        self._wheels = deque()      # (t, left, right) positions [rad] over WHEEL_STILL_S
        self._wheels_still_t = None  # time of the last sample that closed a still window
        self._wheel_last = None     # (left, right) of the last sample
        self._travel = 0.0          # [m] wheel travel since start (pivots count too)
        self._travel_at_bev = 0.0   # _travel at the last used camera fix
        self._zm_released = False   # camera / UWB overruled the wheels in this still episode
        self._zm_disputed = 0       # consecutive camera fixes not used (they leave the hold)
        self._zm_dispute_m = 0.0    # [m] largest such offset this still episode
        self._zm_anchor = None      # (x, y) the hold keeps, once settled
        self._imu_ref = None        # unit specific force of the current quiet stand
        self._imu_disturbed_t = None  # filter time of the last IMU sample showing base motion
        self._imu_hits = 0            # consecutive such samples
        self.imu_releases = 0       # diagnostics: IMU ended a hold
        self._zm_uwb_far = 0        # consecutive far UWB fixes while holding (no camera)
        self.zm_releases = 0        # diagnostics: camera / UWB overruled the wheels
        self.last_fix_disputed = False  # last camera fix was seen but not used (hold dispute)
        self.arm_busy = False       # set by the node: the leader's arm is working (see _frozen)
        self.moved_since_camera = True  # wheels not proven still since the last used camera fix
        self.est = EKFSE2(**kw)
        self.cog_min_move = float(cog_min_move)
        self.qf = 70.0
        self.last_wz = 0.0      # latest gyro z [rad/s]
        self.last_t = None      # filter time [s]; never decreases
        self._cog_ref = None    # (x, y, time) where the current COG segment started
        self._cog_left = (0.0, 0.0)  # (distance, time) when last within COG_STILL of the start
        self._cog_turned = 0.0  # |IMU yaw change| summed over the segment
        self._cog_dyaw = 0.0    # net IMU yaw change
        self._imu_yaw = None    # previous IMU yaw
        self._bev_t = None      # filter time of the last used camera fix
        self._uwb = None        # (filter time, x, y) of the last UWB fix

    def advance(self, t):
        """Predict up to t; returns the lag [s] (>0) when t is older than the filter time."""
        wheels_still = self._stationary(t)
        still = wheels_still and not self._imu_disturbed(t)
        if not still:                       # a new still episode starts with the wheels trusted
            self.moved_since_camera = True
            if self._zm_dispute_m > 0.0:
                self.est.inflate_position(self._zm_dispute_m)
            self._zm_released = False
            self._zm_disputed = self._zm_uwb_far = 0
            self._zm_dispute_m = 0.0
            self._zm_anchor = None
        self.est.stationary = wheels_still and self.wheel_still
        self.est.zero_velocity = still and self.zero_motion and not self._zm_released
        if (self.est.zero_velocity and self._zm_anchor is None and self.est.in_camera_frame
                and self.est.position_sigma <= ZM_ANCHOR_SIGMA):
            self._zm_anchor = self.est.pose[:2]
        if self.last_t is None:
            self.last_t = t
            return 0.0
        if t <= self.last_t:
            return self.last_t - t          # late stamp: caller updates without predict
        self.est.camera_fresh = self._camera_fresh(t)
        self.est.predict(t - self.last_t)
        self.last_t = t
        return 0.0

    def _stationary(self, t):
        return self._wheels_still_t is not None and t - self._wheels_still_t <= WHEEL_FRESH_S

    @property
    def wheels_still(self):
        """Encoders say the rover stands still (False without fresh wheel data)."""
        return self.last_t is not None and self._stationary(self.last_t)

    @property
    def blind_travel(self):
        """[m] wheel travel since the last used camera fix; None without radius or wheel data."""
        if self.wheel_radius is None or self._wheel_last is None:
            return None
        return self._travel - self._travel_at_bev

    def on_wheels(self, t, left, right):
        """Wheel positions [rad]: still test (gates heading freeze and ZUPT) and travel."""
        h = self._wheels
        if h and t < h[-1][0]:
            return                          # out of order: keep the window monotonic
        if self._wheel_last is not None and self.wheel_radius is not None:
            self._travel += 0.5 * self.wheel_radius * (abs(left - self._wheel_last[0])
                                                       + abs(right - self._wheel_last[1]))
        self._wheel_last = (float(left), float(right))
        h.append((t, float(left), float(right)))
        while len(h) > 2 and h[1][0] <= t - WHEEL_STILL_S:
            h.popleft()
        span = max(max(s[i] for s in h) - min(s[i] for s in h) for i in (1, 2))
        covered = t - h[0][0] >= WHEEL_STILL_S
        self._wheels_still_t = t if covered and span < WHEEL_STILL_RAD else None

    def _frozen(self):
        """Leader arm working and wheels standing: hold the position against every source."""
        return self.arm_busy and self.wheels_still

    def _camera_fresh(self, t):
        return self._bev_t is not None and t - self._bev_t <= CAMERA_FRESH_S

    def _fresh_uwb_fix(self):
        if self._uwb is None or self.last_t - self._uwb[0] > UWB_FRESH_S:
            return None
        return self._uwb[1:]

    def on_imu(self, t, yaw, wz, accel=None, rp_rate=None):
        """IMU yaw/rate; accel (specific force, m/s^2) and rp_rate (rad/s) detect hand moves."""
        self.advance(t)
        if accel is not None and not self._frozen():
            self._check_imu_motion(wz, accel, rp_rate)
        if self._imu_yaw is not None:
            step = math.remainder(yaw - self._imu_yaw, 2.0 * math.pi)
            self._cog_turned += abs(step)
            self._cog_dyaw += step
        self._imu_yaw = float(yaw)
        self.est.set_imu_yaw(yaw, wz)
        self.last_wz = float(wz)

    def on_uwb(self, t, x, y):
        self.advance(t)
        if self._frozen():
            return
        if self.est.zero_velocity and not self._camera_fresh(self.last_t):
            self._check_uwb_against_hold(x, y)
        self.est.update_uwb(x, y, self.qf)
        self._uwb = (self.last_t, x, y)

    def on_bev(self, t, x, y, obb_yaw=None):
        """True when the BEV position was used (first sight, passed the gate, or re-acquired)."""
        self.advance(t)
        if self._frozen():
            self.last_fix_disputed = True       # seen, not used
            return False
        if self._disputes_hold(x, y):
            return False
        used = self.est.update_bev(x, y, self._fresh_uwb_fix())
        if used:
            self._camera_used()
            self._rearm_hold(x, y)
            if self.use_obb and obb_yaw is not None:
                self.est.update_obb_heading(obb_yaw)
        return used

    def _camera_used(self):
        self._bev_t = self.last_t
        self._travel_at_bev = self._travel
        self.moved_since_camera = not self.wheels_still

    def _disputes_hold(self, x, y):
        """True when the held anchor is settled and the fix is farther than ZM_DISPUTE_M."""
        self.last_fix_disputed = False
        if not self.est.zero_velocity or self._zm_anchor is None:
            self._zm_disputed = 0
            return False
        dist = math.hypot(x - self._zm_anchor[0], y - self._zm_anchor[1])
        if dist <= ZM_DISPUTE_M:
            self._zm_disputed = 0
            return False
        self._zm_disputed += 1
        if dist <= ZM_DISPUTE_MAX_M:
            self._zm_dispute_m = max(self._zm_dispute_m, dist)
        self.last_fix_disputed = True
        return True

    @property
    def camera_disputes(self):
        """Consecutive camera fixes not used (gate refusal or hold dispute)."""
        return max(self._zm_disputed, self.est.refused_track)

    def _check_imu_motion(self, wz, accel, rp_rate):
        """IMU shows the base moving while wheels stand: motion for the hold, not the heading freeze."""
        a = np.asarray(accel, dtype=float)
        n = float(np.linalg.norm(a))
        if n < 1e-6 or self.last_t is None:
            return
        if self._imu_ref is None:
            self._imu_ref = a / n               # gravity direction
            return
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ self._imu_ref) / n))))
        if not (tilt > ZM_TILT_DEG or abs(wz) > ZM_WZ_RATE
                or (rp_rate is not None and rp_rate > ZM_RP_RATE)):
            self._imu_hits = 0
            return
        self._imu_hits += 1
        if self._imu_hits < ZM_IMU_MIN_SAMPLES and not self._imu_disturbed(self.last_t):
            return                              # an impact, not a moved base
        if not self._imu_disturbed(self.last_t) and self.zero_motion:
            if self._zm_dispute_m > 0.0:
                self.est.inflate_position(self._zm_dispute_m)
            self.imu_releases += 1
        self._imu_disturbed_t = self.last_t
        self._imu_ref = None                    # re-taken when quiet

    def _imu_disturbed(self, t):
        return self._imu_disturbed_t is not None and t - self._imu_disturbed_t <= ZM_IMU_QUIET_S

    def _check_uwb_against_hold(self, x, y):
        """Without the camera, far offset-corrected UWB for ~1 s ends the hold (carried away)."""
        offset = self.est.uwb_offset
        if offset is None:
            return
        px, py, _ = self.est.pose
        far = math.hypot(x - offset[0] - px, y - offset[1] - py) > ZM_UWB_RELEASE_M
        self._zm_uwb_far = self._zm_uwb_far + 1 if far else 0
        if self._zm_uwb_far >= ZM_UWB_RELEASE_N:
            self._release_hold(max(ZM_UWB_RELEASE_M, self._zm_dispute_m))

    def _rearm_hold(self, x, y):
        """The hold returns once the estimate has caught up with the camera."""
        px, py, _ = self.est.pose
        if self._zm_released and math.hypot(x - px, y - py) <= ZM_DISPUTE_M:
            self._zm_released = False

    def _release_hold(self, offset):
        """End the hold for this still episode; inflate position sigma by offset [m]."""
        self._zm_released = True
        self._zm_uwb_far = self._zm_disputed = 0
        self._zm_dispute_m = 0.0
        self._zm_anchor = None
        self.est.zero_velocity = False
        self.est.inflate_position(offset)
        self.zm_releases += 1

    def on_bev_candidate(self, t, x, y, max_dist):
        """Other-class camera fix, used only when it is this rover by position; no box axis."""
        self.advance(t)
        if self._frozen():
            self.last_fix_disputed = True
            return False
        if self._disputes_hold(x, y):
            return False
        used = self.est.update_bev_candidate(x, y, max_dist)
        if used:
            self._camera_used()
        return used

    def uwb_vouches(self, x, y):
        return self.est.uwb_vouches(x, y, self._fresh_uwb_fix())

    def step_cog(self, suppress=False):
        """Feed one COG sample once the position moved cog_min_move.

        The chord is the mid-segment heading (half the yaw change is added); segments more
        curved than straight_curvature are dropped. Speed = segment length / time (not the
        velocity state); time parked at the start is excluded. suppress=True skips the update."""
        if not self.est.initialized:
            return
        if not self._camera_fresh(self.last_t):
            self._cog_ref = None            # camera must see the whole course
            return
        x, y, _ = self.est.pose
        if self._cog_ref is None:
            self._start_cog_segment(x, y)
            return
        dx, dy = x - self._cog_ref[0], y - self._cog_ref[1]
        dist = math.hypot(dx, dy)
        if dist < COG_STILL:
            self._cog_left = (dist, self.last_t)
        if dist < self.cog_min_move:
            return
        if not suppress and self._cog_turned <= self.est.straight_curvature * dist:
            speed = (dist - self._cog_left[0]) / max(self.last_t - self._cog_left[1], 1e-3)
            self.est.update_cog(math.atan2(dy, dx) + 0.5 * self._cog_dyaw, speed, self.last_wz)
        self._start_cog_segment(x, y)

    def _start_cog_segment(self, x, y):
        self._cog_ref = (x, y, self.last_t)
        self._cog_left = (0.0, self.last_t)
        self._cog_turned = self._cog_dyaw = 0.0
