"""Per-robot fusion front-end without ROS (fusion_node, simulator, replay).

No clock reads: t = measurement time, rx = receive time (monotonic), now = output-step time [s].
"""
from typing import NamedTuple, Optional

from . import quality
from .se2nhc import VX, VY
from .tracker import RobotTracker


class Gating(NamedTuple):
    """Output gating settings."""

    pose_timeout: float = quality.POSE_TIMEOUT    # [s] no UWB and no BEV this long -> no pose
    uwb_timeout: float = quality.UWB_TIMEOUT      # [s]
    bev_timeout: float = quality.BEV_TIMEOUT      # [s]
    uwb_only_quality: str = quality.LOST          # quality while only UWB is fresh
    cog_suppress_s: float = 5.0                   # [s] no COG heading update after alignment
    quality_mode: str = quality.CAMERA_AGE        # see quality.MODES
    limits: quality.EstimatorLimits = quality.EstimatorLimits()


class FusedOutput(NamedTuple):
    """One fused sample; heading is None until anchored."""

    x: float
    y: float
    heading: Optional[float]
    vx: float
    vy: float
    wz: float


class RobotFusion:
    """RobotTracker plus receive-time bookkeeping."""

    def __init__(self, kw, use_obb=False, gating=Gating(), wheel_still=False, zero_motion=False,
                 wheel_radius=None):
        self.trk = RobotTracker(kw, use_obb=use_obb, wheel_still=wheel_still, zero_motion=zero_motion,
                                wheel_radius=wheel_radius)
        self.est = self.trk.est
        self.gating = gating
        self._uwb_rx = None               # receive time of the last UWB fix
        self._bev_rx = None               # receive time of the last used BEV fix
        self._aligned = False
        self._cog_suppress_until = 0.0    # in the clock of output(now)

    @property
    def aligned(self):
        """Heading anchored at the last output step."""
        return self._aligned

    def on_qf(self, qf):
        self.trk.qf = qf

    def on_uwb(self, t, rx, x, y):
        self.trk.on_uwb(t, x, y)
        self._uwb_rx = rx

    def on_bev(self, t, rx, x, y, obb_yaw=None):
        """True when the BEV position was used (a gate-rejected one is not present)."""
        used = self.trk.on_bev(t, x, y, obb_yaw)
        if used or self.trk.last_fix_disputed:
            self._bev_rx = rx       # a hold dispute still means the camera sees the rover
        return used

    def on_bev_candidate(self, t, rx, x, y, max_dist):
        """Other-class camera fix; used only if the position says it is this rover."""
        used = self.trk.on_bev_candidate(t, x, y, max_dist)
        if used or self.trk.last_fix_disputed:
            self._bev_rx = rx
        return used

    def advance(self, t):
        """Predict to t (no update)."""
        self.trk.advance(t)

    def bev_gate_distance(self, x, y):
        return self.est.bev_gate_distance(x, y)

    def uwb_vouches(self, x, y):
        return self.trk.uwb_vouches(x, y)

    def on_imu(self, t, yaw, wz, accel=None, rp_rate=None):
        self.trk.on_imu(t, yaw, wz, accel, rp_rate)

    def on_wheels(self, t, left, right):
        """Wheel positions [rad]; no output step."""
        self.trk.on_wheels(t, left, right)

    def ages(self, rx):
        """(uwb_age, bev_age) at rx; None = never received."""
        return (None if self._uwb_rx is None else rx - self._uwb_rx,
                None if self._bev_rx is None else rx - self._bev_rx)

    def quality(self, rx):
        g = self.gating
        if g.quality_mode == quality.ESTIMATOR:
            return quality.classify_estimator(self.estimator_state(rx), g.limits, g.pose_timeout)
        camera_age = dict(uwb_timeout=g.uwb_timeout, bev_timeout=g.bev_timeout, uwb_only=g.uwb_only_quality)
        if g.quality_mode == quality.CAMERA_AGE_HOLD:
            held = self.est.zero_velocity and not self.trk.moved_since_camera
            return quality.classify_held(*self.ages(rx), held=held,
                                         sigma=self.est.position_sigma, sigma_good=g.limits.sigma_good,
                                         pose_timeout=g.pose_timeout, **camera_age)
        return quality.classify(*self.ages(rx), **camera_age)

    def estimator_state(self, rx):
        """Input of quality.classify_estimator at rx."""
        uwb_age, bev_age = self.ages(rx)
        return quality.EstimatorState(
            in_camera_frame=self.est.in_camera_frame,
            position_age=quality.position_age(uwb_age, bev_age),
            camera_age=bev_age,
            sigma=self.est.position_sigma,
            wheels_still=self.trk.wheels_still,
            blind_travel=self.trk.blind_travel,
            camera_refused=self.trk.camera_disputes)

    def position_live(self, rx):
        age = quality.position_age(*self.ages(rx))
        return age is not None and age <= self.gating.pose_timeout

    def output(self, rx, now):
        """Output step after every measurement handler.

        None until the camera placed the rover (UWB frame differs) or when no position
        measurement is within pose_timeout; else feeds COG, updates aligned, returns the sample.
        """
        if not self.est.in_camera_frame or not self.position_live(rx):
            return None
        self.trk.step_cog(suppress=now < self._cog_suppress_until)
        x, y, heading = self.est.pose
        valid = self.est.heading_valid
        sample = FusedOutput(x, y, heading if valid else None,
                             float(self.est.x[VX]), float(self.est.x[VY]), self.trk.last_wz)
        if valid and not self._aligned and self.gating.cog_suppress_s > 0.0:
            # After alignment the rover backs up; that course is heading + 180 deg.
            self._cog_suppress_until = now + self.gating.cog_suppress_s
        self._aligned = valid
        return sample
