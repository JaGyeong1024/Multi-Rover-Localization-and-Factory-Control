"""Sensor-level localization: camera, UWB and IMU models feeding the real RobotFusion.

sensors --(delivery time)--> RobotFusion --> relay (latest sample at a fixed rate, dropped
when stale) --> transport delay --> rover. Events are processed in simulation-time order."""
import dataclasses
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from host_localization.fusion import quality as fusion_quality
from host_localization.fusion.frontend import FusedOutput, Gating, RobotFusion
from host_localization.fusion.tracker import KW, OBB_HEADING

from host_sim.core.faults import FaultState
from host_sim.core.raw_sensors import (CameraFix, CameraParams, CeilingCamera, Imu,
                                       ImuParams, ImuSample, Ticker, UwbFix, UwbParams,
                                       UwbTag)
from host_sim.core.sensors import PoseMeasurement
from host_sim.core.util import build_params

MODE_IDEAL = 'ideal'
MODE_FUSED = 'fused'
_FALLBACK_ROBOT = 'follower'      # fusion constants for an unknown rover name


@dataclass
class RelayParams:
    """Path from a fused sample to the rover."""

    rate_hz: float = 30.0             # republish rate
    stale_timeout: float = 0.5        # s, no pose once older
    transport_delay: float = 0.03     # s
    quality_period: float = 0.1       # s


@dataclass
class StartParams:
    """State of the heading estimate when the run begins."""

    aligned: bool = False             # False: heading comes with the first straight stretch
    heading_error: float = 0.0        # rad, left over by the alignment


@dataclass
class FusedParams:
    """Sensor-level localization of one rover (``localization.fused`` in the world yaml)."""

    start: StartParams = field(default_factory=StartParams)
    camera: CameraParams = field(default_factory=CameraParams)
    uwb: UwbParams = field(default_factory=UwbParams)
    imu: ImuParams = field(default_factory=ImuParams)
    relay: RelayParams = field(default_factory=RelayParams)

    @classmethod
    def from_dict(cls, values):
        """Build from {start, camera, uwb, imu, relay} dicts (all optional)."""
        values = dict(values or {})
        sections = {f.name: f.type for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - set(sections))
        if unknown:
            raise ValueError(f'FusedParams: unknown sections {unknown}')
        return cls(**{name: build_params(kind, values.get(name))
                      for name, kind in sections.items()})


class _Emitted(NamedTuple):
    t: float                  # s, emit time
    sample: FusedOutput
    aligned: bool


class _Relay:
    """Republishes the latest fused sample at a fixed rate and delivers it after
    ``transport_delay``."""

    def __init__(self, params):
        self.params = params
        self.clock = Ticker(params.rate_hz)
        self._outbox = deque()            # (arrival time, PoseMeasurement or None, status)
        self.status = (False, fusion_quality.LOST)

    def tick(self, t, emitted, quality):
        pose = None
        if emitted is not None and t - emitted.t <= self.params.stale_timeout:
            pose = _pose_measurement(emitted, quality)
        aligned = emitted is not None and emitted.aligned      # latched /heading_aligned
        self._outbox.append((t + self.params.transport_delay, pose, (aligned, quality)))

    def receive(self, t):
        newest = None
        while self._outbox and self._outbox[0][0] <= t + 1e-9:
            _, pose, self.status = self._outbox.popleft()
            newest = pose or newest
        return newest


def _pose_measurement(emitted, quality):
    sample = emitted.sample
    theta = 0.0 if sample.heading is None else sample.heading   # identity until anchored
    return PoseMeasurement(emitted.t, sample.x, sample.y, theta, emitted.aligned, quality)


def _feed_camera(fusion, fix, t, rx):
    fusion.on_bev(t, rx, fix.x, fix.y, fix.axis)


def _feed_uwb(fusion, fix, t, rx):
    fusion.on_qf(fix.qf)                 # /uwb/{robot}_qf arrives with the fix
    fusion.on_uwb(t, rx, fix.x, fix.y)


def _feed_imu(fusion, sample, t, _rx):
    fusion.on_imu(t, sample.yaw, sample.wz)


_FEEDERS = {CameraFix: _feed_camera, UwbFix: _feed_uwb, ImuSample: _feed_imu}
_MEASUREMENT, _QUALITY_TICK, _RELAY_TICK = range(3)   # order of simultaneous events
_ALIGN_COURSE_SAMPLES = 10    # course samples offered at an aligned start


class FusedLocalization:
    """Same outputs as LocalizationSensor, produced by the real fusion code. ``roi``: camera
    field of view when the params give none."""

    def __init__(self, name, params=None, roi=None, rng=None):
        self.name = name
        self.params = params or FusedParams()
        self._roi = roi
        self._rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        p = self.params
        use_obb = OBB_HEADING.get(self.name, False)
        kw = dict(KW.get(self.name, KW[_FALLBACK_ROBOT]))
        self.fusion = RobotFusion(kw, use_obb=use_obb, gating=Gating())
        self.camera = CeilingCamera(p.camera, self._roi, use_obb, self._child_rng())
        self.uwb = UwbTag(p.uwb, self._child_rng())
        self.imu = Imu(p.imu, self._child_rng())
        self._relay = _Relay(p.relay)
        self._quality_clock = Ticker(1.0 / p.relay.quality_period)
        self._in_flight = []              # measurements captured, not yet delivered
        self._emitted = None
        self._quality = fusion_quality.LOST
        self._align_at_start = p.start.aligned

    def status(self, fault=None):
        fault = fault or FaultState()
        aligned, quality = self._relay.status
        return aligned and not fault.hold_unaligned, fault.quality or quality

    def poll(self, t, robot, fault=None, others=()):
        """Feed the true state at t; the newest PoseMeasurement that reached the rover, else
        None. ``others``: other rovers' tracked points (camera occlusion)."""
        fault = fault or FaultState()
        self._capture(t, robot, others, fault)
        self._process_until(t)
        pose = self._relay.receive(t)
        if pose is None or fault.pose_blackout:
            return None
        return _apply_output_faults(pose, fault, *self.status(fault))

    def _capture(self, t, robot, others, fault):
        point = robot.tracked_point()
        self._true_heading = robot.theta
        captured = (self.camera.capture(t, point, others, fault.camera_lost),
                    self.uwb.capture(t, point),
                    self.imu.capture(t, robot.theta, robot.w * robot.traction))
        self._in_flight += [m for m in captured if m is not None]

    def _process_until(self, t):
        due = [m for m in self._in_flight if m.deliver_at <= t + 1e-9]
        self._in_flight = [m for m in self._in_flight if m.deliver_at > t + 1e-9]
        events = [(m.deliver_at, _MEASUREMENT, m) for m in due]
        events += [(tick, _QUALITY_TICK, None) for tick in self._quality_clock.ticks_until(t)]
        events += [(tick, _RELAY_TICK, None) for tick in self._relay.clock.ticks_until(t)]
        for when, kind, measurement in sorted(events, key=lambda e: e[:2]):
            if kind == _MEASUREMENT:
                self._fuse(measurement, when)
            elif kind == _QUALITY_TICK:
                self._quality = self.fusion.quality(when)
            else:
                self._relay.tick(when, self._emitted, self._quality)

    def _fuse(self, measurement, rx):
        _FEEDERS[type(measurement)](self.fusion, measurement, min(measurement.stamp, rx), rx)
        if self._align_at_start and isinstance(measurement, ImuSample):
            self._align_at_start = not self._align_heading()
        sample = self.fusion.output(rx, rx)
        if sample is not None:
            self._emitted = _Emitted(rx, sample, self.fusion.aligned)

    def _align_heading(self):
        """``start.aligned``: offer the estimator a straight course until the heading is
        anchored; True once valid."""
        est = self.fusion.est
        if not est.initialized:
            return False
        course = self._true_heading + self.params.start.heading_error
        for _ in range(_ALIGN_COURSE_SAMPLES):
            if est.heading_valid:
                break
            est.update_cog(course, est.cog_vref)
        return est.heading_valid

    def _child_rng(self):
        return np.random.default_rng(self._rng.integers(0, 2 ** 32))


def _apply_output_faults(pose, fault, aligned, quality):
    jump = fault.jump_once or (0.0, 0.0)
    fault.jump_once = None
    return dataclasses.replace(pose, x=pose.x + fault.jump_offset[0] + jump[0],
                               y=pose.y + fault.jump_offset[1] + jump[1],
                               theta=pose.theta if aligned else 0.0,
                               heading_aligned=aligned, quality=quality)
