"""Sensor contracts, 69D observation and conservative physical evidence fusion."""
from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np
from scipy.spatial.transform import Rotation
from .commissioning import commissioning_issues, thresholds_calibrated


class Fault(RuntimeError):
    pass


class Phase(IntEnum):
    RECOGNIZE = 0
    OPEN = 1
    REACH = 2
    GRIP = 3
    RETRACT = 4
    ROTATE_RIGHT = 5
    EXTEND_LOWER = 6
    PLACE = 7
    RETURN_STAY = 8


def finite(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise Fault(f'Expected finite shape {shape}')
    return result


@dataclass
class RobotSample:
    stamp: float
    q: np.ndarray
    dq: np.ndarray
    finger: float
    finger_velocity: float
    gripper_current_raw: float
    voltage: float
    wheel_ticks: tuple
    millis: int


@dataclass
class ObjectPose:
    physical_id: int
    stamp: float
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    reprojection_px: float


class PoseCache:
    def __init__(self):
        self.items = [None, None]
        self.previous = [None, None]
        self.visible = np.zeros(2)
        self.frame_stamp = -math.inf

    def begin_frame(self, stamp):
        if stamp <= self.frame_stamp:
            raise Fault('Repeated/reordered perception frame')
        self.frame_stamp = stamp
        self.visible[:] = 0

    def update(self, pose, now):
        if type(pose.physical_id) is not int or pose.physical_id not in (0, 1):
            raise Fault('Unknown physical ID; no largest-marker fallback')
        finite(pose.position, (3,))
        q = finite(pose.quaternion_xyzw, (4,))
        if not math.isfinite(pose.stamp) or pose.stamp > now+.05 or now-pose.stamp > .4:
            raise Fault('Stale/future marker acquisition')
        if abs(np.linalg.norm(q)-1) > .01 or not 0 <= pose.reprojection_px <= 1.5:
            raise Fault('Invalid marker orientation or reprojection quality')
        old = self.items[pose.physical_id]
        if old is not None and pose.stamp <= old.stamp:
            return False
        if old is not None and now-old.stamp < .5 and np.linalg.norm(pose.position-old.position) > .05:
            raise Fault('Marker position jump')
        if old is not None and now-old.stamp < .5:
            angle = (Rotation.from_quat(pose.quaternion_xyzw)*Rotation.from_quat(old.quaternion_xyzw).inv()).magnitude()
            if angle > .8:
                raise Fault('Marker pose ambiguity/orientation jump')
        self.previous[pose.physical_id] = old
        self.items[pose.physical_id] = pose
        self.visible[pose.physical_id] = 1
        return True

    def fresh(self, index, now, age=.5):
        item = self.items[index]
        return item is not None and 0 <= now-item.stamp <= age

    def motion(self, index):
        a, b = self.previous[index], self.items[index]
        if a is None or b is None or not .01 <= b.stamp-a.stamp <= .4:
            return math.inf, math.inf
        dt = b.stamp-a.stamp
        linear = np.linalg.norm(b.position-a.position)/dt
        qa, qb = Rotation.from_quat(a.quaternion_xyzw), Rotation.from_quat(b.quaternion_xyzw)
        return linear, (qb*qa.inv()).magnitude()/dt


def observation(sample, eef, goal, phase, waypoint, count, cache, selected,
                place, contact_estimate, deck_estimate, previous_action,
                last_plan, recognition_stage, now):
    """Exact v5/v6 feature order/scaling; estimates replace two contact bits explicitly."""
    if selected not in (0, 1) or recognition_stage not in (0, 1):
        raise Fault('Invalid policy slot or recognition stage')
    positions, quaternions = np.zeros((2, 3)), np.zeros((2, 4))
    errors, ages = np.full(2, 1.5), np.full(2, 40.)
    valid, visible = np.zeros(2), np.zeros(2)
    for i, item in enumerate(cache.items):
        if item is not None:
            ages[i] = np.clip(now-item.stamp, 0, 40)
            errors[i] = item.reprojection_px
            if 0 <= now-item.stamp <= 40:
                valid[i], positions[i], quaternions[i] = 1., item.position, item.quaternion_xyzw
                visible[i] = cache.visible[i] if 0 <= now-cache.frame_stamp <= .4 else 0.
    target = positions[selected]
    values = [sample.q, sample.dq*.1, [sample.finger]*2, eef, goal,
              np.eye(9)[int(phase)], [waypoint/max(1, count-1)], positions.ravel(), valid,
              np.eye(2)[selected], target, target-eef if valid[selected] else np.zeros(3),
              place, [float(contact_estimate), float(deck_estimate)], previous_action,
              quaternions.ravel(), errors, [min(40., max(0., now-last_plan))],
              np.eye(2)[recognition_stage], visible, ages/40.]
    return finite(np.concatenate(values), (69,)).astype(np.float32)


def residual_step(target, goal, action, phase, waypoint, count, measured, training, dt):
    """Training reference/residual formula, scaled by real elapsed time."""
    action = np.clip(finite(action, (4,)), -1, 1).copy()
    speed = 1.
    if phase in (Phase.RECOGNIZE, Phase.ROTATE_RIGHT):
        speed = 1.2
    elif phase == Phase.REACH and waypoint < count-2:
        speed = 1.2
    elif phase == Phase.RETRACT and waypoint > 0:
        speed = 1.2
    elif phase == Phase.EXTEND_LOWER and waypoint == 0:
        speed = 1.2
    elif phase == Phase.RETURN_STAY and waypoint > 0:
        speed = 1.2
    if phase == Phase.ROTATE_RIGHT:
        action[1:] = 0
        if abs(measured[0]+math.pi/2) < .03:
            action[0] = 0
    scale = training['joint_step']*speed
    difference = goal-target
    reference = difference/max(scale, float(np.max(np.abs(difference))))
    delta = scale*np.clip(reference+training['residual_scale']/speed*action, -1, 1)
    delta *= dt/(training['physics_timestep']*training['frame_skip'])
    if phase == Phase.ROTATE_RIGHT:
        delta[1:] = 0  # Hold pitch joints exactly, not just zero their residual.
    return delta, action


class SlewLimiter:
    def __init__(self, config):
        self.cfg = config
        self.velocity = np.zeros(4)

    def step(self, delta, dt):
        if not 0 < dt <= self.cfg['max_cycle_s']:
            raise Fault('Control deadline exceeded')
        requested = np.clip(delta/dt, -self.cfg['joint_velocity_rad_s'], self.cfg['joint_velocity_rad_s'])
        dv = self.cfg['joint_acceleration_rad_s2']*dt
        self.velocity += np.clip(requested-self.velocity, -dv, dv)
        return self.velocity*dt


class Evidence:
    """Not a force sensor: estimated hold + visual proof; release never uses a timer alone."""
    def __init__(self, config):
        self.c = config
        self.reset()

    def reset(self):
        self.contact = self.grasp_confirmed = self.placed = False
        self.contact_since = self.lost_since = self.place_since = None
        self.grasp_eef = self.grasp_object = self.grasp_relative = None
        self.follow_frames = self.place_frames = 0
        self.last_follow_stamp = self.last_place_stamp = -math.inf
        self.reason = 'WAIT_SIGNALS'

    def calibrated(self):
        return thresholds_calibrated(self.c)

    def update(self, phase, sample, eef, cache, selected, now, closed=-.01, opened=.019):
        if not self.calibrated():
            self.reason = 'CURRENT_THRESHOLDS_UNCALIBRATED'
            return
        if now-sample.stamp > .15 or sample.stamp > now+.05:
            raise Fault('Stale hardware evidence')
        c = self.c
        current = abs(sample.gripper_current_raw-c['current_idle_raw'])
        gap = 2*(c['finger_base_half_gap_m']+sample.finger)
        electrical = (sample.finger > closed+c['grip_min_position_above_closed_m']
                      and abs(gap-c['object_width_m']) <= c['grip_width_tolerance_m']
                      and abs(sample.finger_velocity) < c['grip_stall_velocity_m_s']
                      and current >= c['current_grip_delta_raw'])
        fresh = cache.fresh(selected, now)
        obj = cache.items[selected]
        if phase == Phase.GRIP and not self.contact:
            near_target = fresh and np.linalg.norm(obj.position-eef) < .025
            if electrical and near_target:
                if self.contact_since is None:
                    self.contact_since = now
                if now-self.contact_since >= c['contact_confirm_s']:
                    self.contact = True
                    self.grasp_eef, self.grasp_object = eef.copy(), obj.position.copy()
                    self.grasp_relative = obj.position-eef
                    self.reason = 'CONTACT_ESTIMATED_REQUIRE_LIFT_PROOF'
            else:
                self.contact_since = None
        if self.contact and Phase.GRIP <= phase <= Phase.EXTEND_LOWER:
            if not electrical:
                self.lost_since = now if self.lost_since is None else self.lost_since
                if now-self.lost_since >= c['grip_lost_s']:
                    self.contact = False
                    raise Fault('Gripper load/aperture evidence lost')
            else:
                self.lost_since = None
            if phase == Phase.RETRACT and fresh and obj.stamp != self.last_follow_stamp:
                self.last_follow_stamp = obj.stamp
                lifted = (eef[2]-self.grasp_eef[2] >= c['proof_lift_m']
                          and obj.position[2]-self.grasp_object[2] >= c['proof_lift_m'])
                follows = np.linalg.norm((obj.position-eef)-self.grasp_relative) < c['follow_error_m']
                self.follow_frames = self.follow_frames+1 if lifted and follows and electrical else 0
                if self.follow_frames >= c['visual_follow_samples']:
                    self.grasp_confirmed = True
                    self.reason = 'GRASP_VERIFIED_LOAD_APERTURE_AND_VISUAL_LIFT'
        if phase == Phase.PLACE:
            self.contact = electrical
            released = sample.finger >= opened-.002 and current <= c['current_release_delta_raw']
            stable = False
            if fresh and released:
                # Full rotated 50mm cube footprint, not just its centre.
                half_extent = np.abs(Rotation.from_quat(obj.quaternion_xyzw).as_matrix()) @ np.full(3, c['object_width_m']/2)
                inside = np.all(np.abs(obj.position[:2]-c['deck_center_xy'])+half_extent[:2]
                                <= np.asarray(c['deck_half_size_xy'])-c['deck_inset_m'])
                supported = abs(obj.position[2]-half_extent[2]-c['deck_top_z']) < c['place_position_tolerance_m']
                linear, angular = cache.motion(selected)
                stable = inside and supported and linear < c['place_speed_m_s'] and angular < c['place_angular_speed_rad_s']
            if not stable:
                self.place_since, self.place_frames = None, 0
                self.reason = 'WAIT_VISUAL_PLACEMENT_AND_RELEASE'
            elif obj.stamp != self.last_place_stamp:
                self.last_place_stamp = obj.stamp
                self.place_since = now if self.place_since is None else self.place_since
                self.place_frames += 1
                if (now-self.place_since >= c['release_confirm_s']
                        and self.place_frames >= c['release_min_frames']):
                    self.placed = True
                    self.reason = 'PLACE_VERIFIED_VISUAL_DECK_AND_RELEASE'
