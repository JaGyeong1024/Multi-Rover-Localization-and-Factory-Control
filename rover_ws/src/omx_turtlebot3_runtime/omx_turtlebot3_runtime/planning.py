"""Frozen training geometry for FK and collision checks, NEVER simulated feedback.

Only mj_forward is used. No mj_step, simulated contact bit, or random object pose
may enter the policy. Object poses are overwritten with real camera observations.
"""
import math
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .core import Fault, Phase, finite
from .robot_contract import RobotUrdf


class Planner:
    def __init__(self, scene, training, runtime):
        self.t, self.cfg = training, runtime
        self.scene_path = Path(scene)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.arm = [self.model.joint(f'joint{i}').qposadr[0] for i in range(1, 5)]
        self.fingers = [self.model.joint(f'gripper_{s}_joint').qposadr[0] for s in ('left', 'right')]
        self.objects = [self.model.joint(f'object_{i}_free').qposadr[0] for i in (1, 2)]
        self.object_geoms = [self.model.geom(f'object_{i}_geom').id for i in (1, 2)]
        self.arm_geoms = set(np.flatnonzero(self.model.geom_contype == 2))
        self.pads = {self.model.geom(f'{s}_finger_pad').id for s in ('left', 'right')}
        self.static = set(np.flatnonzero(np.isin(self.model.geom_contype, [1, 8])))
        self.deck = self.model.geom('follower_follower_base_link_collision_0').id
        self.eef = self.model.site('end_effector').id
        self.link5 = self.model.body('link5').id
        self.camera_id = self.model.camera('eef_usb_camera').id
        self._configure_camera_geometry()
        self.origin = self.model.body('link2').pos.copy()
        self.low = self.model.actuator_ctrlrange[:4, 0]+training['joint_margin']
        self.high = self.model.actuator_ctrlrange[:4, 1]-training['joint_margin']
        self.stay = np.asarray(training['stay'])
        self.poses = None
        self.selected = None
        self._validate_robot_chain()
        # No invented pedestals: commissioned real environment must match this scene.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        c = runtime['estimation']
        if (not np.allclose(self.data.geom_xpos[self.deck][:2], c['deck_center_xy'], atol=1e-6)
                or not np.allclose(self.model.geom_size[self.deck][:2], c['deck_half_size_xy'], atol=1e-6)
                or not np.allclose(self.data.geom_xmat[self.deck].reshape(3, 3), np.eye(3), atol=1e-6)
                or not np.isclose(self.data.geom_xpos[self.deck][2]+self.model.geom_size[self.deck][2], c['deck_top_z'], atol=1e-6)
                or any(not np.allclose(self.model.geom_size[g], c['object_width_m']/2, atol=1e-6) for g in self.object_geoms)):
            raise Fault('Physical estimator geometry differs from collision scene; update both')

    def _configure_camera_geometry(self):
        """Keep optical FK and the camera collision body on one rigid mount."""
        v = self.cfg['vision']
        body = self.model.body('eef_usb_camera_link').id
        if self.model.body_parentid[body] != self.link5 or self.model.cam_bodyid[self.camera_id] != body:
            raise Fault('Unexpected camera mounting body in scene')
        keys = ('camera_body_position_in_link5', 'camera_body_rotation_in_link5')
        self.robot_urdf = None
        if 'robot_urdf_asset' in v:
            if any(k in v for k in keys):
                raise Fault('URDF and manual camera overrides cannot be mixed')
            name = v['robot_urdf_asset']
            if Path(name).name != name:
                raise Fault('Robot URDF must be a scene-local asset')
            self.robot_urdf = RobotUrdf(self.scene_path.parent/name, v['robot_urdf_sha256'])
            mount, optical_mount = self.robot_urdf.camera_mount()
            self.model.body_pos[body] = mount[:3, 3]
            self.model.body_quat[body] = Rotation.from_matrix(mount[:3, :3]).as_quat()[[3, 0, 1, 2]]
            self.model.cam_pos[self.camera_id] = optical_mount[:3, 3]
            self.model.cam_quat[self.camera_id] = Rotation.from_matrix(
                optical_mount[:3, :3] @ np.diag([1., -1., -1.])).as_quat()[[3, 0, 1, 2]]
        elif any(k in v for k in keys):
            if not all(k in v for k in keys):
                raise Fault('Camera mount override requires position and rotation together')
            position = finite(v[keys[0]], (3,))
            rotation = finite(v[keys[1]], (3, 3))
            if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
                raise Fault('Invalid camera mounting rotation')
            self.model.body_pos[body] = position
            self.model.body_quat[body] = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        body_r = Rotation.from_quat(self.model.body_quat[body][[1, 2, 3, 0]]).as_matrix()
        camera_r = Rotation.from_quat(self.model.cam_quat[self.camera_id][[1, 2, 3, 0]]).as_matrix()
        lens = self.model.body_pos[body]+body_r@self.model.cam_pos[self.camera_id]
        optical = body_r@camera_r@np.diag([1., -1., -1.])
        if (not np.allclose(lens, finite(v['lens_position_in_link5'], (3,)), atol=1e-6)
                or not np.allclose(optical, finite(v['optical_rotation_in_link5'], (3, 3)), atol=1e-6)):
            raise Fault('Optical calibration differs from camera collision/render mount')

    def _validate_robot_chain(self):
        self.policy_from_base = None
        if self.robot_urdf is None:
            return
        zero = {f'joint{i}': 0. for i in range(1, 5)}
        base_link2 = self.robot_urdf.frame('link2', zero)
        self.forward(np.zeros(4))
        world_link2 = np.eye(4)
        world_link2[:3, :3] = self.data.xmat[self.model.body('link2').id].reshape(3, 3)
        world_link2[:3, 3] = self.data.xpos[self.model.body('link2').id]
        self.policy_from_base = world_link2 @ np.linalg.inv(base_link2)
        # Validate the complete moving arm, not just the fixed camera mount.
        for values in ([0., 0., 0., 0.], [.3, -.4, .2, -.6], [-.2, .7, -.3, .1]):
            self.forward(np.asarray(values))
            angles = dict(zip(zero, values))
            for link in ('link2', 'link3', 'link4', 'link5'):
                expected = self.policy_from_base @ self.robot_urdf.frame(link, angles)
                body = self.model.body(link).id
                if (not np.allclose(expected[:3, 3], self.data.xpos[body], atol=1e-7)
                        or not np.allclose(expected[:3, :3], self.data.xmat[body].reshape(3, 3), atol=1e-7)):
                    raise Fault('Robot/training joint-coordinate mismatch at '+link)

    def update_objects(self, cache, selected):
        if any(p is None for p in cache.items):
            raise Fault('Both measured object poses required for collision model')
        self.poses = list(cache.items)
        self.selected = selected

    def forward(self, q, finger=0., carry=False):
        q = finite(q, (4,))
        self.data.qpos[self.arm] = q
        self.data.qpos[self.fingers] = finger
        self.data.qvel[:] = 0
        if self.poses is not None:
            for adr, pose in zip(self.objects, self.poses):
                self.data.qpos[adr:adr+7] = np.r_[pose.position, pose.quaternion_xyzw[[3, 0, 1, 2]]]
        mujoco.mj_forward(self.model, self.data)
        if carry and self.selected is not None:
            # Swept payload envelope ONLY, never an observation or grasp confirmation.
            rotation = self.data.xmat[self.link5].reshape(3, 3)
            center = self.data.site_xpos[self.eef]-rotation @ np.array([0., 0., self.t['grasp_offset_z']])
            quat = Rotation.from_matrix(rotation).as_quat()
            adr = self.objects[self.selected]
            self.data.qpos[adr:adr+7] = np.r_[center, quat[[3, 0, 1, 2]]]
            mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.eef].copy()

    def fk(self, q):
        return self.forward(q)

    def camera_pose(self, q):
        self.forward(q)
        # MuJoCo camera looks along -Z with +Y up; ROS optical is +Z/+Y down.
        return (self.data.cam_xpos[self.camera_id].copy(),
                self.data.cam_xmat[self.camera_id].reshape(3, 3)@np.diag([1., -1., -1.]))

    def object_from_marker(self, q, marker_position, marker_rotation):
        camera, optical = self.camera_pose(q)
        v = self.cfg['vision']
        rotation = optical @ marker_rotation @ np.asarray(v['object_to_marker_rotation']).T
        center = camera+optical @ marker_position-rotation @ v['object_to_marker_translation']
        return center, Rotation.from_matrix(rotation).as_quat()

    def check(self, q, phase, finger=.019, carry=False):
        if np.any(q < self.low) or np.any(q > self.high):
            raise Fault('Joint outside commissioned model limits')
        self.forward(q, finger, carry)
        selected_geom = self.object_geoms[self.selected] if self.selected is not None else -1
        other = set(self.object_geoms)-{selected_geom}
        for c in self.data.contact:
            if c.dist >= -.0005:
                continue
            pair = {int(c.geom1), int(c.geom2)}
            arm = bool(pair & self.arm_geoms)
            forbidden = arm and (bool(pair & self.static) or len(pair & self.arm_geoms) == 2 or bool(pair & other))
            if arm and selected_geom in pair:
                permitted = bool(pair & self.pads) and Phase.REACH <= phase <= Phase.PLACE
                forbidden |= not permitted
            if carry and selected_geom in pair and pair & self.static:
                permitted = self.deck in pair and phase in (Phase.EXTEND_LOWER, Phase.PLACE)
                forbidden |= not permitted
            if forbidden:
                raise Fault(f'Collision: {self.model.geom(c.geom1).name}/{self.model.geom(c.geom2).name}')

    def path(self, start, goal, phase, carry=False):
        count = max(7, int(np.max(np.abs(goal-start))/.025)+2)
        for t in np.linspace(0, 1, count):
            self.check(start+(goal-start)*t, phase, .0035 if carry else .019, carry)

    def stay_path(self, start, finger):
        """Validate startup at the measured finger opening, without PPO or IK.

        Before camera recognition this uses the commissioned scene geometry.
        It is a collision preflight, never a substitute for measured feedback.
        """
        start = finite(start, (4,))
        if not math.isfinite(finger) or not -.010 <= finger <= .019:
            raise Fault('Invalid startup gripper feedback')
        count = max(7, int(np.max(np.abs(self.stay-start))/.025)+2)
        for t in np.linspace(0, 1, count):
            self.check(start+(self.stay-start)*t, Phase.RECOGNIZE, finger, False)

    def ik(self, point, pitch=0.):
        point = finite(point, (3,))
        yaw = math.atan2(point[1]-self.origin[1], point[0]-self.origin[0])
        if not self.low[0] <= yaw <= self.high[0]:
            raise Fault('Target bearing outside joint1 limits')
        best = None
        for seed in ([.8, .4, -1.2], [1.3, -.3, -.5], [-.3, .8, -.5]):
            solution = least_squares(lambda v: np.r_[20*(self.fk([yaw, *v])-point), sum(v)-pitch],
                                     np.clip(seed, self.low[1:], self.high[1:]),
                                     bounds=(self.low[1:], self.high[1:]), max_nfev=80)
            q = np.r_[yaw, solution.x]
            error = np.linalg.norm(self.fk(q)-point)
            score = error+abs(sum(q[1:])-pitch)*.1
            if best is None or score < best[0]:
                best = score, error, q
            if error < .0002 and abs(sum(q[1:])-pitch) < .005:
                break
        _, error, q = best
        if error > .001 or abs(sum(q[1:])-pitch) > .01:
            raise Fault('Target not reachable at required pitch')
        return q

    def observe_pose(self, pose):
        v, p = self.cfg['vision'], self.t['perception']
        s = v['marker_size_m']/2
        marker_corners = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
        cube_r = Rotation.from_quat(pose.quaternion_xyzw).as_matrix()
        corners = (marker_corners @ np.asarray(v['object_to_marker_rotation']).T+v['object_to_marker_translation']) @ cube_r.T+pose.position
        yaw = math.atan2(pose.position[1]-self.origin[1], pose.position[0]-self.origin[0])
        best = None
        start = np.asarray(self.t['motion']['scan_pose'])
        fx, fy, cx, cy = p['intrinsics']
        width, height = p['image_size']
        for a in (-.3, 0., .3, .55, .8):
            for b in (.3, .6, .9, 1.1):
                for pitch in (-.1, .1, .3, .5, .7, .9, 1.1):
                    q = np.array([yaw, a, b, pitch-a-b])
                    try:
                        self.check(q, Phase.RECOGNIZE)
                        camera, rotation = self.camera_pose(q)
                        local = (corners-camera) @ rotation
                        if np.any(local[:, 2] < .07):
                            continue
                        uv = local[:, :2]/local[:, 2:]*[fx, fy]+[cx, cy]
                        if np.any(uv < 5) or np.any(uv[:, 0] > width-5) or np.any(uv[:, 1] > height-5):
                            continue
                        score = np.linalg.norm(uv.mean(axis=0)-[cx, cy])+.5*np.linalg.norm(q-start)
                        if best is None or score < best[0]:
                            self.path(start, q, Phase.RECOGNIZE)
                            best = score, q
                    except Fault:
                        continue
        if best is None:
            raise Fault('No collision-free target observation pose')
        return best[1]

    def _plan_ik(self, stage, point, pitch):
        try:
            return self.ik(point, pitch)
        except Fault as exc:
            raise Fault(f'{stage}: {exc}; target={np.asarray(point).tolist()}, pitch={pitch}') from exc

    def plan(self, cache, selected):
        self.update_objects(cache, selected)
        target = cache.items[selected].position.copy()
        target[2] += self.t['grasp_offset_z']
        pitch = self.t['motion']['upper_pick_pitch']
        grasp = self._plan_ik('grasp', target, pitch)
        height = target[2]+self.t['motion']['upper_lift']
        lift = self._plan_ik('lift', [target[0], target[1], height], pitch)
        pre = self._plan_ik('pregrasp', [target[0]-.045, target[1], height], pitch)
        outside = self._plan_ik('outside_shelf', [.095, target[1], height], pitch)
        carry = self._plan_ik('carry', [self.origin[0]+.155*np.cos(grasp[0]), .155*np.sin(grasp[0]), height], pitch)
        right = carry.copy(); right[0] = -math.pi/2
        c = self.cfg['estimation']
        z = c['deck_top_z']+c['object_width_m']*(np.cos(.45)+np.sin(.45))/2+.002+self.t['grasp_offset_z']*np.cos(.45)
        place = release = None
        for radius in np.arange(.34, .23-.0001, -.002):
            candidate = np.array([self.origin[0], -radius, z])
            if np.any(np.abs(candidate[:2]-c['deck_center_xy'])+.036 > np.asarray(c['deck_half_size_xy'])-c['deck_inset_m']):
                continue
            try:
                release = self.ik(candidate, .45)
                self.check(release, Phase.PLACE, .0035, True)
                place = candidate
                break
            except Fault:
                continue
        if place is None:
            raise Fault('No reachable collision-free inset follower release')
        over = self._plan_ik('over_follower', [self.origin[0], place[1], height], pitch)
        raised = self._plan_ik('raised_from_follower', place+[0, 0, .080], .45)
        scan = np.asarray(self.t['motion']['scan_pose'])
        observe = self.observe_pose(cache.items[selected])
        plans = {Phase.RECOGNIZE: [scan, observe], Phase.OPEN: [observe],
                 Phase.REACH: [outside, pre, lift, grasp], Phase.GRIP: [grasp],
                 Phase.RETRACT: [lift, pre, outside, carry], Phase.ROTATE_RIGHT: [right],
                 Phase.EXTEND_LOWER: [over, raised, release], Phase.PLACE: [release],
                 Phase.RETURN_STAY: [raised, over, right, np.array([self.stay[0], *carry[1:]]), self.stay.copy()]}
        previous = self.stay.copy()
        for phase, points in plans.items():
            for point in points:
                self.path(previous, point, phase, Phase.RETRACT <= phase <= Phase.PLACE)
                previous = point
        return plans, place
