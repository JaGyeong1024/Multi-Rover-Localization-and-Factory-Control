"""Explicit mp_control controller/raw coordinates and installed-URDF FK.

The command conversion is equivalent to mp_control's controller-target -> raw
pre-inversion -> joint_trajectory_transformer. It is NOT a new joint zero/sign
calibration, nor a claim that the training scene matches the physical scene.
"""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .core import Fault, finite


def mp_controller_trajectory(reference, target, max_joint3_delta=.20):
    reference, target = finite(reference, (4,)), finite(target, (4,))
    if abs(target[2]-reference[2]) > max_joint3_delta+1e-12:
        raise Fault('mp_control joint3 delta limit exceeded; no silent clipping')
    raw = target.copy()
    raw[2] = 2*reference[2]-target[2]
    # Both raw and controller targets must remain in the installed mp_control range.
    if not all(-2.70 <= x <= 1.38 for x in (reference[2], raw[2], target[2])):
        raise Fault('mp_control joint3 trajectory bounds exceeded')
    output = raw.copy()
    output[2] = 2*reference[2]-raw[2]
    # joint4 is already the policy/planner's final controller target. The installed
    # transformer uses preserve_roll_sum=False; compensating it again is wrong.
    if not np.allclose(output, target, atol=1e-12, rtol=0):
        raise Fault('mp_control trajectory conversion mismatch')
    return output, {'reference_q': reference.tolist(), 'raw_q': raw.tolist(),
                    'controller_q': output.tolist(), 'joint4_compensated_again': False,
                    'route': 'mp_control_equivalent_in_process_not_ros_controller'}


class RobotUrdf:
    def __init__(self, path, expected_sha256):
        data = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        if self.sha256 != expected_sha256:
            raise Fault('Robot URDF snapshot checksum mismatch')
        root = ET.fromstring(data)
        self.joints = {j.find('child').get('link'): j for j in root.findall('joint')}

    def frame(self, link, q, root='base_footprint', fixed_only=False, _seen=None):
        if link == root:
            return np.eye(4)
        seen = set() if _seen is None else set(_seen)
        if link in seen or link not in self.joints:
            raise Fault('Invalid/disconnected URDF chain')
        seen.add(link)
        j = self.joints[link]
        origin = j.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        tf = np.eye(4)
        tf[:3, 3] = finite(xyz, (3,))
        tf[:3, :3] = Rotation.from_euler('xyz', finite(rpy, (3,))).as_matrix()
        if j.get('type') != 'fixed':
            if fixed_only or j.get('type') != 'revolute' or j.get('name') not in q:
                raise Fault('Unexpected movable URDF joint')
            axis = finite(np.fromstring(j.find('axis').get('xyz'), sep=' '), (3,))
            if not np.isclose(np.linalg.norm(axis), 1.):
                raise Fault('Invalid URDF joint axis')
            angle = float(q[j.get('name')])
            if not np.isfinite(angle):
                raise Fault('Nonfinite URDF joint angle')
            tf[:3, :3] = tf[:3, :3] @ Rotation.from_rotvec(axis*angle).as_matrix()
        return self.frame(j.find('parent').get('link'), q, root, fixed_only, seen) @ tf

    def camera_mount(self):
        body = self.frame('eef_usb_camera_link', {}, 'link5', fixed_only=True)
        optical = self.frame('eef_usb_camera_optical_frame', {}, 'eef_usb_camera_link', fixed_only=True)
        return body, optical
