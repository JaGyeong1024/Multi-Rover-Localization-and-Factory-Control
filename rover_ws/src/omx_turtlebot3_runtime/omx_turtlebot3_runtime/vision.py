"""ArUco detection on the EEF camera with per-ID PnP quality."""
from collections import Counter
from copy import deepcopy
import cv2
import numpy as np
from .core import Fault


# ArUco dictionary and detector parameters (OpenCV 4.5 and 4.7+ APIs).
def get_aruco_dictionary(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module is not available. Install opencv-contrib-python or ros-humble-vision-opencv with aruco support.")
    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def paired_camera_info(image, info, expected_frame):
    """Bind a driver-omitted info frame ONLY to its exact stamped image pair.

    v4l2_camera 0.6.2 on the robot labels Image but leaves CameraInfo.frame_id
    empty. This carries that image's optical-frame label; it creates no TF and
    does not assert that camera extrinsics are calibrated.
    """
    if (image.header.stamp.sec, image.header.stamp.nanosec) != (info.header.stamp.sec, info.header.stamp.nanosec):
        raise Fault('Image/CameraInfo timestamps are not an exact pair')
    if (image.width, image.height) != (info.width, info.height):
        raise Fault('Image/CameraInfo resolution mismatch')
    if (not expected_frame or image.header.frame_id != expected_frame
            or info.header.frame_id not in ('', expected_frame)):
        raise Fault('Image/CameraInfo optical frame mismatch')
    result = deepcopy(info)
    result.header.frame_id = image.header.frame_id
    return result


class Detector:
    def __init__(self, config):
        self.c = config
        self.dictionary = get_aruco_dictionary(config['dictionary'])
        self.parameters = make_detector_parameters()
        # OpenCV 4.5 can merge/reject nested near-duplicate candidates with its
        # default grouping distance. Real 320x240 captures recover both IDs at
        # 0.01 (20/20 vs 0/20 for ID0), without changing dictionary decoding,
        # duplicate-ID rejection, corner/depth/front-face or reprojection gates.
        self.parameters.minMarkerDistanceRate = .01
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters) if hasattr(cv2.aruco, 'ArucoDetector') else None

    def detect(self, image, matrix, distortion):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.detector is None:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        else:
            corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        result, counts = [], Counter(int(i) for i in ids.flatten())
        s = self.c['marker_size_m']/2
        points = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=float)
        height, width = image.shape[:2]
        for corner, marker in zip(corners, ids.flatten()):
            marker = int(marker)
            uv = corner.reshape(4, 2).astype(float)
            if marker not in self.c['physical_ids'] or counts[marker] != 1:
                continue
            if (np.any(uv < 3) or np.any(uv[:, 0] >= width-3) or np.any(uv[:, 1] >= height-3)
                    or np.min(np.linalg.norm(uv-np.roll(uv, 1, axis=0), axis=1)) < self.c['minimum_marker_side_px']):
                continue
            candidates = []
            # OpenCV 4.5's IPPE has a near-frontal degeneracy. Independently solve
            # iterative PnP as well, keeping the same positive-depth/error gates.
            for method in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE):
                try:
                    values = cv2.solvePnPGeneric(points, uv, matrix, distortion, flags=method)
                    if values[0]:
                        for rvec, tvec in zip(values[1], values[2]):
                            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all() or not .04 <= tvec[2, 0] <= 1.2:
                                continue
                            rotation = cv2.Rodrigues(rvec)[0]
                            if np.any((points @ rotation.T+tvec.reshape(1, 3))[:, 2] <= 0):
                                continue
                            if np.dot(rotation[:, 2], tvec.flatten()) >= 0:
                                continue  # Reject a back-facing printed surface.
                            projected, _ = cv2.projectPoints(points, rvec, tvec, matrix, distortion)
                            error = float(np.mean(np.linalg.norm(projected.reshape(4, 2)-uv, axis=1)))
                            if error <= self.c['max_reprojection_error_px']:
                                candidates.append((error, rvec, tvec))
                except cv2.error:
                    continue
            if candidates:
                error, rvec, tvec = min(candidates, key=lambda c: c[0])
                result.append((marker, tvec.flatten(), cv2.Rodrigues(rvec)[0], error))
        return result


def synchronized_joints(history, stamp, tolerance):
    """Interpolate only between nearby samples; never use an arbitrary latest TF."""
    if not history:
        raise Fault('No encoder history for image timestamp')
    samples = list(history)
    before = [s for s in samples if s.stamp <= stamp]
    after = [s for s in samples if s.stamp >= stamp]
    if before and after:
        a, b = before[-1], after[0]
        if max(stamp-a.stamp, b.stamp-stamp) > tolerance:
            raise Fault('Image/encoder synchronization exceeded tolerance')
        if b.stamp == a.stamp:
            return a.q.copy()
        return a.q+(b.q-a.q)*(stamp-a.stamp)/(b.stamp-a.stamp)
    nearest = min(samples, key=lambda s: abs(s.stamp-stamp))
    if abs(nearest.stamp-stamp) > tolerance or np.max(np.abs(nearest.dq)) > .03:
        raise Fault('Cannot extrapolate a moving arm at the image timestamp')
    return nearest.q.copy()
