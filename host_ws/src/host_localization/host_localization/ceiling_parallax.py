"""Height-parallax correction for the ceiling camera (numpy + yaml only).

The detector sees the top of a rover at height h, but the pixel->world map assumes the floor,
so the point lands farther from the nadir by 1 / (1 - h / H). The camera centre
C = (nadir_x, nadir_y, H) comes from the calibration files; exact for any tilt (the point
lies on the ray from C to its apparent floor point).
"""
from dataclasses import dataclass

import numpy as np
import yaml

# K^-1 G must normalise to a rotation: equal column norms, orthogonal.
MAX_NORM_MISMATCH = 0.05     # |n1 - n2| / mean(n1, n2)
MAX_ORTHO_RESIDUAL = 0.05    # |r1 . r2| after normalisation


@dataclass(frozen=True)
class CameraCentre:
    """Camera centre in uwb_global [m] plus the rotation-fit residuals."""
    nadir_x: float
    nadir_y: float
    height: float            # H above the floor [m]
    norm_mismatch: float
    ortho_residual: float
    tilt_deg: float          # optical axis from straight down


def floor_to_undistorted_pixel_homography(h_img_to_bev, ppm, origin_x, origin_y, bev_height):
    """World metres (z=0) -> undistorted pixels (inverse of the calibrator's pixel->BEV map; BEV y flipped)."""
    world_to_bev = np.array([[ppm, 0.0, -origin_x * ppm],
                             [0.0, -ppm, bev_height + origin_y * ppm],
                             [0.0, 0.0, 1.0]])
    return np.linalg.inv(np.asarray(h_img_to_bev, float)) @ world_to_bev


def camera_centre_from_floor_homography(g_world_to_pixel, k):
    """Decompose K^-1 G ~ [r1 r2 t] (up to scale) into the camera centre."""
    m = np.linalg.inv(np.asarray(k, float)) @ np.asarray(g_world_to_pixel, float)
    n1, n2 = np.linalg.norm(m[:, 0]), np.linalg.norm(m[:, 1])
    scale = 0.5 * (n1 + n2)
    m = m / scale
    r1, r2, t = m[:, 0], m[:, 1], m[:, 2]
    r3 = np.cross(r1, r2)
    rot = np.column_stack([r1, r2, r3])
    centre = -rot.T @ t
    if centre[2] < 0:                       # sign of G is arbitrary: put the camera above the floor
        rot = np.column_stack([-r1, -r2, r3])
        centre = -rot.T @ (-t)
    return CameraCentre(
        nadir_x=float(centre[0]), nadir_y=float(centre[1]), height=float(centre[2]),
        norm_mismatch=float(abs(n1 - n2) / scale),
        ortho_residual=float(abs(r1 @ r2)),
        tilt_deg=float(np.degrees(np.arccos(np.clip(abs(r3[2]), 0.0, 1.0)))),
    )


def camera_centre_from_calibration(calibration_yaml, intrinsics_npz):
    """Validated CameraCentre from the calibration files; ValueError on an implausible pose."""
    with open(calibration_yaml) as f:
        c = yaml.safe_load(f)
    with np.load(intrinsics_npz) as z:
        k = z["K"]
    g = floor_to_undistorted_pixel_homography(
        np.array(c["homography_matrix"], float).reshape(3, 3), c["pixels_per_meter"],
        c["real_origin_x"], c["real_origin_y"], c["bev_height"])
    cam = camera_centre_from_floor_homography(g, k)
    if cam.height <= 0 or cam.norm_mismatch > MAX_NORM_MISMATCH or cam.ortho_residual > MAX_ORTHO_RESIDUAL:
        raise ValueError(f"calibration does not give a valid camera pose: {cam}")
    return cam


def correct_height_parallax(x, y, top_height, nadir_x, nadir_y, camera_height):
    """Floor position under a point seen at (x, y) at height top_height."""
    if not 0.0 <= top_height < camera_height:
        raise ValueError(f"top_height {top_height} m must be in [0, camera height {camera_height} m)")
    s = 1.0 - top_height / camera_height
    return nadir_x + (x - nadir_x) * s, nadir_y + (y - nadir_y) * s
