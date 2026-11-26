"""Height-parallax correction tests (pure numpy/cv2, no ROS, no camera): pytest test/test_ceiling_parallax.py"""
import os

import cv2
import numpy as np
import pytest

from host_localization.ceiling_parallax import (
    camera_centre_from_calibration, camera_centre_from_floor_homography,
    correct_height_parallax, floor_to_undistorted_pixel_homography)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
CALIB_YAML = os.path.join(REPO, "data/calibration/calibration.yaml")
INTRINSICS = os.path.join(REPO, "data/calibration/charuco_intrinsics.npz")

K = np.array([[1178.0, 0.0, 956.5], [0.0, 1178.1, 571.0], [0.0, 0.0, 1.0]])
PPM, ORIGIN_X, ORIGIN_Y, BEV_H = 200.0, -0.0255, -0.0255, 698
TRUE_CENTRE = np.array([2.50, 1.70, 3.00])


def look_down_rotation(tilt_deg, azimuth_deg):
    """World->camera rotation (rows xc, yc, zc) for an optical axis tilted from straight down."""
    t, a = np.radians(tilt_deg), np.radians(azimuth_deg)
    zc = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)])
    xc = np.cross(np.array([0.0, 0.0, 1.0]), zc)
    if np.linalg.norm(xc) < 1e-9:                      # straight down: any horizontal x axis
        xc = np.array([1.0, 0.0, 0.0])
    xc /= np.linalg.norm(xc)
    return np.vstack([xc, np.cross(zc, xc), zc])


def project(rot, world_pts):
    cam = (np.atleast_2d(world_pts) - TRUE_CENTRE) @ rot.T
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


def calibrator_homography(rot):
    """Pixel->BEV homography fitted from a floor grid exactly like unified_calibrator does."""
    gx, gy = np.meshgrid(np.linspace(0.5, 5.0, 6), np.linspace(0.3, 3.5, 5))
    floor = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    bev = np.column_stack([(floor[:, 0] - ORIGIN_X) * PPM, BEV_H - (floor[:, 1] - ORIGIN_Y) * PPM])
    h, _ = cv2.findHomography(project(rot, floor).astype(np.float64), bev, 0)
    return h


def apparent_floor_position(h_img_to_bev, pixel):
    """What perception_node.px2world reports for an undistorted pixel (BEV px -> metres)."""
    bx, by, bw = h_img_to_bev @ np.array([*pixel, 1.0])
    bx, by = bx / bw, by / bw
    return bx / PPM + ORIGIN_X, (BEV_H - by) / PPM + ORIGIN_Y


@pytest.mark.parametrize("tilt,azimuth", [(0.0, 0.0), (8.0, 0.0), (8.0, 137.0), (15.0, -60.0)])
def test_synthetic_camera_centre_correction(tilt, azimuth):
    rot = look_down_rotation(tilt, azimuth)
    h_img = calibrator_homography(rot)
    g = floor_to_undistorted_pixel_homography(h_img, PPM, ORIGIN_X, ORIGIN_Y, BEV_H)
    cam = camera_centre_from_floor_homography(g, K)

    assert cam.nadir_x == pytest.approx(TRUE_CENTRE[0], abs=1e-6)
    assert cam.nadir_y == pytest.approx(TRUE_CENTRE[1], abs=1e-6)
    assert cam.height == pytest.approx(TRUE_CENTRE[2], abs=1e-6)
    assert cam.tilt_deg == pytest.approx(tilt, abs=1e-4)
    assert cam.norm_mismatch < 1e-6 and cam.ortho_residual < 1e-6

    for top_height in (0.1, 0.2, 0.35):
        for floor_xy in [(0.8, 0.5), (4.5, 3.0), (2.5, 1.7), (1.0, 3.3), (4.8, 0.6)]:
            top = np.array([[*floor_xy, top_height]])
            apparent = apparent_floor_position(h_img, project(rot, top)[0])
            fixed = correct_height_parallax(*apparent, top_height, cam.nadir_x, cam.nadir_y, cam.height)
            assert fixed == pytest.approx(floor_xy, abs=1e-3)


def test_uncorrected_error_exceeds_tolerance():
    rot = look_down_rotation(8.0, 0.0)
    h_img = calibrator_homography(rot)
    apparent = apparent_floor_position(h_img, project(rot, np.array([[4.5, 3.0, 0.2]]))[0])
    assert np.hypot(apparent[0] - 4.5, apparent[1] - 3.0) > 0.1


def test_zero_height_keeps_position():
    assert correct_height_parallax(4.321, 0.987, 0.0, 2.5, 1.7, 3.0) == (4.321, 0.987)


def test_correction_pulls_toward_nadir():
    x, y = correct_height_parallax(4.5, 1.7, 0.3, 2.5, 1.7, 3.0)
    assert (x, y) == pytest.approx((2.5 + 2.0 * 0.9, 1.7))


def test_height_at_or_above_camera_rejected():
    with pytest.raises(ValueError):
        correct_height_parallax(1.0, 1.0, 3.0, 2.5, 1.7, 3.0)


def test_real_calibration_plausible_camera():
    cam = camera_centre_from_calibration(CALIB_YAML, INTRINSICS)
    assert 2.9 < cam.height < 3.1
    assert cam.norm_mismatch < 0.01
    assert cam.ortho_residual < 0.02
    assert 0.0 < cam.tilt_deg < 15.0
