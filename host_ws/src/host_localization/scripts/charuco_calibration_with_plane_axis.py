#!/usr/bin/env python3
"""
ChArUco intrinsic calibration + ground-plane coordinate helper.

Board default:
    squares_x     = 11
    squares_y     = 8
    square_length = 0.07   # meter
    marker_length = 0.052  # meter

Modes:
    capture    Live camera view, ChArUco overlay, save raw calibration frames.
    calibrate  Detect ChArUco corners from saved frames and estimate K, D.
    plane      Load K,D, undistort live view, estimate image<->ground homography,
               draw principal-point axes and a translated/rotated ground frame.

Recommended flow:
    1) Capture calibration images at the SAME width/height/focus used at runtime.
       python3 charuco_calibration_with_plane_axis.py capture \
           --device /dev/global_cam --focus 40 --save-dir ~/intrinsic_shots

    2) Calibrate intrinsics.
       python3 charuco_calibration_with_plane_axis.py calibrate \
           --image-dir ~/intrinsic_shots --output ~/charuco_intrinsics.npz

    3) Put the ChArUco board flat on the floor plane and run plane setup.
       python3 charuco_calibration_with_plane_axis.py plane \
           --device /dev/global_cam --focus 40 \
           --calib ~/charuco_intrinsics.npz \
           --translate-x 0.00 --translate-y 0.00 --roll-deg 0.0 \
           --output ~/ground_plane_frame.npz

Keys:
    capture: s = save raw frame, z = toggle 1:1 crop/full-fit, q/ESC = quit
    plane:   h = lock/update homography, u = unlock live homography,
             p = save plane setup, z = toggle 1:1 crop/full-fit, q/ESC = quit
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Optional project camera wrapper. If this script is placed under scripts/ in
# your repository, this mirrors the camera_on.py behavior you already have.
# -----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
try:
    from host_localization.camera_io import (  # type: ignore
        GEOMETRIC_CONTROLS,
        open_camera,
        resolve_device,
    )

    HAS_FACTORY_CAMERA = True
except Exception:
    GEOMETRIC_CONTROLS = {}
    open_camera = None
    resolve_device = None
    HAS_FACTORY_CAMERA = False


# -----------------------------------------------------------------------------
# ChArUco board specification
# -----------------------------------------------------------------------------
DEFAULT_SQUARES_X = 11
DEFAULT_SQUARES_Y = 8
DEFAULT_SQUARE_LENGTH = 0.07
DEFAULT_MARKER_LENGTH = 0.052
DEFAULT_DICTIONARY = "DICT_4X4_50"   # 실측: 이 보드는 DICT_4X4_50 (5X5 아님)

DICT_NAMES: Dict[str, int] = {
    name: getattr(cv2.aruco, name)
    for name in (
        "DICT_4X4_50",
        "DICT_4X4_100",
        "DICT_4X4_250",
        "DICT_5X5_50",
        "DICT_5X5_100",
        "DICT_5X5_250",
        "DICT_6X6_100",
        "DICT_6X6_250",
        "DICT_6X6_1000",
        "DICT_7X7_1000",
    )
    if hasattr(cv2.aruco, name)
}


@dataclass
class BoardConfig:
    squares_x: int = DEFAULT_SQUARES_X
    squares_y: int = DEFAULT_SQUARES_Y
    square_length: float = DEFAULT_SQUARE_LENGTH
    marker_length: float = DEFAULT_MARKER_LENGTH
    dictionary_name: str = DEFAULT_DICTIONARY
    legacy_pattern: bool = False


@dataclass
class CameraCalibration:
    K: np.ndarray
    D: np.ndarray
    image_size: Tuple[int, int]  # (width, height)
    rms: Optional[float] = None


@dataclass
class PlaneFrame:
    H_img_to_plane: np.ndarray
    H_plane_to_img: np.ndarray
    principal_pixel: np.ndarray
    principal_plane: np.ndarray
    origin_plane: np.ndarray
    roll_deg: float
    translate_xy: np.ndarray
    flip_y: bool = False


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_dictionary(dictionary_name: str) -> cv2.aruco.Dictionary:
    if dictionary_name not in DICT_NAMES:
        valid = ", ".join(sorted(DICT_NAMES))
        raise ValueError(f"Unknown dictionary '{dictionary_name}'. Valid names: {valid}")
    return cv2.aruco.getPredefinedDictionary(DICT_NAMES[dictionary_name])


def create_charuco_board(cfg: BoardConfig):
    dictionary = get_dictionary(cfg.dictionary_name)
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (cfg.squares_x, cfg.squares_y),
            cfg.square_length,
            cfg.marker_length,
            dictionary,
        )
    elif hasattr(cv2.aruco, "CharucoBoard_create"):
        # Older OpenCV-contrib API.
        board = cv2.aruco.CharucoBoard_create(
            cfg.squares_x,
            cfg.squares_y,
            cfg.square_length,
            cfg.marker_length,
            dictionary,
        )
    else:
        raise RuntimeError("Your cv2.aruco build does not include CharucoBoard.")

    # Important only when the physical board was generated by OpenCV <= 4.6 and
    # the row count is even. Your board has squares_y=8, so keep this option.
    if cfg.legacy_pattern and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board, dictionary


def make_charuco_detector(board, dictionary):
    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            detector_params = cv2.aruco.DetectorParameters()
        except Exception:
            detector_params = None
        try:
            charuco_params = cv2.aruco.CharucoParameters()
        except Exception:
            charuco_params = None

        try:
            if charuco_params is not None and detector_params is not None:
                return cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
            return cv2.aruco.CharucoDetector(board)
        except Exception:
            return cv2.aruco.CharucoDetector(board)

    # Legacy path uses cv2.aruco.detectMarkers + interpolateCornersCharuco.
    return None


def detect_charuco(gray: np.ndarray, board, dictionary, detector=None):
    """Return charuco_corners, charuco_ids, marker_corners, marker_ids."""
    if detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        return charuco_corners, charuco_ids, marker_corners, marker_ids

    # Older OpenCV-contrib API fallback.
    try:
        params = cv2.aruco.DetectorParameters_create()
    except Exception:
        params = cv2.aruco.DetectorParameters()

    marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
        gray, dictionary, parameters=params
    )
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids

    try:
        cv2.aruco.refineDetectedMarkers(gray, board, marker_corners, marker_ids, rejected)
    except Exception:
        pass

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    if retval is None or retval <= 0:
        return None, None, marker_corners, marker_ids
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def count_ids(ids: Optional[np.ndarray]) -> int:
    if ids is None:
        return 0
    return int(np.asarray(ids).reshape(-1).shape[0])


def draw_charuco_overlay(
    image_bgr: np.ndarray,
    charuco_corners,
    charuco_ids,
    marker_corners=None,
    marker_ids=None,
) -> np.ndarray:
    out = image_bgr
    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, marker_corners, marker_ids)
    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids)
    return out


def get_board_chessboard_corners(board) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        corners = board.chessboardCorners
    else:
        raise RuntimeError("Cannot access board chessboard corners from this OpenCV build.")
    return np.asarray(corners, dtype=np.float32).reshape(-1, 3)


def object_and_image_points_from_charuco(
    board, charuco_corners: np.ndarray, charuco_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    all_obj = get_board_chessboard_corners(board)
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    img = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)

    valid = (ids >= 0) & (ids < all_obj.shape[0])
    ids = ids[valid]
    img = img[valid]
    obj = all_obj[ids]
    return obj.astype(np.float32), img.astype(np.float32)


def apply_homography(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    out_h = (H @ pts_h.T).T
    denom = out_h[:, 2:3]
    denom[np.abs(denom) < 1e-12] = 1e-12
    return (out_h[:, :2] / denom).astype(np.float64)


def rot2(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def frame_matrix(roll_deg: float, flip_y: bool = False) -> np.ndarray:
    """Ground-frame basis (2x2): in-plane roll (about z) + optional 180 deg about
    the x-axis (mirror y: (x,y)->(x,-y)). Orthogonal, so M.T is its inverse."""
    M = rot2(np.deg2rad(roll_deg))
    if flip_y:
        M = M @ np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    return M


def pixel_to_new_xy(
    pixel_xy: np.ndarray,
    H_img_to_plane: np.ndarray,
    origin_plane: np.ndarray,
    roll_deg: float,
    flip_y: bool = False,
) -> np.ndarray:
    """Map undistorted image pixel(s) to the translated/rotated(/flipped) ground frame."""
    plane_xy = apply_homography(H_img_to_plane, pixel_xy)
    M = frame_matrix(roll_deg, flip_y)
    return ((plane_xy - origin_plane.reshape(1, 2)) @ M).astype(np.float64)


def new_xy_to_pixel(
    new_xy: np.ndarray,
    H_plane_to_img: np.ndarray,
    origin_plane: np.ndarray,
    roll_deg: float,
    flip_y: bool = False,
) -> np.ndarray:
    """Map point(s) from the translated/rotated(/flipped) ground frame to undistorted image pixels."""
    pts = np.asarray(new_xy, dtype=np.float64).reshape(-1, 2)
    M = frame_matrix(roll_deg, flip_y)
    plane_xy = origin_plane.reshape(1, 2) + (pts @ M.T)
    return apply_homography(H_plane_to_img, plane_xy)


def compute_plane_homography_from_charuco(
    board,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    min_corners: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute H_img_to_plane using undistorted image pixels and board XY in meters."""
    obj3d, img2d = object_and_image_points_from_charuco(board, charuco_corners, charuco_ids)
    if len(obj3d) < max(4, min_corners):
        raise ValueError(f"Need at least {max(4, min_corners)} corners, got {len(obj3d)}")

    plane_xy = obj3d[:, :2].astype(np.float32)
    img_xy = img2d.astype(np.float32)

    # H maps undistorted image pixel -> physical board/floor coordinates in meters.
    # RANSAC threshold is in destination-plane units here; 0.01 m = 1 cm.
    H, mask = cv2.findHomography(img_xy, plane_xy, cv2.RANSAC, 0.01)
    if H is None or H.shape != (3, 3):
        raise RuntimeError("findHomography failed.")
    inliers = int(mask.sum()) if mask is not None else len(img_xy)
    return H.astype(np.float64), mask, inliers


def principal_pixel_from_K(K: Optional[np.ndarray], image_size: Tuple[int, int]) -> np.ndarray:
    w, h = image_size
    if K is None:
        return np.array([(w - 1) * 0.5, (h - 1) * 0.5], dtype=np.float64)
    return np.array([float(K[0, 2]), float(K[1, 2])], dtype=np.float64)


def draw_principal_axes(
    image_bgr: np.ndarray,
    K: Optional[np.ndarray],
    label: str = "principal point",
    axis_len_px: int = 160,
    show_image_center: bool = True,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    pp = principal_pixel_from_K(K, (w, h))
    cx, cy = int(round(pp[0])), int(round(pp[1]))

    # Optional image center marker, useful to compare against calibrated principal point.
    if show_image_center and K is not None:
        icx, icy = int(round((w - 1) * 0.5)), int(round((h - 1) * 0.5))
        cv2.drawMarker(image_bgr, (icx, icy), (180, 180, 180), cv2.MARKER_CROSS, 24, 1)
        cv2.putText(
            image_bgr,
            "image center",
            (icx + 8, icy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    cv2.line(image_bgr, (cx, 0), (cx, h - 1), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image_bgr, (0, cy), (w - 1, cy), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(image_bgr, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)

    # Image coordinate axes at the principal point: +u/right, +v/down.
    cv2.arrowedLine(
        image_bgr,
        (cx, cy),
        (min(w - 1, cx + axis_len_px), cy),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )
    cv2.arrowedLine(
        image_bgr,
        (cx, cy),
        (cx, min(h - 1, cy + axis_len_px)),
        (255, 0, 0),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )
    cv2.putText(image_bgr, "+x image", (min(w - 1, cx + axis_len_px + 8), cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, "+y image", (cx + 8, min(h - 1, cy + axis_len_px + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(
        image_bgr,
        f"{label}: ({pp[0]:.1f}, {pp[1]:.1f})",
        (8, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image_bgr


def draw_ground_frame(
    image_bgr: np.ndarray,
    H_plane_to_img: np.ndarray,
    origin_plane: np.ndarray,
    roll_deg: float,
    axis_len_m: float = 0.20,
    label_prefix: str = "new",
    flip_y: bool = False,
    x_color: Tuple[int, int, int] = (0, 255, 0),
    y_color: Tuple[int, int, int] = (255, 0, 255),
) -> None:
    origin = np.array([[0.0, 0.0]], dtype=np.float64)
    x_axis = np.array([[axis_len_m, 0.0]], dtype=np.float64)
    y_axis = np.array([[0.0, axis_len_m]], dtype=np.float64)

    p0 = new_xy_to_pixel(origin, H_plane_to_img, origin_plane, roll_deg, flip_y)[0]
    px = new_xy_to_pixel(x_axis, H_plane_to_img, origin_plane, roll_deg, flip_y)[0]
    py = new_xy_to_pixel(y_axis, H_plane_to_img, origin_plane, roll_deg, flip_y)[0]

    def as_int(p: np.ndarray) -> Tuple[int, int]:
        return int(round(float(p[0]))), int(round(float(p[1])))

    p0i, pxi, pyi = as_int(p0), as_int(px), as_int(py)
    h, w = image_bgr.shape[:2]

    def extend_axis(a, b, color):
        # draw the moved-frame axis as a FULL line across the image (the projected
        # axis is straight under a homography), like the principal-point crosshair.
        d = np.array(b, dtype=np.float64) - np.array(a, dtype=np.float64)
        n = float(np.hypot(d[0], d[1]))
        if n < 1e-6:
            return
        d /= n
        big = float(w + h)
        pa = (int(a[0] - d[0] * big), int(a[1] - d[1] * big))
        pb = (int(a[0] + d[0] * big), int(a[1] + d[1] * big))
        ok, q1, q2 = cv2.clipLine((0, 0, w, h), pa, pb)
        if ok:
            cv2.line(image_bgr, q1, q2, color, 1, cv2.LINE_AA)

    # axis colors are parameters (default green +X / magenta +Y)
    extend_axis(p0i, pxi, x_color)
    extend_axis(p0i, pyi, y_color)
    cv2.arrowedLine(image_bgr, p0i, pxi, x_color, 3, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(image_bgr, p0i, pyi, y_color, 3, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(image_bgr, p0i, 7, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(image_bgr, p0i, 7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image_bgr, f"+X {label_prefix}", (pxi[0] + 8, pxi[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, x_color, 2, cv2.LINE_AA)
    cv2.putText(image_bgr, f"+Y {label_prefix}", (pyi[0] + 8, pyi[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, y_color, 2, cv2.LINE_AA)
    cv2.putText(image_bgr, f"{label_prefix} origin px=({p0i[0]},{p0i[1]})",
                (8, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)


def draw_ground_grid(
    image_bgr: np.ndarray,
    H_plane_to_img: np.ndarray,
    origin_plane: np.ndarray,
    roll_deg: float,
    flip_y: bool = False,
    rng: float = 10.0,
    step: float = 1.0,
    color: Tuple[int, int, int] = (0, 190, 190),
) -> None:
    """Draw integer-meter grid points of the NEW frame that land inside the image,
    each labelled (x,y). Lets you read/verify the frame even when its origin is
    off-screen (e.g. a 2 m translate puts the origin outside the camera view)."""
    h, w = image_bgr.shape[:2]
    g = np.arange(-rng, rng + 1e-9, step)
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    px = new_xy_to_pixel(pts, H_plane_to_img, origin_plane, roll_deg, flip_y)
    for (mx, my), (u, v) in zip(pts, px):
        iu, iv = int(round(float(u))), int(round(float(v)))
        if 0 <= iu < w and 0 <= iv < h:
            cv2.circle(image_bgr, (iu, iv), 2, color, -1, cv2.LINE_AA)
            cv2.putText(image_bgr, f"({mx:.0f},{my:.0f})", (iu + 3, iv - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def show_status_line(image_bgr: np.ndarray, text: str, y: int = 26) -> None:
    cv2.putText(
        image_bgr,
        text,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (60, 230, 60),
        2,
        cv2.LINE_AA,
    )


def display_frame(frame: np.ndarray, one_to_one: bool, window_name: str) -> None:
    if one_to_one:
        h0, w0 = frame.shape[:2]
        cw, ch = min(1280, w0), min(720, h0)
        x0, y0 = (w0 - cw) // 2, (h0 - ch) // 2
        disp = frame[y0:y0 + ch, x0:x0 + cw]
    else:
        disp = cv2.resize(frame, (1280, 720))
    cv2.imshow(window_name, disp)


def load_calibration(path: str) -> CameraCalibration:
    data = np.load(os.path.expanduser(path), allow_pickle=True)
    K = np.asarray(data["K"], dtype=np.float64)
    D = np.asarray(data["D"], dtype=np.float64)
    if "image_size" in data:
        image_size_arr = np.asarray(data["image_size"]).reshape(-1).astype(int)
        image_size = (int(image_size_arr[0]), int(image_size_arr[1]))
    else:
        image_size = (int(data["image_width"]), int(data["image_height"]))
    rms = float(data["rms"]) if "rms" in data else None
    return CameraCalibration(K=K, D=D, image_size=image_size, rms=rms)


def save_calibration(path: str, calib: CameraCalibration, cfg: BoardConfig, extra: Dict[str, object]) -> None:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        K=calib.K,
        D=calib.D,
        image_size=np.array(calib.image_size, dtype=np.int32),
        image_width=calib.image_size[0],
        image_height=calib.image_size[1],
        rms=np.array(calib.rms if calib.rms is not None else np.nan, dtype=np.float64),
        squares_x=cfg.squares_x,
        squares_y=cfg.squares_y,
        square_length=cfg.square_length,
        marker_length=cfg.marker_length,
        dictionary_name=cfg.dictionary_name,
        legacy_pattern=cfg.legacy_pattern,
        **extra,
    )


def save_plane_frame(path: str, plane: PlaneFrame, calib: CameraCalibration, cfg: BoardConfig) -> None:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        H_img_to_plane=plane.H_img_to_plane,
        H_plane_to_img=plane.H_plane_to_img,
        K=calib.K,
        D=calib.D,
        image_size=np.array(calib.image_size, dtype=np.int32),
        principal_pixel=plane.principal_pixel,
        principal_plane=plane.principal_plane,
        origin_plane=plane.origin_plane,
        translate_xy=plane.translate_xy,
        roll_deg=np.array(plane.roll_deg, dtype=np.float64),
        flip_y=np.array(bool(plane.flip_y)),
        squares_x=cfg.squares_x,
        squares_y=cfg.squares_y,
        square_length=cfg.square_length,
        marker_length=cfg.marker_length,
        dictionary_name=cfg.dictionary_name,
        legacy_pattern=cfg.legacy_pattern,
    )


def open_geometric_camera(args):
    if HAS_FACTORY_CAMERA:
        dev = resolve_device(args.device)  # type: ignore[misc]
        controls = dict(GEOMETRIC_CONTROLS)
        if args.focus is not None:
            controls["focus_absolute"] = args.focus
        if getattr(args, "exposure", None) is not None:
            # manual exposure — darken to recover marker contrast (glare/washout)
            controls["auto_exposure"] = 1
            controls["exposure_dynamic_framerate"] = 0
            controls["exposure_time_absolute"] = args.exposure
        else:
            controls["auto_exposure"] = 3  # auto; does not change K,D
        cap = open_camera(dev, args.width, args.height, controls=controls, verbose=True)  # type: ignore[misc]
        return cap, controls

    # Generic fallback. Useful for laptop webcam tests outside your repo.
    dev_arg = int(args.device) if str(args.device).isdigit() else args.device
    cap = cv2.VideoCapture(dev_arg)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, args.focus)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {args.device}")
    return cap, {"fallback_cv2_video_capture": True, "focus": args.focus}


# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------
def run_capture(args) -> None:
    cfg = board_config_from_args(args)
    board, dictionary = create_charuco_board(cfg)
    detector = make_charuco_detector(board, dictionary)

    calib = load_calibration(args.calib) if args.calib else None
    cap, controls = open_geometric_camera(args)
    os.makedirs(os.path.expanduser(args.save_dir), exist_ok=True)

    print(
        f"[capture] board: {cfg.squares_x}x{cfg.squares_y}, square={cfg.square_length} m, "
        f"marker={cfg.marker_length} m, dict={cfg.dictionary_name}, legacy={cfg.legacy_pattern}"
    )
    print(f"[capture] camera controls: {controls}")
    print("[capture] keys: s=save  z=toggle 1:1/full  q/ESC=quit")

    cv2.namedWindow("charuco_capture", cv2.WINDOW_NORMAL)
    one_to_one = False  # default: full-fit (see whole frame for board/edge coverage); z = 1:1
    saved = 0
    t0, fc, fps = time.monotonic(), 0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        fc += 1
        now = time.monotonic()
        if now - t0 >= 0.5:
            fps, t0, fc = fc / (now - t0), now, 0

        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(
            gray, board, dictionary, detector
        )
        n_charuco = count_ids(charuco_ids)
        draw_charuco_overlay(vis, charuco_corners, charuco_ids, marker_corners, marker_ids)
        draw_principal_axes(
            vis,
            calib.K if calib is not None else None,
            label="K principal" if calib is not None else "image center proxy",
            show_image_center=(calib is not None),
        )
        tag = "1:1 center crop" if one_to_one else "full fit"
        show_status_line(
            vis,
            f"fps {fps:4.1f}  saved {saved}  charuco {n_charuco}  [{tag}]  s=save z=toggle q=quit",
        )
        display_frame(vis, one_to_one, "charuco_capture")

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("z"):
            one_to_one = not one_to_one
        if key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(os.path.expanduser(args.save_dir), f"charuco_{stamp}_{saved:03d}.png")
            cv2.imwrite(path, frame)
            saved += 1
            quality = "OK" if n_charuco >= args.min_corners else "LOW_CORNER_COUNT"
            print(f"[capture] saved {path}  charuco={n_charuco}  {quality}")

    cap.release()
    cv2.destroyAllWindows()


def run_calibrate(args) -> None:
    cfg = board_config_from_args(args)
    board, dictionary = create_charuco_board(cfg)
    detector = make_charuco_detector(board, dictionary)

    image_dir = os.path.expanduser(args.image_dir)
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_paths: List[str] = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pat)))
    image_paths = sorted(image_paths)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    debug_dir = os.path.expanduser(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    all_charuco_corners: List[np.ndarray] = []
    all_charuco_ids: List[np.ndarray] = []
    all_object_points: List[np.ndarray] = []
    all_image_points: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None
    used_paths: List[str] = []

    print(f"[calibrate] reading {len(image_paths)} images from {image_dir}")
    for idx, path in enumerate(image_paths):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[calibrate] skip unreadable: {path}")
            continue
        h, w = frame.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            print(f"[calibrate] skip size mismatch {path}: {(w, h)} != {image_size}")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(
            gray, board, dictionary, detector
        )
        n = count_ids(charuco_ids)
        if n >= args.min_corners:
            obj, img = object_and_image_points_from_charuco(board, charuco_corners, charuco_ids)
            all_charuco_corners.append(np.asarray(charuco_corners, dtype=np.float32))
            all_charuco_ids.append(np.asarray(charuco_ids, dtype=np.int32))
            all_object_points.append(obj)
            all_image_points.append(img)
            used_paths.append(path)
            status = "USED"
        else:
            status = "skip"

        print(f"[calibrate] {idx:03d}: charuco={n:02d}  {status}  {os.path.basename(path)}")

        if debug_dir:
            vis = frame.copy()
            draw_charuco_overlay(vis, charuco_corners, charuco_ids, marker_corners, marker_ids)
            show_status_line(vis, f"charuco={n} {status}")
            cv2.imwrite(os.path.join(debug_dir, f"debug_{idx:03d}_{status}.jpg"), vis)

    if image_size is None:
        raise RuntimeError("No readable images.")
    if len(all_image_points) < args.min_views:
        raise RuntimeError(
            f"Need at least {args.min_views} valid views, got {len(all_image_points)}. "
            f"Capture more varied views or lower --min-corners after checking quality."
        )

    flags = 0
    if args.rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-9,
    )

    print(f"[calibrate] valid views: {len(all_image_points)}  image_size={image_size}")
    print(f"[calibrate] using {'cv2.aruco.calibrateCameraCharuco' if hasattr(cv2.aruco, 'calibrateCameraCharuco') else 'cv2.calibrateCamera fallback'}")

    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        # Some OpenCV builds still expose this function. If it fails, fall back to
        # calibrateCamera with object/image points reconstructed from Charuco IDs.
        try:
            rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_charuco_corners,
                all_charuco_ids,
                board,
                image_size,
                None,
                None,
                flags=flags,
                criteria=criteria,
            )
        except Exception as exc:
            print(f"[calibrate] calibrateCameraCharuco failed: {exc}")
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                all_object_points,
                all_image_points,
                image_size,
                None,
                None,
                flags=flags,
                criteria=criteria,
            )
    else:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            all_object_points,
            all_image_points,
            image_size,
            None,
            None,
            flags=flags,
            criteria=criteria,
        )

    calib = CameraCalibration(K=np.asarray(K, dtype=np.float64), D=np.asarray(D, dtype=np.float64), image_size=image_size, rms=float(rms))
    save_calibration(
        args.output,
        calib,
        cfg,
        extra={
            "valid_views": len(all_image_points),
            "used_images": np.array(used_paths, dtype=object),
            "opencv_version": cv2.__version__,
        },
    )

    print("\n[calibrate] DONE")
    print(f"  RMS reprojection error: {rms:.6f} px")
    print("  K =")
    print(K)
    print("  D =")
    print(D.reshape(-1))
    print(f"  saved: {os.path.expanduser(args.output)}")


def run_plane(args) -> None:
    cfg = board_config_from_args(args)
    board, dictionary = create_charuco_board(cfg)
    detector = make_charuco_detector(board, dictionary)
    calib = load_calibration(args.calib)

    cap, controls = open_geometric_camera(args)
    print(f"[plane] loaded calibration: {args.calib}")
    print(f"[plane] K principal point: ({calib.K[0, 2]:.3f}, {calib.K[1, 2]:.3f})")
    print(f"[plane] camera controls: {controls}")
    print("[plane] Put the board flat on the ground plane. keys: h=lock/update H  u=unlock  p=save  z=toggle  q=quit")

    cv2.namedWindow("plane_axis", cv2.WINDOW_NORMAL)
    one_to_one = False
    locked_H: Optional[np.ndarray] = None
    last_H: Optional[np.ndarray] = None
    last_inliers = 0
    t0, fc, fps = time.monotonic(), 0, 0.0

    map1 = map2 = None
    expected_w, expected_h = calib.image_size
    principal_pixel = principal_pixel_from_K(calib.K, calib.image_size)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        fc += 1
        now = time.monotonic()
        if now - t0 >= 0.5:
            fps, t0, fc = fc / (now - t0), now, 0

        h, w = frame.shape[:2]
        if (w, h) != calib.image_size:
            # Scale K if the runtime frame size changed. Better: avoid this and use
            # the same resolution for calibration and runtime.
            sx = w / float(expected_w)
            sy = h / float(expected_h)
            K_runtime = calib.K.copy()
            K_runtime[0, :] *= sx
            K_runtime[1, :] *= sy
            image_size_runtime = (w, h)
        else:
            K_runtime = calib.K
            image_size_runtime = calib.image_size

        if map1 is None or map1.shape[:2] != (h, w):
            map1, map2 = cv2.initUndistortRectifyMap(
                K_runtime, calib.D, None, K_runtime, (w, h), cv2.CV_16SC2
            )
            principal_pixel = principal_pixel_from_K(K_runtime, image_size_runtime)

        undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        vis = undist.copy()
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(
            gray, board, dictionary, detector
        )
        n = count_ids(charuco_ids)
        draw_charuco_overlay(vis, charuco_corners, charuco_ids, marker_corners, marker_ids)
        draw_principal_axes(vis, K_runtime, label="calibrated principal", show_image_center=True)

        current_H = None
        inliers = 0
        if n >= args.min_corners:
            try:
                current_H, mask, inliers = compute_plane_homography_from_charuco(
                    board, charuco_corners, charuco_ids, args.min_corners
                )
                last_H = current_H
                last_inliers = inliers
            except Exception as exc:
                show_status_line(vis, f"H failed: {exc}", y=54)

        active_H = locked_H if locked_H is not None else last_H
        if active_H is not None:
            try:
                H_plane_to_img = np.linalg.inv(active_H)
                principal_plane = apply_homography(active_H, principal_pixel.reshape(1, 2))[0]
                translate_xy = np.array([args.translate_x, args.translate_y], dtype=np.float64)
                origin_plane = principal_plane + translate_xy

                # Mark where the optical/principal ray intersects the ground plane.
                pp_back = apply_homography(H_plane_to_img, principal_plane.reshape(1, 2))[0]
                cv2.circle(vis, (int(round(pp_back[0])), int(round(pp_back[1]))), 8,
                           (0, 255, 255), 2, cv2.LINE_AA)

                draw_ground_frame(
                    vis,
                    H_plane_to_img,
                    origin_plane,
                    args.roll_deg,
                    axis_len_m=args.axis_len_m,
                    label_prefix="ground",
                    flip_y=args.flip_y,
                )
                draw_ground_grid(vis, H_plane_to_img, origin_plane, args.roll_deg, args.flip_y)
                show_status_line(
                    vis,
                    f"fps {fps:4.1f} charuco {n} H_inliers {last_inliers} "
                    f"mode={'LOCKED' if locked_H is not None else 'LIVE'} "
                    f"pp_plane=({principal_plane[0]:+.3f},{principal_plane[1]:+.3f})m "
                    f"origin=({origin_plane[0]:+.3f},{origin_plane[1]:+.3f})m roll={args.roll_deg:+.2f}deg",
                    y=26,
                )
                # visible range = the 4 image corners mapped into each frame
                corners_px = np.array([[0., 0.], [w - 1, 0.], [w - 1, h - 1], [0., h - 1]])
                pc = apply_homography(active_H, corners_px)            # original (board/plane) frame
                nc = pixel_to_new_xy(corners_px, active_H, origin_plane, args.roll_deg, args.flip_y)
                show_status_line(
                    vis,
                    f"visible  plane x[{pc[:, 0].min():+.2f},{pc[:, 0].max():+.2f}] "
                    f"y[{pc[:, 1].min():+.2f},{pc[:, 1].max():+.2f}]   |   "
                    f"new x[{nc[:, 0].min():+.2f},{nc[:, 0].max():+.2f}] "
                    f"y[{nc[:, 1].min():+.2f},{nc[:, 1].max():+.2f}]  (m)",
                    y=52,
                )
            except Exception as exc:
                show_status_line(vis, f"active H draw failed: {exc}", y=26)
        else:
            show_status_line(
                vis,
                f"fps {fps:4.1f} charuco {n}  no homography yet; show board on floor",
                y=26,
            )

        display_frame(vis, one_to_one, "plane_axis")
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("z"):
            one_to_one = not one_to_one
        if key == ord("h"):
            if last_H is not None:
                locked_H = last_H.copy()
                print(f"[plane] homography locked/updated, inliers={last_inliers}")
            else:
                print("[plane] no valid H to lock")
        if key == ord("u"):
            locked_H = None
            print("[plane] homography unlocked; using live H")
        if key == ord("p"):
            active_H = locked_H if locked_H is not None else last_H
            if active_H is None:
                print("[plane] cannot save: no valid H")
            else:
                H_plane_to_img = np.linalg.inv(active_H)
                principal_plane = apply_homography(active_H, principal_pixel.reshape(1, 2))[0]
                translate_xy = np.array([args.translate_x, args.translate_y], dtype=np.float64)
                origin_plane = principal_plane + translate_xy
                plane = PlaneFrame(
                    H_img_to_plane=active_H,
                    H_plane_to_img=H_plane_to_img,
                    principal_pixel=principal_pixel,
                    principal_plane=principal_plane,
                    origin_plane=origin_plane,
                    roll_deg=float(args.roll_deg),
                    translate_xy=translate_xy,
                    flip_y=bool(args.flip_y),
                )
                save_plane_frame(args.output, plane, calib, cfg)
                print(f"[plane] saved: {os.path.expanduser(args.output)}")

    cap.release()
    cv2.destroyAllWindows()


def board_config_from_args(args) -> BoardConfig:
    return BoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary_name=args.dictionary,
        legacy_pattern=args.legacy_pattern,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def add_board_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X)
    p.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y)
    p.add_argument("--square-length", type=float, default=DEFAULT_SQUARE_LENGTH)
    p.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH)
    p.add_argument("--dictionary", default=DEFAULT_DICTIONARY, choices=sorted(DICT_NAMES.keys()))
    p.add_argument(
        "--legacy-pattern",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ON by default — this board needs it (even rows / old-OpenCV gen): "
             "legacy=False gives charuco=0, legacy=True gives valid corners. --no-legacy-pattern to disable.",
    )


def add_camera_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="/dev/global_cam")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--focus", type=int, default=0, help="fixed focus_absolute (0=far/infinity); must match calibration/runtime")
    p.add_argument("--exposure", type=int, default=None,
                   help="manual exposure_time_absolute (default: auto). Lower it if markers wash out (glare/too bright).")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ChArUco calibration and ground-plane axis helper")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_cap = sub.add_parser("capture", help="live camera capture for ChArUco intrinsic calibration")
    add_camera_args(p_cap)
    add_board_args(p_cap)
    p_cap.add_argument("--save-dir", default="~/intrinsic_shots")
    p_cap.add_argument("--calib", default=None, help="optional .npz calibration to draw true principal point")
    p_cap.add_argument("--min-corners", type=int, default=12)
    p_cap.set_defaults(func=run_capture)

    p_cal = sub.add_parser("calibrate", help="compute camera intrinsics from captured images")
    add_board_args(p_cal)
    p_cal.add_argument("--image-dir", default="~/intrinsic_shots")
    p_cal.add_argument("--output", default="~/charuco_intrinsics.npz")
    p_cal.add_argument("--debug-dir", default=None, help="optional directory for detection debug images")
    p_cal.add_argument("--min-corners", type=int, default=12)
    p_cal.add_argument("--min-views", type=int, default=15)
    p_cal.add_argument("--rational-model", action=argparse.BooleanOptionalAction, default=True,
                       help="k4,k5,k6 terms — default ON for the 90 wide-angle (--no-rational-model to disable)")
    p_cal.set_defaults(func=run_calibrate)

    p_plane = sub.add_parser("plane", help="estimate ground homography and draw principal/new axes")
    add_camera_args(p_plane)
    add_board_args(p_plane)
    p_plane.add_argument("--calib", default="~/charuco_intrinsics.npz", help="intrinsics .npz from calibrate mode")
    p_plane.add_argument("--output", default="~/ground_plane_frame.npz")
    p_plane.add_argument("--min-corners", type=int, default=12)
    p_plane.add_argument("--translate-x", type=float, default=0.0, help="meters, in board/floor X direction, added to principal ground point")
    p_plane.add_argument("--translate-y", type=float, default=0.0, help="meters, in board/floor Y direction, added to principal ground point")
    p_plane.add_argument("--roll-deg", type=float, default=0.0, help="2D in-plane rotation of the new ground frame, degrees")
    p_plane.add_argument("--flip-y", action="store_true",
                         help="180 deg about x-axis (mirror y: (x,y)->(x,-y)) — for handedness alignment")
    p_plane.add_argument("--axis-len-m", type=float, default=0.20)
    p_plane.set_defaults(func=run_plane)

    return ap


def main() -> None:
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()