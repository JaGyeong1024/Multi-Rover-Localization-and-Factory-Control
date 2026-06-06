#!/usr/bin/env python3
"""camera_io — single source for opening the ceiling BRIO + freezing v4l2 controls.

Why this module: the camera was opened in 6 places with inconsistent settings,
and the v4l2 freeze table (focus/exposure/WB) from the calibration manual §4 was
applied NOWHERE — only autofocus+zoom via OpenCV, which is unreliable. Calibration
requires identical settings at calib-time and run-time, so every consumer (tracker,
calibrator, test scripts) should open through here.

Device: prefer the stable udev symlink /dev/global_cam (→ BRIO RGB node; see
99-factory-devices.rules). Falls back to /dev/video2. Open by PATH (not int index)
so the symlink survives /dev/videoN reordering.

Freeze table: Logitech BRIO 4K (046d:085e) real control names confirmed via
`v4l2-ctl --list-ctrls` on this kernel. Applied AFTER open (opening can reset
controls) and every session (focus etc. reset on unplug/reboot). The photometric
values were frozen 2026-05-31 from camera_tuner.py (camera ceiling-mounted).
"""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2

DEFAULT_DEVICE = '/dev/global_cam'
FALLBACK_DEVICE = '/dev/video2'

# ── FROZEN GEOMETRY (determines the intrinsic matrix K, D) ──────────────────────
# These 4 fix the lens projection and MUST be identical at intrinsic-calibration
# time AND at runtime, or the computed K,D won't fit. Photometric settings
# (exposure/gain/WB/brightness/...) do NOT affect K,D.
#   1. FOV        = 90°             <- Windows Logi Tune ONLY (not settable on Linux);
#                                      set once, never change. Re-calibrate if changed.
#   2. resolution = 1920x1080 MJPG  <- open_camera(width,height) / node params
#   3. zoom       = 1x (100)        <- GEOMETRIC_CONTROLS below (v4l2)
#   4. focus      = autofocus off + focus_absolute 0 (far/infinity)  <- below (v4l2)
#
# Layer 1 — GEOMETRIC (the v4l2-settable part of the above: zoom + focus).
# (order: each *_automatic set BEFORE its absolute partner.)
GEOMETRIC_CONTROLS = {
    'zoom_absolute': 100,             # 1x — digital zoom crops → different focal length
    'focus_automatic_continuous': 0,  # autofocus OFF — focus breathing shifts distortion
    'focus_absolute': 0,              # 0 = far/infinity (suits ceiling ~3m); same for calib + runtime
}
# Layer 2 — PHOTOMETRIC: brightness / colour only. Does NOT change K,D, so it is
# irrelevant to intrinsic calibration. Tune for the scene (exposure/detection/YOLO)
# and freeze for runtime reproducibility.
# Values frozen 2026-05-31 from camera_tuner.py (camera ceiling-mounted, scene lit).
PHOTOMETRIC_CONTROLS = {
    'auto_exposure': 1,               # 1 = Manual
    'exposure_dynamic_framerate': 0,
    'exposure_time_absolute': 105,    # short -> less rover motion blur
    'gain': 150,                      # raised to offset the short exposure
    'backlight_compensation': 0,
    'white_balance_automatic': 0,
    'white_balance_temperature': 5100,
    'brightness': 85,
    'contrast': 150,
    'saturation': 40,
    'sharpness': 255,                 # MAX — watch fiducial subpixel (over-sharpening)
    'power_line_frequency': 2,        # 60 Hz (Korea) — anti-flicker banding
}
# Full freeze (runtime): geometric + photometric. Intrinsic phase uses GEOMETRIC only.
DEFAULT_CONTROLS = {**GEOMETRIC_CONTROLS, **PHOTOMETRIC_CONTROLS}


def resolve_device(device):
    """int -> /dev/videoN; prefer an existing symlink; fall back to /dev/video2."""
    if isinstance(device, int):
        device = f'/dev/video{device}'
    if device and os.path.exists(device):
        return device
    for alt in (DEFAULT_DEVICE, FALLBACK_DEVICE):
        if os.path.exists(alt):
            return alt
    return device  # let cv2 fail with a clear error


def device_index(path):
    """Follow a symlink (e.g. /dev/global_cam) to /dev/videoN and return N.

    OpenCV's V4L2 backend cannot capture *by name* — it needs the integer index.
    We keep the stable symlink for v4l2-ctl + provenance, but open by index.
    """
    base = os.path.basename(os.path.realpath(path))
    if base.startswith('video') and base[5:].isdigit():
        return int(base[5:])
    return None


def apply_v4l2_controls(device, controls=None, verbose=True):
    """Apply the freeze table via v4l2-ctl. Per-control so one bad name doesn't
    block the rest. Returns True if all succeeded."""
    controls = DEFAULT_CONTROLS if controls is None else controls
    if shutil.which('v4l2-ctl') is None:
        if verbose:
            print('[camera_io] v4l2-ctl 없음 — 컨트롤 미적용 (sudo apt install v4l-utils)')
        return False
    ok = True
    for k, v in controls.items():
        r = subprocess.run(['v4l2-ctl', '-d', str(device), f'--set-ctrl={k}={v}'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            ok = False
            if verbose:
                print(f'[camera_io] set {k}={v} 실패: {r.stderr.strip() or r.stdout.strip()}')
    if verbose and ok:
        print(f'[camera_io] v4l2 동결 {len(controls)} ctrls 적용: {device}')
    return ok


def open_camera(device=DEFAULT_DEVICE, width=1920, height=1080, fps=30,
                fourcc='MJPG', controls=DEFAULT_CONTROLS, warmup=5, verbose=True):
    """Open the camera by path, set MJPG/resolution/fps, apply the v4l2 freeze
    table, warm up, and return the cv2.VideoCapture."""
    path = resolve_device(device)
    idx = device_index(path)
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if idx is not None else cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open camera {path} (index={idx})')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if controls:
        apply_v4l2_controls(path, controls, verbose)
    for _ in range(max(0, warmup)):
        cap.read()
    if verbose:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'[camera_io] opened {path} @ {w}x{h} {fourcc}')
    return cap


def main():
    """Headless smoke test: open, grab one frame, save it. No display needed."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', default=DEFAULT_DEVICE)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--save', default='/tmp/ceiling_frame.png')
    ap.add_argument('--no-controls', action='store_true', help='skip v4l2 freeze table')
    args = ap.parse_args()
    cap = open_camera(args.device, args.width, args.height,
                      controls=None if args.no_controls else DEFAULT_CONTROLS)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit('[camera_io] frame grab failed')
    cv2.imwrite(args.save, frame)
    print(f'[camera_io] saved {args.save}  shape={frame.shape}')


if __name__ == '__main__':
    main()
