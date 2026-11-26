#!/usr/bin/env python3
"""Open the ceiling BRIO (Logitech 4K, 046d:085e) and freeze its v4l2 controls.

Calibration and runtime need identical settings, so every consumer opens the camera here.
Prefers /dev/global_cam (falls back to /dev/video2). Controls are applied after every open.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2

DEFAULT_DEVICE = '/dev/global_cam'
FALLBACK_DEVICE = '/dev/video2'

# Geometry (FOV 90 deg set outside Linux, 1920x1080 MJPG, zoom, focus) fixes the intrinsics K, D
# and must match calibration time. Each *_automatic is set BEFORE its absolute partner.
GEOMETRIC_CONTROLS = {
    'zoom_absolute': 100,             # 1x — digital zoom crops → different focal length
    'focus_automatic_continuous': 0,  # autofocus OFF — focus breathing shifts distortion
    'focus_absolute': 0,              # 0 = far/infinity (suits ceiling ~3m); same for calib + runtime
}
# Photometric: no effect on K, D; frozen for reproducibility.
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
# Runtime freeze = both; the intrinsic phase uses GEOMETRIC only.
DEFAULT_CONTROLS = {**GEOMETRIC_CONTROLS, **PHOTOMETRIC_CONTROLS}


def resolve_device(device):
    """int -> /dev/videoN; prefer an existing path, else the defaults."""
    if isinstance(device, int):
        device = f'/dev/video{device}'
    if device and os.path.exists(device):
        return device
    for alt in (DEFAULT_DEVICE, FALLBACK_DEVICE):
        if os.path.exists(alt):
            return alt
    return device  # let cv2 fail with a clear error


def device_index(path):
    """Follow a symlink to /dev/videoN and return N (OpenCV V4L2 opens by index)."""
    base = os.path.basename(os.path.realpath(path))
    if base.startswith('video') and base[5:].isdigit():
        return int(base[5:])
    return None


def apply_v4l2_controls(device, controls=None, verbose=True):
    """Apply the freeze table via v4l2-ctl, one control at a time; True if all succeeded."""
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
    """Open by path, set format/fps, apply the freeze table, warm up; return the capture."""
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


# 30 consecutive failed grabs (~1 s at 30 fps) mean the device is gone and needs a reopen.
REOPEN_AFTER_FAILURES = 30
# Spacing between reopen attempts while the device stays away.
REOPEN_RETRY_SEC = 2.0


class ReconnectingCapture:
    """VideoCapture wrapper that reopens after repeated grab failures.

    open_fn() returns a fully configured capture (used for every open). read() never raises:
    (False, None) while the device is down.
    """

    def __init__(self, open_fn, clock, fail_threshold=REOPEN_AFTER_FAILURES,
                 retry_sec=REOPEN_RETRY_SEC, on_event=None):
        self._open_fn = open_fn
        self._clock = clock                 # callable -> seconds (monotonic)
        self._threshold = fail_threshold
        self._retry_sec = retry_sec
        self._on_event = on_event or (lambda kind, msg: None)
        self._cap = open_fn()
        self._fails = 0
        self._next_try = 0.0
        self._down = False

    def isOpened(self):
        return self._cap is not None and self._cap.isOpened()

    def release(self):
        if self._cap is not None:
            self._cap.release()

    def read(self):
        ret, frame = (False, None)
        if self._cap is not None:
            ret, frame = self._cap.read()
        if ret:
            self._fails = 0
            if self._down:
                self._down = False
                self._on_event('recovered', 'camera recovered')
            return True, frame
        self._fails += 1
        if self._fails >= self._threshold and self._clock() >= self._next_try:
            self._reopen()
        return False, None

    def _reopen(self):
        self._down = True
        self._next_try = self._clock() + self._retry_sec
        self._on_event('reopen', f'camera lost ({self._fails} failed grabs) — reopening')
        self.release()
        self._cap = None
        try:
            self._cap = self._open_fn()
        except Exception as exc:  # noqa: BLE001 - device may still be absent; retry later
            self._on_event('reopen_failed', f'camera reopen failed: {exc}')


def main():
    """Headless smoke test: grab one frame and save it."""
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
