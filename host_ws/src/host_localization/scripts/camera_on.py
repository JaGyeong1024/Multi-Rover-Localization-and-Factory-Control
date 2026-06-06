#!/usr/bin/env python3
"""camera_on — minimal GEOMETRIC-only camera-on for intrinsic-calibration capture.

Intrinsic K,D depend ONLY on geometry (zoom + focus + resolution/FOV), not on
exposure/gain/white-balance. So this fixes just the geometric layer (zoom 1x,
autofocus OFF, fixed focus, 1080p MJPG) and leaves exposure on AUTO so the
checkerboard stays visible. Use it to view / capture checkerboard or ChArUco
shots for intrinsics. The SAME fixed focus/zoom must be used at runtime.

    python3 scripts/camera_on.py                       # live view
    python3 scripts/camera_on.py --focus 40 --save-dir ~/intrinsic_shots

Keys:  s = save frame   z = toggle 1:1 native crop / full-fit   q/ESC = quit
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from host_localization.camera_io import open_camera, resolve_device, GEOMETRIC_CONTROLS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', default='/dev/global_cam')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--focus', type=int, default=None, help='fixed focus_absolute (default from GEOMETRIC_CONTROLS)')
    ap.add_argument('--save-dir', default='/tmp/intrinsic_shots')
    args = ap.parse_args()

    dev = resolve_device(args.device)
    controls = dict(GEOMETRIC_CONTROLS)              # zoom 1x + autofocus off + fixed focus
    if args.focus is not None:
        controls['focus_absolute'] = args.focus
    controls['auto_exposure'] = 3                    # AUTO exposure — photometric ⊥ K,D
    cap = open_camera(dev, args.width, args.height, controls=controls, verbose=True)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'[camera_on] geometric-only (zoom 1x, autofocus off, focus={controls["focus_absolute"]}, '
          f'exposure AUTO). s=save  q=quit')
    n, t0, fc, fps = 0, time.monotonic(), 0, 0.0
    one_to_one = True  # default: native pixels (true sharpness for focus check)
    cv2.namedWindow('camera_on', cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        fc += 1
        now = time.monotonic()
        if now - t0 >= 0.5:
            fps, t0, fc = fc / (now - t0), now, 0
        if one_to_one:
            # native-resolution CENTER CROP — 1 screen px = 1 sensor px (real sharpness)
            h0, w0 = frame.shape[:2]
            cw, ch = min(1280, w0), min(720, h0)
            x0, y0 = (w0 - cw) // 2, (h0 - ch) // 2
            disp = frame[y0:y0 + ch, x0:x0 + cw].copy()
            tag = f'1:1 center {cw}x{ch} (z=full)'
        else:
            disp = cv2.resize(frame, (1280, 720))  # downscaled overview for framing
            tag = 'full fit (z=1:1)'
        cv2.putText(disp, f'fps {fps:4.1f}  saved {n}  [{tag}]  s=save q=quit', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 230, 60), 2, cv2.LINE_AA)
        cv2.imshow('camera_on', disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord('z'):
            one_to_one = not one_to_one
        if k == ord('s'):
            p = os.path.join(args.save_dir, f'shot_{n:03d}.png')
            cv2.imwrite(p, frame)
            n += 1
            print('[camera_on] saved', p)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
