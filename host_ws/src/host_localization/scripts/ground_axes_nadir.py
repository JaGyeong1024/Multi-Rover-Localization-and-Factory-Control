#!/usr/bin/env python3
"""ground_axes_nadir.py — board-free ground coordinate-axis viewer from intrinsics.

Intrinsics (K, D) alone do NOT fix the camera<->floor relation, so this ASSUMES the
camera looks straight down (nadir, no tilt) at a given height above the floor and
builds an APPROXIMATE ground homography from K + height:

    H_ground->img = [[fx/H, 0, cx],
                     [0, fy/H, cy],
                     [0,  0,   1 ]]

It then undistorts the live image and draws a movable ground coordinate frame
(translate / flip-y / roll), ALWAYS — no ChArUco board needed.

Caveats: nadir + no-tilt approximation; the frame's principal/origin sits under the
camera. For an EXACT frame matching a real board / uwb_global, use the plane mode +
saved homography instead. This is for quick board-free placement / visualization.

    python3 scripts/ground_axes_nadir.py --calib ~/charuco_intrinsics.npz --height 3.0 \
        --translate-x -2.01 --translate-y 1.18 --flip-y

Live keys:
    d / a = +x / -x        w / s = +y / -y     (translate, 0.05 m steps)
    [ / ] = roll -/+ 1deg  f = toggle flip-y   = / - = height +/- 0.05 m
    g = toggle grid        v = save npz        p = print params    q / ESC = quit
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)                          # sibling import (charuco_...)
sys.path.insert(0, os.path.join(_THIS, ".."))      # host_localization.camera_io

from charuco_calibration_with_plane_axis import (   # noqa: E402  (reuse proven draw helpers)
    load_calibration, draw_ground_frame, draw_ground_grid,
)
from host_localization.camera_io import (        # noqa: E402
    open_camera, resolve_device, GEOMETRIC_CONTROLS,
)

WIN = "ground_axes_nadir"


def nadir_homography(K, height):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H_g2i = np.array([[fx / height, 0.0, cx],
                      [0.0, fy / height, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    return H_g2i


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calib", default="~/charuco_intrinsics.npz", help="intrinsic .npz (K, D)")
    ap.add_argument("--device", default="/dev/global_cam")
    ap.add_argument("--height", type=float, default=3.0, help="camera height above floor (m)")
    ap.add_argument("--translate-x", type=float, default=0.0)
    ap.add_argument("--translate-y", type=float, default=0.0)
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--roll-deg", type=float, default=0.0)
    ap.add_argument("--axis-len-m", type=float, default=0.5)
    ap.add_argument("--exposure", type=int, default=None, help="manual exposure (default auto)")
    ap.add_argument("--output", default="~/ground_frame_nadir.npz",
                    help="npz save path for the moved frame (press 'v')")
    args = ap.parse_args()

    calib = load_calibration(args.calib)
    K, D = calib.K, calib.D
    w, h = calib.image_size
    print(f"[nadir] loaded {os.path.expanduser(args.calib)}  K fx={K[0,0]:.1f} cx={K[0,2]:.1f} "
          f"cy={K[1,2]:.1f}  size {w}x{h}")

    dev = resolve_device(args.device)
    controls = dict(GEOMETRIC_CONTROLS)
    if args.exposure is not None:
        controls.update({"auto_exposure": 1, "exposure_dynamic_framerate": 0,
                         "exposure_time_absolute": args.exposure})
    else:
        controls["auto_exposure"] = 3
    cap = open_camera(dev, w, h, controls=controls, verbose=True)

    # live-adjustable state
    tx, ty, roll = args.translate_x, args.translate_y, args.roll_deg
    flip, cam_h, grid = args.flip_y, args.height, True

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print("[nadir] keys: d/a/w/s=move  f=flip  [ ]=roll  =/-=height  g=grid  v=save  p=print  q=quit")
    map1 = map2 = None
    t0, fc, fps = time.monotonic(), 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        fc += 1
        now = time.monotonic()
        if now - t0 >= 0.5:
            fps, t0, fc = fc / (now - t0), now, 0
        fh, fw = frame.shape[:2]
        if map1 is None or map1.shape[:2] != (fh, fw):
            map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (fw, fh), cv2.CV_16SC2)
        vis = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        H_g2i = nadir_homography(K, cam_h)             # ground meters -> undistorted px
        origin = np.array([tx, ty], dtype=np.float64)  # moved-frame origin = base + translate

        # BASE frame: principal point at z=0 = origin (0,0), identity. GRAY = reference.
        draw_ground_frame(vis, H_g2i, np.array([0.0, 0.0]), 0.0, axis_len_m=args.axis_len_m,
                          label_prefix="base(z0)", flip_y=False,
                          x_color=(210, 210, 210), y_color=(210, 210, 210))
        # MOVED frame: base + translate + flip + roll. GREEN +X / MAGENTA +Y = your new frame.
        draw_ground_frame(vis, H_g2i, origin, roll, axis_len_m=args.axis_len_m,
                          label_prefix="moved", flip_y=flip)
        if grid:
            draw_ground_grid(vis, H_g2i, origin, roll, flip)

        cv2.putText(vis, f"fps {fps:4.1f}  height {cam_h:.2f}m  scale {K[0,0]/cam_h:.0f}px/m  "
                         f"origin=({tx:+.2f},{ty:+.2f})m roll={roll:+.1f} flip={flip}  [NADIR approx]",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 230, 60), 2, cv2.LINE_AA)
        cv2.imshow(WIN, cv2.resize(vis, (1280, 720)))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('d'):
            tx += 0.05
        elif k == ord('a'):
            tx -= 0.05
        elif k == ord('w'):
            ty += 0.05
        elif k == ord('s'):
            ty -= 0.05
        elif k == ord('f'):
            flip = not flip
        elif k == ord('['):
            roll -= 1.0
        elif k == ord(']'):
            roll += 1.0
        elif k == ord('='):
            cam_h += 0.05
        elif k == ord('-'):
            cam_h = max(0.3, cam_h - 0.05)
        elif k == ord('g'):
            grid = not grid
        elif k == ord('p'):
            print(f"[nadir] --height {cam_h:.2f} --translate-x {tx:.2f} "
                  f"--translate-y {ty:.2f} --roll-deg {roll:.1f}" + (" --flip-y" if flip else ""))
        elif k == ord('v'):
            outp = os.path.expanduser(args.output)
            H_g2i_save = nadir_homography(K, cam_h)
            np.savez(
                outp,
                model="nadir",
                K=K, D=D,
                image_width=w, image_height=h,
                height=cam_h,
                H_ground_to_img=H_g2i_save,         # base ground meters -> undistorted px
                H_img_to_ground=np.linalg.inv(H_g2i_save),  # undistorted px -> base ground meters
                translate_xy=np.array([tx, ty], dtype=np.float64),  # base -> moved origin
                roll_deg=float(roll),
                flip_y=bool(flip),
                scale_px_per_m=float(K[0, 0] / cam_h),
            )
            print(f"[nadir] saved -> {outp}  height {cam_h:.2f}  translate ({tx:.2f},{ty:.2f})  "
                  f"roll {roll:.1f}  flip {flip}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
