#!/usr/bin/env python3
"""camera_tuner — live v4l2 tuner for the global/ceiling BRIO with a readable panel.

OpenCV's built-in trackbars truncate names and hide values, so this draws its own
control panel (name · draggable bar · value · min..max) below the live feed and
applies each change to the camera in real time. Use it to set exposure / gain /
brightness / white-balance / focus for the (gray floor, black rover, black lines)
scene, then press `s` to dump a freeze table.

All AUTO controls (auto-exposure, auto-WB, autofocus, dynamic-framerate, backlight
compensation) are forced OFF at startup and NOT shown — the goal is one fixed,
reproducible pixel pipeline (calib == YOLO == runtime), so nothing may auto-adjust.

Needs a display (run on the desktop):
    python3 scripts/camera_tuner.py            # /dev/global_cam (→ video2 fallback)

Mouse: click / drag a bar to set its value.   Keys:  s = save freeze table   q/ESC = quit

Sliders set the camera's response to light (exposure/gain), not room illumination.
Long exposure brightens a dim scene but blurs a moving rover — prefer more light +
a short fixed exposure.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from host_localization.camera_io import open_camera, resolve_device  # noqa: E402

WIN = 'camera_tuner'
# Manual value controls shown as sliders.
INT_CONTROLS = ['exposure_time_absolute', 'gain', 'brightness', 'contrast',
                'saturation', 'sharpness', 'white_balance_temperature', 'focus_absolute']
MENU_CONTROLS = ['power_line_frequency']      # real setting (50/60 Hz), not "auto"
# Auto controls: forced to manual/off at startup, never adjustable here.
FORCE_MANUAL = {'auto_exposure': 1, 'focus_automatic_continuous': 0,
                'white_balance_automatic': 0, 'exposure_dynamic_framerate': 0,
                'backlight_compensation': 0}

W, CAM_H, ROW_H = 1000, 562, 28
BAR_X, BAR_W = 300, 540
FONT = cv2.FONT_HERSHEY_SIMPLEX


def list_ctrls(dev):
    out = subprocess.run(['v4l2-ctl', '-d', dev, '--list-ctrls'], capture_output=True, text=True).stdout
    ctrls = {}
    for line in out.splitlines():
        m = re.match(r'\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)', line)
        if not m:
            continue
        name, _typ, rest = m.groups()
        d = {k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', rest)}
        if 'value' in d or 'default' in d:
            ctrls[name] = d
    return ctrls


def set_ctrl(dev, name, val):
    subprocess.run(['v4l2-ctl', '-d', dev, f'--set-ctrl={name}={val}'], capture_output=True, text=True)


class Tuner:
    def __init__(self, dev, cap, ctrls):
        self.dev, self.cap = dev, cap
        # force every auto OFF first so manual values take effect and stay fixed
        for n, v in FORCE_MANUAL.items():
            if n in ctrls:
                set_ctrl(dev, n, v)
        self.order = [n for n in INT_CONTROLS + MENU_CONTROLS if n in ctrls]
        self.rng, self.val = {}, {}
        for n in self.order:
            c = ctrls[n]
            self.rng[n] = (c.get('min', 0), c.get('max', 1), max(1, c.get('step', 1)))
            self.val[n] = c.get('value', c.get('default', 0))
        self.panel_y0 = CAM_H
        self.dragging = None
        self.fps = 0.0

    def _row_at(self, y):
        if y < self.panel_y0:
            return None
        r = (y - self.panel_y0) // ROW_H
        return self.order[r] if 0 <= r < len(self.order) else None

    def _set_from_x(self, name, x):
        lo, hi, st = self.rng[name]
        frac = min(1.0, max(0.0, (x - BAR_X) / BAR_W))
        v = int(min(hi, max(lo, lo + round(frac * (hi - lo) / st) * st)))
        if self.val[name] != v:
            self.val[name] = v
            set_ctrl(self.dev, name, v)

    def on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            n = self._row_at(y)
            if n:
                self.dragging = n
                self._set_from_x(n, x)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self._set_from_x(self.dragging, x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    def draw(self, frame):
        canvas = np.full((CAM_H + len(self.order) * ROW_H + 8, W, 3), 30, np.uint8)
        canvas[:CAM_H, :W] = cv2.resize(frame, (W, CAM_H))
        mean = float(frame.mean())
        blk = float((frame < 5).mean()) * 100
        wht = float((frame > 250).mean()) * 100
        cv2.rectangle(canvas, (0, 0), (430, 24), (0, 0, 0), -1)
        cv2.putText(canvas, f'mean {mean:5.1f}  black% {blk:4.1f}  white% {wht:4.1f}  fps {self.fps:4.1f}',
                    (6, 17), FONT, 0.5, (60, 230, 60), 1, cv2.LINE_AA)
        for i, n in enumerate(self.order):
            y = self.panel_y0 + i * ROW_H
            lo, hi, _ = self.rng[n]
            v = self.val[n]
            hot = (n == self.dragging)
            cv2.putText(canvas, n, (12, y + 19), FONT, 0.48,
                        (120, 230, 120) if hot else (225, 225, 225), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (BAR_X, y + 7), (BAR_X + BAR_W, y + 20), (70, 70, 78), -1)
            frac = (v - lo) / (hi - lo) if hi > lo else 0
            cv2.rectangle(canvas, (BAR_X, y + 7), (int(BAR_X + BAR_W * frac), y + 20), (90, 175, 90), -1)
            cv2.putText(canvas, str(v), (BAR_X + BAR_W + 10, y + 19), FONT, 0.5, (120, 230, 120), 1, cv2.LINE_AA)
            cv2.putText(canvas, f'{lo}..{hi}', (BAR_X + BAR_W + 62, y + 19), FONT, 0.4, (135, 135, 135), 1, cv2.LINE_AA)
        return canvas

    def save(self, out):
        lines = ['# camera freeze table — paste into config/global_cam_params.yaml']
        # include the forced-manual autos so the node freezes them too
        lines += [f'{n}: {v}' for n, v in FORCE_MANUAL.items()]
        lines += [f'{n}: {self.val[n]}' for n in self.order]
        block = '\n'.join(lines)
        print('\n' + block + '\n')
        with open(out, 'w') as f:
            f.write(block + '\n')
        print(f'[tuner] saved {out}')

    def run(self, out):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        print(f'[tuner] {len(self.order)} controls on {self.dev}. drag bars · s=save · q=quit')
        t0, n = time.monotonic(), 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                continue
            n += 1
            now = time.monotonic()
            if now - t0 >= 0.5:
                self.fps = n / (now - t0)
                t0, n = now, 0
            cv2.imshow(WIN, self.draw(frame))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                self.save(out)
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', default='/dev/global_cam')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--out', default='/tmp/camera_freeze.yaml')
    args = ap.parse_args()
    dev = resolve_device(args.device)
    cap = open_camera(dev, args.width, args.height, controls=None, verbose=True)
    ctrls = list_ctrls(dev)
    if not ctrls:
        print('[tuner] v4l2 컨트롤을 못 읽음 (v4l-utils?)')
        return
    Tuner(dev, cap, ctrls).run(args.out)


if __name__ == '__main__':
    main()
