#!/usr/bin/env python3
"""Step 5: BEV 실시간 미리보기
~/.factory_calibration/ 에서 remap 자동 로드
키: g=그리드  s=스냅샷  q=종료  클릭=좌표출력
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_CALIBRATION_DIR  = os.path.join(str(Path.home()), '.factory_calibration')
DEFAULT_CALIBRATION_PATH = os.path.join(DEFAULT_CALIBRATION_DIR, 'calibration.yaml')
DEFAULT_REMAP_PATH        = os.path.join(DEFAULT_CALIBRATION_DIR, 'remap.npz')


class BEVPreview:
    def __init__(self, camera_index=2, cal_path=None, remap_path=None):
        self.camera_index = camera_index
        self.cal_path   = cal_path   or DEFAULT_CALIBRATION_PATH
        self.remap_path = remap_path or DEFAULT_REMAP_PATH
        self.show_grid = True
        self.map1 = self.map2 = None
        self.bev_w = self.bev_h = 1080
        self.scale_x = self.scale_y = 0.005
        self.real_origin_x = self.real_origin_y = 0.0
        self.ppm = 200.0

    def load(self):
        if not os.path.isfile(self.cal_path) or not os.path.isfile(self.remap_path):
            print(f"[경고] 캘리브레이션 파일 없음 — 원본 표시")
            return
        with open(self.cal_path) as f:
            d = yaml.safe_load(f)
        self.bev_w = int(d['bev_width'])
        self.bev_h = int(d['bev_height'])
        self.scale_x = float(d['scale_x'])
        self.scale_y = float(d['scale_y'])
        self.ppm = float(d['pixels_per_meter'])
        self.real_origin_x = float(d['real_origin_x'])
        self.real_origin_y = float(d['real_origin_y'])
        npz = np.load(self.remap_path)
        self.map1 = npz['map1']
        self.map2 = npz['map2']
        print(f"캘리브레이션 로드 (BEV {self.bev_w}×{self.bev_h}, ppm={self.ppm})")

    def pixel_to_meter(self, px, py):
        x_m = px * self.scale_x + self.real_origin_x
        y_m = (self.bev_h - py) * self.scale_y + self.real_origin_y
        return x_m, y_m

    def draw_grid(self, img):
        overlay = img.copy()
        # 1m 간격 그리드
        step_px = int(self.ppm)
        if step_px < 5:
            return img
        h, w = img.shape[:2]
        # 원점 픽셀
        ox = int(-self.real_origin_x * self.ppm)
        oy = int(self.bev_h + self.real_origin_y * self.ppm)
        for i in range(-10, 50):
            x = ox + i * step_px
            if 0 <= x < w:
                cv2.line(overlay, (x, 0), (x, h), (80, 80, 80), 1)
                cv2.putText(overlay, f"{i}m", (x + 2, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        for j in range(-10, 50):
            y = oy - j * step_px
            if 0 <= y < h:
                cv2.line(overlay, (0, y), (w, y), (80, 80, 80), 1)
                cv2.putText(overlay, f"{j}m", (2, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        # 원점 강조
        if 0 <= ox < w and 0 <= oy < h:
            cv2.circle(overlay, (ox, oy), 6, (0, 255, 255), -1)
        return cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xm, ym = self.pixel_to_meter(x, y)
            print(f"클릭: px=({x},{y}) → m=({xm:.3f},{ym:.3f})")

    def run(self):
        self.load()
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[오류] 카메라 {self.camera_index} 열기 실패")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        cv2.namedWindow('BEV Preview')
        cv2.setMouseCallback('BEV Preview', self.mouse_cb)
        snap_idx = 0
        print("키: g=그리드  s=스냅샷  q=종료  / 클릭=좌표")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if self.map1 is not None:
                bev = cv2.remap(frame, self.map1, self.map2,
                                cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            else:
                bev = frame

            display = self.draw_grid(bev) if self.show_grid else bev.copy()
            cv2.imshow('BEV Preview', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                self.show_grid = not self.show_grid
            elif key == ord('s'):
                path = f'bev_snapshot_{snap_idx:03d}.png'
                cv2.imwrite(path, display)
                print(f"저장: {path}")
                snap_idx += 1

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=2)
    parser.add_argument('--calibration', default=None)
    parser.add_argument('--remap', default=None)
    args = parser.parse_args()
    BEVPreview(args.camera, args.calibration, args.remap).run()


if __name__ == '__main__':
    main()
