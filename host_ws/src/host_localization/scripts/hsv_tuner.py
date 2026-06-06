#!/usr/bin/env python3
"""Step 5: HSV 범위 실시간 튜너 (트랙바)
BEV 이미지에서 HSV 범위를 조정하여 최적값을 찾음
키: p=범위출력  q=종료
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_CALIBRATION_DIR  = os.path.join(str(Path.home()), '.factory_calibration')
DEFAULT_CALIBRATION_PATH = os.path.join(DEFAULT_CALIBRATION_DIR, 'calibration.yaml')
DEFAULT_REMAP_PATH        = os.path.join(DEFAULT_CALIBRATION_DIR, 'remap.npz')

WIN = 'HSV Tuner'
WIN_MASK = 'Mask'
WIN_RESULT = 'Masked Result'


def load_remap(cal_path, remap_path):
    if not os.path.isfile(cal_path) or not os.path.isfile(remap_path):
        return None, None
    npz = np.load(remap_path)
    return npz['map1'], npz['map2']


def nothing(_):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=2)
    parser.add_argument('--image', default=None, help='정적 이미지 경로 (카메라 대신)')
    parser.add_argument('--calibration', default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument('--remap', default=DEFAULT_REMAP_PATH)
    args = parser.parse_args()

    map1, map2 = load_remap(args.calibration, args.remap)
    static_frame = None

    if args.image:
        static_frame = cv2.imread(args.image)
        if static_frame is None:
            print(f"[오류] 이미지 로드 실패: {args.image}")
            return
        cap = None
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[오류] 카메라 {args.camera} 열기 실패")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    cv2.namedWindow(WIN)
    cv2.namedWindow(WIN_MASK)
    cv2.namedWindow(WIN_RESULT)

    # 트랙바 생성
    for name, val in [('H_low', 35), ('S_low', 100), ('V_low', 100),
                       ('H_high', 85), ('S_high', 255), ('V_high', 255)]:
        cv2.createTrackbar(name, WIN, val, 180 if 'H' in name else 255, nothing)

    print("키: p=범위 출력  q=종료")

    while True:
        if static_frame is not None:
            frame = static_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                continue

        if map1 is not None:
            bev = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        else:
            bev = frame

        h_low  = cv2.getTrackbarPos('H_low',  WIN)
        s_low  = cv2.getTrackbarPos('S_low',  WIN)
        v_low  = cv2.getTrackbarPos('V_low',  WIN)
        h_high = cv2.getTrackbarPos('H_high', WIN)
        s_high = cv2.getTrackbarPos('S_high', WIN)
        v_high = cv2.getTrackbarPos('V_high', WIN)

        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([h_low, s_low, v_low]),
                           np.array([h_high, s_high, v_high]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        result = cv2.bitwise_and(bev, bev, mask=mask)

        # 윈도우 크기 조정 (디스플레이용)
        disp_w = 800
        scale = disp_w / bev.shape[1]
        disp_h = int(bev.shape[0] * scale)
        bev_small    = cv2.resize(bev, (disp_w, disp_h))
        mask_small   = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (disp_w, disp_h))
        result_small = cv2.resize(result, (disp_w, disp_h))

        cv2.imshow(WIN, bev_small)
        cv2.imshow(WIN_MASK, mask_small)
        cv2.imshow(WIN_RESULT, result_small)

        key = cv2.waitKey(1 if static_frame is None else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"\n# HSV 범위 (ceiling_tracker_params.yaml에 복사):")
            print(f"  color_xxx_hsv_low:  [{h_low}, {s_low}, {v_low}]")
            print(f"  color_xxx_hsv_high: [{h_high}, {s_high}, {v_high}]")

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
