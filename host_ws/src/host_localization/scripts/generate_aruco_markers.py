#!/usr/bin/env python3
"""Step 1: ArUco 마커 생성 (DICT_4X4_50)
ID 0~5: 물체, ID 10: 벽, ID 20~25: 캘리브레이션
"""

import cv2
import numpy as np
import os
import sys


MARKER_SIZE_PX = 200
BORDER_BITS = 1
SHEET_COLS = 4

MARKER_GROUPS = {
    'calibration': [0],                # ID 0: 4모서리 공용 (4장 인쇄)
    'objects': list(range(1, 6)),      # 1~5: 물체
    'wall': [10],
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'markers')


def get_aruco_dict():
    # OpenCV 4.x / 4.13 호환
    if hasattr(cv2, 'aruco'):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    raise RuntimeError("cv2.aruco 모듈 없음. opencv-contrib-python 설치 필요")


def generate_marker(aruco_dict, marker_id, size=MARKER_SIZE_PX):
    img = np.zeros((size, size), dtype=np.uint8)
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size, img, BORDER_BITS)
    return img


def make_sheet(images, cols=SHEET_COLS, label=True, aruco_dict_ids=None):
    rows = (len(images) + cols - 1) // cols
    cell = MARKER_SIZE_PX + 60  # 마커 + 레이블 여백
    sheet = np.ones((rows * cell, cols * cell), dtype=np.uint8) * 255

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        y0, x0 = r * cell, c * cell
        sheet[y0:y0 + MARKER_SIZE_PX, x0:x0 + MARKER_SIZE_PX] = img
        if label and aruco_dict_ids:
            cv2.putText(sheet, f"ID:{aruco_dict_ids[idx]}",
                        (x0 + 5, y0 + MARKER_SIZE_PX + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    return sheet


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    aruco_dict = get_aruco_dict()

    all_ids = []
    all_imgs = []

    for group_name, ids in MARKER_GROUPS.items():
        group_imgs = []
        for mid in ids:
            img = generate_marker(aruco_dict, mid)
            path = os.path.join(OUTPUT_DIR, f'marker_{mid:02d}.png')
            cv2.imwrite(path, img)
            print(f"  저장: {path}")
            group_imgs.append(img)
            all_ids.append(mid)
            all_imgs.append(img)

        sheet = make_sheet(group_imgs, aruco_dict_ids=ids)
        sheet_path = os.path.join(OUTPUT_DIR, f'sheet_{group_name}.png')
        cv2.imwrite(sheet_path, sheet)
        print(f"  시트: {sheet_path}")

    # 전체 시트
    full_sheet = make_sheet(all_imgs, aruco_dict_ids=all_ids)
    full_path = os.path.join(OUTPUT_DIR, 'sheet_all.png')
    cv2.imwrite(full_path, full_sheet)
    print(f"\n전체 시트: {full_path}")
    print(f"총 {len(all_ids)}개 마커 생성 완료 → {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
