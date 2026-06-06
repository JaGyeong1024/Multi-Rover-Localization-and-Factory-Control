#!/usr/bin/env python3
"""Step 1 테스트: BRIO 카메라 연결 확인 및 FPS 측정"""

import subprocess
import cv2
import time
import sys


def list_v4l2_devices():
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'],
                                capture_output=True, text=True)
        print("=== v4l2 장치 목록 ===")
        print(result.stdout if result.stdout else "(장치 없음)")
    except FileNotFoundError:
        print("[경고] v4l2-ctl 없음. apt install v4l-utils 로 설치하세요.")


def configure_brio(cap):
    """BRIO: AF OFF, zoom 최소, MJPG 1080p 설정"""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_ZOOM, 10)


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    list_v4l2_devices()

    print(f"\n=== 카메라 {camera_index} 열기 ===")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[오류] 카메라 {camera_index} 열기 실패")
        sys.exit(1)

    configure_brio(cap)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"해상도: {actual_w}x{actual_h}")

    print("FPS 측정 중 (5초)... 'q' 키로 종료")

    frame_count = 0
    start = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[오류] 프레임 읽기 실패")
            break

        frame_count += 1
        elapsed = time.time() - start
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            start = time.time()

        cv2.putText(frame, f"FPS: {fps_display:.1f}  {actual_w}x{actual_h}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow('BRIO Camera Test', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print(f"\n최종 FPS: {fps_display:.1f} ({'OK' if fps_display >= 28 else 'NG - 28fps 미만'})")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
