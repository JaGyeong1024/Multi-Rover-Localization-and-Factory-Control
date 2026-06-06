#!/usr/bin/env python3
"""
[현장 다회용] Extrinsic 캘리브레이션 — AprilTag 4모서리로 BEV remap 생성
카메라를 옮기거나 각도가 바뀌면 재실행 (Intrinsic 재실행 불필요).

전제:
  - ~/.factory_calibration/intrinsic.yaml 존재 (calibrate_intrinsic.py 실행 후)
  - AprilTag tag36h11 ID 1~4를 활동영역 4모서리에 배치 (코너 정렬)
  - config/calibration_points.yaml 에 태그 중심 좌표 입력 완료

출력:
  ~/.factory_calibration/calibration.yaml
  ~/.factory_calibration/remap.npz

동작 (헤드리스 자동 — 화면/키 입력 없음):
  카메라를 열어 AprilTag(36h11) ID 1~4를 프레임마다 누적 검출하고, 4개가 다
  모이면 H+remap을 계산·검증·저장한다. timeout 안에 4개를 못 모으면 어떤 ID가
  빠졌는지 경고하고 미저장으로 종료(exit 1). 카메라·태그가 고정이라 한 프레임에
  4개가 동시에 안 잡혀도 됨(ID별 누적).

사용법:
  python3 scripts/calibrate_extrinsic.py
  python3 scripts/calibrate_extrinsic.py --timeout 30 --min-samples 20
"""

import argparse
import os
import sys
from pathlib import Path

_PKG_SRC = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _PKG_SRC)

from host_localization.unified_calibrator import (
    BEVCalibrator,
    DEFAULT_INTRINSIC_PATH,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_REMAP_PATH,
)

_PKG_DIR = os.path.dirname(os.path.abspath(_PKG_SRC))
DEFAULT_CAL_POINTS = os.path.join(
    os.path.dirname(__file__), '..', 'config', 'calibration_points.yaml'
)


def _resolve_intrinsic(path):
    """Use `path` if present, else fall back to the charuco intrinsic npz in
    standard spots so extrinsic runs without a separate intrinsic.yaml.
    _load_intrinsic reads either .npz(K,D) or .yaml.
    (restructure로 calibration/ → data/calibration/ 이동 — 새 위치 우선, 구 위치 폴백.)"""
    root = os.path.join(_PKG_DIR, '..', '..')   # ~/capstone
    candidates = (
        path,
        os.path.expanduser('~/charuco_intrinsics.npz'),
        os.path.normpath(os.path.join(root, 'data', 'calibration', 'charuco_intrinsics.npz')),
        os.path.normpath(os.path.join(root, 'calibration', 'charuco_intrinsics.npz')),
    )
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return path


def main():
    parser = argparse.ArgumentParser(
        description='[현장 다회용] AprilTag Extrinsic 캘리브레이션 (BEV remap 생성)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--camera',     type=int,   default=2,
                        help='카메라 인덱스 (기본: 2)')
    parser.add_argument('--ppm',        type=int,   default=200,
                        help='pixels per meter — BEV 해상도 (기본: 200)')
    parser.add_argument('--intrinsic',  default=DEFAULT_INTRINSIC_PATH,
                        help=f'intrinsic.yaml 경로 (기본: {DEFAULT_INTRINSIC_PATH})')
    parser.add_argument('--cal-points', default=DEFAULT_CAL_POINTS,
                        help='calibration_points.yaml 경로')
    parser.add_argument('--output',     default=DEFAULT_CALIBRATION_PATH,
                        help=f'calibration.yaml 출력 경로')
    parser.add_argument('--remap-out',  default=DEFAULT_REMAP_PATH,
                        help=f'remap.npz 출력 경로')
    parser.add_argument('--timeout',     type=float, default=20.0,
                        help='4개 ID 누적 대기 시간(초) (기본: 20)')
    parser.add_argument('--min-samples', type=int,   default=15,
                        help='태그당 최소 누적 샘플 수 (기본: 15)')
    args = parser.parse_args()

    # intrinsic 존재 확인 (.yaml 없으면 charuco npz로 폴백)
    args.intrinsic = _resolve_intrinsic(args.intrinsic)
    if not os.path.isfile(args.intrinsic):
        print(f"[오류] intrinsic 없음: {args.intrinsic}")
        print("  charuco npz(K,D) 또는 calibrate_intrinsic.py의 intrinsic.yaml 필요.")
        sys.exit(1)

    # calibration_points.yaml 존재 확인
    cal_pts = os.path.normpath(args.cal_points)
    if not os.path.isfile(cal_pts):
        print(f"[오류] calibration_points.yaml 없음: {cal_pts}")
        sys.exit(1)

    print("=" * 55)
    print("  Extrinsic 캘리브레이션 (현장 다회용)")
    print(f"  카메라: /dev/video{args.camera}")
    print(f"  Intrinsic: {args.intrinsic}")
    print(f"  마커 좌표: {cal_pts}")
    print(f"  PPM: {args.ppm}  (BEV {args.ppm*4}×{args.ppm*4}px 예상)")
    print(f"  출력: {args.output}")
    print(f"         {args.remap_out}")
    print("=" * 55)
    print("  AprilTag(36h11) ID 1~4가 활동영역 4모서리에 보이게 두면 자동 진행.")
    print(f"  {args.timeout:.0f}s 안에 4개 ID 누적 시 자동 저장, 못 모으면 경고 후 종료.")
    print()

    cal = BEVCalibrator(
        camera_index=args.camera,
        intrinsic_path=args.intrinsic,
        cal_points_path=cal_pts,
        pixels_per_meter=args.ppm,
    )
    ok = cal.run(output_path=args.output, remap_output_path=args.remap_out,
                 timeout=args.timeout, min_samples=args.min_samples)

    if ok:
        print(f"\n완료.")
        print(f"  {args.output}")
        print(f"  {args.remap_out}")
        print("\n다음: ros2 launch host_localization global_cam_runtime.launch.py")
    else:
        print("\n[경고] 저장되지 않음 (위 [실패] 메시지 참조).")

    # pupil_apriltags' native detector segfaults during interpreter teardown
    # (known upstream issue). Work is already done + saved above, so exit hard
    # to skip the crashing C cleanup and the scary core-dump message.
    # flush 먼저 — os._exit은 stdout 버퍼를 비우지 않아, 출력을 파이프/리다이렉트하면
    # 위 로그(특히 [실패] 메시지)가 통째로 사라진다.
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    main()
