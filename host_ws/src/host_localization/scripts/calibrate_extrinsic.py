#!/usr/bin/env python3
"""
Extrinsic calibration: builds the BEV remap from four AprilTags at the corners of the
arena. Re-run whenever the camera is moved (intrinsics need not be redone).

Needs ~/.factory_calibration/intrinsic.yaml, tag36h11 IDs 1-4 and their centres in
config/site/calibration_points.yaml. Writes calibration.yaml and remap.npz there.
Headless: tags accumulate per ID; on timeout it names the missing IDs and exits 1.

Usage:
  python3 scripts/calibrate_extrinsic.py
  python3 scripts/calibrate_extrinsic.py --timeout 30 --min-samples 20
"""

import argparse
import os
import sys

_PKG_SRC = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _PKG_SRC)

from host_localization.unified_calibrator import (
    BEVCalibrator,
    DEFAULT_INTRINSIC_PATH,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_REMAP_PATH,
)

_PKG_DIR = os.path.dirname(os.path.abspath(_PKG_SRC))
DEFAULT_CAL_POINTS = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', 'config', 'site', 'calibration_points.yaml'
))


def _resolve_intrinsic(path):
    """Use `path` if present, else fall back to the charuco intrinsic npz in
    standard spots so extrinsic runs without a separate intrinsic.yaml.
    _load_intrinsic reads either .npz(K,D) or .yaml. data/calibration/ is preferred
    over calibration/."""
    root = os.path.join(_PKG_DIR, '..', '..')   # repo root
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
    parser.add_argument('--margin',     type=float, default=0.3,
                        help='태그 중심 사각형 바깥으로 남길 바닥 폭 m (기본: 0.3). '
                             '바꾸면 BEV 크기·원점이 바뀐다(ROI는 미터 기준이라 유지)')
    parser.add_argument('--intrinsic',  default=DEFAULT_INTRINSIC_PATH,
                        help=f'intrinsic.yaml 경로 (기본: {DEFAULT_INTRINSIC_PATH})')
    parser.add_argument('--cal-points', default=DEFAULT_CAL_POINTS,
                        help='calibration_points.yaml 경로')
    parser.add_argument('--output',     default=DEFAULT_CALIBRATION_PATH,
                        help='calibration.yaml 출력 경로')
    parser.add_argument('--remap-out',  default=DEFAULT_REMAP_PATH,
                        help='remap.npz 출력 경로')
    parser.add_argument('--timeout',     type=float, default=20.0,
                        help='4개 ID 누적 대기 시간(초) (기본: 20)')
    parser.add_argument('--min-samples', type=int,   default=15,
                        help='태그당 최소 누적 샘플 수 (기본: 15)')
    args = parser.parse_args()

    # falls back to the charuco npz when intrinsic.yaml is missing
    args.intrinsic = _resolve_intrinsic(args.intrinsic)
    if not os.path.isfile(args.intrinsic):
        print(f"[오류] intrinsic 없음: {args.intrinsic}")
        print("  charuco npz(K,D) 또는 calibrate_intrinsic.py의 intrinsic.yaml 필요.")
        sys.exit(1)

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
        margin=args.margin,
    )
    ok = cal.run(output_path=args.output, remap_output_path=args.remap_out,
                 timeout=args.timeout, min_samples=args.min_samples)

    if ok:
        print("\n완료.")
        print(f"  {args.output}")
        print(f"  {args.remap_out}")
        print("\n다음: ros2 launch host_localization global_cam_runtime.launch.py")
    else:
        print("\n[경고] 저장되지 않음 (위 [실패] 메시지 참조).")

    # pupil_apriltags' native detector segfaults during interpreter teardown
    # (known upstream issue). Work is already done + saved above, so exit hard
    # to skip the crashing C cleanup and the scary core-dump message.
    # flush first: os._exit skips buffered stdout, which would drop the log when piped.
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    main()
