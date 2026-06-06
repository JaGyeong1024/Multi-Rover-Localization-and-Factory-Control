#!/usr/bin/env python3
"""
[1회용] Intrinsic 캘리브레이션 — 체커보드로 K, D 추출
카메라 기종이 바뀌지 않는 한 재실행 불필요.

출력: ~/.factory_calibration/intrinsic.yaml

사용법:
  python3 scripts/calibrate_intrinsic.py
  python3 scripts/calibrate_intrinsic.py --cols 9 --rows 6 --square-mm 25

키:
  s  — 체커보드 검출 성공 시 프레임 캡처
  c  — 캡처된 장수로 calibrateCamera 실행 후 저장
  r  — 캡처 초기화
  q  — 종료
"""

import argparse
import os
import sys
from pathlib import Path

_PKG_SRC = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _PKG_SRC)

from host_localization.unified_calibrator import (
    IntrinsicCalibrator, DEFAULT_INTRINSIC_PATH
)


def main():
    parser = argparse.ArgumentParser(
        description='[1회용] 체커보드 Intrinsic 캘리브레이션',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--camera',    type=int,   default=2,    help='카메라 인덱스 (기본: 2)')
    parser.add_argument('--cols',      type=int,   default=9,    help='체커보드 내부 코너 가로 (기본: 9)')
    parser.add_argument('--rows',      type=int,   default=6,    help='체커보드 내부 코너 세로 (기본: 6)')
    parser.add_argument('--square-mm', type=float, default=25.0, help='한 칸 크기 mm (기본: 25)')
    parser.add_argument('--target',    type=int,   default=25,   help='목표 캡처 수 (기본: 25)')
    parser.add_argument('--output',    default=DEFAULT_INTRINSIC_PATH,
                        help=f'출력 경로 (기본: {DEFAULT_INTRINSIC_PATH})')
    args = parser.parse_args()

    print("=" * 55)
    print("  Intrinsic 캘리브레이션 (1회용)")
    print(f"  카메라: /dev/video{args.camera}")
    print(f"  체커보드: {args.cols}×{args.rows}, 한 칸 {args.square_mm}mm")
    print(f"  목표 캡처: {args.target}장")
    print(f"  출력: {args.output}")
    print("=" * 55)
    print("  체커보드를 다양한 각도/위치로 이동하며 's' 로 캡처하세요.")
    print("  25장 이상 모이면 'c' 로 캘리브레이션 실행.")
    print()

    cal = IntrinsicCalibrator(
        camera_index=args.camera,
        board_cols=args.cols,
        board_rows=args.rows,
        square_mm=args.square_mm,
    )
    cal.run(output_path=args.output, num_target=args.target)

    if os.path.isfile(args.output):
        print(f"\n완료. 결과 파일: {args.output}")
        print("다음 단계: python3 scripts/calibrate_extrinsic.py")
    else:
        print("\n[경고] 파일이 저장되지 않았습니다. 'c' 키로 캘리브레이션을 실행했는지 확인하세요.")


if __name__ == '__main__':
    main()
