# scripts/ — 일회성 캘리브레이션 · 디버그 툴

런타임 노드가 **아니다**. 전부 `from host_localization...`를 import(패키지 결합)하므로
패키지 밖으로 못 옮긴다. 실행은 colcon install이 source된 상태여야 한다.
(물리적으로 calibration/debug 하위폴더로 안 나눈 이유: 각 스크립트가
`dirname(__file__)/..`로 패키지 루트를 잡아, 더 깊은 폴더로 옮기면 import가 깨짐. 위험 대비 실익 낮음.)

## 캘리브레이션 (런타임 **전** 1회, 산출물을 런타임이 소비)
| 스크립트 | 역할 | 산출물 |
|---|---|---|
| `calibrate_intrinsic.py` | 카메라 intrinsic(K,D) | npz/yaml |
| `calibrate_extrinsic.py` | extrinsic + remap | remap.npz |
| `charuco_calibration_with_plane_axis.py` | charuco intrinsic + 평면축 | charuco_intrinsics.npz |
| `ground_axes_nadir.py` | 지면 축(nadir) 추정 (charuco 헬퍼 재사용) | ground_frame_nadir.npz |
| `generate_aruco_markers.py` | ArUco 마커 PNG 생성 | markers/ |

> 산출물은 `data/calibration/` (repo) + `~/.factory_calibration/` (repo 밖). 런타임 config가 절대경로로 참조.

## 디버그 · 브링업 (인터랙티브 점검/튜닝)
| 스크립트 | 역할 |
|---|---|
| `camera_on.py` | 카메라 빠른 켜기/확인 |
| `camera_tuner.py` | 카메라 파라미터 튜너 |
| `hsv_tuner.py` | HSV 임계 튜너 |
| `test_brio_camera.py` | BRIO 카메라 점검 |
| `test_bev_preview.py` | BEV 워프 프리뷰 |
| `test_uwb_serial.py` | UWB 시리얼 raw 점검 |
| `spike_uwb_lec.py` | UWB `les`/`lec` raw range 스파이크 |
