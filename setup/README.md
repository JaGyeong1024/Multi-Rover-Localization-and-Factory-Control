# 팀원 노트북 세팅 (호스트 작업 이식)

호스트 PC의 글로벌 위치추정 파이프라인을 팀원 노트북에서 **UWB·카메라 꽂으면 돌아가게** 하는 1회 세팅.
로버 담당자가 호스트 없이도 `/{robot}/global/position`을 받아 테스트할 수 있게 하는 게 목적.

## 전제
- **Ubuntu 22.04 + ROS 2 Humble** (ROS Humble 네이티브 OS — 라이브 USB/GPU엔 네이티브가 정석, Docker 아님)
- **NVIDIA GPU** (실시간 YOLO perception용; 드라이버 CUDA 12.4↑ 권장)
- repo를 **`~/capstone`** 에 clone (일부 경로가 `~/capstone` 가정)

## clone만으로 self-contained
런타임 모델(`data/models/yolo11n-obb_2class_v3.pt`)과 카메라 캘리(`data/calibration/`: calibration.yaml·remap.npz·intrinsics)는 **git에 포함**(force-add). → **별도 데이터 전달 불필요**, `git clone` 한 번이면 됨.

## 순서 (팀원 노트북)
```bash
git clone <repo> ~/capstone && cd ~/capstone
bash setup/setup.sh            # 1회 (sudo: apt·udev)
# (그룹 추가됐으면 재로그인 1회)
# UWB 동글 + BRIO 카메라 USB 꽂기  → /dev/ttyUWB·/dev/global_cam 자동 생성
bash host_ws/run_pipeline.sh   # 파이프라인 기동 → /{robot}/global/position 발행
```

## setup.sh가 하는 일
1. udev 규칙(`udev/99-factory-devices.rules`) → 꽂으면 `/dev/ttyUWB`(UWB, ModemManager 무시)·`/dev/global_cam`(BRIO RGB) 안정 생성 + dialout/video 그룹 추가
2. 시스템 python deps(colcon 노드용: serial·numpy·opencv·scipy·yaml) + rosdep
3. `.venv_yolo` (torch 2.6.0+cu124 + ultralytics — perception)
4. **캘리 배치**: `data/calibration/{calibration.yaml,remap.npz}` → `~/.factory_calibration/`(런타임이 거기서 읽음). 모델·intrinsics는 clone에 이미 포함됨.
5. `colcon build --symlink-install`

## 트러블슈팅
- **GPU/CUDA**: 드라이버가 12.4 미만이면 `setup.sh`의 torch 줄을 `--index-url .../whl/cu121`로(또는 드라이버 업데이트). `torch.cuda.is_available()=False`면 GPU 인식 실패.
- **GPU 없음**: 실시간 perception 불가 → perception 빼고 UWB+fusion만 써도 position은 나옴(UWB 기반).
- **장치 안 뜸**: USB 재플러그(udev는 plug 시 적용). `ls -l /dev/ttyUWB /dev/global_cam` 확인. UWB가 ModemManager에 잡히면 `sudo udevadm control --reload-rules && sudo udevadm trigger`.
- **권한**: 시리얼/카메라 접근 안 되면 `dialout`·`video` 그룹 추가 후 **재로그인**.
- **카메라/마운트 교체 시**: 캘리는 같은 카메라·천장 마운트 전제(1세트 공용). 바뀌면 재캘리(`scripts/`) 후 `data/calibration/` 갱신·재커밋.

## 비-Ubuntu / 하드웨어 없는 팀원
라이브 USB+GPU+ROS는 네이티브 Ubuntu 전용. Win/Mac이나 하드웨어 없는 인원은 **뷰/분석만**(웹 path_editor·대시보드는 OS 무관, bag 분석은 Docker) — 별도 안내.
