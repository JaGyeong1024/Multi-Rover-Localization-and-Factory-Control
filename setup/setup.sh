#!/usr/bin/env bash
# 팀원 노트북 1회 세팅 — 전제: Ubuntu 22.04 + ROS Humble + NVIDIA GPU.
#   사용: git clone ... ~/capstone && cd ~/capstone && bash setup/setup.sh
#   (모델·캘리는 git에 포함 → 별도 데이터 전달 불필요. clone만으로 self-contained)
#   끝나면: UWB·BRIO 카메라 USB 꽂고 → bash host_ws/run_pipeline.sh
# 멱등(여러 번 실행 안전). sudo 필요(apt·udev).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

echo "── 0) 전제 점검 ───────────────────────────────"
. /etc/os-release 2>/dev/null || true
[ "${VERSION_ID:-}" = "22.04" ] || echo "[warn] Ubuntu 22.04 아님(${VERSION_ID:-?}) — ROS Humble 호환 확인 필요"
[ -f /opt/ros/humble/setup.bash ] || { echo "[err] ROS Humble 없음(/opt/ros/humble). 먼저 설치."; exit 1; }
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' || echo "?")
  echo "[ok] GPU 드라이버 CUDA=$CUDA (torch cu124엔 12.4↑ 권장; 낮으면 README의 cu121 참고)"
else
  echo "[warn] nvidia-smi 없음 — 실시간 perception엔 NVIDIA GPU 필요(없으면 perception 빼고 UWB+fusion만)"
fi

echo "── 1) udev 규칙(UWB·카메라 안정 심링크) ─────────"
sudo cp "$HERE/udev/99-factory-devices.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout,video "$USER" || true   # 시리얼·카메라 접근(재로그인 필요할 수 있음)
echo "  → 재플러그 시 /dev/ttyUWB·/dev/global_cam 생성"

echo "── 2) 시스템 python deps(colcon 노드용) ─────────"
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv python3-serial python3-numpy python3-opencv python3-scipy python3-yaml

echo "── 3) ROS deps(rosdep) ─────────────────────────"
sudo rosdep init 2>/dev/null || true
rosdep update
rosdep install --from-paths host_ws/src --ignore-src -y

echo "── 4) .venv_yolo(perception, CUDA torch) ────────"
[ -d .venv_yolo ] || python3 -m venv .venv_yolo
.venv_yolo/bin/pip install --upgrade pip -q
.venv_yolo/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv_yolo/bin/pip install -r "$HERE/requirements_yolo.txt"
.venv_yolo/bin/python -c "import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available())"

echo "── 4b) 대시보드 deps(rosbridge·web_video·Node18) ─"
sudo apt-get install -y ros-humble-rosbridge-suite ros-humble-web-video-server
# vite 프론트는 Node 18+ 필요(22.04 기본 apt node는 구버전) → NodeSource로 설치.
NODEMAJ="$(node -v 2>/dev/null | sed 's/v//;s/\..*//')"
if ! command -v node >/dev/null 2>&1 || [ "${NODEMAJ:-0}" -lt 18 ]; then
  echo "  Node 18+ 설치(NodeSource)..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
command -v node >/dev/null 2>&1 && echo "  node $(node -v) / npm $(npm -v 2>/dev/null)"

echo "── 5) 캘리 배치(런타임은 ~/.factory_calibration 읽음) ──"
# 모델(data/models/*.pt)·intrinsics(data/calibration/*.npz)는 git에 포함 → clone에 이미 있음.
# calibration.yaml·remap.npz만 런타임이 ~/.factory_calibration서 읽으므로 거기로 복사.
mkdir -p "$HOME/.factory_calibration"
if [ -f data/calibration/calibration.yaml ] && [ -f data/calibration/remap.npz ]; then
  cp -n data/calibration/calibration.yaml data/calibration/remap.npz "$HOME/.factory_calibration/"
  echo "  → ~/.factory_calibration 배치 완료"
else
  echo "  [warn] data/calibration/{calibration.yaml,remap.npz} 없음 — git에 포함됐는지 확인(perception/BEV 안 됨)"
fi

echo "── 6) colcon 빌드 ──────────────────────────────"
source /opt/ros/humble/setup.bash
( cd host_ws && colcon build --packages-select host_localization --symlink-install )

echo
echo "✅ 세팅 완료. (그룹 추가했으면 한 번 재로그인 권장)"
echo "   UWB·BRIO 카메라 USB 꽂고 → bash host_ws/run_pipeline.sh"
