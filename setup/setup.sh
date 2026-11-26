#!/usr/bin/env bash
# One-time host setup. Requires Ubuntu 22.04, ROS Humble and an NVIDIA GPU.
#   Usage: git clone ... && cd <repo> && bash setup/setup.sh
#   Models and calibration are in git, so a clone is self-contained.
#   Afterwards plug in the UWB and camera USB devices, then: bash host_ws/run_host.sh
# Idempotent. Needs sudo (apt, udev).
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
sudo cp "$REPO/config/hardware/udev/99-factory-devices.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout,video "$USER" || true   # serial/camera access (may need re-login)
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
# The vite frontend needs Node 18+ (the 22.04 apt node is older); install via NodeSource.
NODEMAJ="$(node -v 2>/dev/null | sed 's/v//;s/\..*//')"
if ! command -v node >/dev/null 2>&1 || [ "${NODEMAJ:-0}" -lt 18 ]; then
  echo "  Node 18+ 설치(NodeSource)..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
command -v node >/dev/null 2>&1 && echo "  node $(node -v) / npm $(npm -v 2>/dev/null)"

echo "── 5) 캘리 배치(런타임은 ~/.factory_calibration 읽음) ──"
# Models and calibration files are in git. The runtime reads calibration.yaml, remap.npz and
# charuco_intrinsics.npz from ~/.factory_calibration, so copy them there.
mkdir -p "$HOME/.factory_calibration"
if [ -f data/calibration/calibration.yaml ] && [ -f data/calibration/remap.npz ]; then
  cp -n data/calibration/calibration.yaml data/calibration/remap.npz data/calibration/charuco_intrinsics.npz "$HOME/.factory_calibration/"
  echo "  → ~/.factory_calibration 배치 완료"
else
  echo "  [warn] data/calibration/{calibration.yaml,remap.npz} 없음 — git에 포함됐는지 확인(perception/BEV 안 됨)"
fi

echo "── 6) colcon 빌드 ──────────────────────────────"
source /opt/ros/humble/setup.bash
( cd host_ws && colcon build --symlink-install )

echo
echo "✅ 세팅 완료. (그룹 추가했으면 한 번 재로그인 권장)"
echo "   UWB·BRIO 카메라 USB 꽂고 → bash host_ws/run_host.sh"
