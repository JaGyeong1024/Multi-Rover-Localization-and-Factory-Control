#!/usr/bin/env bash
# Control dashboard — single launcher (one terminal for everything).
#   Live pipeline: UWB + camera + fusion, GPU YOLO perception, dashboard services.
#   Needs the camera (/dev/global_cam) and UWB (/dev/ttyUWB); exits at once if one is missing.
#   Chrome opens a new tab once the dev server is up; one Ctrl+C stops everything.
#
#   bash run_dashboard.sh
#
# One-time prerequisite: sudo apt install -y ros-humble-rosbridge-suite ros-humble-web-video-server
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"   # repo root (relative to this script)
DASH="$ROOT/dashboard"
URL="http://localhost:5173"
CAM_DEV="/dev/global_cam"
UWB_DEV="/dev/ttyUWB"
PERCEPTION="$ROOT/host_ws/perception/run_perception.sh"

# setup.bash uses unset variables; relax nounset while sourcing.
set +u
source /opt/ros/humble/setup.bash
source "$ROOT/host_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID=30

# --- prerequisites ---
miss=0
ros2 pkg list 2>/dev/null | grep -qx rosbridge_server || { echo "[dash] ✗ rosbridge_server 미설치"; miss=1; }
ros2 pkg list 2>/dev/null | grep -qx web_video_server || { echo "[dash] ✗ web_video_server 미설치"; miss=1; }
[ "$miss" = 1 ] && { echo "  → sudo apt install -y ros-humble-rosbridge-suite ros-humble-web-video-server"; exit 1; }
[ -d "$DASH/node_modules" ] || { echo "[dash] npm install (최초 1회)..."; ( cd "$DASH" && npm install ) || exit 1; }
# turtlebot3 URDF meshes (high-poly, gitignored); fetched once if missing.
[ -f "$DASH/public/urdf/turtlebot3_manipulation_description/urdf/turtlebot3_manipulation.urdf" ] \
  || { echo "[dash] URDF 메시 가져오기(최초 1회)..."; bash "$DASH/fetch_urdf.sh" || exit 1; }

# --- live preflight: real sensors required ---
miss=0
[ -e "$CAM_DEV" ] || { echo "[dash] ✗ 카메라 없음: $CAM_DEV"; miss=1; }
[ -e "$UWB_DEV" ] || { echo "[dash] ✗ UWB 없음:    $UWB_DEV"; miss=1; }
[ "$miss" = 1 ] && { echo "[dash] 센서 연결 후 재실행."; exit 1; }

# --- kill previous instances (safe to re-run; a script file has a clean argv, so no self-kill) ---
for pat in rosbridge_websocket web_video_server localization_diagnostics_node \
           "lib/host_localization/fusion_node" "lib/host_localization/global_position_node" \
           "host_ws/perception/perception_node.py" \
           mission_coordinator_node; do
  pkill -f "$pat" 2>/dev/null || true
done
sleep 1

PIDS=(); VITE=""
cleanup() {
  echo; echo "[dash] 종료 — 전체 정리..."
  [ -n "$VITE" ] && kill -- -"$VITE" 2>/dev/null || true   # vite = setsid process group
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  for pat in rosbridge_websocket web_video_server localization_diagnostics_node \
             "lib/host_localization/fusion_node" "lib/host_localization/global_position_node" \
             "perception_node.py" \
             mission_coordinator_node; do
    pkill -f "$pat" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# --- ROS backend ---
echo "[dash] LIVE 파이프라인 (uwb + global_cam + rover_bridge + fusion + global_position)..."
ros2 launch host_localization pipeline.launch.py & PIDS+=($!)
sleep 3

# --- perception (GPU YOLO) ---
echo "[dash] perception (.venv YOLO · GPU)..."
bash "$PERCEPTION" & PIDS+=($!)

# --- dashboard services (rosbridge :9090 / web_video :8080 / diagnostics) ---
echo "[dash] 대시보드 서비스 (rosbridge + web_video + diagnostics)..."
ros2 launch host_localization dashboard.launch.py & PIDS+=($!)
sleep 2

# --- map data from the mission settings (same files as mission.launch.py; pass the same
# MISSION_SETTINGS / MISSION_SCENARIO / MISSION_GOALS if it used others). Goals are replaced
# at runtime by /mission/goals. Arena and rover overlays are merged over mission.yaml; with a
# custom MISSION_SETTINGS they are skipped unless MISSION_OVERLAYS="<file> <file>" is given.
MCFG="$ROOT/host_ws/src/host_mission/config"
SITE="$ROOT/config/site"
if [ -z "${MISSION_SETTINGS:-}" ]; then
  MISSION_OVERLAYS="$SITE/arena.yaml $ROOT/config/hardware/rovers.yaml"
fi
OVERLAY_ARGS=()
for f in ${MISSION_OVERLAYS:-}; do OVERLAY_ARGS+=(--overlay "$f"); done
python3 "$DASH/gen_mission_map.py" \
  --settings "${MISSION_SETTINGS:-$MCFG/mission.yaml}" "${OVERLAY_ARGS[@]}" \
  --scenario "${MISSION_SCENARIO:-$SITE/mission_scenario.yaml}" \
  --goals "${MISSION_GOALS:-$SITE/goals.yaml}" -o "$DASH/public/mission_map.json" \
  || { echo "[dash] ✗ 지도 생성 실패 (위 오류) — 임무 설정 파일 확인"; exit 1; }

# --- frontend (vite dev; own session/group, killed as a group on exit) ---
echo "[dash] 프론트 dev 서버..."
setsid bash -c "cd '$DASH' && exec npm run dev -- --port 5173 --strictPort" \
  >/tmp/dash_vite.log 2>&1 & VITE=$!

# --- wait for the dev server, then open Chrome ---
echo -n "[dash] dev 서버 대기"
ready=0
for _ in $(seq 1 40); do
  curl -s -o /dev/null "$URL" 2>/dev/null && { ready=1; break; }
  echo -n "."; sleep 0.5
done
echo
if [ "$ready" = 1 ]; then
  browser=$(command -v google-chrome || command -v google-chrome-stable \
            || command -v chromium-browser || command -v chromium || command -v xdg-open || true)
  if [ -n "$browser" ]; then echo "[dash] Chrome 새탭: $URL"; ( "$browser" "$URL" >/dev/null 2>&1 ) &
  else echo "[dash] 브라우저 못 찾음 — 수동: $URL"; fi
else
  echo "[dash] ⚠ dev 서버 미기동 (/tmp/dash_vite.log 확인). 수동: $URL"
fi

echo "──────────────────────────────────────────────"
echo " 대시보드: $URL   (rosbridge :9090 · web_video :8080)"
echo " 중지: Ctrl+C  (백엔드/프론트/브라우저 전부 정리)"
echo "──────────────────────────────────────────────"
wait
