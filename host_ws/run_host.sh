#!/usr/bin/env bash
# run_host.sh — master launcher for the host (one terminal for everything).
#   Dashboard + localization: run_dashboard.sh (UWB + camera + fusion + perception).
#   Mission: mission.launch.py (path commander + domain bridge 30 <-> 25 / 73).
#   Real sensors only; exits at once if camera or UWB is missing.
#
#   bash run_host.sh                 # dashboard + localization + mission + obstacle map
#   MAPPER=false bash run_host.sh    # no obstacle map (boxes are NOT avoided; not for demos)
#   MISSION=off bash run_host.sh     # dashboard + localization only
#
#   One Ctrl+C stops everything.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"   # repo root (relative to this script)
CAM_DEV="/dev/global_cam"
UWB_DEV="/dev/ttyUWB"
MISSION="${MISSION:-on}"         # on | off
case "$MISSION" in
  on|off) ;;
  *) echo "[host] ✗ MISSION=$MISSION — on | off 중 하나"; exit 2 ;;
esac
MAPPER="${MAPPER:-true}"         # obstacle map; without it the planner does not know the boxes
case "$MAPPER" in
  true|false) ;;
  *) echo "[host] ✗ MAPPER=$MAPPER — true | false 중 하나"; exit 2 ;;
esac

# setup.bash uses unset variables; relax nounset while sourcing.
set +u
source /opt/ros/humble/setup.bash
source "$ROOT/host_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID=30

# --- LIVE preflight: real sensors required ---
miss=0
[ -e "$CAM_DEV" ] || { echo "[host] ✗ 카메라 없음: $CAM_DEV"; miss=1; }
[ -e "$UWB_DEV" ] || { echo "[host] ✗ UWB 없음:    $UWB_DEV"; miss=1; }
if [ "$miss" = 1 ]; then
  echo "[host] 센서 연결 후 재실행."
  exit 1
fi
if [ "$MISSION" != off ]; then
  ros2 pkg list 2>/dev/null | grep -qx host_mission || {
    echo "[host] ✗ host_mission 미빌드 — colcon build --packages-select host_mission --symlink-install"; exit 1; }
fi

# --- cleanup ---
# Background children start with SIGINT ignored and cannot re-trap it, so they get SIGTERM.
PIDS=()
_cleaned=0
cleanup() {
  [ "$_cleaned" = 1 ] && return 0; _cleaned=1
  echo; echo "[host] 종료 — 전체 정리..."
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
  # bounded grace (~5 s) so child traps can stop vite/perception
  for _ in $(seq 1 20); do
    a=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && a=1; done
    [ "$a" = 0 ] && break
    sleep 0.25
  done
  # last resort: SIGKILL, plus leftovers started with setsid
  for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null || true; done
  for pat in "node_modules/.bin/vite" "perception_node.py" \
             mission_coordinator_node obstacle_mapper_node "domain_bridge"; do
    pkill -f "$pat" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

# --- dashboard + localization (MISSION is not passed on) ---
echo "[host] 대시보드 + 위치추정 (LIVE)..."
env -u MISSION bash "$ROOT/host_ws/run_dashboard.sh" & PIDS+=($!)

# --- mission stack, after localization is up ---
if [ "$MISSION" = on ]; then
  sleep 6
  echo "[host] 미션 경로 지휘기 + 도메인 브릿지 (obstacle_mapper=$MAPPER)..."
  ros2 launch host_mission mission.launch.py obstacle_mapper:="$MAPPER" & PIDS+=($!)
else
  echo "[host] MISSION=off — 대시보드 + 위치추정만."
fi

echo "──────────────────────────────────────────────"
echo " 호스트 통합 가동 (LIVE)     대시보드: http://localhost:5173"
case "$MISSION" in
  on)   echo " 미션: on (경로 지휘)        박스 지도: $MAPPER"
        echo " 큐 추가: ros2 topic pub --once /task/add std_msgs/String \"data: 'fetch:gear_01'\" (bolt_01, 또는 대시보드 버튼)"
        echo " 이벤트: ros2 topic echo /mission/events" ;;
  off)  echo " 미션: off" ;;
esac
echo " 중지: Ctrl+C  (전부 정리)"
echo "──────────────────────────────────────────────"
wait
