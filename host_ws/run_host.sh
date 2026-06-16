#!/usr/bin/env bash
# run_host.sh — 호스트 통합 마스터 런처 (터미널 하나로 전부).
#   대시보드 + 위치추정 : run_dashboard.sh 를 LIVE 강제로 재사용 (로직 복붙 X)
#   미션               : mission_runtime.launch.py (coordinator + 도메인 브릿지, mock 없음)
#
#   ★ bag 폴백 없음 — 실센서(카메라+UWB) 전용. 둘 중 하나라도 없으면 즉시 중단.
#     (bag으로 보고 싶으면 run_dashboard.sh 를 직접 쓸 것.)
#
#   bash run_host.sh                 # 대시보드 + 위치추정 + 미션 (실데모공간 맵)
#   MISSION=off bash run_host.sh     # 미션 빼고 대시보드 + 위치추정만
#   GRAPH=lane_graph.json SCENARIO=mission_scenario.yaml bash run_host.sh   # 그래프 교체
#
#   Ctrl+C 한 번 → 대시보드/위치추정/미션 전부 정리.
set -u

ROOT="$HOME/capstone"
CAM_DEV="/dev/global_cam"
UWB_DEV="/dev/ttyUWB"
MISSION="${MISSION:-on}"
GRAPH="${GRAPH:-lane_graph_map.json}"
SCENARIO="${SCENARIO:-mission_scenario_map.yaml}"

# ROS setup.bash는 미설정 변수를 참조 → set -u에서 죽음. 소싱 구간만 nounset 해제.
set +u
source /opt/ros/humble/setup.bash
source "$ROOT/host_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID=30

# --- LIVE 선행점검 (bag 폴백 없음 → 실센서 필수, fail-fast) --------------------
miss=0
[ -e "$CAM_DEV" ] || { echo "[host] ✗ 카메라 없음: $CAM_DEV"; miss=1; }
[ -e "$UWB_DEV" ] || { echo "[host] ✗ UWB 없음:    $UWB_DEV"; miss=1; }
if [ "$miss" = 1 ]; then
  echo "[host] 실데이터 전용 런처 — 센서 연결 후 재실행."
  echo "       (bag으로 보려면: bash run_dashboard.sh)"
  exit 1
fi
if [ "$MISSION" = on ]; then
  ros2 pkg list 2>/dev/null | grep -qx host_mission || {
    echo "[host] ✗ host_mission 미빌드 — colcon build --packages-select host_mission --symlink-install"; exit 1; }
fi

# --- 정리 ---------------------------------------------------------------------
# ⚠ 자식은 &(비동기)로 띄운다 → bash가 비대화형 셸에서 비동기 자식의 SIGINT를 진입 시
#   SIG_IGN으로 박고, *진입 시 무시된 시그널은 trap으로 다시 못 잡는다*. 그래서 SIGINT(Ctrl+C)
#   로는 run_dashboard.sh의 trap이 영영 안 떠 setsid된 vite가 살아남고 wait가 멈춘다(이 버그).
#   → 자식에는 반드시 **SIGTERM**을 보낸다(TERM은 무시 대상 아님 → run_dashboard trap 동작).
PIDS=()
_cleaned=0
cleanup() {
  [ "$_cleaned" = 1 ] && return 0; _cleaned=1
  echo; echo "[host] 종료 — 전체 정리..."
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
  # 바운디드 grace(최대 ~5s) — 무한 wait 금지(멈춤 방지). 자식 trap이 vite/perception 내릴 시간.
  for _ in $(seq 1 20); do
    a=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && a=1; done
    [ "$a" = 0 ] && break
    sleep 0.25
  done
  # 최후 방어: 응답 없으면 SIGKILL + setsid 잔존(터미널/그룹 시그널이 안 닿는 것) 패턴 정리.
  for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null || true; done
  for pat in "node_modules/.bin/vite" "perception_node.py" \
             mission_coordinator_node "domain_bridge"; do
    pkill -f "$pat" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

# --- 대시보드 + 위치추정 (run_dashboard.sh, LIVE 강제) ------------------------
echo "[host] 대시보드 + 위치추정 (LIVE)..."
MODE=live bash "$ROOT/host_ws/run_dashboard.sh" & PIDS+=($!)

# --- 미션 스택 (coordinator + 도메인 브릿지) — 위치추정/diagnostics 뜬 뒤 -------
if [ "$MISSION" = on ]; then
  sleep 6
  echo "[host] 미션 coordinator + 도메인 브릿지 (graph=$GRAPH)..."
  ros2 launch host_mission mission_runtime.launch.py \
      graph_file:="$GRAPH" scenario_file:="$SCENARIO" & PIDS+=($!)
else
  echo "[host] MISSION=off — 대시보드 + 위치추정만."
fi

echo "──────────────────────────────────────────────"
echo " 호스트 통합 가동 (LIVE)     대시보드: http://localhost:5173"
echo " 미션: $MISSION              graph: $GRAPH"
[ "$MISSION" = on ] && \
echo " 큐 추가: ros2 topic pub --once /task/add std_msgs/String \"data: 'fetch:gear_01'\" (또는 대시보드 버튼)"
echo " 중지: Ctrl+C  (전부 정리)"
echo "──────────────────────────────────────────────"
wait
