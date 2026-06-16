#!/usr/bin/env bash
# 관제 대시보드 — 단일 통합 런처 (터미널 하나로 전부).
#   • 카메라(/dev/global_cam) + UWB(/dev/ttyUWB) 둘 다 연결 → LIVE 실시간 파이프라인
#   • 하나라도 없으면                                    → test2 bag 자동 폴백
#   • GPU(YOLO perception)는 항상 사용
#   • dev 서버 뜨면 Chrome 새탭 자동 (맵에디터 컨벤션)
#   • Ctrl+C 한 번 → 백엔드/프론트/브라우저 백그라운드 전부 정리
#
#   bash run_dashboard.sh                  # 자동(LIVE 가능하면 LIVE, 아니면 bag)
#   MODE=bag  bash run_dashboard.sh        # 강제 bag
#   MODE=live bash run_dashboard.sh        # 강제 live
#   BAG=/path/to/bag bash run_dashboard.sh # bag 경로 지정
#   MISSION=on bash run_dashboard.sh       # mock 미션 데모(map_demo) — 큐/상태/cancel/하역완료
#                                          #   UI 테스트용. mock 로버가 pose 발행 →
#                                          #   위치추정/bag/perception 대신 동작(pose 충돌 회피).
#
# 선행(최초 1회): sudo apt install -y ros-humble-rosbridge-suite ros-humble-web-video-server
set -u

DASH="$HOME/capstone/dashboard"
URL="http://localhost:5173"
CAM_DEV="/dev/global_cam"
UWB_DEV="/dev/ttyUWB"
BAG="${BAG:-$HOME/capstone/data/bags/uwb_20260604_003149_leader_follower_test2}"
QOS="$HOME/capstone/host_ws/dashboard_bag_qos.yaml"
PERCEPTION="$HOME/capstone/host_ws/perception/run_perception.sh"
MISSION="${MISSION:-off}"   # on → mock 미션 데모(map_demo): mock 로버가 pose 발행, 위치추정/bag 대신

# ROS setup.bash는 미설정 변수(AMENT_TRACE_SETUP_FILES 등)를 참조 → set -u에서 죽음.
# 소싱 구간만 nounset 해제(맵에디터 컨벤션).
set +u
source /opt/ros/humble/setup.bash
source "$HOME/capstone/host_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID=30

# --- 선행 점검 ----------------------------------------------------------------
miss=0
ros2 pkg list 2>/dev/null | grep -qx rosbridge_server || { echo "[dash] ✗ rosbridge_server 미설치"; miss=1; }
ros2 pkg list 2>/dev/null | grep -qx web_video_server || { echo "[dash] ✗ web_video_server 미설치"; miss=1; }
[ "$miss" = 1 ] && { echo "  → sudo apt install -y ros-humble-rosbridge-suite ros-humble-web-video-server"; exit 1; }
[ -d "$DASH/node_modules" ] || { echo "[dash] npm install (최초 1회)..."; ( cd "$DASH" && npm install ) || exit 1; }
# turtlebot3 URDF 메시(고폴리, gitignore) — 없으면 rover 레포에서 1회 fetch.
[ -f "$DASH/public/urdf/turtlebot3_manipulation_description/urdf/turtlebot3_manipulation.urdf" ] \
  || { echo "[dash] URDF 메시 가져오기(최초 1회)..."; bash "$DASH/fetch_urdf.sh" || exit 1; }

# --- 모드 판정: 카메라+UWB 둘 다 있으면 LIVE, 아니면 bag (MODE로 강제 가능) -------
camO=$([ -e "$CAM_DEV" ] && echo O || echo X)
uwbO=$([ -e "$UWB_DEV" ] && echo O || echo X)
if [ "${MODE:-auto}" = auto ]; then
  if [ -e "$CAM_DEV" ] && [ -e "$UWB_DEV" ]; then MODE=live; else MODE=bag; fi
fi
echo "[dash] 모드=$MODE  (카메라 $camO / UWB $uwbO)"
[ "$MODE" = bag ] && echo "[dash] bag: $BAG"

# --- 이전 인스턴스 정리 (재실행 안전; 파일 스크립트라 argv 깨끗 → self-kill 없음) ----
for pat in rosbridge_websocket web_video_server localization_diagnostics_node \
           "lib/host_localization/fusion_node" "lib/host_localization/global_position_node" \
           "host_ws/perception/perception_node.py" "ros2 bag play" \
           mock_rover_node mission_coordinator_node mock_pickplace_node; do
  pkill -f "$pat" 2>/dev/null || true
done
sleep 1

PIDS=(); VITE=""
cleanup() {
  echo; echo "[dash] 종료 — 전체 정리..."
  [ -n "$VITE" ] && kill -- -"$VITE" 2>/dev/null || true   # vite = setsid 프로세스그룹
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  for pat in rosbridge_websocket web_video_server localization_diagnostics_node \
             "lib/host_localization/fusion_node" "lib/host_localization/global_position_node" \
             "perception_node.py" "ros2 bag play" \
             mock_rover_node mission_coordinator_node mock_pickplace_node; do
    pkill -f "$pat" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# --- ROS 백엔드 (모드별) ------------------------------------------------------
if [ "$MISSION" = on ]; then
  echo "[dash] MOCK 미션 데모 (map_demo: mock 로버×2 + coordinator + mock pnp)..."
  echo "       (mock 로버가 pose 발행 → 위치추정/bag/perception 미실행: pose 충돌 회피)"
  ros2 launch host_mission map_demo.launch.py & PIDS+=($!)
  sleep 2
elif [ "$MODE" = live ]; then
  echo "[dash] LIVE 파이프라인 (uwb + global_cam + rover_bridge + fusion + global_position)..."
  ros2 launch host_localization pipeline.launch.py & PIDS+=($!)
  sleep 3
else
  echo "[dash] BAG 추정 노드 (fusion + global_position, follower=raw UWB)..."
  ros2 launch host_localization bag_estimation.launch.py & PIDS+=($!)
  sleep 2
fi

# --- perception (GPU YOLO) — mock 미션 모드 아니면 (이미지 소스 필요) -----------
if [ "$MISSION" != on ]; then
  echo "[dash] perception (.venv YOLO · GPU)..."
  bash "$PERCEPTION" & PIDS+=($!)
fi

# --- 대시보드 서비스 (rosbridge :9090 / web_video :8080 / diagnostics) — 공통 --
echo "[dash] 대시보드 서비스 (rosbridge + web_video + diagnostics)..."
ros2 launch host_localization dashboard.launch.py & PIDS+=($!)
sleep 2

# --- bag 모드면 리플레이 (카메라 QoS reliable override → web_video_server 호환) --
if [ "$MODE" = bag ] && [ "$MISSION" != on ]; then
  echo "[dash] bag 리플레이(--loop)..."
  ros2 bag play --loop --qos-profile-overrides-path "$QOS" "$BAG" & PIDS+=($!)
fi

# --- 프론트 (vite dev; 자기 세션/그룹 = 종료 시 그룹 통째 정리) -----------------
echo "[dash] 프론트 dev 서버..."
setsid bash -c "cd '$DASH' && exec npm run dev -- --port 5173 --strictPort" \
  >/tmp/dash_vite.log 2>&1 & VITE=$!

# --- dev 서버 대기 후 Chrome 새탭 --------------------------------------------
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
echo " 모드: $MODE   미션: $MISSION   대시보드: $URL   (rosbridge :9090 · web_video :8080)"
[ "$MISSION" = on ] && \
echo " 큐 추가: ros2 topic pub --once /task/add std_msgs/String \"data: 'fetch:gear_01'\" (또는 대시보드 버튼)"
echo " 중지: Ctrl+C  (백엔드/프론트/브라우저 전부 정리)"
echo "──────────────────────────────────────────────"
wait
