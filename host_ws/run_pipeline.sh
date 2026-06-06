#!/usr/bin/env bash
# 호스트 런타임 위치추정 파이프라인 전체 기동 (한 명령).
#   colcon(pipeline.launch: uwb+global_cam+rover_bridge+fusion+global_position)
#   + Perception(.venv YOLO: /perception/{robot}/bev_pose)
# 두 인터프리터(system / .venv_yolo)라 한 ros2 launch엔 못 섞임 → 이 스크립트가 오케스트레이션.
# Ctrl+C로 둘 다 종료.
#
#   기본:           bash run_pipeline.sh   (v3 2클래스: 0=follower,1=leader 발행)
#   가중치 교체:     bash run_pipeline.sh --ros-args -p weights:=<other.pt> -p class_map:="0:follower,1:leader"
#                    (class_map 순서는 그 가중치 data.yaml 의 names 순서에 맞출 것 — Roboflow=알파벳순)
set -e
source /opt/ros/humble/setup.bash
source ~/capstone/host_ws/install/setup.bash
export ROS_DOMAIN_ID=30

# 이전 잔존 인스턴스 정리 (중복 publisher 오염 방지 — /{robot}/pose에 헤딩/위치 섞여 튐).
pkill -f "host_localization/lib/host_localization/fusion_node" 2>/dev/null || true
pkill -f "host_ws/perception/perception_node.py" 2>/dev/null || true
sleep 1

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM

echo "[run_pipeline] colcon 파이프라인 기동..."
ros2 launch host_localization pipeline.launch.py & PIDS+=($!)
sleep 4
echo "[run_pipeline] Perception(.venv YOLO) 기동..."
bash ~/capstone/host_ws/perception/run_perception.sh "$@" & PIDS+=($!)
echo "[run_pipeline] 가동 중. Ctrl+C로 전체 종료."
wait
