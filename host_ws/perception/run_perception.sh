#!/usr/bin/env bash
# Perception 노드 실행 — 런타임 파이프라인의 YOLO 검출 역할 (.venv_yolo python + ROS source).
#
#   단독:           bash run_perception.sh   (기본 = v2 2클래스: 0=follower,1=leader)
#   가중치 교체:     bash run_perception.sh --ros-args \
#                       -p weights:=<other.pt> -p class_map:="0:follower,1:leader"
#                    (class_map 순서는 그 가중치 data.yaml names 순서에 — Roboflow=알파벳순)
set -e
# 스크립트 위치 기준 절대경로(이식성: clone 위치/사용자 무관)
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source /opt/ros/humble/setup.bash
source "$REPO/host_ws/install/setup.bash"
export ROS_DOMAIN_ID=30
exec "$REPO/.venv_yolo/bin/python" "$HERE/perception_node.py" "$@"
