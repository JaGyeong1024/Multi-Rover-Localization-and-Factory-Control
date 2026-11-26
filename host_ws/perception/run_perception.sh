#!/usr/bin/env bash
# Run the perception node (YOLO detection of the runtime pipeline; .venv_yolo python + ROS sourced).
#
#   bash run_perception.sh       # default weights (0=follower, 1=leader)
#   bash run_perception.sh --ros-args -p weights:=<other.pt> -p class_map:="0:follower,1:leader"
#       (class_map order must match the names order of the weights' data.yaml; Roboflow = alphabetical)
set -e
# absolute paths relative to this script (portable)
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source /opt/ros/humble/setup.bash
source "$REPO/host_ws/install/setup.bash"
export ROS_DOMAIN_ID=30
# publish_all_boxes: every detection per frame on /perception/<class>/bev_all, read by
# fusion_node when {robot}.bev_frames is on (pipeline.launch.py).
exec "$REPO/.venv_yolo/bin/python" "$HERE/perception_node.py" --ros-args -p publish_all_boxes:=true "$@"
