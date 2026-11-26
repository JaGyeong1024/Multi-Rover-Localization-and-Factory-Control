#!/usr/bin/env bash
# Launch RViz2 with the factory UWB visualization layout.
set -eo pipefail

source /opt/ros/humble/setup.bash
source "$(dirname "$(readlink -f "$0")")/install/setup.bash"

RVIZ_CFG="$(ros2 pkg prefix host_localization)/share/host_localization/rviz/uwb.rviz"

exec rviz2 -d "$RVIZ_CFG" --geometry 1820x1020+10+5
