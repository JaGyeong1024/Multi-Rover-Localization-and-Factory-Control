#!/usr/bin/env bash
# Full-lap layout scenarios (map layout x box layout x part), in-process, one at a time.
# Usage: host_ws/run_scenarios.sh [pytest args]   e.g. -k "box1 and bolt"
set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
cd "$WS/src/host_mission"
nice -n 19 python3 -m pytest scenarios -q -p no:cacheprovider "$@"
