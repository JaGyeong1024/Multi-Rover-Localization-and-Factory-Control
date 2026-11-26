#!/usr/bin/env bash
# Simulator + dashboard without hardware: two simulated rovers, the mission coordinator, rosbridge
# and the web UI on an isolated ROS domain. Open http://localhost:5173 and queue a part.
#
#   bash host_ws/run_sim.sh                  # empty floor
#   bash host_ws/run_sim.sh layout:=box1     # boxes on the floor (scenarios/data/layouts.yaml)
#
# Needs host_ws and rover_ws built, rosbridge_server, and `npm install` in dashboard/.
# One Ctrl+C stops everything.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
source "$ROOT/host_ws/install/setup.bash"
source "$ROOT/rover_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-45}" ROS_LOCALHOST_ONLY=1

MCFG="$ROOT/host_ws/src/host_mission/config"
SITE="$ROOT/config/site"
python3 "$ROOT/dashboard/gen_mission_map.py" --settings "$MCFG/mission.yaml" \
  --overlay "$SITE/arena.yaml" --overlay "$ROOT/config/hardware/rovers.yaml" \
  --scenario "$SITE/mission_scenario.yaml" --goals "$SITE/goals.yaml" \
  -o "$ROOT/dashboard/public/mission_map.json" || exit 1

PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -INT -- -"$p" 2>/dev/null; done
  sleep 2
  for p in "${PIDS[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done
}
trap cleanup EXIT INT TERM

setsid ros2 launch host_mission sim_dashboard.launch.py "$@" & PIDS+=($!)
setsid ros2 launch rosbridge_server rosbridge_websocket_launch.xml > /dev/null 2>&1 & PIDS+=($!)
setsid bash -c "cd '$ROOT/dashboard' && exec npm run dev -- --port 5173 --strictPort" & PIDS+=($!)
echo "[sim] dashboard: http://localhost:5173 (Ctrl+C to stop)"
wait
