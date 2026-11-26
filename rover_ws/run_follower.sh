#!/usr/bin/env bash
# Follower rover start/stop.
#
#   run_follower.sh start     safety chain, then (after 5 s) base bringup, state publisher, LiDAR
#   run_follower.sh stop | status
#   PRESETS="pivot_slow xtrack_off" run_follower.sh start   drive parameter overlays
#       (space or comma separated; config/presets/follower_<name>.yaml, later wins)
#
# Deploy rover_ws/ and config/ side by side (repository layout): rover_drive's setup.py installs
# the hardware parameters from ../../../config/hardware.
#
# The base has no command timeout, so the safety chain (tracker, gate = the only /cmd_vel
# publisher, watchdog) starts first. Only the hardware driver and robot_state_publisher are
# started for the base (not robot.launch.py, which adds a second /scan publisher).
# TURTLEBOT3_WS: workspace with the TurtleBot3 packages (default ~/Turtlebot3_Follower).
# LIDAR_PARAMS: YDLIDAR G6 driver parameters (default config/hardware/ydlidar_g6_follower.yaml, an example).
source /opt/ros/humble/setup.bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB3_WS="${TURTLEBOT3_WS:-$HOME/Turtlebot3_Follower}"
LIDAR_PARAMS="${LIDAR_PARAMS:-$HERE/../config/hardware/ydlidar_g6_follower.yaml}"
source "$TB3_WS/install/setup.bash"
source "$HERE/install/setup.bash"
export ROS_DOMAIN_ID=73 TURTLEBOT3_MODEL=waffle
L=~/logs; P=$L/follower.pids
mkdir -p "$L"

bg() {  # bg <log name> <command...>: own process group, pid recorded
  local log="$L/$1_$T.log"; shift
  setsid nohup "$@" > "$log" 2>&1 < /dev/null &
  echo $! >> "$P"
}


case "$1" in
  start)
    [ -s "$P" ] && { echo "already started (pids: $(tr '\n' ' ' < "$P")); run stop first"; exit 1; }
    PRESET_LIST=$(echo "${PRESETS:-}" | tr -s ' ,' ',,' | sed 's/^,//; s/,$//')
    PRESET_DIR=$(ros2 pkg prefix rover_drive)/share/rover_drive/config/presets
    for n in ${PRESET_LIST//,/ }; do
      [ -f "$PRESET_DIR/follower_$n.yaml" ] || { echo "unknown preset '$n' (no $PRESET_DIR/follower_$n.yaml)"; exit 1; }
    done
    echo "presets: ${PRESET_LIST:-none}"
    T=$(date +%Y%m%d_%H%M%S); : > "$P"
    bg drive ros2 launch rover_drive follower.launch.py ${PRESET_LIST:+presets:=$PRESET_LIST}  # no empty presets:= (launch rejects it)
    D=$(tail -1 "$P"); sleep 5
    # the base has no command timeout: never bring it up without the safety chain
    kill -0 "$D" 2>/dev/null || { echo "drive launch died ($L/drive_$T.log): base NOT started"; "$0" stop; exit 1; }
    # robot_state_publisher: waffle URDF (namespace empty) passed through a params file
    URDF=$(ros2 pkg prefix turtlebot3_description)/share/turtlebot3_description/urdf/turtlebot3_waffle.urdf
    RSP=$L/rsp_params_$T.yaml
    xacro "$URDF" namespace:= | python3 -c 'import json,sys; print("robot_state_publisher:\n  ros__parameters:\n    robot_description: "+json.dumps(sys.stdin.read()))' > "$RSP"
    bg rsp ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file "$RSP"
    bg base ros2 run turtlebot3_node turtlebot3_ros --ros-args \
      --params-file "$TB3_WS/install/turtlebot3_bringup/share/turtlebot3_bringup/param/humble/waffle_pi.yaml" \
      -p 'namespace:=""' -- -i /dev/ttyACM0
    bg lidar ros2 run ydlidar_ros2_driver ydlidar_ros2_driver_node --ros-args \
      --params-file "$LIDAR_PARAMS"
    echo "started $1 at $T: $(tr '\n' ' ' < "$P")";;
  stop)
    for p in $(tac "$P" 2>/dev/null); do kill -INT -- -"$p" 2>/dev/null; done; sleep 4
    for p in $(cat "$P" 2>/dev/null); do kill -TERM -- -"$p" 2>/dev/null; done
    rm -f "$P"; echo stopped;;
  status)
    for p in $(cat "$P" 2>/dev/null); do ps -o pid,etime,cmd -p "$p" | tail -1 | cut -c1-120; done;;
  *) echo "usage: $0 start|stop|status"; exit 2;;
esac
