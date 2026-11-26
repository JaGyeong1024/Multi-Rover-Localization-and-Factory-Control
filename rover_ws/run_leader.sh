#!/usr/bin/env bash
# Leader rover start/stop.
#
#   run_leader.sh start    safety chain, then base + arm hardware, depth camera, command mux and
#                          the arm adapter (runs the v6 pick and place when the leader holds at a pick)
#   LEADER_CAMERA=false run_leader.sh start   same without the front depth camera driver
#   run_leader.sh stop | status
#   PRESETS="pivot_slow xtrack_off" run_leader.sh start   drive parameter overlays
#       (space or comma separated; config/presets/leader_<name>.yaml, later wins)
#
# Deploy rover_ws/ and config/ side by side (repository layout): rover_drive's setup.py installs
# the hardware parameters from ../../../config/hardware.
#
# The safety gate is the only non-zero /cmd_vel publisher; the mux publishes /cmd_vel_mux.
# The arm motors are energised with the wheels. The arm is not moved at startup: bring it to
# the Stay pose before the first pick (the pick program waits for it).
# stop does not kill a running pick program (stopping it mid-motion freezes the arm).
# TURTLEBOT3_WS: optional underlay with the source-built externals (astra_camera); sourced
#   before rover_ws (rover_ws/README.md).
export PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB3_SETUP="${TURTLEBOT3_WS:-$HOME/turtlebot3_ws}/install/setup.bash"
[ -f "$TB3_SETUP" ] && source "$TB3_SETUP"
source "$HERE/install/setup.bash"
export ROS_DOMAIN_ID=25 TURTLEBOT3_MODEL=burger LDS_MODEL=LDS-01
L=~/logs; P=$L/leader.pids
mkdir -p "$L"

# Pick program started by the arm adapter ({marker}: shelf marker ID, {stamp}: time stamp).
ARM_CMD="exec ros2 launch omx_turtlebot3_runtime ros_policy_pick_place.launch.py \
ros_domain_id:=$ROS_DOMAIN_ID marker_id:={marker} execute:=true mission_timeout_s:=90 \
log_dir:=$HOME/omx_runs/pick_{stamp}"

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
      [ -f "$PRESET_DIR/leader_$n.yaml" ] || { echo "unknown preset '$n' (no $PRESET_DIR/leader_$n.yaml)"; exit 1; }
    done
    echo "presets: ${PRESET_LIST:-none}"
    T=$(date +%Y%m%d_%H%M%S); : > "$P"
    bg drive ros2 launch rover_drive leader.launch.py ${PRESET_LIST:+presets:=$PRESET_LIST}  # no empty presets:= (launch rejects it)
    D=$(tail -1 "$P"); sleep 5
    # the base has no command timeout: never bring it up without the safety chain
    kill -0 "$D" 2>/dev/null || { echo "drive launch died ($L/drive_$T.log): base NOT started"; "$0" stop; exit 1; }
    bg hardware ros2 launch turtlebot3_manipulation_bringup hardware.launch.py \
      start_rviz:=false start_camera:=${LEADER_CAMERA:-true} start_lidar:=false \
      move_to_stay_pose:=false start_eef_camera_driver:=false
    bg mux ros2 run leader_cmd_mux cmd_vel_mux --ros-args -r __node:=cmd_vel_mux \
      -p output_topic:=/cmd_vel_mux -p align_topic:=/cmd_vel_align -p pnp_topic:=/cmd_vel_pnp \
      -p nav_topic:=/leader/cmd_vel -p pnp_working_topic:=/pnp/working
    bg arm python3 "$HERE/arm_signal_adapter.py" --ros-args -p launch_cmd:="$ARM_CMD"
    echo "started at $T: $(tr '\n' ' ' < "$P")";;
  stop)
    for p in $(tac "$P" 2>/dev/null); do kill -INT -- -"$p" 2>/dev/null; done; sleep 4
    for p in $(cat "$P" 2>/dev/null); do kill -TERM -- -"$p" 2>/dev/null; done
    rm -f "$P"; echo stopped;;
  status)
    for p in $(cat "$P" 2>/dev/null); do ps -o pid,etime,cmd -p "$p" | tail -1 | cut -c1-120; done;;
  *) echo "usage: $0 start|stop|status"; exit 2;;
esac
