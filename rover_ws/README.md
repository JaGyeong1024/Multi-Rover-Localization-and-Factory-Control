# rover_ws

Code that runs on the rovers. Copy `rover_ws/` and `config/` side by side onto a rover,
build with `colcon build --symlink-install` (with `host_ws` built and sourced first, for
`host_mission_interfaces`), then use `run_leader.sh start` or `run_follower.sh start`.

| Package | Rover | Role |
|---|---|---|
| `rover_drive` | both | Path tracker, safety gate (`/cmd_vel`), depth front distance |
| `leader_cmd_mux` | leader | `cmd_vel_mux`: align > pick > nav -> `/cmd_vel_mux` (safety gate input) |
| `turtlebot3_manipulation_bringup` | leader | Base + arm hardware launch, front depth camera launch and parameters |
| `turtlebot3_manipulation_description` | leader | Robot model: upper deck, depth camera, EEF camera frames |
| `turtlebot3_manipulation_hardware` | leader | ros2_control system for the OpenCR (wheels, arm, gripper) |
| `leader_platooning_beacon` | leader | `/dynamic_joint_states` -> `/battery_state`, `/sensor_state` |
| `omx_turtlebot3_runtime` | leader | v6 pick and place launch: EEF ArUco observer, trial gate |
| `omx_turtlebot3_control_cpp` | leader | v6 policy relay (ONNX residual on the arm trajectories), ONNX check |
| `omx_turtlebot3_policy` | leader | Frozen v6 policy model (`actor.onnx`, contract, parity vectors) |
| `mp_control` | leader | Grasp sequence and handoff onto the follower |

The `turtlebot3_manipulation_*` packages are ROBOTIS packages modified for this rover
(see the "Modified" notes in the files). `arm_signal_adapter.py` maps the arm status to the
`/leader/work_state` and `/leader/pick_place_done` topics the host waits for, and starts the
v6 pick and place when the leader holds at a pick goal (marker ID 0 or 1 by the shelf).

## Leader start

`run_leader.sh start` runs `rover_drive leader.launch.py`, `hardware.launch.py`, `cmd_vel_mux` and
the adapter. Before the first pick the arm must be at the Stay pose
(`0.104311 0.027612 -0.001534 -1.638291`, e.g. `ros2 topic pub --once /arm_controller/joint_trajectory ...`):
the pick program waits for it and gives up after 45 s.

## Leader external dependencies

- apt: `ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-v4l2-camera
  ros-humble-dynamixel-sdk ros-humble-turtlebot3-msgs ros-humble-xacro ros-humble-urdf
  libyaml-cpp-dev libssl-dev python3-scipy python3-opencv` (OpenCV with `dnn` and `aruco`)
- Python for the v6 observer: MuJoCo 3.10.0 with NumPy 1.x. Use `rover_ws/.venv_omx_cpp`
  (`python3 -m venv --system-site-packages`, then `pip install mujoco==3.10.0`); it is picked up
  automatically, or set `OMX_CPP_PYTHON`.
- Source (in the `TURTLEBOT3_WS` underlay): [ros2_astra_camera](https://github.com/orbbec/ros2_astra_camera)
  (`astra_camera`, `astra_camera_msgs`; needs libuvc from source and its udev rules) for the
  front depth camera. `LEADER_CAMERA=false` starts without it.

## Hardware values in the leader packages

- Arm Stay pose: `turtlebot3_manipulation_bringup/launch/base.launch.py`, `mp_control_aruco_params.yaml`,
  `omx_turtlebot3_runtime` trial gate, `arm_signal_adapter.py`
- EEF camera: device, calibration and mount (`ros_policy_pick_place.launch.py`,
  `omx_turtlebot3_runtime/config/eef_usb_camera.yaml`, `runtime.yaml` `vision`, `hardware.launch.py`)
- Front depth camera: `turtlebot3_manipulation_bringup/config/astra_mini_params.yaml`
- Markers (IDs 0/1, 40 mm on a 50 mm cube), deck pose: `omx_turtlebot3_runtime/config/runtime.yaml`
- Grasp and handoff tuning: `mp_control/config/mp_control_aruco_params.yaml` and the overrides in
  `ros_policy_pick_place.launch.py`
- Shelf split for the marker ID: `arm_signal_adapter.py` `marker_split_y`
- Model files are checksum-locked (`omx_turtlebot3_policy/.../SHA256SUMS`, `runtime.yaml`
  `robot_urdf_sha256`): update the sums when one of them changes.
