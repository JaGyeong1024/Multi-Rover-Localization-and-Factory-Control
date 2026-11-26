# config

Everything that depends on **the site** (the arena floor) or **the hardware** (devices, rover
geometry, sensor mounts). Algorithm tuning stays in each package's own `config/` folder.
The packages install these files into their share directories (`colcon build --symlink-install`),
so an edit here takes effect on the next launch.

All poses are in `uwb_global` (metres, radians, yaw counter-clockwise).

## site/ — change when the arena is rebuilt or re-measured

| File | Contents |
|---|---|
| `arena.yaml` | Area the rovers may drive in, shelf and corner fixtures, idle heading (overlay for `host_mission/config/mission.yaml`) |
| `goals.yaml` | Named goal poses: pick, load, standby, unload (`/mission/teach_goal` rewrites it) |
| `mission_scenario.yaml` | Parts, the goals each rover uses for them, pick and phase timing |
| `calibration_points.yaml` | Positions of the four floor AprilTags used to calibrate the ceiling camera |
| `uwb_config.yaml` | UWB serial port, tag IDs and heights, anchor positions, affine correction |
| `obstacle_mapper.yaml` | Zones the obstacle map ignores (shelves), size of the boxes on the floor |

## hardware/ — change when a rover, sensor or mount changes

| File | Contents |
|---|---|
| `rovers.yaml` | Rover footprints, rotation-centre offset, front-stop zone, cargo deck, arm drop point (overlay for `mission.yaml`) |
| `sensor_mounts.yaml` | Pose of the follower LiDAR and the leader depth camera on their rovers |
| `rover_leader.yaml`, `rover_follower.yaml` | Rover-side `rover_drive` keys: motor sign and speed quantisation, tracked-point offset, LiDAR / depth-camera geometry, device topics |
| `fusion_wheels.yaml` | Wheel radius and wheel joint names per rover (wheel odometry used by the EKF) |
| `global_cam_params.yaml` | Ceiling camera device, resolution, fixed exposure / focus, processed ROI |
| `ydlidar_g6_follower.yaml` | Follower LiDAR driver parameters (example: stock G6 values, set your port) |
| `udev/99-factory-devices.rules` | Stable `/dev/ttyUWB` and `/dev/global_cam` names (installed by `setup/setup.sh`) |

## Elsewhere

- **Camera calibration** (`data/calibration/`): intrinsics, floor homography and remap, produced by
  `host_ws/src/host_localization/scripts/`. Redo it when the ceiling camera or its mount moves.
- **ROS domains**: host 30, leader 25, follower 73, bridged by `domain_bridge`
  (`host_localization/config/bridge_*.yaml`, `host_mission/config/bridge_mission_*.yaml`).
  The rover IDs are also set in `rover_ws/run_leader.sh` and `run_follower.sh`.
- **Rovers**: copy `rover_ws/` and `config/` side by side onto each rover, then
  `colcon build --symlink-install` in `rover_ws`.
