# Host setup

One-time setup of a host PC: Ubuntu 22.04, ROS 2 Humble, NVIDIA GPU (driver with CUDA 12.4+).
The YOLO weights and the camera calibration are in the repository, so a clone is self-contained.

```bash
bash setup/setup.sh      # needs sudo; log in again once if it added you to dialout/video
```

`setup.sh` is idempotent and does the following:

1. Installs the udev rules (`config/hardware/udev/`), so the UWB dongle appears as `/dev/ttyUWB`
   and the ceiling camera as `/dev/global_cam`, and adds the user to `dialout` and `video`.
2. Installs the system Python packages the nodes use and the ROS dependencies (`rosdep`).
3. Creates `.venv_yolo` with CUDA PyTorch and Ultralytics for the perception node.
4. Installs rosbridge, web_video_server and Node.js 20 for the dashboard.
5. Copies the camera calibration from `data/calibration/` to `~/.factory_calibration/`.
6. Builds `host_ws` with `colcon build --symlink-install`.

Then plug in the UWB dongle and the camera and run `bash host_ws/run_host.sh`.

## Troubleshooting

- **CUDA driver older than 12.4**: install the `cu121` PyTorch wheels instead (edit the `torch`
  line in `setup.sh`). Without a GPU the perception node cannot run in real time; localization
  still works from UWB and the IMU.
- **Device missing**: re-plug it (udev rules apply on plug-in) and check
  `ls -l /dev/ttyUWB /dev/global_cam`.
- **Camera or mount changed**: recalibrate with `host_ws/src/host_localization/scripts/` and
  replace the files in `data/calibration/`.
