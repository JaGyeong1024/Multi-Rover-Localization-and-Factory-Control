# Camera calibration scripts

Not runtime nodes. They import `host_localization`, so run them with the workspace sourced.

## Calibration (once per camera mount)

1. **Intrinsics**: `charuco_calibration_with_plane_axis.py` writes `charuco_intrinsics.npz` (K, D).
   `calibrate_intrinsic.py` is a checkerboard alternative that writes `intrinsic.yaml`.
2. **Top-down view**: `calibrate_extrinsic.py` finds the four floor AprilTags at the positions in
   `config/site/calibration_points.yaml` and writes `calibration.yaml` and `remap.npz`.
   It uses `intrinsic.yaml` if present, otherwise `charuco_intrinsics.npz`.

`generate_aruco_markers.py` makes printable marker images.
Results go to `data/calibration/`; `setup/setup.sh` copies them to `~/.factory_calibration/`,
where the camera node reads them.

## Camera settings

`camera_tuner.py` finds fixed exposure and focus values for
`config/hardware/global_cam_params.yaml`.
