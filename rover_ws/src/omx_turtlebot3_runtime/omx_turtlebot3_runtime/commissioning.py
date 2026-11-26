"""Physical readiness metadata shared by launch and the runtime, without ROS imports."""
import math

REQUIRED_VERIFICATIONS = (
    'joint_mapping_verified', 'camera_extrinsics_verified', 'marker_mount_verified',
    'scene_alignment_verified', 'grasp_thresholds_verified',
    'placement_thresholds_verified', 'hardware_watchdog_verified',
)


def thresholds_calibrated(config):
    fields = ('current_idle_raw', 'current_grip_delta_raw', 'current_release_delta_raw')
    return (all(type(config.get(key)) in (int, float) and math.isfinite(config[key])
                for key in fields)
            and 0 <= config[fields[2]] < config[fields[1]])


def commissioning_issues(config):
    commissioning = config.get('commissioning', {})
    issues = [key for key in REQUIRED_VERIFICATIONS if commissioning.get(key) is not True]
    if config.get('vision', {}).get('camera_mount_candidate', False) is not False:
        issues.append('camera_mount_is_unmeasured_candidate')
    if config.get('vision', {}).get('marker_size_confirmed') is not True:
        issues.append('marker_size_unconfirmed')
    if not commissioning.get('evidence_directory'):
        issues.append('evidence_directory_missing')
    if not thresholds_calibrated(config.get('estimation', {})):
        issues.append('current_thresholds_missing')
    return issues
