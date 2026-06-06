from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'host_localization'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
        # launch/ is organized into runtime/ and debug/ subdirs at source level,
        # but installed FLAT into share/<pkg>/launch/ so all `ros2 launch <pkg> X`
        # commands and FindPackageShare-based includes keep working unchanged.
        ('share/' + package_name + '/launch',
            glob('launch/**/*.py', recursive=True)),
        ('share/' + package_name + '/rviz',
            glob('rviz/*.rviz')),
    ],
    # pupil-apriltags: official AprilTag library. cv2.aruco's apriltag dict does
    # not reliably decode real tags from apriltag_ros/online generators; pinned
    # for graduation-demo reproducibility. Install: pip install pupil-apriltags==1.0.4.post11
    install_requires=['setuptools', 'pupil-apriltags==1.0.4.post11'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Ceiling camera BEV-based rover localization',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ceiling_tracker_node = host_localization.ceiling_tracker_node:main',
            'unified_calibrator = host_localization.unified_calibrator:main',
            'uwb_reader_node = host_localization.uwb_reader_node:main',
            'uwb_correction_node = host_localization.uwb_correction_node:main',
            'uwb_trail_cloud_node = host_localization.uwb_trail_cloud_node:main',
            'anchor_publisher_node = host_localization.anchor_publisher_node:main',
            'waypoint_marker_node = host_localization.waypoint_marker_node:main',
            'camera_publisher_node = host_localization.camera_publisher_node:main',
            'global_cam_node = host_localization.global_cam_node:main',
            'check_imu_sync = host_localization.check_imu_sync:main',
            'localization_node = host_localization.localization_node:main',
            'fusion_node = host_localization.fusion.fusion_node:main',
            'global_position_node = host_localization.global_position_node:main',
            'localization_diagnostics_node = host_localization.localization_diagnostics_node:main',
        ],
    },
)
