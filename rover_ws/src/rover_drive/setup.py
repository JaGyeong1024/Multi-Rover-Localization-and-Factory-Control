from glob import glob

from setuptools import find_packages, setup

package_name = 'rover_drive'
# Hardware files live in the repository's config/ folder, next to rover_ws/.
REPO_CONFIG = '../../../config'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/hardware',
         [REPO_CONFIG + '/hardware/rover_leader.yaml', REPO_CONFIG + '/hardware/rover_follower.yaml']),
        ('share/' + package_name + '/config/presets', glob('config/presets/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Path tracker and safety gate shared by the leader and follower rovers.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_tracker_node = rover_drive.path_tracker_node:main',
            'safety_gate_node = rover_drive.safety_gate_node:main',
            'cmd_watchdog_node = rover_drive.cmd_watchdog_node:main',
            'depth_front_node = rover_drive.depth_front_node:main',
        ],
    },
)
