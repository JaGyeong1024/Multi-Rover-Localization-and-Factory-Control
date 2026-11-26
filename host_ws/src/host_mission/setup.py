from glob import glob

from setuptools import find_packages, setup

package_name = 'host_mission'
# Site and hardware files live in the repository's config/ folder (three levels up).
REPO_CONFIG = '../../../config'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
         glob('config/*.yaml') + glob('config/*.json')
         + [REPO_CONFIG + '/site/goals.yaml', REPO_CONFIG + '/site/mission_scenario.yaml']),
        ('share/' + package_name + '/config/site',
         [REPO_CONFIG + '/site/arena.yaml', REPO_CONFIG + '/site/obstacle_mapper.yaml']),
        ('share/' + package_name + '/config/hardware',
         [REPO_CONFIG + '/hardware/rovers.yaml', REPO_CONFIG + '/hardware/sensor_mounts.yaml']),
        ('share/' + package_name + '/launch', glob('launch/**/*.py', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Host-side mission coordination (fleet orchestration) for the multi-rover system.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission_coordinator_node = host_mission.mission_coordinator_node:main',
            'obstacle_mapper_node = host_mission.obstacle_mapper_node:main',
        ],
    },
)
