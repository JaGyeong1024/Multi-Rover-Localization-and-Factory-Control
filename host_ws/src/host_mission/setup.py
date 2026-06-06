from glob import glob

from setuptools import find_packages, setup

package_name = 'host_mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.json')),
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
            'mock_pickplace_node = host_mission.mock_pickplace_node:main',
            'mock_rover_node = host_mission.mock_rover_node:main',
        ],
    },
)
