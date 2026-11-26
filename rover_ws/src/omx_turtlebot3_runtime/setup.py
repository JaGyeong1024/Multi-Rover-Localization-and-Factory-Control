from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'omx_turtlebot3_runtime'

# assets/: MuJoCo scene, meshes and marker textures, robot URDF snapshot.
assets = []
for directory, _, files in os.walk('assets'):
    if files:
        assets.append(('share/' + package_name + '/' + directory,
                       [directory + '/' + name for name in files]))
setup(name=package_name, version='0.1.0', packages=find_packages(),
      data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                  ('share/' + package_name, ['package.xml']),
                  ('share/' + package_name + '/config', glob('config/*.yaml')),
                  ('share/' + package_name + '/launch', glob('launch/*.launch.py'))] + assets,
      install_requires=['setuptools'], zip_safe=False,
      maintainer='user', maintainer_email='user@example.com',
      description='v6 pick and place for the leader arm: observer, trial gate and launch.',
      license='Apache-2.0')
