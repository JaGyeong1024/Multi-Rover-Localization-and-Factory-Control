from setuptools import find_packages, setup

package_name = 'rover_nav_ref'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Rover-side reference (planning guide) consuming host TargetCommand; also the host mock.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rover_nav_node = rover_nav_ref.rover_nav_node:main',
        ],
    },
)
