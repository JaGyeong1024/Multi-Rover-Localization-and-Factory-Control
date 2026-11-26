from setuptools import find_packages, setup


package_name = 'leader_cmd_mux'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Leader base command mux (align > pick > nav).',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'cmd_vel_mux = leader_cmd_mux.cmd_vel_mux:main',
        ],
    },
)
