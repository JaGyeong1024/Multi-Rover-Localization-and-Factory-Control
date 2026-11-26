"""One simulated mission lap in node mode, for the web dashboard (single domain).

The node-mode counterpart of host_mission/scenarios/layout_runner.py: the same world,
layout, goals, mission settings and mapper parameters, but every piece runs as its
ROS node:

  host_sim sim_node            rover bases + localization stand-in, layout obstacles
  mission_coordinator_node     nav_mode path
  obstacle_mapper_node         /follower/scan + /leader/depth_scan -> mapped obstacles
  rover_drive (per rover)      path_tracker + safety_gate (+ cmd_watchdog on the follower),
                               config/<robot>.yaml + hardware/rover_<robot>.yaml with only
                               the topic wiring changed:
                               gate output -> /{robot}/cmd_vel_out (simulator base input)
  sim_stand_in_node            pick&place, leader front distance, unload button, dispatch

    ROS_DOMAIN_ID=45 ROS_LOCALHOST_ONLY=1 ros2 launch <this file> layout:=box1 part:=bolt_01

NEVER on the real domains 25/30/73. ``part:=''`` leaves dispatch to the dashboard;
``auto_unload:=false`` waits for the dashboard's unload button. The generated configs
are written to ``out_dir``. Needs rover_drive built (any colcon install on the path).
"""
import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(os.path.realpath(__file__)).parent          # source tree (symlink install)
PACKAGE = HERE.parent                                   # host_ws/src/host_mission
DATA = PACKAGE / 'scenarios' / 'data'
MISSION_CONFIG = PACKAGE / 'config'
SITE_CONFIG = PACKAGE.parents[2] / 'config' / 'site'
HARDWARE_CONFIG = PACKAGE.parents[2] / 'config' / 'hardware'
SIM_CONFIG = PACKAGE.parent / 'host_sim' / 'config'

ARGS = (
    ('layout', 'empty', 'box layout of scenarios/data/layouts.yaml'),
    ('part', '', "part to dispatch once both rovers are up ('' = dashboard buttons)"),
    ('auto_unload', 'true', 'press unload-done by itself (false = dashboard button)'),
    ('pick_duration', '5.0', 's, pick&place stand-in time (layout_runner PICK_DURATION)'),
    ('pick_cooldown', '0.0', 's, per shelf (layout_runner sets 0)'),
    ('profile', '', 'host_sim sim_profiles.yaml profile overlaid on the world'),
    ('out_dir', '/tmp/sim_dashboard', 'where the generated config files go'),
)


def _read(path):
    return yaml.safe_load(Path(path).read_text()) or {}


def _write(out_dir, name, data):
    path = out_dir / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def sim_world(layout, profile):
    """Layout world as layout_runner.load_layout builds it, plus the leader depth scanner
    of the default world (the mapper's /leader/depth_scan; in-process it is cut from the
    leader LiDAR instead)."""
    layouts = _read(DATA / 'layouts.yaml')
    if layout not in layouts['boxes']:
        raise ValueError(f'unknown layout {layout!r}; expected one of {sorted(layouts["boxes"])}')
    world = _read(DATA / 'sim_world.yaml')
    world['obstacles'] = layouts['shelves'] + layouts['boxes'][layout]
    leader = world['robots']['leader']
    if 'depth_scan' not in leader:
        leader['depth_scan'] = _read(SIM_CONFIG / 'sim_default.yaml')['robots']['leader']['depth_scan']
    if profile:
        from host_sim.config import apply_profile
        world = apply_profile(world, profile, str(SIM_CONFIG / 'sim_profiles.yaml'))
    fixtures = [{k: v for k, v in s.items() if k != 'type'} for s in layouts['shelves']]
    return world, fixtures


def _rover_nodes(robot, overrides, watchdog):
    share = get_package_share_directory('rover_drive')
    params = [os.path.join(share, 'config', f'{robot}.yaml'),
              os.path.join(share, 'config', 'hardware', f'rover_{robot}.yaml'),
              {'robot': robot, **overrides}]
    names = ['path_tracker_node', 'safety_gate_node'] + (['cmd_watchdog_node'] if watchdog else [])
    return [Node(package='rover_drive', executable=exe, name=exe.replace('_node', ''),
                 namespace=f'sim_{robot}', parameters=params, output='screen')
            for exe in names]


def _setup(context):
    arg = {name: LaunchConfiguration(name).perform(context) for name, _d, _h in ARGS}
    out_dir = Path(arg['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    world, fixtures = sim_world(arg['layout'], arg['profile'])
    from host_mission.settings import read_settings_yaml
    settings = read_settings_yaml(MISSION_CONFIG / 'mission.yaml',
                                  [SITE_CONFIG / 'arena.yaml', HARDWARE_CONFIG / 'rovers.yaml'])
    settings['fixtures'] = fixtures
    scenario = _read(SITE_CONFIG / 'mission_scenario.yaml')
    scenario['pick_cooldown'] = float(arg['pick_cooldown'])
    mapper = _read(DATA / 'obstacle_mapper.yaml')
    files = {
        'sim': _write(out_dir, 'sim_world.yaml', world),
        'settings': _write(out_dir, 'mission.yaml', settings),
        'scenario': _write(out_dir, 'mission_scenario.yaml', scenario),
        'goals': _write(out_dir, 'goals.yaml', _read(DATA / 'goals.yaml')),  # teach writes here
    }
    # The written settings are already merged: no overlays on top of them.
    mapper_params = mapper['obstacle_mapper_node']['ros__parameters']
    mapper_params['settings_file'] = files['settings']
    mapper_params['settings_overlays'] = ['']
    files['mapper'] = _write(out_dir, 'obstacle_mapper.yaml', mapper)

    stand_in_params = ['--ros-args', '-r', '__node:=sim_stand_in',
                       '-p', f"auto_unload:={arg['auto_unload']}",
                       '-p', f"pick_duration:={float(arg['pick_duration'])}",
                       '-p', f"dispatch:='{arg['part']}'"]
    return [
        Node(package='host_sim', executable='sim_node', name='host_sim', output='screen',
             parameters=[{'config_file': files['sim']}]),
        Node(package='host_mission', executable='mission_coordinator_node',
             name='mission_coordinator_node', output='screen',
             parameters=[{'nav_mode': 'path', 'settings_file': files['settings'],
                          'settings_overlays': [''], 'scenario_file': files['scenario'],
                          'goals_file': files['goals']}]),
        Node(package='host_mission', executable='obstacle_mapper_node',
             name='obstacle_mapper_node', output='screen', parameters=[files['mapper']]),
        # leader: tracker -> /leader/cmd_vel -> gate (no cmd_vel_mux in the sim) -> base;
        # front stop from /leader/front_distance (stand-in; no depth image in the sim)
        *_rover_nodes('leader', {'input_topic': '/leader/cmd_vel',
                                 'output_topic': '/leader/cmd_vel_out'}, watchdog=False),
        # follower: gate reads the simulated LiDAR; watchdog zeroes the same base input
        *_rover_nodes('follower', {'output_topic': '/follower/cmd_vel_out',
                                   'scan_topic': '/follower/scan'}, watchdog=True),
        ExecuteProcess(cmd=['python3', '-m', 'host_mission.sim_stand_in_node', *stand_in_params],
                       output='screen'),
    ]


def generate_launch_description():
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=default, description=text)
          for name, default, text in ARGS],
        OpaqueFunction(function=_setup),
    ])
