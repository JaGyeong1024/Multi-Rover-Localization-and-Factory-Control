"""Pieces shared by ``leader.launch.py`` and ``follower.launch.py``."""
import os
import re

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

_RESPAWN_DELAY_S = 1.0
PARAMS_ARG = 'params_file'
HARDWARE_ARG = 'hardware_file'
PRESETS_ARG = 'presets'


def params_argument(robot: str) -> DeclareLaunchArgument:
    """``params_file`` launch argument, default ``config/<robot>.yaml``."""
    default = PathJoinSubstitution([FindPackageShare('rover_drive'), 'config', robot + '.yaml'])
    return DeclareLaunchArgument(PARAMS_ARG, default_value=default,
                                 description=f'parameter YAML (default: config/{robot}.yaml)')


def hardware_argument(robot: str) -> DeclareLaunchArgument:
    """``hardware_file`` launch argument, default ``config/hardware/rover_<robot>.yaml``."""
    default = PathJoinSubstitution([FindPackageShare('rover_drive'), 'config', 'hardware',
                                    f'rover_{robot}.yaml'])
    return DeclareLaunchArgument(
        HARDWARE_ARG, default_value=default,
        description=f'hardware parameter YAML, applied after params_file '
                    f'(default: config/hardware/rover_{robot}.yaml)')


def presets_argument(robot: str) -> DeclareLaunchArgument:
    """``presets`` launch argument: overlays applied on top of the hardware file."""
    return DeclareLaunchArgument(
        PRESETS_ARG, default_value='',
        description=f'comma-separated preset names; each is config/presets/{robot}_<name>.yaml, '
                    'applied after the hardware file in the order given (later wins)')


def preset_names(text: str) -> list:
    """``'a, b c'`` -> ``['a', 'b', 'c']`` (commas and/or spaces separate names)."""
    return [name for name in re.split(r'[,\s]+', text) if name]


def preset_files(robot: str, names: list, share_dir: str) -> list:
    """Paths of the named presets; an unknown name raises (no silent fallback to the base)."""
    preset_dir = os.path.join(share_dir, 'config', 'presets')
    files = [os.path.join(preset_dir, f'{robot}_{name}.yaml') for name in names]
    missing = [name for name, path in zip(names, files) if not os.path.isfile(path)]
    if missing:
        prefix = robot + '_'
        known = sorted(f[len(prefix):-len('.yaml')] for f in os.listdir(preset_dir)
                       if f.startswith(prefix) and f.endswith('.yaml'))
        raise RuntimeError(f'unknown {robot} preset(s) {missing}: no {preset_dir}/'
                           f'{robot}_<name>.yaml; known presets: {known}')
    return files


def toggle_argument(name: str, default: bool, description: str) -> DeclareLaunchArgument:
    return DeclareLaunchArgument(name, default_value=str(default).lower(),
                                 description=description)


def drive_node(executable: str, robot: str, toggle: str = '') -> OpaqueFunction:
    """One rover_drive node reading the shared parameter file, the hardware file, then the
    chosen presets."""
    return OpaqueFunction(function=_make_drive_node, args=[executable, robot, toggle])


def _make_drive_node(context, executable: str, robot: str, toggle: str) -> list:
    names = preset_names(LaunchConfiguration(PRESETS_ARG).perform(context))
    presets = preset_files(robot, names, get_package_share_directory('rover_drive')) if names else []
    condition = IfCondition(LaunchConfiguration(toggle)) if toggle else None
    return [Node(
        package='rover_drive', executable=executable, name=executable.replace('_node', ''),
        parameters=[LaunchConfiguration(PARAMS_ARG), LaunchConfiguration(HARDWARE_ARG), *presets,
                    {'robot': robot}],
        output='screen', respawn=True, respawn_delay=_RESPAWN_DELAY_S, condition=condition)]
