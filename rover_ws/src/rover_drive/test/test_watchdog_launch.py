"""Command watchdog logic, launch files and the shipped YAML files."""
import dataclasses
import importlib.util
import os

import pytest
import yaml

from rover_drive.core.config import DepthFrontConfig, DriveConfig, GateConfig, WatchdogConfig
from rover_drive.core.watchdog import CmdWatchdog

from sim import gate_config, load_config, load_params

_PKG = os.path.join(os.path.dirname(__file__), '..')
_CONFIGS = (DriveConfig, GateConfig, WatchdogConfig, DepthFrontConfig)


def test_watchdog_zero_until_gate_heard():
    dog = CmdWatchdog(0.5)
    assert dog.must_stop(0.0) and dog.must_stop(10.0)
    dog.on_alive(10.0)
    assert not dog.must_stop(10.1)


def test_watchdog_takes_over_then_releases():
    dog = CmdWatchdog(0.5)
    for t in (0.0, 0.1, 0.2):
        dog.on_alive(t)
    assert not dog.must_stop(0.7)
    assert dog.must_stop(0.71)
    assert dog.must_stop(5.0)
    dog.on_alive(5.05)
    assert not dog.must_stop(5.1)


def test_watchdog_timeout_under_ten_heartbeats():
    """The heartbeat rate must leave room for a few lost messages inside the timeout."""
    for robot in ('leader', 'follower'):
        dog = load_config(WatchdogConfig, robot)
        gate = gate_config(robot)
        assert dog.watchdog_timeout_s * gate.alive_rate_hz >= 3.0
        assert dog.output_topic == gate.output_topic == '/cmd_vel'
        assert dog.alive_topic == gate.alive_topic == ''


def _yaml_params(robot):
    return load_params(robot)


@pytest.mark.parametrize('robot', ['leader', 'follower'])
def test_yaml_keys_match_declared_parameters(robot):
    known = {f.name for cls in _CONFIGS for f in dataclasses.fields(cls)}
    params = _yaml_params(robot)
    assert set(params) - known == set()
    needed = {f.name for cls in (DriveConfig, GateConfig, WatchdogConfig)
              for f in dataclasses.fields(cls)}
    assert needed - set(params) == set()


@pytest.mark.parametrize('robot', ['leader', 'follower'])
def test_yaml_types_match_the_dataclasses(robot):
    """ROS parameter types are strict: 1 instead of 1.0 stops the node at start."""
    params = _yaml_params(robot)
    for cls in _CONFIGS:
        for field in dataclasses.fields(cls):
            if field.name not in params:
                continue
            default, value = getattr(cls(), field.name), params[field.name]
            expected = list if isinstance(default, tuple) else type(default)
            assert type(value) is expected, f'{robot} yaml {field.name}: {value!r}'
            if expected is list:
                assert value and all(type(v) is float for v in value)
                assert len(value) % 2 == 0


def test_yaml_holds_the_measured_values():
    leader, follower = _yaml_params('leader'), _yaml_params('follower')
    # leader base turns opposite to the command, follower does not
    assert leader['angular_sign'] == -1.0 and follower['angular_sign'] == 1.0
    assert follower['lidar_yaw_offset'] == -0.5236
    assert (leader['omega_quantum'], leader['omega_quantize']) == (0.0697, 'truncate')
    assert (leader['input_topic'], leader['output_topic']) == ('/cmd_vel_mux', '/cmd_vel')
    assert leader['front_distance_topic'] == '/leader/front_distance'
    assert leader['lidar_stop_base'] + leader['lidar_to_front'] > 0.47
    assert leader['lidar_stop_base'] < leader['pick_approach_dist'] + 0.47
    assert follower['lidar_enabled'] is True and follower['scan_topic'] == '/scan'
    assert (follower['input_topic'], follower['output_topic']) == ('', '/cmd_vel')


def _launch_nodes(name, presets=''):
    """Nodes of a launch file as they are built at launch time with ``presets:=...``."""
    launch_ros = pytest.importorskip('launch_ros')
    import launch
    path = os.path.join(_PKG, 'launch', name)
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entities = module.generate_launch_description().entities
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {'params_file': 'BASE.yaml', 'hardware_file': 'HARDWARE.yaml', 'presets': presets})
    built = [n for e in entities if isinstance(e, launch.actions.OpaqueFunction)
             for n in e.execute(context)]
    nodes = {e.node_executable: e for e in entities + built
             if isinstance(e, launch_ros.actions.Node)}
    defaults = {e.name: e.default_value[0].text for e in entities
                if hasattr(e, 'default_value')
                if e.name not in ('params_file', 'hardware_file', 'presets')}
    return nodes, defaults


def _parameter_files(node):
    """The parameter entries a node is started with: files as paths, dicts as plain dicts."""
    import launch
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {'params_file': 'BASE.yaml', 'hardware_file': 'HARDWARE.yaml'})

    def text(subs):
        return launch.utilities.perform_substitutions(context, list(subs))
    return [{text(k): yaml.safe_load(text(v)) for k, v in entry.items()}
            if isinstance(entry, dict)
            else text(entry.param_file) for entry in node._Node__parameters]


@pytest.fixture
def share_is_source_tree(monkeypatch):
    """Preset names resolve against this source tree (same layout as the installed share)."""
    from rover_drive import launch_common
    monkeypatch.setattr(launch_common, 'get_package_share_directory', lambda pkg: _PKG)


# Second preset per robot (its base is 0).
@pytest.mark.parametrize('robot, second', [('leader', 'xtrack_off'), ('follower', 'speed_025')])
def test_presets_follow_base_in_order(robot, second, share_is_source_tree):
    nodes, _ = _launch_nodes(robot + '.launch.py', presets='pivot_slow, ' + second)
    assert nodes
    presets = os.path.join(_PKG, 'config', 'presets')
    for executable, node in nodes.items():
        assert _parameter_files(node) == [
            'BASE.yaml', 'HARDWARE.yaml', os.path.join(presets, f'{robot}_pivot_slow.yaml'),
            os.path.join(presets, f'{robot}_{second}.yaml'), {'robot': robot}], executable


def test_leader_presets_speed_then_pivot_slow(share_is_source_tree):
    nodes, _ = _launch_nodes('leader.launch.py', presets='speed_015,pivot_slow')
    assert set(nodes) == {'path_tracker_node', 'safety_gate_node', 'depth_front_node',
                          'cmd_watchdog_node'}
    presets = os.path.join(_PKG, 'config', 'presets')
    for node in nodes.values():
        assert _parameter_files(node) == [
            'BASE.yaml', 'HARDWARE.yaml', os.path.join(presets, 'leader_speed_015.yaml'),
            os.path.join(presets, 'leader_pivot_slow.yaml'), {'robot': 'leader'}]


def test_no_presets_gives_base_and_hardware():
    # The default must not even look up the package (it works without presets installed).
    nodes, _ = _launch_nodes('follower.launch.py')
    for node in nodes.values():
        assert _parameter_files(node) == ['BASE.yaml', 'HARDWARE.yaml', {'robot': 'follower'}]


def test_presets_argument_defaults_to_empty():
    pytest.importorskip('launch_ros')
    for robot in ('leader', 'follower'):
        path = os.path.join(_PKG, 'launch', robot + '.launch.py')
        spec = importlib.util.spec_from_file_location(robot + '_launch', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = {e.name: e for e in module.generate_launch_description().entities
                if hasattr(e, 'default_value')}
        assert ''.join(s.text for s in args['presets'].default_value) == ''


@pytest.mark.parametrize('presets', ['speed_15', 'pivot_slow,nope'])
def test_unknown_preset_fails_launch(presets, share_is_source_tree):
    with pytest.raises(RuntimeError, match='unknown leader preset') as error:
        _launch_nodes('leader.launch.py', presets=presets)
    assert 'speed_015' in str(error.value)


def test_leader_only_preset_unknown_to_follower(share_is_source_tree):
    with pytest.raises(RuntimeError, match='unknown follower preset'):
        _launch_nodes('follower.launch.py', presets='speed_015')


def test_follower_launch_starts_all_nodes():
    nodes, defaults = _launch_nodes('follower.launch.py')
    assert set(nodes) == {'path_tracker_node', 'safety_gate_node', 'cmd_watchdog_node'}
    assert defaults == {'watchdog': 'true'}
    assert nodes['cmd_watchdog_node'].condition is not None
    assert nodes['safety_gate_node'].condition is None


def test_leader_launch_starts_all_nodes():
    nodes, defaults = _launch_nodes('leader.launch.py')
    assert set(nodes) == {'path_tracker_node', 'safety_gate_node', 'depth_front_node',
                          'cmd_watchdog_node'}
    assert defaults == {'depth': 'true', 'watchdog': 'false'}


def test_entry_points_exist_for_executables():
    with open(os.path.join(_PKG, 'setup.py')) as f:
        setup_text = f.read()
    for launch in ('leader.launch.py', 'follower.launch.py'):
        for executable in _launch_nodes(launch)[0]:
            module = executable
            assert f"'{executable} = rover_drive.{module}:main'" in setup_text
            assert os.path.isfile(os.path.join(_PKG, 'rover_drive', module + '.py'))
    assert not os.path.exists(os.path.join(_PKG, 'launch', 'rover_drive.launch.py'))
