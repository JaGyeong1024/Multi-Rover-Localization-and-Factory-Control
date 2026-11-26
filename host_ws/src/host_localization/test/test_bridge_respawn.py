"""Every domain_bridge process in the localization launch files must respawn.

Loads the real launch files (launch_ros only, no hardware); skipped without ROS.
"""
import importlib.util
import os

import pytest

_LAUNCH = os.path.join(os.path.dirname(__file__), '..', 'launch', 'runtime')


@pytest.mark.parametrize('name', ['global_position.launch.py', 'rover_bridge.launch.py'])
def test_every_domain_bridge_respawns(name):
    launch_ros = pytest.importorskip('launch_ros')
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'),
                                                  os.path.join(_LAUNCH, name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bridges = [e for e in module.generate_launch_description().entities
               if isinstance(e, launch_ros.actions.Node)
               and e.node_package == 'domain_bridge']
    assert len(bridges) == 2
    for b in bridges:
        assert b._ExecuteLocal__respawn is True or getattr(b, 'respawn', True)
