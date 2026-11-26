"""World config: localization mode selection, overlays and named profiles."""
import pytest

from host_sim.config import apply_profile, load_config, overlay, with_localization_mode
from host_sim.core import FusedLocalization, World
from host_sim.core.sensors import LocalizationSensor


def test_default_world_ideal_output_model():
    world = World.from_config(load_config(), seed=1)
    assert all(isinstance(s, LocalizationSensor) for s in world.sensors.values())


def test_mode_fused_selects_sensor_model():
    config = overlay(load_config(), {'robots': {'follower': {'localization': {
        'mode': 'fused', 'fused': {'uwb': {'offset_x': 0.5}}}}}})
    world = World.from_config(config, seed=1)
    assert isinstance(world.sensors['leader'], LocalizationSensor)
    follower = world.sensors['follower']
    assert isinstance(follower, FusedLocalization)
    assert follower.params.uwb.offset_x == 0.5
    assert follower.params.camera.rate_hz == 21.0                  # default kept


@pytest.mark.parametrize('localization', [
    {'mode': 'perfect'},                                         # unknown mode
    {'mode': 'fused', 'fused': {'lidar': {}}},                  # unknown section
    {'mode': 'fused', 'fused': {'uwb': {'offst_x': 1.0}}},      # typo in a key
])
def test_bad_localization_rejected(localization):
    config = overlay(load_config(), {'robots': {'leader': {'localization': localization}}})
    with pytest.raises(ValueError):
        World.from_config(config)


def test_overlay_merges_without_touching_base():
    base = {'a': {'b': 1, 'c': [1, 2]}, 'd': 3}
    merged = overlay(base, {'a': {'c': [9]}, 'e': {'f': 1}})
    assert merged == {'a': {'b': 1, 'c': [9]}, 'd': 3, 'e': {'f': 1}}
    assert base == {'a': {'b': 1, 'c': [1, 2]}, 'd': 3}


def test_localization_mode_sets_every_rover():
    config = with_localization_mode(load_config(), 'fused')
    assert {r['localization']['mode'] for r in config['robots'].values()} == {'fused'}


def test_field_profile_sets_site_conditions():
    config = apply_profile(load_config(), 'field_site')
    world = World.from_config(config, seed=1)
    for name in ('leader', 'follower'):
        params = world.sensors[name].params
        assert (params.uwb.offset_x, params.uwb.offset_y) == (1.27, 1.23)
        assert params.camera.occlusion_dist == 0.34
        assert params.imu.yaw_rate_scale == 0.965
    assert config['robots']['leader']['start'] == load_config()['robots']['leader']['start']


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match='unknown sim profile'):
        apply_profile(load_config(), 'no_such_day')
