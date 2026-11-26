"""Config file loading (YAML -> dict) for the node and for scenario tests.

A profile is a named overlay of world keys (config/sim_profiles.yaml), e.g. the site
conditions of one field day, applied to any world config with apply_profile.
"""
import copy
import os

import yaml

DEFAULT_CONFIG_NAME = 'sim_default.yaml'
PROFILES_NAME = 'sim_profiles.yaml'


def packaged_config_path(name):
    """Locate a file of config/ in the source tree, else in the installed share."""
    here = os.path.dirname(os.path.realpath(__file__))
    source_path = os.path.join(here, '..', 'config', name)
    if os.path.isfile(source_path):
        return os.path.normpath(source_path)
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('host_sim'), 'config', name)


def default_config_path():
    """Locate sim_default.yaml in the source tree, else in the installed share."""
    return packaged_config_path(DEFAULT_CONFIG_NAME)


def load_config(path=None):
    """Read a simulator config; an empty path selects the packaged default."""
    with open(path or default_config_path()) as stream:
        return yaml.safe_load(stream) or {}


def overlay(base, patch):
    """Copy of ``base`` with ``patch`` merged in: dicts merge key by key, other values replace."""
    merged = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = overlay(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_profile(config, name, path=None):
    """World config with the named profile of sim_profiles.yaml (or ``path``) overlaid.
    ``name`` may list several profiles separated by commas; they are overlaid in order."""
    with open(path or packaged_config_path(PROFILES_NAME)) as stream:
        profiles = yaml.safe_load(stream) or {}
    for part in name.split(','):
        if part not in profiles:
            raise ValueError(f'unknown sim profile {part!r}; expected one of {sorted(profiles)}')
        config = overlay(config, profiles[part])
    return config


def with_localization_mode(config, mode):
    """World config with ``localization.mode`` set for every rover (ideal | fused)."""
    robots = {name: {'localization': {'mode': mode}} for name in config.get('robots') or {}}
    return overlay(config, {'robots': robots})
