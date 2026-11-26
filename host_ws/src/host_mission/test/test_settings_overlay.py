"""Settings overlays: dicts merge recursively, any other value replaces."""
import yaml

from host_mission.settings import load_settings, merge_dicts, read_settings_yaml


def test_merge_dicts_merges_nested_replaces_rest():
    base = {'a': {'b': 1, 'c': [1, 2]}, 'd': 1, 'e': {'x': 1}}
    overlay = {'a': {'c': [3]}, 'd': {'y': 2}, 'e': 5}
    assert merge_dicts(base, overlay) == {'a': {'b': 1, 'c': [3]}, 'd': {'y': 2}, 'e': 5}
    assert base['a']['c'] == [1, 2]            # inputs are not modified


def test_overlays_equal_one_full_file(tmp_path):
    full = {'tick_rate': 4.0, 'bounds': {'xmin': 0.0, 'xmax': 5.0, 'ymin': 0.0, 'ymax': 3.0},
            'robots': {'leader': {'speed_cap': 0.1, 'plan': {'spacing': 0.05,
                                                             'footprint': [0.1, 0.1, 0.1]}}}}
    base = {'tick_rate': 4.0, 'robots': {'leader': {'speed_cap': 0.1, 'plan': {'spacing': 0.05}}}}
    arena = {'bounds': full['bounds']}
    rovers = {'robots': {'leader': {'plan': {'footprint': [0.1, 0.1, 0.1]}}}}
    paths = []
    for name, data in (('full', full), ('base', base), ('arena', arena), ('rovers', rovers)):
        paths.append(tmp_path / f'{name}.yaml')
        paths[-1].write_text(yaml.safe_dump(data))
    assert read_settings_yaml(paths[1], paths[2:]) == full
    assert load_settings(paths[1], paths[2:]) == load_settings(paths[0])
