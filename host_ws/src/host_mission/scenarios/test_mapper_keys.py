"""make_mapper must refuse a mapper yaml key no mapper field accepts (no silent default)."""

import pytest
import yaml

import layout_runner
from layout_runner import make_mapper
from helpers import settings


def _data_with(tmp_path, edit):
    raw = yaml.safe_load((layout_runner.DATA / 'obstacle_mapper.yaml').read_text())
    edit(raw['obstacle_mapper_node']['ros__parameters'])
    (tmp_path / 'obstacle_mapper.yaml').write_text(yaml.safe_dump(raw))
    return tmp_path


def test_shipped_mapper_yaml_builds_mapper():
    assert make_mapper(settings())


@pytest.mark.parametrize('name, edit, token', [
    ('param', lambda p: p.__setitem__('cell_siz', 0.05), 'cell_siz'),
    ('source', lambda p: p['leader_depth'].__setitem__('max_use_rang', 4.0),
     'leader_depth.max_use_rang')])
def test_make_mapper_raises_on_unknown_key(tmp_path, monkeypatch, name, edit, token):
    monkeypatch.setattr(layout_runner, 'DATA', _data_with(tmp_path, edit))
    with pytest.raises(ValueError, match=token):
        make_mapper(settings())
