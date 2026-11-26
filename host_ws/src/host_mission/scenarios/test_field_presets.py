"""One site lap per rover_drive preset (rover_ws/src/rover_drive/config/presets).

Every preset file found is covered, so one added later is lapped too. The lap and its checks
are those of test_layout_scenarios.test_box_lap_under_site_conditions (box1 / gear_01, site
profile, seed 1, its _judge), with the preset applied on top of the shipped rover config of
its robot (the other rover runs its shipped config). A preset that fails here is a preset the
operator must not pick on site without knowing why.
"""
import pytest

from layout_runner import PRESET_DIR, preset_overrides, rover_stack, run_layout
from lap_checks import summary
from test_layout_scenarios import SITE_PROFILE, TIME_LIMIT, _judge

LAYOUT, PART = 'box1', 'gear_01'
PRESET_FILES = sorted(p.stem for p in PRESET_DIR.glob('*.yaml'))


def test_presets_are_found():
    assert PRESET_FILES


@pytest.mark.parametrize('preset_file', PRESET_FILES)
def test_preset_values_reach_sim_rover(preset_file):
    # Cheap guard: a lap that passes because the preset never got applied proves nothing.
    robot, name = preset_file.split('_', 1)
    stack = rover_stack(robot, [name])
    values = preset_overrides(robot, [name])
    shipped = rover_stack(robot)
    configs = (lambda s: s.ctrl._cfg, lambda s: s.gate._cfg, lambda s: s.depth_cfg)
    for key, value in values.items():
        owners = [(cfg(stack), cfg(shipped)) for cfg in configs
                  if cfg(stack) is not None and hasattr(cfg(stack), key)]
        assert owners, f'{preset_file}: {key} reaches no simulated config'
        for with_preset, without in owners:
            assert getattr(with_preset, key) == value != getattr(without, key), key


def test_later_preset_wins_unknown_raises(tmp_path, monkeypatch):
    import layout_runner
    for name, gain in (('a', 1.0), ('b', 2.0)):
        (tmp_path / f'leader_{name}.yaml').write_text(
            f'/**:\n  ros__parameters:\n    pivot_omega: {gain}\n')
    monkeypatch.setattr(layout_runner, 'PRESET_DIR', tmp_path)
    assert preset_overrides('leader', ['a', 'b']) == {'pivot_omega': 2.0}
    assert preset_overrides('leader', ['b', 'a']) == {'pivot_omega': 1.0}
    with pytest.raises(ValueError, match='unknown leader preset'):
        preset_overrides('leader', ['a', 'nope'])


@pytest.mark.parametrize('preset_file', PRESET_FILES)
def test_site_lap_with_preset(preset_file, record_row):
    robot, name = preset_file.split('_', 1)
    result = run_layout(LAYOUT, PART, duration=TIME_LIMIT[LAYOUT], seed=1, profile=SITE_PROFILE,
                        presets={robot: [name]})
    record_row(f'{preset_file:24s} ' + result.row())
    record_row('    ' + summary(result))
    _judge(result, LAYOUT, bound_key=f'{LAYOUT}-site')
