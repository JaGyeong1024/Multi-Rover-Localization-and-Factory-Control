"""Full-lap regression over the real layout: box layout x part, each run in-process.

Judged per run: mission done (both rovers home), parts delivered, no collision, both rovers
PARKED, finished within TIME_LIMIT sim seconds. Then, from the recorded TRUE state
(lap_checks): both rovers inside the goal tolerances of the true goal, no more distinct
tracker commands than MAX_LOADED_COMMANDS, no mission pause / fault, no LIDAR_STOP of the
leader, the follower never first at a goal nor ahead of the leader on its path, and the
planner footprints never overlap each other or a box. Run via host_ws/run_scenarios.sh."""
import pytest

from lap_checks import lap_failures, loaded_commands, summary
from layout_runner import LAYOUTS, PARTS, run_layout
from map_quality import map_quality

# s sim time allowed for one lap (also the run length).
TIME_LIMIT = {'empty': 130.0, 'box1': 240.0, 'box2': 240.0}
FINAL_NAV_STATE = 'PARKED'    # tracker state of a rover standing on its final goal
# Distinct tracker commands per lap (the larger of the two rovers). Every extra command is
# a re-load the tracker slows down for: a re-plan, a turn-in-place hold (STOP), or a
# re-issued path with changed flags. The bounds are the measured maxima.
MAX_PATH_COMMANDS = {'empty': 2}
MAX_LOADED_COMMANDS = {'empty': 6, 'box1': 8, 'box2': 9, 'box1-site': 10}

SITE_PROFILE = 'field_site,aligned_start'
SITE_LAPS = (('box1', 'gear_01'), ('box1', 'bolt_01'))   # seed 1; every lap costs minutes

# Mapped boxes against the layout truth (scenarios/map_quality.py), every publication.
MAX_COVERAGE_GAP = 0.03  # m a true corner may lie outside shape + pad: the planner's
# clearance absorbs this much; more is a box the rover may clip
MAX_FINAL_EXCESS = 0.10     # m beyond a true side in the lap's last publication: the rovers
#                             have looked around by then; more = free space not given back
#                             (shrink_threshold 0.10)
MAX_PUBLICATIONS = 12       # per box (measured <= 8); each changed set may re-plan a rover
MIN_CORRIDOR = 0.56         # m between two mapped boxes + pads (box2: 0.846 true, 0.646 with
#                             both pads): the grid search needs 0.56 to plan through it


def _cases():
    for layout in LAYOUTS:
        for part in PARTS:
            yield pytest.param(layout, part, id=f'{layout}-{part}')


def _site_cases():
    for layout, part in SITE_LAPS:
        yield pytest.param(layout, part, id=f'{layout}-{part}')


def _judge(result, layout, bound_key=None):
    """The checks of every full lap (see the module docstring)."""
    assert result.completed, result.reason()
    assert result.delivered == 1
    assert not result.collisions, result.collisions
    assert result.nav_states == {'leader': FINAL_NAV_STATE, 'follower': FINAL_NAV_STATE}
    if layout in MAX_PATH_COMMANDS:
        assert max(result.path_commands.values()) <= MAX_PATH_COMMANDS[layout], \
            result.path_commands
    fails = lap_failures(result, front_stop=layout in ('empty', 'box1'))
    loaded = {rover: len(ids) for rover, ids in loaded_commands(result).items()}
    bound = MAX_LOADED_COMMANDS[bound_key or layout]
    if max(loaded.values()) > bound:
        fails['loaded_cmds'] = f'{loaded} against {bound}'
    assert not fails, fails
    assert_map_quality(result)


def assert_map_quality(result):
    quality = map_quality(result.mapped_log, result.true_boxes, result.shelves)
    rows = '\n'.join(quality.rows(f'{result.layout} {result.part}'))
    assert not quality.phantoms, rows
    for box in quality.boxes.values():
        if not box.pubs:
            continue                    # a box no rover looked at is not mapped: nothing to judge
        assert box.coverage_max <= MAX_COVERAGE_GAP, rows
        assert box.excess_final <= MAX_FINAL_EXCESS, rows
        assert box.pubs <= MAX_PUBLICATIONS, rows
    if len(result.true_boxes) > 1 and quality.min_gap < float('inf'):
        assert quality.min_gap >= MIN_CORRIDOR, rows


@pytest.mark.parametrize('layout,part', _cases())
def test_full_lap(layout, part, record_row):
    result = run_layout(layout, part, duration=TIME_LIMIT[layout])
    record_row(result.row())
    record_row('    ' + summary(result))
    _judge(result, layout)


@pytest.mark.parametrize('layout,part', _site_cases())
def test_box_lap_under_site_conditions(layout, part, record_row):
    """Box layout with the site profile (fused localization, UWB offset, camera occlusion)."""
    result = run_layout(layout, part, duration=TIME_LIMIT[layout], seed=1, profile=SITE_PROFILE)
    record_row('site  ' + result.row())
    record_row('    ' + summary(result))
    _judge(result, layout, bound_key=f'{layout}-site')


SLOW_PLAN = 0.6  # s; slowest planner call seen


def test_box_lap_with_slow_path_compute(record_row):
    """On the node paths are computed beside the mission tick (plan_jobs): a path is there
    some ticks after it was asked for, and the rovers drive on meanwhile. Same lap, same
    checks, with every path ``SLOW_PLAN`` late."""
    result = run_layout('box1', 'gear_01', duration=TIME_LIMIT['box1'], seed=1,
                        profile=SITE_PROFILE, plan_delay=SLOW_PLAN)
    record_row('slow plans  ' + result.row())
    record_row('    ' + summary(result))
    _judge(result, 'box1', bound_key='box1-site')
