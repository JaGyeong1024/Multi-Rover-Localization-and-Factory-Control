"""Site demo acceptance suite (scenarios/site_demo.py): one test per case, judged by the
clean-take criteria (delivered, platoon, no_stop, stop_go, contact, ends_east).

The cases run once per session, in SITE_DEMO_JOBS processes (default 6), before the first
test reads its verdict. A test passes when none of the criteria fails.

The cases are the four demo scenarios (site_demo.PRIMARY_LAPS; map frozen before dispatch,
east start, +- arm occlusion). Run:
  cd host_ws/src/host_mission && python3 -m pytest scenarios/test_site_demo.py -q
(ROS sourced, PYTHONPATH as for layout_runner)."""
import os

import pytest

from site_demo import primary_cases, run_cases

JOBS = int(os.environ.get('SITE_DEMO_JOBS', '6'))


def _cases():
    for case in primary_cases():
        yield pytest.param(case, id=case.id)


@pytest.fixture(scope='session')
def verdicts(request):
    wanted = [item.callspec.params['case'] for item in request.session.items
              if getattr(item, 'callspec', None) and 'case' in item.callspec.params]
    return {v.case: v for v in run_cases(wanted, jobs=JOBS)}


@pytest.mark.parametrize('case', _cases())
def test_site_demo(case, verdicts, record_row):
    verdict = verdicts[case.id]
    record_row(verdict.row())
    fails = verdict.failed()
    assert not fails, f'{case.id}: {fails}'
