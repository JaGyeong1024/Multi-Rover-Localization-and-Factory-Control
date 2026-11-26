"""Golden test: the planner returns exactly what it returned when the fixture was made.

Guards refactors that must not change behaviour. A fixed seeded set of scenarios
(free, boxes, rectangular footprint, platoon, pivot departures) is compared with
``golden_paths.npz``: None-ness, pose count, every pose to 1e-9, platoon flags and arc
lengths. Regenerate the fixture only for an intended behaviour change:

    python3 test/test_golden.py
"""
import math
import os
import random

import numpy as np

from checks import AREA
from host_planning import (Box, PlanParams, Pose2D, plan_departure, plan_follower_path,
                           plan_path)
from host_planning.geometry import to_array

import test_random as tr

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden_paths.npz')
TOL = 1e-9
SEED = 20261120
BODY = PlanParams(footprint=(0.13, 0.23, 0.18), pivot_step=math.radians(15))


def _cases():
    """(name, thunk) pairs; each thunk returns {key: array} for one scenario."""
    rng = random.Random(SEED)
    cases = []
    for i in range(12):
        start, goal, boxes = tr._scenario(rng, 0.3, 0)
        cases.append((f'free{i}', _path_case(start, goal, boxes, tr.PARAMS)))
    for i in range(36):
        start, goal, boxes = tr._scenario(rng, 0.3, rng.randint(2, 4))
        cases.append((f'box{i}', _path_case(start, goal, boxes, tr.PARAMS)))
    for i in range(8):
        start, goal, boxes = tr._scenario(rng, 0.4, rng.randint(1, 3))
        cases.append((f'body{i}', _path_case(start, goal, boxes, BODY)))
    for i in range(4):
        start, goal, boxes = tr._scenario(rng, 0.4, 2)
        shelf = [Box(b.x, b.y, 2.0 * b.r, 1.5 * b.r) for b in boxes]
        cases.append((f'pivot{i}', _departure_case(start, goal, shelf)))
    leader = plan_path(Pose2D(0.6, 0.6, 0.0), Pose2D(4.0, 2.0, math.pi / 2), (), AREA,
                       tr.PARAMS)
    for i in range(8):
        follower = tr._near(rng, leader[rng.randrange(0, 20)], 0.3, 0.6)
        goal = tr._near(rng, leader[rng.randrange(50, len(leader))], 0.4, 0.8)
        cases.append((f'platoon{i}', _platoon_case(follower, leader, goal)))
    return cases


def _poses(path):
    return np.zeros((0, 3)) if path is None else to_array(path)


def _path_case(start, goal, boxes, params):
    return lambda: {'path': _poses(plan_path(start, goal, boxes, AREA, params))}


def _departure_case(start, goal, boxes):
    def run():
        path, heading = plan_departure(start, goal, boxes, AREA, BODY)
        return {'path': _poses(path),
                'heading': np.array(np.nan if heading is None else heading)}
    return run


def _platoon_case(follower, leader, goal):
    def run():
        plan = plan_follower_path(follower, leader, goal, (), AREA, tr.PARAMS)
        if plan is None:
            return {'path': _poses(None)}
        arcs = [plan.join_s, plan.branch_s, plan.leader_join_s, plan.leader_branch_s]
        return {'path': _poses(plan.path), 'shared': np.array(plan.shared),
                'arcs': np.array([np.nan if a is None else a for a in arcs])}
    return run


def _results():
    out = {}
    for name, run in _cases():
        for key, value in run().items():
            out[f'{name}/{key}'] = value
    return out


def test_planner_matches_golden_fixture():
    golden = np.load(FIXTURE)
    now = _results()
    assert sorted(now) == sorted(golden.files)
    for key in golden.files:
        want, got = golden[key], now[key]
        assert want.shape == got.shape, key            # None-ness and pose count
        if want.dtype == bool:
            assert np.array_equal(want, got), key
        else:
            np.testing.assert_allclose(got, want, rtol=0.0, atol=TOL, equal_nan=True,
                                       err_msg=key)


if __name__ == '__main__':
    results = _results()
    np.savez_compressed(FIXTURE, **results)
    solved = sum(len(v) > 0 for k, v in results.items() if k.endswith('/path'))
    print(f'wrote {FIXTURE}: {len(results)} arrays, {solved} non-empty paths')
