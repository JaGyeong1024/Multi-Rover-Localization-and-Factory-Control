"""Seeded random scenarios: every returned path obeys the contract.

Prints failure rate and planning-time statistics (``pytest -s``). Random poses near
walls/boxes include infeasible cases, so the success-rate thresholds are regression
guards. Planning time is CPU time (``time.process_time``), not wall time.
"""
import math
import random
import time

import numpy as np
import pytest

from checks import AREA, assert_path_ok
from host_planning import (Circle, PlanParams, Pose2D, min_clearance,
                           plan_follower_path, plan_path)

PARAMS = PlanParams()
SEED = 20261117
RUNS = 300


def _pose(rng, margin):
    return Pose2D(rng.uniform(AREA.xmin + margin, AREA.xmax - margin),
                  rng.uniform(AREA.ymin + margin, AREA.ymax - margin),
                  rng.uniform(-math.pi, math.pi))


def _boxes(rng, count):
    return [Circle(rng.uniform(0.6, 4.2), rng.uniform(0.5, 2.2),
                   rng.uniform(0.15, 0.30)) for _ in range(count)]


def _scenario(rng, margin, box_count):
    """Start/goal inside the area whose run-in is in bounds and box-free."""
    boxes = _boxes(rng, box_count)
    while True:
        start, goal = _pose(rng, margin), _pose(rng, margin)
        entry = Pose2D(goal.x - PARAMS.approach_len * math.cos(goal.theta),
                       goal.y - PARAMS.approach_len * math.sin(goal.theta), 0.0)
        in_area = (AREA.xmin <= entry.x <= AREA.xmax
                   and AREA.ymin <= entry.y <= AREA.ymax)
        if in_area and min_clearance([start, goal, entry], boxes) >= margin:
            return start, goal, boxes


def _campaign(label, margin, with_boxes):
    rng = random.Random(SEED)
    times, solved = [], 0
    for _ in range(RUNS):
        count = rng.randint(2, 4) if with_boxes else 0
        start, goal, boxes = _scenario(rng, margin, count)
        began = time.process_time()
        path = plan_path(start, goal, boxes, AREA, PARAMS)
        times.append(time.process_time() - began)
        if path is not None:
            solved += 1
            assert_path_ok(path, start, goal, PARAMS, obstacles=boxes)
    times = np.array(times)
    rate = solved / RUNS
    print(f'\n[{label}] solved {solved}/{RUNS} ({100 * rate:.1f}%, '
          f'fail {100 * (1 - rate):.1f}%)  CPU ms: median '
          f'{1e3 * np.median(times):.1f}  p95 {1e3 * np.percentile(times, 95):.1f}'
          f'  max {1e3 * times.max():.1f}')
    return rate, times


@pytest.mark.parametrize('label,margin,with_boxes,min_rate', [
    ('free, 0.3 m margin', 0.3, False, 0.88),
    ('free, 0.5 m margin', 0.5, False, 0.93),
    ('2-4 boxes, 0.3 m margin', 0.3, True, 0.72),
    ('2-4 boxes, 0.5 m margin', 0.5, True, 0.88),
])
def test_random_scenarios(label, margin, with_boxes, min_rate):
    rate, times = _campaign(label, margin, with_boxes)
    assert rate >= min_rate
    assert np.percentile(times, 95) < 0.5


def _near(rng, pose, offset, turn):
    """Random pose around ``pose``: lateral/longitudinal offset, heading turn."""
    return Pose2D(pose.x + rng.uniform(-offset, offset),
                  pose.y + rng.uniform(-offset, offset),
                  pose.theta + rng.uniform(-turn, turn))


def test_random_platoon():
    """Follower starts near the early leader path, goal near the late part."""
    rng = random.Random(SEED)
    leader = plan_path(Pose2D(0.6, 0.6, 0.0), Pose2D(4.0, 2.0, math.pi / 2),
                       (), AREA, PARAMS)
    times, shared, solved = [], 0, 0
    for _ in range(RUNS):
        follower = _near(rng, leader[rng.randrange(0, 20)], 0.3, 0.6)
        goal = _near(rng, leader[rng.randrange(50, len(leader))], 0.4, 0.8)
        began = time.process_time()
        plan = plan_follower_path(follower, leader, goal, (), AREA, PARAMS)
        times.append(time.process_time() - began)
        if plan is None:
            continue
        solved += 1
        shared += plan.shared
        assert_path_ok(plan.path, follower, goal, PARAMS)
        if plan.shared:
            assert 0.0 <= plan.join_s < plan.branch_s
    times = np.array(times)
    print(f'\n[platoon] solved {solved}/{RUNS}, shared {shared}  CPU ms: median '
          f'{1e3 * np.median(times):.1f}  p95 {1e3 * np.percentile(times, 95):.1f}'
          f'  max {1e3 * times.max():.1f}')
    assert solved / RUNS >= 0.9
    assert shared / RUNS >= 0.5
    assert np.percentile(times, 95) < 0.5
