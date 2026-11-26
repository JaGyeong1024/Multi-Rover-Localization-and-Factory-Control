"""Exhaustive check of the two-rover yield / deadlock rules of PathCommander.

Rules under test (path_commander._blocked_only_by_peer, _lead_waits, _follower_plan):
the follower always yields to the lead; the lead waits only for a follower that will
move and can leave with the lead where it stands; a mutual box is NO_PATH for the lead.

Property, judged on a frozen world after the commander has had two plan-retry periods
to settle: a rover that waits for the other (YIELD) always has something to wait for.
At least one of these holds:
  * a rover drives (FOLLOW with a real path),
  * a rover reports NO_PATH (the mission pauses for the operator),
  * a rover has no usable pose (STALE / NO_POSE: the supervisor pauses the mission),
  * nobody waits (each rover is idle or parked at its goal).
A violation is a YIELD whose peer waits too ("mutual") or will never move: idle or
parked at its goal ("starved"). Both are silent stalls.

Part 1 enumerates the commander's decision inputs as a discrete space (a flag-driven
planner and path-conflict test replace the geometry). Part 2 samples real poses of the
two rovers in the installation layout (real planner, real geometry)."""
import dataclasses
import functools
import itertools
import math
import os
from dataclasses import dataclass

import numpy as np
import pytest
from host_planning import api as planning
from host_planning import pivot
from host_planning.geometry import obstacle_array, shape_gaps, wrap_angle
from host_planning.region import DropRegion, feasible_goals
from host_planning.types import Box, PlatoonPlan, Pose2D

from host_mission import path_commander
from host_mission.goals import GoalPose
from host_mission.path_commander import FOLLOW, Desired, PathCommander, footprint_box
from host_mission.world_model import WorldModel

import helpers

ROBOTS = ('leader', 'follower')
GO, DONE, IDLE, NO_PATH, POSE, YIELD = 'GO', 'DONE', 'IDLE', 'NO_PATH', 'POSE', 'YIELD'
SURFACED = (GO, NO_PATH, POSE)      # a rover that drives, or a stall someone is told of
STAYS = (IDLE, DONE)                # a rover that will not move by itself


def rover_status(commander, world, want, robot, now):
    """What one rover does after an update: GO / DONE / IDLE / NO_PATH / POSE / YIELD.

    A wanted rover that is stopped although its pose is usable and it is not blocked
    can only be held by a YIELD (the commander's other stop reasons are pose reasons)."""
    if not want.follow or want.goal is None:
        return IDLE
    cmd = commander.cmds.get(robot)
    if cmd is not None and cmd.mode == FOLLOW:
        return DONE if len(cmd.poses) == 1 else GO
    if commander.blocked(robot):
        return NO_PATH
    if world.usable_pose(robot, now, need_heading=False)[0] is None:
        return POSE
    return YIELD


def silent_stall(statuses):
    """None, or 'mutual' / 'starved' when a waiting rover has nothing to wait for."""
    values = list(statuses.values())
    if YIELD not in values or any(s in SURFACED for s in values):
        return None
    return 'mutual' if values.count(YIELD) == 2 else 'starved'


@dataclass(frozen=True)
class RoverCase:
    """One rover's share of the decision inputs.

    task: idle | pose (named goal) | region (load region; follower only)
    pose: ok | unaligned (provisional plan, no pivot) | stale
    at_goal: stands at its goal (the region's best pose); arrival is reported
    free: a path exists with the other rover ignored (False: boxes / shelves block it)
    by_peer_now / by_peer_goal: the other rover's footprint blocks every path while it
        stands where it is now / at the goal it drives to
    pivot_only: no forward path from the current heading; a departure pivot works
        (a provisional plan fails too: its guessed heading is not a pivot)
    shared: the follower's plan can ride on the lead path (else stand-alone)
    """

    task: str = 'idle'
    pose: str = 'ok'
    at_goal: bool = False
    free: bool = True
    by_peer_now: bool = False
    by_peer_goal: bool = False
    pivot_only: bool = False
    shared: bool = False


@dataclass(frozen=True)
class Case:
    """Both rovers' inputs plus what the commander remembers from earlier ticks.

    cross: the two paths come within conflict_dist (paths_conflict, both directions)
    history: fresh | moving (both got paths, follower riding on the lead) |
        lead_boxed (the lead waited and the follower left first, stand-alone)
    replan: '' | leader | follower — a re-entry requested before the first tick
    """

    leader: RoverCase
    follower: RoverCase
    cross: bool
    history: str
    replan: str


HISTORIES = ('fresh', 'moving', 'lead_boxed')
REPLANS = ('', 'leader', 'follower')
LEADER_CASES = 38       # idle 1 + stale 1 + {ok, unaligned} x at_goal x (blocked 1 + 2^3)
FOLLOWER_CASES = 139    # idle 1 + {pose, region} x (stale 1 + 4 x (blocked 1 + 2^4))
SPACE_SIZE = LEADER_CASES * FOLLOWER_CASES * 2 * len(HISTORIES) * len(REPLANS)

CLEAR = RoverCase(task='pose', shared=True)
PIVOT_STEP = helpers.settings().robots['leader'].plan.pivot_step
# Abstract layout: two lanes 1.1 m apart, far from each other and from the bounds, so
# the flags alone decide. Start / goal headings and the bearing to the goal (the
# provisional heading) are no multiple of the pivot step: the flag planner tells a
# pivot candidate by its heading.
START = {'leader': (2.5, 2.9, 0.3), 'follower': (2.5, 4.0, 0.3)}
GOAL = {'leader': GoalPose(4.5, 2.95, 0.05), 'follower': GoalPose(4.5, 4.05, 0.05)}
CARGO = (0.0, 0.0, 0.20, 0.20)


def _rover_cases(tasks, shared_values):
    yield RoverCase()
    for task in tasks:
        yield RoverCase(task, 'stale')
        for pose, at_goal in itertools.product(('ok', 'unaligned'), (False, True)):
            yield RoverCase(task, pose, at_goal, free=False)
            for now, goal, pivot_only, shared in itertools.product(
                    (False, True), (False, True), (False, True), shared_values):
                yield RoverCase(task, pose, at_goal, True, now, goal, pivot_only, shared)


def all_cases():
    leaders = list(_rover_cases(('pose',), (False,)))
    followers = list(_rover_cases(('pose', 'region'), (False, True)))
    for history, replan, lead, follow, cross in itertools.product(
            HISTORIES, REPLANS, leaders, followers, (False, True)):
        yield Case(lead, follow, cross, history, replan)


def _is_pivot_heading(theta):
    return abs(float(wrap_angle(theta - round(theta / PIVOT_STEP) * PIVOT_STEP))) < 1e-9


class FlagPlanner:
    """Planner stand-in answering from the RoverCase flags of the current case.

    The planning rover is told by its footprint (params); the other rover's footprint
    box among the obstacles by its size, and whether it stands at that rover's current
    pose or at its goal by its centre."""

    def __init__(self, settings):
        self.settings = settings
        self.flags = {}
        self.cross = False
        self.poses = {}

    def _robot(self, params):
        return next(r for r in ROBOTS
                    if self.settings.robots[r].plan.footprint == params.footprint)

    def _peer_blocks(self, robot, obstacles):
        peer = ROBOTS[1 - ROBOTS.index(robot)]
        here = footprint_box(self.poses[peer], self.settings.robots[peer].plan.footprint)
        size = (round(here.width, 6), round(here.height, 6))
        flags = self.flags[robot]
        for o in obstacles:
            if isinstance(o, Box) and (round(o.width, 6), round(o.height, 6)) == size:
                now = math.hypot(o.x - here.x, o.y - here.y) < 1e-6
                if flags.by_peer_now if now else flags.by_peer_goal:
                    return True
        return False

    def plan_path(self, start, goal, obstacles, bounds=None, params=None):
        robot = self._robot(params)
        flags = self.flags[robot]
        if not flags.free or self._peer_blocks(robot, obstacles):
            return None
        if flags.pivot_only and not _is_pivot_heading(start.theta):
            return None
        count = 4                   # only the end poses are ever looked at
        return [Pose2D(start.x + (goal.x - start.x) * k / count,
                       start.y + (goal.y - start.y) * k / count,
                       start.theta if k == 0 else goal.theta) for k in range(count + 1)]

    def plan_follower_path(self, start, lead_path, goal, obstacles, bounds=None,
                           params=None):
        path = self.plan_path(start, goal, obstacles, bounds, params)
        return None if path is None else PlatoonPlan(path, self.flags['follower'].shared)

    def paths_conflict(self, path, lead_path, min_dist):
        return len(path) >= 2 and bool(lead_path) and self.cross


@functools.lru_cache(maxsize=None)
def _region_poses(region):
    return tuple(region.candidates())


def _desired(rover, robot):
    if rover.task == 'idle':
        return Desired()
    goal = GOAL[robot]
    region = DropRegion((goal.x, goal.y), goal.theta, CARGO) if rover.task == 'region' \
        else None
    return Desired(robot, goal, True, region=region)


def _abstract_pose(robot, rover):
    goal = GOAL[robot]
    return (goal.x, goal.y, goal.theta) if rover.at_goal else START[robot]


class AbstractRun:
    """One case: a fresh commander driven through its history, then the settle ticks
    with the world frozen (nobody moves: the worst case for a wait)."""

    TICKS = (0.2, 2.3, 4.4)         # further apart than plan_retry_period (2.0 s)

    def __init__(self, settings, planner):
        self.settings, self.planner = settings, planner

    def statuses(self, case):
        """Statuses of both rovers after each settle tick."""
        rovers = {'leader': case.leader, 'follower': case.follower}
        self.planner.poses = {r: _abstract_pose(r, rovers[r]) for r in ROBOTS}
        world = WorldModel(pose_timeout=0.5)
        commander = PathCommander(self.settings, helpers.EPOCH, self.planner)
        self._history(case.history, commander, world, rovers)
        self.planner.flags, self.planner.cross = rovers, case.cross
        if case.replan:
            commander.request_replan(case.replan, reentry=True)
        want = {r: _desired(rovers[r], r) for r in ROBOTS}
        arrived = {r: rovers[r].at_goal for r in ROBOTS}
        seen = []
        for now in self.TICKS:
            self._place(world, rovers, now)
            commander.update(want, world, arrived, now)
            seen.append({r: rover_status(commander, world, want[r], r, now) for r in ROBOTS})
        return seen

    def _history(self, history, commander, world, rovers):
        """One earlier tick with every path free; both rovers wanted at their goals."""
        if history == 'fresh':
            return
        boxed = history == 'lead_boxed'     # lead boxed in by the follower: it waits
        self.planner.flags = {'leader': dataclasses.replace(CLEAR, by_peer_now=boxed),
                              'follower': dataclasses.replace(CLEAR, shared=not boxed)}
        self.planner.cross = False
        wanted = {r: dataclasses.replace(rovers[r], task=rovers[r].task.replace(
            'idle', 'pose')) for r in ROBOTS}
        self._place(world, rovers, 0.0)
        commander.update({r: _desired(wanted[r], r) for r in ROBOTS}, world,
                         {r: False for r in ROBOTS}, 0.0)

    @staticmethod
    def _place(world, rovers, now):
        for robot, rover in rovers.items():
            stamp = now - 1.0 if rover.pose == 'stale' else now
            helpers.place(world, robot, *_abstract_pose(robot, rover), stamp,
                          aligned=rover.pose != 'unaligned')


@pytest.fixture(scope='module')
def abstract_results():
    """case -> statuses per settle tick, for the whole space (computed once, ~30 s).

    Geometry is not the subject here: the flag planner decides every plan, so the
    pivot sweep check and the region validator are replaced by permissive stand-ins
    (the abstract lanes are clear anyway; this only saves time)."""
    settings = helpers.settings()
    planner = FlagPlanner(settings)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(path_commander, 'paths_conflict', planner.paths_conflict)
        patch.setattr(path_commander, 'feasible_goals',
                      lambda region, *args: list(_region_poses(region)))
        patch.setattr(pivot, 'pivot_clear', lambda *args: True)
        run = AbstractRun(settings, planner)
        return {case: run.statuses(case) for case in all_cases()}


def _final_stalls(results):
    return {case: seen[-1] for case, seen in results.items() if silent_stall(seen[-1])}


def test_abstract_space_whole_all_outcomes(abstract_results):
    assert len(abstract_results) == SPACE_SIZE == 95076
    for robot in ROBOTS:
        final = {seen[-1][robot] for seen in abstract_results.values()}
        assert {GO, DONE, IDLE, NO_PATH, POSE, YIELD} <= final, robot


def test_abstract_rovers_never_wait_each_other(abstract_results):
    mutual = [case for case, statuses in _final_stalls(abstract_results).items()
              if silent_stall(statuses) == 'mutual']
    assert not mutual, mutual[:3]


def test_abstract_mutual_waits_clear_in_one_retry(abstract_results):
    """Both rovers may YIELD on the first tick (each decided on the other's last
    command); one plan-retry period later the lead re-decides and drives or reports
    NO_PATH."""
    late = [case for case, seen in abstract_results.items()
            if any(silent_stall(statuses) == 'mutual' for statuses in seen[1:])]
    assert not late, late[:3]


def test_abstract_space_has_no_silent_stall(abstract_results):
    stalls = _final_stalls(abstract_results)
    assert not stalls, f'{len(stalls)} silent stalls, e.g. {next(iter(stalls.items()))}'


# Part 2: real layout, real planner
# Installation layout: standby / unload side by side at the west wall facing it, two
# 0.4 m shelves on the east side.
SHELVES = (Box(5.386, 2.693, 0.40, 0.40), Box(5.40, 4.15, 0.40, 0.40))
LAYOUT_GOALS = {
    'leader_standby': GoalPose(1.82, 3.29, math.pi),
    'follower_unload': GoalPose(1.82, 3.79, math.pi),
    'pick_gear': GoalPose(4.89, 2.69, 0.0),
    'pick_bolt': GoalPose(4.90, 4.15, 0.0),
}
LAYOUT = dataclasses.replace(helpers.settings(), fixtures=SHELVES)
SWEEP_SEED = 20261118
# Each sample costs ~1-3 s of real planning, so the default keeps the file near a minute.
# More: YIELD_SWEEP_SAMPLES=200.
SWEEP_SAMPLES = int(os.environ.get('YIELD_SWEEP_SAMPLES', '12'))
GRID = 0.3          # m
HEADINGS = (0.0, math.pi / 2, math.pi, -math.pi / 2)
NEAR = 0.9          # m: farthest offset of the second rover (beyond it nobody blocks)


def _load_region(pick_key):
    """The follower's load region of a pick goal (as mission_engine._load_region)."""
    pick = LAYOUT_GOALS[pick_key]
    dx, dy = LAYOUT.pick.drop_point
    c, s = math.cos(pick.theta), math.sin(pick.theta)
    drop = (pick.x + c * dx - s * dy, pick.y + s * dx + c * dy)
    return DropRegion(drop, pick.theta, LAYOUT.robots['follower'].cargo,
                      LAYOUT.pick.drop_margin)


def _go(key):
    return Desired(key, LAYOUT_GOALS[key], True)


def _go_load(pick_key):
    region = _load_region(pick_key)
    return Desired(f'load_{pick_key}', GoalPose(*region.drop, region.heading), True,
                   region=region)


PHASES = {      # mission phases: what each rover is wanted at
    'to_gear': {'leader': _go('pick_gear'), 'follower': _go_load('pick_gear')},
    'to_bolt': {'leader': _go('pick_bolt'), 'follower': _go_load('pick_bolt')},
    'home': {'leader': _go('leader_standby'), 'follower': _go('follower_unload')},
    'follower_home': {'leader': Desired(), 'follower': _go('follower_unload')},
    'leader_home': {'leader': _go('leader_standby'), 'follower': Desired()},
}


class MemoPlanner:
    """The real planner, memoized: the settle ticks repeat the same plans."""

    def __init__(self):
        self._memo = {}

    def _cached(self, key, plan):
        if key not in self._memo:
            self._memo[key] = plan()
        return self._memo[key]

    def plan_path(self, start, goal, obstacles, bounds=None, params=None):
        obstacles = tuple(obstacles)
        return self._cached(('path', start, goal, obstacles, bounds, params),
                            lambda: planning.plan_path(start, goal, obstacles, bounds, params))

    def plan_follower_path(self, start, lead_path, goal, obstacles, bounds=None,
                           params=None):
        lead_path, obstacles = tuple(lead_path), tuple(obstacles)
        return self._cached(
            ('follow', start, lead_path, goal, obstacles, bounds, params),
            lambda: planning.plan_follower_path(start, list(lead_path), goal, obstacles,
                                                bounds, params))


PLANNER = MemoPlanner()


def _arrived(robot, pose, want):
    if not want.follow or want.goal is None:
        return False
    cfg, here = LAYOUT.robots[robot], Pose2D(*pose)
    if want.region is not None:
        return want.region.holds(here, cfg.goal_yaw_tol)
    goal = want.goal
    turn = math.atan2(math.sin(here.theta - goal.theta), math.cos(here.theta - goal.theta))
    return math.hypot(here.x - goal.x, here.y - goal.y) <= cfg.arrive_pos_tol \
        and abs(turn) <= cfg.arrive_yaw_tol


def layout_statuses(poses, phase):
    """Statuses after each tick of a fresh commander, world frozen; stops early once
    neither rover waits or is held (a frozen world then never changes)."""
    want = PHASES[phase]
    world = WorldModel(pose_timeout=0.5)
    commander = PathCommander(LAYOUT, helpers.EPOCH, PLANNER)
    arrived = {r: _arrived(r, poses[r], want[r]) for r in ROBOTS}
    seen = []
    for now in (0.0, 2.1, 4.2):
        for robot in ROBOTS:
            helpers.place(world, robot, *poses[robot], now)
        commander.update(want, world, arrived, now)
        seen.append({r: rover_status(commander, world, want[r], r, now) for r in ROBOTS})
        if not set(seen[-1].values()) & {YIELD, NO_PATH, POSE}:
            break
    return seen


def _bodies_clear(poses):
    """Both footprints keep off the shelves and each other (a physical state)."""
    lead, follow = (footprint_box(poses[r], LAYOUT.robots[r].plan.footprint) for r in ROBOTS)
    gaps = shape_gaps(obstacle_array([lead, follow]), obstacle_array(list(SHELVES) + [follow]))
    return bool(np.all(gaps[:, :len(SHELVES)] > 0.0)) and gaps[0, len(SHELVES)] > 0.0


def _inside(pose):
    b = LAYOUT.bounds
    return b.xmin <= pose[0] <= b.xmax and b.ymin <= pose[1] <= b.ymax


def _offsets():
    steps = range(-3, 4)
    return [(i * GRID, j * GRID, h) for i, j, h in itertools.product(steps, steps, HEADINGS)
            if 0.0 < math.hypot(i, j) * GRID <= NEAR]


def _pairs(anchors):
    """Both role assignments of every anchor pose with a near second pose."""
    for (x, y, yaw), (dx, dy, h) in itertools.product(anchors, _offsets()):
        other = (x + dx, y + dy, h)
        yield {'leader': (x, y, yaw), 'follower': other}
        yield {'leader': other, 'follower': (x, y, yaw)}


def sweep_samples(count, seed=SWEEP_SEED):
    """Seeded sample of near rover pairs: half on the 0.3 m x 4 heading grid, half with
    one rover parked at a layout goal (where the mission leaves it standing)."""
    b = LAYOUT.bounds
    grid = [(float(x), float(y), h) for x, y, h in itertools.product(
        np.arange(b.xmin + 0.15, b.xmax, GRID), np.arange(b.ymin + 0.15, b.ymax, GRID),
        HEADINGS)]
    parked = [(g.x, g.y, g.theta) for g in LAYOUT_GOALS.values()]
    params = LAYOUT.robots['follower'].plan       # best load pose (the gear drop point
    for key in ('pick_gear', 'pick_bolt'):        # itself lies outside the bounds)
        load = feasible_goals(_load_region(key), SHELVES, b, params)[0]
        parked.append((load.x, load.y, load.theta))
    rng = np.random.default_rng(seed)
    phases = sorted(PHASES)
    out = []
    for anchors, share in ((grid, count - count // 2), (parked, count // 2)):
        pool = [p for p in _pairs(anchors) if _inside(p['leader']) and _inside(p['follower'])]
        picked = 0
        for k in rng.permutation(len(pool)):
            if picked == share:
                break
            if _bodies_clear(pool[k]):
                out.append((pool[k], phases[rng.integers(len(phases))]))
                picked += 1
    return out


@pytest.fixture(scope='module')
def layout_results():
    return [(poses, phase, layout_statuses(poses, phase))
            for poses, phase in sweep_samples(SWEEP_SAMPLES)]


def test_layout_sample_has_no_silent_stall(layout_results):
    assert len(layout_results) == SWEEP_SAMPLES
    for poses, phase, seen in layout_results:
        statuses = seen[-1]
        assert silent_stall(statuses) is None, (poses, phase, seen)


# The counterexample, reproduced in the real layout. Each: the follower is wanted
# somewhere, every path of it passes the leader, and the leader stays where it is. The
# follower YIELDs through every retry; with the leader parked at its pick goal the
# mission itself waits for the follower at its load pose, so the two layers wait on each other.
REPROS = {
    # follower coming home 0.45 m behind the leader parked at standby
    'home_behind_parked_lead': (
        {'leader': (1.82, 3.29, math.pi), 'follower': (2.27, 3.44, math.pi)}, 'home'),
    # follower 0.15 m short of its load pose, leader parked at the gear pick pose
    'load_beside_parked_lead': (
        {'leader': (4.89, 2.69, 0.0), 'follower': (4.74, 2.24, 0.0)}, 'to_gear'),
    # leader without a task in the south-west corner, follower wanted home
    'home_past_idle_lead': (
        {'leader': (1.59, 2.62, 0.0), 'follower': (1.89, 2.32, math.pi)}, 'follower_home'),
}


@pytest.fixture(scope='module')
def repro_results():
    return {name: layout_statuses(*repro) for name, repro in REPROS.items()}


def test_follower_blocked_by_staying_lead_surfaced(repro_results):
    stalled = {name: seen[-1] for name, seen in repro_results.items()
               if silent_stall(seen[-1])}
    assert not stalled, stalled
