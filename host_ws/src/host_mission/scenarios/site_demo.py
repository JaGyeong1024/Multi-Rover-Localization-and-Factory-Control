"""Site demo acceptance suite: one full lap per case, judged by the clean-take criteria.

A case is one lap of layout_runner's LayoutLoop (production mission.yaml, real planner,
supervisor, mapper and rover stacks in host_sim) with demo-site conditions set on top:

  heading    both rovers start at home facing east (0) or west (pi)
  map        frozen = the follower LiDAR maps for PREMAP_S with both rovers standing, then
             scans count only while the mission is IDLE; premap = same, map stays live;
             empty = dispatch at t = 0 on an empty map
  occlusion  the arm and the part cover the follower during the pick (ArmOcclusion)
  loc        held (default) = zero-motion hold of the host localization (HeldLocalization);
             ideal = the sensor model alone
  bounds / drop_y / home_y   overrides of mission bounds, pick.drop_point lateral and
             the home goal y

Criteria (``judge``), each pass / fail with its numbers:
  delivered   mission done, parts delivered, both rovers PARKED within the goal tolerances
  platoon     every leg is ridden behind the leader (``platoon_legs``)
  no_stop     no paused / plan_failed / hold / halt / fault event
  stop_go     no rover stands still on the way for more than STAND_LIMIT s outside the
              designed waits (``standing_report``)
  contact     planner footprints never overlap each other or a box
  ends_east   both rovers end facing east

Command line (from host_mission, ROS + PYTHONPATH as for layout_runner):
  python3 scenarios/site_demo.py [--fast] [--jobs 6] [-k SUBSTRING] [--events]
prints one row per case. pytest: scenarios/test_site_demo.py."""
import argparse
import contextlib
import dataclasses
import math
import multiprocessing
import pickle
import re
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

import layout_runner as lr
from host_sim.scenario import run_scenario
from lap_checks import final_goal_errors, footprint_gaps, lead_excess
from sim_harness import FakePicker

from host_mission.obstacle_mapper import (MapperParams, ObstacleMapper, Region, ScanSource,
                                          integrates_scans)
from host_mission.obstacle_mapper_node import parse_zones

PREMAP_S = 5.0          # s standing at home before the dispatch (map modes frozen / premap)
LAP_S = 300.0           # s of sim time allowed after the dispatch (one part)
PICK_SECONDS = 48.4  # s FakePicker pick time
START_LAG = 2.0         # s, outbound: follower's first path at most this after the leader's
STAND_LIMIT = 3.0       # s, longest stand-still on the way outside the designed waits
STAND_MIN = 0.5         # s, shorter stand-stills are not counted
DOCK_GRACE = 3.0        # s after the leader parks: the follower's dock path is designed
STILL_SPEED = 0.01      # m/s, true speed below this ...
STILL_TURN = 0.05       # rad/s, ... and yaw rate below this = standing still
EAST_TOL = 0.15         # rad, 'facing east' (= arrive_yaw_tol)
HOLD_NOISE = 0.002      # m, largest noise on a held (standing) pose
SITE_NOISE = 0.011  # m per axis: parked leader, x std
SITE_LATENCY = 0.18  # s: camera + fusion lag
# Body origin x where the idle turn leaves each rover at home facing east (sim):
# the leader tracked point is 0.05 m ahead of its body origin; the follower tracked
# point is its body origin. The sim leader turns about a point slightly behind its
# tracked point, so it ends a few cm east of its goal.
EAST_START_X = {'leader': 0.92, 'follower': 0.74}
IDLE_TURN_LIMIT = 60.0  # s after 'done' the lap waits for the idle turns
# Production state the suite models (--step): 1 = held localization only (idle turn off);
# 2 = + production mapper (values, LiDAR mount, map frozen while a mission runs);
# 3 = everything as on disk.
STEP = 3
PRODUCTION_LIDAR_MOUNT_X = 0.10  # m, config/hardware/sensor_mounts.yaml follower_lidar mount_x
STUCK_TOTAL = 60.0  # s in all with the mission PAUSED / FAULT / ABORTED: the lap is given up
MISSION_STOPS = ('paused', 'plan_failed', 'hold', 'halt', 'fault', 'aborted', 'phase_timeout')
CRITERIA = ('delivered', 'platoon', 'no_stop', 'stop_go', 'contact', 'ends_east')
TAG_BOUNDS = (0.66, 4.78, 0.66, 2.78)  # AprilTag outer edge, vs mission

_PATH_ISSUED = re.compile(r'(\w+) -> (\S+) \(cmd (\d+),')


# ---------------------------------------------------------------- arm occlusion
# Camera gaps of the follower during a recorded pick, s from the start of the first gap
# (a camera gap turns LOST after 1 s, so each gap starts 1 s before its LOST interval).
REAL_GAPS = ((0.0, 8.6), (11.0, 12.1), (12.4, 14.4), (16.1, 18.8), (21.4, 22.5),
             (23.7, 25.1), (27.5, 30.2), (30.5, 31.8), (35.9, 37.1), (38.6, 39.9))


@dataclass(frozen=True)
class OcclusionParams:
    after_pick_start: float = 34.1  # s, first gap after the pick began
    lost_after: float = 1.0         # s of camera gap before the quality reads LOST
    bias_toward_leader: float = 0.18  # m, UWB position error while LOST (+ = towards the leader); assumed
    release_dist: float = 0.5       # m the leader moves away before the gaps end
    gaps: Tuple[Tuple[float, float], ...] = REAL_GAPS


@dataclass
class _Episode:
    t0: float                       # first gap begins (sim time)
    anchor: Tuple[float, float]     # leader's tracked point then
    bias: Tuple[float, float]       # UWB position error while LOST
    released_at: Optional[float] = None   # s after t0 the leader was release_dist away


class ArmOcclusion:
    """Replays the camera gaps of the follower once per pick (ideal localization: the
    quality is forced LOST and the pose carries the UWB bias while a gap is older than
    ``lost_after``). No new gap starts once the leader stands ``release_dist`` from where
    it was when the first gap began; a gap already begun runs to its end."""

    def __init__(self, loop, params=OcclusionParams(), hold=None):
        self.loop, self.params, self.hold = loop, params, hold
        self.episodes = []
        self.lost = []              # [[t0, t1]] LOST windows applied (sim time)

    def step(self, t):
        picks = self.loop.picker.starts
        if len(self.episodes) < len(picks) and \
                t >= picks[len(self.episodes)] + self.params.after_pick_start:
            self.episodes.append(self._begin(t))
        lost = [e for e in self.episodes if self._lost_in(e, t)]
        if self.hold is not None and self.hold.standing('follower'):
            lost = []               # held model: a standing rover stays GOOD through a gap
        fault = self.loop.world.faults['follower']
        if lost and fault.quality != 'LOST':
            self.lost.append([t, None])
        if not lost and fault.quality == 'LOST':
            self.lost[-1][1] = t
        fault.quality = 'LOST' if lost else None
        fault.jump_offset = lost[-1].bias if lost else (0.0, 0.0)

    def _begin(self, t):
        robots = self.loop.world.robots
        lx, ly, _ = robots['leader'].tracked_point()
        fx, fy, _ = robots['follower'].tracked_point()
        d = math.hypot(lx - fx, ly - fy) or 1.0
        size = self.params.bias_toward_leader
        return _Episode(t, (lx, ly), ((lx - fx) / d * size, (ly - fy) / d * size))

    def _lost_in(self, episode, t):
        rel = t - episode.t0
        if episode.released_at is None:
            lx, ly, _ = self.loop.world.robots['leader'].tracked_point()
            if math.hypot(lx - episode.anchor[0], ly - episode.anchor[1]) >= \
                    self.params.release_dist:
                episode.released_at = rel
        limit = math.inf if episode.released_at is None else episode.released_at
        return any(a + self.params.lost_after <= rel < b and a <= limit
                   for a, b in self.params.gaps)


class HeldLocalization:
    """Zero-motion hold of the host localization: while a rover's wheels stand still its
    reported pose is frozen at the pose it had when it stopped, plus at most HOLD_NOISE of
    noise, and (ArmOcclusion) its quality stays GOOD through camera gaps. Once it moves
    the sensor's own output returns. The frozen value is the true tracked point at the
    moment of stopping (the EKF averages camera jitter out at rest)."""

    def __init__(self, world, rng, shift=None):
        self.world, self.rng = world, rng
        self.shift = shift or (lambda name: (0.0, 0.0))   # camera offset at the stop (ArmShift)
        self.held = {}
        for name, sensor in world.sensors.items():
            sensor.poll = self._held_poll(name, sensor.poll)

    def standing(self, name):
        robot = self.world.robots[name]
        return abs(robot.v) < STILL_SPEED and abs(robot.w) < STILL_TURN

    def _held_poll(self, name, poll):
        def held_poll(t, robot, fault=None, others=()):
            sample = poll(t, robot, fault, others)
            if not self.standing(name):
                self.held.pop(name, None)
                return sample
            if name not in self.held:  # the filter holds what it believed: truth + offset
                (x, y, theta), (ox, oy) = robot.tracked_point(), self.shift(name)
                self.held[name] = (x + ox, y + oy, theta)
            x, y, theta = self.held[name]
            if sample is None:
                return None
            dx, dy = np.clip(self.rng.normal(0.0, HOLD_NOISE / 2.0, 2), -HOLD_NOISE, HOLD_NOISE)
            return dataclasses.replace(
                sample, x=x + dx, y=y + dy,
                theta=theta if sample.heading_aligned else sample.theta)
        return held_poll


CAMERA_NADIR = (2.417, 2.110)  # m, floor point under the ceiling camera


class ArmShift:
    """The leader's camera position moved by its own arm: after a pick the box centre sits
    0.01-0.11 m off, and the arm rests in a stow pose, so the offset stays for the laps
    that follow. From the first pick on the leader's reported position carries it:

      ('radial', k)      k * (position - CAMERA_NADIR): height parallax, grows with the
                         distance from the point under the camera (assumed shape)
      ('const', dx, dy)  the same world offset everywhere

    A standing rover is not moved by it (HeldLocalization): the pose steps by the offset
    when the leader drives off, and the hold of the next stop keeps the shifted position."""

    def __init__(self, loop, spec):
        self.loop, self.spec = loop, spec

    def offset(self, name='leader'):
        if name != 'leader' or not self.loop.picker.starts:
            return (0.0, 0.0)
        if self.spec[0] == 'const':
            return (self.spec[1], self.spec[2])
        x, y, _ = self.loop.world.robots['leader'].tracked_point()
        return (self.spec[1] * (x - CAMERA_NADIR[0]), self.spec[1] * (y - CAMERA_NADIR[1]))

    def step(self):
        self.loop.world.faults['leader'].jump_offset = self.offset()


class TimedPicker(FakePicker):
    """FakePicker that keeps the start time of every pick (the occlusion trigger)."""

    def __init__(self, engine, duration):
        super().__init__(engine, duration=duration)
        self.starts = []

    def step(self, nav_state, now):
        before = self._started
        super().step(nav_state, now)
        if before is None and self._started is not None:
            self.starts.append(now)


# ---------------------------------------------------------------- cases
@dataclass(frozen=True)
class Case:
    layout: str  # site | site_two (layouts.yaml)
    part: str                           # gear_01 | bolt_01 | gear_01+bolt_01
    heading: str = 'east'               # east | west: both rovers at home
    map: str = 'frozen'  # frozen | premap | empty (module docstring)
    occlusion: bool = False
    bounds: Optional[Tuple[float, float, float, float]] = None   # xmin, xmax, ymin, ymax
    drop_y: Optional[float] = None  # pick.drop_point lateral
    home_y: Optional[Tuple[float, float]] = None   # leader_standby y, follower_unload y
    seed: int = 1
    bias: Optional[float] = None        # occlusion bias override (diagnosis)
    loc: str = 'held'  # held (zero-motion hold) | ideal
    box_shift: Optional[Tuple[str, float, float]] = None  # (box name, dx, dy) robustness
    box_at: Tuple[Tuple[str, float, float], ...] = ()   # (box name, x, y): a placement
    lead_clearance: Optional[float] = None   # leader plan.clearance override
    # Robustness variations, (key, value) pairs; ``label`` names them in the id:
    #   start      ((robot, dx, dy, dtheta), ...) hand placement at home (body origin)
    #   boxes      ((name, x, y, yaw, width, height), ...) box placement and size
    #   pick_s     pick time (s);  occl  'early' (gaps from the pick start) | 'late' (no release)
    # loc_site   site-like localization: 0.011 m noise, 0.18 s latency (+ the hold)
    # lead_ref   sim leader's tracked point ahead of its rotation centre (m)
    # park_bias  ((robot, dx, dy, dtheta), ...) the rover's own pose offset: it parks that
    # much off while the host sees the truth
    # arm_shift  ('radial', k) | ('const', dx, dy): leader's camera offset (ArmShift)
    var: Tuple[Tuple[str, object], ...] = ()
    label: str = ''

    @property
    def id(self):
        parts = [self.layout, self.part.replace('_01', '').replace('+', '_').replace('>', '_then_')
                 .replace('|', '_again_'),
                 self.heading,
                 self.map]
        if self.occlusion:
            parts.append('occl' if self.bias is None else f'occl{self.bias:+.2f}')
        if self.bounds:
            parts.append('tagbounds' if self.bounds == TAG_BOUNDS else f'bounds{self.bounds}')
        if self.drop_y is not None:
            parts.append(f'drop{abs(self.drop_y):.2f}')
        if self.home_y:
            parts.append(f'home{self.home_y[0]:.2f}-{self.home_y[1]:.2f}')
        if self.loc != 'held':
            parts.append(self.loc)
        if self.box_shift:
            name, dx, dy = self.box_shift
            parts.append(f'{name}{dx:+.2f},{dy:+.2f}')
        if self.lead_clearance is not None:
            parts.append(f'lclr{self.lead_clearance:.2f}')
        parts += [f'{name}@{x:.2f},{y:.2f}' for name, x, y in self.box_at]
        if self.label:
            parts.append(self.label)
        if self.seed != 1:
            parts.append(f's{self.seed}')
        return '-'.join(parts)


# The four demo scenarios: 1 site gear; 2 bolt, then gear as a second trip queued at the
# bolt's 'loaded'; 3 two boxes, bolt; 4 boxes moved, gear.
PRIMARY_LAPS = (('site', 'gear_01'), ('site', 'bolt_01>gear_01'), ('site_s3', 'bolt_01'),
                ('site_s4', 'gear_01'))


def primary_cases():
    """The demo scenarios (map frozen before dispatch, east start), +- arm occlusion."""
    return [Case(layout, part, occlusion=occ) for occ in (False, True)
            for layout, part in PRIMARY_LAPS]


def secondary_cases():
    """Evidence: the site_two layout (its north box blocks the bolt approach), one trip | gear+bolt, single parts, map / heading / override variants."""
    out = [Case('site_two', 'gear_01'), Case('site_two', 'bolt_01'),
           Case('site', 'bolt_01'), Case('site', 'gear_01+bolt_01')]
    laps = (('site', 'gear_01'), ('site', 'bolt_01'), ('site_two', 'gear_01'))
    out += [Case(layout, part, map=mode) for layout, part in laps for mode in ('empty', 'premap')]
    out += [Case(layout, part, heading='west') for layout, part in laps]
    return out


def all_cases():
    return primary_cases() + secondary_cases()


def robustness_cases():
    """site_two gear with one box moved +-0.05 / +-0.10 m in x or y, and the leader's clearance 0.03."""
    shifts = [(d, 0.0) for d in (-0.10, -0.05, 0.05, 0.10)] + \
        [(0.0, d) for d in (-0.10, -0.05, 0.05, 0.10)]
    out = [Case('site_two', 'gear_01', box_shift=(box, dx, dy))
           for box in ('box_n', 'box_s') for dx, dy in shifts]
    return out + [Case('site_two', 'gear_01', lead_clearance=0.03),
                  Case('site', 'gear_01', lead_clearance=0.03)]


def arm_shift_cases():
    """The demo scenarios with the leader's camera position moved by its arm: the largest
    offset after a pick (0.11 m at the shelf), outward and inward, a typical one (0.05 m),
    and 0.06 m sideways."""
    specs = (('out11', ('radial', 0.054)), ('in11', ('radial', -0.054)),
             ('out05', ('radial', 0.025)), ('in05', ('radial', -0.025)),
             ('north06', ('const', 0.0, 0.06)), ('south06', ('const', 0.0, -0.06)))
    return [Case(layout, part, var=(('arm_shift', spec),), label=f'arm_{name}')
            for name, spec in specs for layout, part in PRIMARY_LAPS]


@contextlib.contextmanager
def _patched(module, **values):
    saved = {name: getattr(module, name) for name in values}
    for name, value in values.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def _settings_loader(case):
    def load(fixtures):
        raw = lr.production_settings_raw()
        layout_names = {f['name'] for f in fixtures}
        raw['fixtures'] = fixtures + [f for f in raw.get('fixtures') or []
                                      if f['name'] not in layout_names]
        if case.bounds:
            raw['bounds'] = dict(zip(('xmin', 'xmax', 'ymin', 'ymax'), case.bounds))
        if case.drop_y is not None:
            raw['pick']['drop_point'][1] = case.drop_y
        if STEP < 3:
            raw['idle_heading'] = None
        if case.lead_clearance is not None:
            raw['robots']['leader']['plan']['clearance'] = case.lead_clearance
        return lr.settings_from_dict(raw)
    return load


def _layout_loader(case, original):
    def load(name):
        config, fixtures = original(name)
        if STEP >= 2:  # sim LiDAR where production places its scans (mount_x)
            config['robots']['follower']['lidar']['mount_x'] = PRODUCTION_LIDAR_MOUNT_X
        if case.box_shift:
            name, dx, dy = case.box_shift
            box = next(o for o in config['obstacles'] if o.get('name') == name)
            box['x'], box['y'] = box['x'] + dx, box['y'] + dy
        for name, x, y in case.box_at:
            box = next(o for o in config['obstacles'] if o.get('name') == name)
            box['x'], box['y'] = x, y
        for name, x, y, yaw, width, height in _var(case, 'boxes', ()):
            box = next(o for o in config['obstacles'] if o.get('name') == name)
            box.update(x=x, y=y, yaw=yaw, width=width, height=height)
        heading = 0.0 if case.heading == 'east' else math.pi
        for robot, start in config['robots'].items():
            start['start'][2] = heading
            if case.heading == 'east' and STEP >= 3:
                start['start'][0] = EAST_START_X[robot]
            if case.home_y:
                start['start'][1] = case.home_y[0 if robot == 'leader' else 1]
        for robot, dx, dy, dth in _var(case, 'start', ()):
            start = config['robots'][robot]['start']
            start[0], start[1], start[2] = start[0] + dx, start[1] + dy, start[2] + dth
        if _var(case, 'lead_ref', None) is not None:  # the leader's rotation centre
            config['robots']['leader']['base']['ref_offset_x'] = _var(case, 'lead_ref', None)
        if _var(case, 'loc_site', False):
            for robot in config['robots'].values():
                robot['localization'].update(pos_noise_std=SITE_NOISE, delay=SITE_LATENCY)
        return config, fixtures
    return load


def _var(case, key, default):
    return dict(case.var).get(key, default)


class _BiasedStack:
    """A rover stack that sees its own pose offset by (dx, dy, dtheta) in its body frame:
    it parks that far off its goal while the host (and the world) see the truth."""

    def __init__(self, stack, name, bias):
        self._stack, self._name, self._bias = stack, name, bias

    def __getattr__(self, attr):
        return getattr(self._stack, attr)

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            object.__setattr__(self, attr, value)
        else:
            setattr(self._stack, attr, value)

    def __call__(self, obs):
        pose = obs.pose
        if pose is not None:
            dx, dy, dth = self._bias
            c, s = math.cos(pose.theta), math.sin(pose.theta)
            pose = dataclasses.replace(pose, x=pose.x + c * dx - s * dy,
                                       y=pose.y + s * dx + c * dy, theta=pose.theta + dth)
            obs = dataclasses.replace(obs, pose=pose, poses={**obs.poses, self._name: pose})
        return self._stack(obs)


def production_mapper(settings):
    """ObstacleMapper with the production mapper parameters (what the node
    builds from it; sections of sources not listed in ``sources`` are ignored)."""
    raw = lr.production_mapper_raw()
    params = MapperParams(**{k: v for k, v in raw.items()
                             if k in {f.name for f in dataclasses.fields(MapperParams)}})
    source_fields = {f.name for f in dataclasses.fields(ScanSource)} - {'name'}
    sources = [ScanSource(name=n, **{k: v for k, v in raw[n].items() if k in source_fields})
               for n in raw['sources']]
    b = settings.bounds
    return ObstacleMapper(params, Region(b.xmin, b.xmax, b.ymin, b.ymax),
                          parse_zones(raw['ignore_zones']), sources,
                          {r: raw['robot_radius'][r] for r in raw['robots']},
                          fixtures=settings.fixtures, heading_accuracy=raw['heading_accuracy'])


def _set_home_goals(loop, case):
    if not case.home_y:
        return
    goals = loop.engine.world.goals
    for name, y in zip(('leader_standby', 'follower_unload'), case.home_y):
        goal = goals.get(name)
        goals.set(name, type(goal)(goal.x, y, goal.theta))


class DemoLap:
    """One case: a LayoutLoop with the case's hooks (dispatch time, frozen map, LiDAR-only
    map, occlusion) wrapped around its controllers."""

    def __init__(self, case):
        self.case = case
        with _patched(lr, load_mission_settings=_settings_loader(case),
                      load_layout=_layout_loader(case, lr.load_layout)):
            self.loop = loop = lr.LayoutLoop(case.layout, case.seed)
        _set_home_goals(loop, case)
        loop.picker = TimedPicker(loop.engine, _var(case, 'pick_s', PICK_SECONDS))
        for robot, dx, dy, dth in _var(case, 'park_bias', ()):
            loop.stacks[robot] = _BiasedStack(loop.stacks[robot], robot, (dx, dy, dth))
        if STEP >= 2:
            loop.mapper = production_mapper(loop._settings)
        self.done_turned = None  # s from 'done' until both idle turns were finished
        params = OcclusionParams() if case.bias is None else OcclusionParams(
            bias_toward_leader=case.bias)
        if _var(case, 'occl', '') == 'early':  # the arm covers the follower from the start
            params = dataclasses.replace(params, after_pick_start=0.0)
        elif _var(case, 'occl', '') == 'late':  # flickers on while the rovers leave
            params = dataclasses.replace(params, release_dist=99.0)
        spec = _var(case, 'arm_shift', None)
        self.arm_shift = ArmShift(loop, spec) if spec else None
        self.hold = None if case.loc == 'ideal' else HeldLocalization(
            loop.world, np.random.default_rng(case.seed),
            self.arm_shift.offset if self.arm_shift else None)
        self.occlusion = ArmOcclusion(loop, params, self.hold) if case.occlusion else None
        self.dispatch_t = 0.0 if case.map == 'empty' else PREMAP_S
        self.dispatched = False
        self.later, self.trips, self.session, self.parts_sent = [], 1, [], 0
        self.stuck_s = 0.0
        loop.finish_when = self._given_up
        feed, control = loop._feed_mapper, loop._control
        loop._feed_mapper = lambda obs, name: self._feed_mapper(feed, obs, name)
        loop._control = lambda obs, name: self._control(control, obs, name)

    def _given_up(self, loop):
        """Ends the lap (raises lr._Finished): paused for STUCK_TOTAL in all, or the idle
        turns after 'done' finished (1 s later) or not finished IDLE_TURN_LIMIT after it."""
        if loop.engine.seq.system_state() in lr.STUCK_STATES:
            self.stuck_s += lr.HOST_TICK
        if self.stuck_s >= STUCK_TOTAL:
            raise lr._Finished
        kinds = [e[1] for e in loop.events]
        if self.later and kinds.count('loaded') >= self.trips - len(self.later):
            self._dispatch_trip(loop.engine, self.later.pop(0))  # a second trip, queued
        dones = [e[0] for e in loop.events if e[1] == 'done']
        # a trip queued behind another starts from the unload ('next_batch'), no 'done' between
        delivered = [e[0] for e in loop.events if e[1] == 'delivered']
        if not dones or len(delivered) < self.parts_sent or self.later \
                or dones[-1] < delivered[-1]:
            return False  # the last trip has not ended yet
        since = loop._now - dones[-1]
        engine = loop.engine
        turns = engine.idle_turns
        turned = turns.heading is None or all(
            r in turns.turns and not turns.under_way(r, engine._at_goal_key(r))
            for r in engine.robots)
        if turned and self.done_turned is None:
            self.done_turned = round(since, 1)
            if self.session:  # the next lap of the session
                trip = self.session.pop(0)
                self.trips, self.later = 1, trip.split('>')[1:]
                self._dispatch_trip(loop.engine, trip.split('>')[0])
                self.done_turned = None
                return False
        if (self.done_turned is not None and since >= self.done_turned + 1.0) \
                or since >= IDLE_TURN_LIMIT:
            raise lr._Finished
        return False

    def _feed_mapper(self, feed, obs, name):
        if name == 'leader':
            return  # production map: follower LiDAR only
        state = self.loop.engine.seq.system_state()
        if not integrates_scans(state, self.case.map == 'frozen'):
            return  # production rule: map frozen while a mission runs
        feed(obs, name)

    def _control(self, control, obs, name):
        if name == 'leader' and self.occlusion is not None:
            self.occlusion.step(obs.t)
        if name == 'leader' and self.arm_shift is not None:
            self.arm_shift.step()
        return control(obs, name)

    def _dispatch(self, engine, _now):
        laps = self.case.part.split('|')  # 'a|b': b dispatched once a's idle turn is over
        self.session = laps[1:]
        trips = laps[0].split('>')
        self.later = trips[1:]  # each dispatched at the 'loaded' of the trip before it
        self.trips = len(trips)
        self._dispatch_trip(engine, trips[0])
        self.dispatched = True

    def _dispatch_trip(self, engine, trip):
        self.parts_sent += len(trip.split('+'))
        for name in trip.split('+'):
            if not engine.dispatch(name):
                raise ValueError(f'part {name!r} was not accepted by the mission engine')

    def run(self):
        if self.dispatch_t <= 0.0:
            self._dispatch(self.loop.engine, 0.0)
        else:
            self.loop.operator.append((self.dispatch_t, self._dispatch))
        duration = self.dispatch_t + LAP_S * len(_parts(self.case.part))
        try:  # the lap ends in _given_up, not AFTER_DONE
            with _patched(lr, AFTER_DONE=math.inf):
                run_scenario(self.loop.world, self.loop.controllers(), duration=duration,
                             stop_on_arrival=False)
        except lr._Finished:
            pass
        return self.loop.result(self.case.layout, self.case.part, until_done=False)


@dataclass
class Verdict:
    case: str
    criteria: dict                      # name -> (ok, detail)
    numbers: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    trace: list = field(default_factory=list)

    def failed(self):
        return {k: d for k, (ok, d) in self.criteria.items() if not ok}

    def row(self):
        marks = ' '.join(f'{k}={"ok" if ok else "FAIL"}' for k, (ok, _) in self.criteria.items())
        n = self.numbers
        return (f'{self.case:44s} {marks} | done={n.get("done_t")} paths={n.get("paths")} '
                f'stand={n.get("max_stand")} gap_fp={n.get("gap_fp")} gap_sim={n.get("gap_sim")}')


def issued_paths(result):
    """rover -> [(t, goal key, host PathCmd)] of every newly planned path (path_issued),
    without the one-pose paths of a rover already at its goal (re-issued after a pause)."""
    cmds = {(rover, cmd_id): cmd for (rover, _epoch, cmd_id), cmd in result.host_cmds.items()}
    out = {'leader': [], 'follower': []}
    for event in result.events:
        m = event[1] == 'path_issued' and _PATH_ISSUED.match(event[2])
        cmd = m and cmds.get((m.group(1), int(m.group(3))))
        if cmd and len(cmd.poses) > 1:
            out[m.group(1)].append((event[0], m.group(2), cmd))
    return out


def _legs(paths):
    """[(goal key, [(t, cmd)])] in order: a new leg each time the goal changes (two trips
    serve the same goals twice)."""
    legs = []
    for t, goal, cmd in paths:
        if not legs or legs[-1][0] != goal:
            legs.append((goal, []))
        legs[-1][1].append((t, cmd))
    return legs


def _stands_at(end, shift):
    """Where a rover truly stands once its camera puts it on ``end``: on it, or without the
    camera offset of the leader's arm (``shift``: the ArmShift spec, None for no arm)."""
    spots = [(end.x, end.y)]
    if shift and shift[0] == 'const':
        spots.append((end.x - shift[1], end.y - shift[2]))
    elif shift:
        k = shift[1]
        spots.append(((end.x + k * CAMERA_NADIR[0]) / (1.0 + k),
                      (end.y + k * CAMERA_NADIR[1]) / (1.0 + k)))
    return spots


def _leg_arrival(result, rover, cmds, until, shift=None):
    """First time in [first path of the leg, ``until``) the rover's TRUE tracked point stood
    within arrive_pos_tol of the end of the leg's last non-via path (None = never). A leader
    whose camera carries its arm's offset parks that far off the end (``_stands_at``)."""
    target = next((c for _, c in reversed(cmds) if not c.via), None)
    if target is None:
        return None
    tol = result.robot_settings[rover].arrive_pos_tol
    spots = _stands_at(target.poses[-1], shift if rover == 'leader' else None)
    return next((r.t for r in result.series[rover] if cmds[0][0] <= r.t < until
                 and any(math.hypot(r.x - x, r.y - y) <= tol for x, y in spots)), None)


def _leg_table(result, shift=None):
    """rover -> [(goal, cmds, arrival)] per leg, legs of both rovers paired by order."""
    issued = issued_paths(result)
    out = {}
    for rover in ('leader', 'follower'):
        legs = _legs(issued[rover])
        starts = [cmds[0][0] for _, cmds in legs[1:]] + [math.inf]
        out[rover] = [(goal, cmds, _leg_arrival(result, rover, cmds, until, shift))
                      for (goal, cmds), until in zip(legs, starts)]
    return out


def _turn_made(result, cmd, t0):
    """When the leader had made the leading turn of ``cmd`` issued at ``t0`` (t0 without
    one): its first true heading within goal_yaw_tol of the turn's pivot_heading."""
    if not cmd.pivot_first:
        return t0
    tol = result.robot_settings['leader'].goal_yaw_tol
    return next((r.t for r in result.series['leader'] if r.t >= t0 and
                 abs(math.remainder(r.theta - cmd.pivot_heading, math.tau)) <= tol), None)


def _describe(cmd):
    if cmd.lead_robot:
        return 'hold ride' if cmd.hold else 'turn-in ride' if cmd.via else 'shared'
    return 'stand-alone'


def platoon_legs(result, shift=None):
    """[(leader goal, follower goal, [problems], start lag)] per leg.

    A leg is kept when (1) the follower's first path of the leg names the leader
    (``lead_robot``: a shared path, or a ride to a hold / turn-in point), on an outbound leg
    issued at most START_LAG after the leader could go; (2) every later new path of the
    leg names the leader too, except the last one, which may be the stand-alone dock path
    after a ride to a via point. A stand-alone path after a YIELD wait is the broken
    platoon. Outbound = the leader is not going to its standby."""
    table = _leg_table(result, shift)
    out = []
    for (lgoal, lcmds, arrived), (fgoal, fcmds, _) in zip(table['leader'], table['follower']):
        problems = []
        t0, first = fcmds[0]
        if not first.lead_robot:
            problems.append(f'first path stand-alone (cmd {first.cmd_id} at {t0}; leader path '
                            f'at {lcmds[0][0]}, leader at {lgoal}: {arrived})')
        lag = None
        if not lgoal.startswith('leader_standby'):
            ready = _turn_made(result, lcmds[0][1], lcmds[0][0])
            lag = None if ready is None else round(t0 - ready, 1)
            if lag is not None and lag > START_LAG:
                problems.append(f'first path {lag:.1f} s after the leader could go')
        for j, ((_, prev), (t, cmd)) in enumerate(zip(fcmds, fcmds[1:])):
            if cmd.lead_robot:
                continue
            if not (j == len(fcmds) - 2 and prev.via):  # the dock path after a via ride
                problems.append(f'stand-alone cmd {cmd.cmd_id} at {t} after {_describe(prev)} '
                                f'cmd {prev.cmd_id} (leader at {lgoal}: {arrived})')
        out.append((lgoal, fgoal, problems, lag))
    if len(table['leader']) != len(table['follower']):
        out.append(('?', '?', [f'leader legs {[g for g, _, _ in table["leader"]]} != follower '
                               f'legs {[g for g, _, _ in table["follower"]]}'], None))
    return out


def _still(prev, cur):
    dt = cur.t - prev.t
    if dt <= 0.0:
        return True
    speed = math.hypot(cur.x - prev.x, cur.y - prev.y) / dt
    turn = abs(math.remainder(cur.theta - prev.theta, math.tau)) / dt
    return speed < STILL_SPEED and turn < STILL_TURN


def _still_intervals(rows, t0, t1):
    """[(a, b)] stand-stills of at least STAND_MIN between t0 and t1, after the first move."""
    rows = [s for s in rows if t0 <= s.t <= t1]
    out, start, moved = [], None, False
    for prev, cur in zip(rows, rows[1:]):
        still = _still(prev, cur)
        moved = moved or not still
        if still and moved and start is None:
            start = prev.t
        elif not still and start is not None:
            if cur.t - start >= STAND_MIN:
                out.append((start, cur.t))
            start = None
    if start is not None and rows and rows[-1].t - start >= STAND_MIN:
        out.append((start, rows[-1].t))
    return out


def _designed_wait(a, b, fcmds, lead_arrived):
    """Follower standing at its hold point until the leader has parked (+ DOCK_GRACE)."""
    last = [cmd for t, cmd in fcmds if t <= a]
    if not last or not last[-1].hold:
        return False
    return lead_arrived is None or b <= lead_arrived + DOCK_GRACE


def _start_gap_wait(rows, t0, a, b):
    """Standing behind the leader before having driven at all in the leg that began at
    ``t0`` (the rover's own gap wait, detail GAP: the leader leaves first). A leading pivot
    before it does not make it a stop on the way."""
    driven = any(abs(s.v) >= STILL_SPEED for s in rows if t0 <= s.t <= a)
    return not driven and all(s.detail == 'GAP' for s in rows if a < s.t < b)


def standing_report(result, shift=None):
    """rover -> [(goal, longest stand-still on the way (s), [(a, b)] counted)]: from the
    rover's first path of the leg to its true arrival, after its first move, without the
    designed waits (follower at its hold point until the leader parks; the wait at the
    start of a leg)."""
    table = _leg_table(result, shift)
    out = {}
    for rover in ('leader', 'follower'):
        report = []
        for i, (goal, cmds, arrival) in enumerate(table[rover]):
            rows, t0 = result.series[rover], cmds[0][0]
            spans = [(a, b) for a, b in _still_intervals(rows, t0, arrival or result.sim_time)
                     if not _start_gap_wait(rows, t0, a, b)]
            if rover == 'follower' and i < len(table['leader']):
                lead_arrived = table['leader'][i][2]
                spans = [(a, b) for a, b in spans
                         if not _designed_wait(a, b, cmds, lead_arrived)]
            longest = max((b - a for a, b in spans), default=0.0)
            report.append((goal, round(longest, 1), [(round(a, 1), round(b, 1)) for a, b in spans]))
        out[rover] = report
    return out


def sim_contact(result, world):
    """(smallest sim body-circle gap, time, leader / follower true tracked points there):
    host_sim's own collision rule (circle radius around the body origin)."""
    params = {name: robot.params for name, robot in world.robots.items()}
    best = (math.inf, None, None, None)
    for t, lead, foll in result.truth:
        centres = []
        for name, (x, y, th) in (('leader', lead), ('follower', foll)):
            off = params[name].icr_offset_x + params[name].ref_offset_x
            centres.append((x - off * math.cos(th), y - off * math.sin(th)))
        gap = math.dist(*centres) - params['leader'].radius - params['follower'].radius
        if gap < best[0]:
            best = (gap, t, lead, foll)
    return best


def _parts_done(result, part):
    return result.delivered == len(_parts(part))


def _parts(part):
    """Part names of a case: 'a+b' one trip, 'a>b' two trips (b dispatched at a's 'loaded')."""
    return re.split(r'[+>|]', part)


def _final_tolerances(result):
    """rover -> (pos, yaw) tolerance its last path command was sent with (an idle turn
    step carries the arrival tolerance), else its configured goal tolerances."""
    out = {}
    for rover, rows in result.series.items():
        cfg = result.robot_settings[rover]
        cmd = next((c for s in reversed(rows)
                    if (c := result.host_cmds.get((rover, s.epoch, s.cmd_id))) is not None
                    and c.mode != 'STOP'), None)
        pos = cmd.goal_pos_tol if cmd is not None and cmd.goal_pos_tol else cfg.goal_pos_tol
        out[rover] = (pos, cfg.goal_yaw_tol)
    return out


def judge(case, result, world, dispatch_t):
    criteria, numbers = {}, {}
    errs = final_goal_errors(result) if result.completed else {}
    tols = _final_tolerances(result)
    parked_bad = {r: e for r, e in errs.items() if e[0] > tols[r][0] or e[1] > tols[r][1]}
    ok = result.completed and _parts_done(result, case.part) and not parked_bad \
        and all(s == 'PARKED' for s in result.nav_states.values())
    criteria['delivered'] = (ok, f'{result.reason() or "done"} nav={result.nav_states} '
                                 f'parked_err={ {r: (round(p, 3), round(y, 3)) for r, (p, y) in errs.items()} }')
    shift = _var(case, 'arm_shift', None)
    legs = platoon_legs(result, shift)
    broken = [f'{lg}/{fg}: ' + '; '.join(p) for lg, fg, p, _ in legs if p]
    numbers['start_lags'] = [lag for _, _, _, lag in legs if lag is not None]
    criteria['platoon'] = (not broken, ' | '.join(broken) or 'kept on every leg')
    stops = [e for e in result.events if e[1] in MISSION_STOPS]
    criteria['no_stop'] = (not stops, str(stops[:6]))
    stand = standing_report(result, shift)
    longest = max((s for legs_ in stand.values() for _, s, _ in legs_), default=0.0)
    criteria['stop_go'] = (longest <= STAND_LIMIT, str(stand))
    rr, rb = footprint_gaps(result)
    gap_sim, t_sim, lead, foll = sim_contact(result, world)
    criteria['contact'] = (rr >= 0.0 and min(rb.values()) >= 0.0,
                           f'footprint gap rover-rover {rr:.3f}, rover-box '
                           f'{ {r: round(g, 3) for r, g in rb.items()} }')
    final = {r: rows[-1].theta for r, rows in result.series.items()}
    east = all(abs(math.remainder(th, math.tau)) <= EAST_TOL for th in final.values())
    criteria['ends_east'] = (result.completed and east,
                             f'final true headings { {r: round(th, 2) for r, th in final.items()} }')
    numbers.update(
        done_t=round(result.done_t - dispatch_t, 1) if result.completed else None,
        paths=result.path_commands, max_stand=round(longest, 1), gap_fp=round(rr, 3),
        gap_sim=round(gap_sim, 3), gap_sim_t=t_sim and round(t_sim, 2),
        gap_box=round(min(rb.values()), 3),
        gap_sim_poses=(lead and tuple(round(v, 3) for v in lead),
                       foll and tuple(round(v, 3) for v in foll)),
        sim_collisions=result.collisions, lead_excess=round(lead_excess(result)[0], 3),
        dispatch_t=dispatch_t, yields=sum(1 for e in result.events
                                          if e[1] == 'plan_waiting' and 'YIELD' in e[2]))
    return Verdict(case.id, criteria, numbers, list(result.events))


def run_case(case, trace=False):
    """Verdict of one case (``trace``: the planner decisions, scenarios/plan_trace.py)."""
    lap = DemoLap(case)
    if trace:
        from plan_trace import PlanTrace
        with PlanTrace() as recorder:
            result = lap.run()
    else:
        result = lap.run()
    verdict = judge(case, result, lap.loop.world, lap.dispatch_t)
    verdict.numbers['paused_s'] = round(lap.stuck_s, 1)
    verdict.numbers['turn_s'] = lap.done_turned
    verdict.numbers['end_poses'] = {r: tuple(round(v, 3) for v in rows[-1][1:4])
                                    for r, rows in result.series.items()}
    if lap.occlusion is not None:
        verdict.numbers['lost_windows'] = [(round(a, 2), b and round(b, 2))
                                           for a, b in lap.occlusion.lost]
    if trace:
        verdict.trace = recorder.records
    return verdict


def _run_traced(case):
    return run_case(case, trace=True)


def iter_cases(cases, jobs=6, trace=False):
    """Verdicts in case order as they finish, ``jobs`` processes at once."""
    run = _run_traced if trace else run_case
    if jobs <= 1 or len(cases) == 1:
        yield from map(run, cases)
        return
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(jobs, maxtasksperchild=1) as pool:
        yield from pool.imap(run, cases, chunksize=1)


def run_cases(cases, jobs=6, trace=False):
    return list(iter_cases(cases, jobs, trace))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fast', action='store_true', help='primary cases only (demo config)')
    ap.add_argument('--robust', action='store_true', help='robustness cases (box shifts)')
    ap.add_argument('--arm', action='store_true', help="leader camera moved by its arm")
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('-k', default='', help='only cases whose id contains this')
    ap.add_argument('--events', action='store_true', help='print the events of each case')
    ap.add_argument('--trace', action='store_true', help='print the planner decisions')
    ap.add_argument('--save', help='pickle the verdicts to this file')
    ap.add_argument('--loc', choices=('held', 'ideal'), help='localization model of every case')
    ap.add_argument('--step', type=int, choices=(1, 2, 3), default=3, help='see STEP')
    args = ap.parse_args(argv)
    global STEP
    STEP = args.step
    pool = arm_shift_cases() if args.arm else robustness_cases() if args.robust \
        else primary_cases() if args.fast else all_cases()
    cases = [c for c in pool if args.k in c.id]
    if args.loc:
        cases = [dataclasses.replace(c, loc=args.loc) for c in cases]
    verdicts = []
    for verdict in iter_cases(cases, args.jobs, args.trace):
        verdicts.append(verdict)
        if args.save:
            with open(args.save, 'wb') as stream:
                pickle.dump(verdicts, stream)
        print(verdict.row())
        for name, detail in verdict.failed().items():
            print(f'    {name}: {detail}')
        if args.events:
            for event in verdict.events:
                print('      ', event)
        for record in verdict.trace:
            print('      T', record)
        sys.stdout.flush()


if __name__ == '__main__':
    import site_demo  # pickled verdicts then name site_demo.Verdict, not __main__'s
    site_demo.main()
