"""In-process full-lap scenario: MissionEngine + two rover stacks + obstacle mapper in host_sim.

Real code throughout (host_mission engine and mapper, host_planning, host_sim world,
rover_drive controller and safety gate); only the leader's pick&place is a FakePicker.
Run one case: python3 layout_runner.py LAYOUT PART [DURATION]   (LAYOUT = empty|box1|box2|site,
PART = gear_01|bolt_01|gear_01+bolt_01; VERBOSE=1 prints the mission events; LOCALIZATION=ideal|fused
and PROFILE=<host_sim sim_profiles.yaml name> as for run_layout). Needs ROS sourced.
Run the suite: host_ws/run_scenarios.sh."""
import dataclasses
import math
import os
import sys
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent                      # host_ws/src
ROVER_DRIVE = SRC.parents[1] / 'rover_ws' / 'src' / 'rover_drive'
sys.path[:0] = [str(p) for p in (
    SRC / 'host_planning', SRC / 'host_sim', SRC / 'host_localization', HERE.parent,
    HERE.parent / 'test',
    ROVER_DRIVE, ROVER_DRIVE / 'test')]

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from host_sim.config import apply_profile, load_config, with_localization_mode  # noqa: E402
from host_sim.core import World  # noqa: E402
from host_sim.scenario import run_scenario  # noqa: E402

from rover_drive.core import state as st  # noqa: E402
from rover_drive.core.config import DepthFrontConfig, DriveConfig, GateConfig  # noqa: E402

from host_loop import LEADER_QUANTUM, RoverStack  # noqa: E402
from sim import CMD_PERIOD, drive_config, gate_config  # noqa: E402
from sim_harness import FakePicker  # noqa: E402

from host_mission import plan_jobs  # noqa: E402
from host_mission.goals import GoalStore  # noqa: E402
from host_mission.mission_engine import MissionEngine  # noqa: E402
from host_mission.obstacle_mapper import MapperParams, ObstacleMapper, Region, ScanData, ScanSource  # noqa: E402
from host_mission.obstacle_mapper_node import parse_zones  # noqa: E402
from host_mission.settings import (load_scenario, merge_dicts, read_settings_yaml,  # noqa: E402
                                   settings_from_dict)

DATA = HERE / 'data'
MISSION_CONFIG = HERE.parent / 'config'
SITE_CONFIG = SRC.parents[1] / 'config' / 'site'
HARDWARE_CONFIG = SRC.parents[1] / 'config' / 'hardware'
PRESET_DIR = ROVER_DRIVE / 'config' / 'presets'
LAYOUTS = ('empty', 'box1', 'box2')
SITE = 'site'                    # production shelves, goals and homes (not in the suite)
PARTS = ('gear_01', 'bolt_01')

HOST_TICK = 0.2                  # s, mission engine period (as the ROS node)
PICK_DURATION = 5.0              # s, FakePicker pick&place time
AFTER_DONE = 3.0                 # s, kept running after 'done' to read the rovers' final state
STUCK_AFTER = 60.0  # s, mission PAUSED/FAULT/ABORTED this long: abandon the run
STUCK_STATES = ('PAUSED', 'FAULT', 'ABORTED')
# A/B of the rover gate's path stop zone: off; rover = both rovers with their front_path
# preset; rover_host = as rover, and the host plans without a front-stop zone.
FRONT_PATH_AB_CHOICES = ('off', 'rover', 'rover_host')


class Sample(NamedTuple):
    """One control tick of one rover: its TRUE tracked point and what it did."""
    t: float
    x: float                  # true tracked point (host_sim world truth, not the estimate)
    y: float
    theta: float
    v: float                  # the gated linear speed the base received
    w: float
    nav: str                  # tracker state
    quality: str              # localization quality the rover had at this tick ('' = no pose yet)
    epoch: int                # the command the tracker had loaded ((0, 0) = the idle one)
    cmd_id: int
    lead_robot: str
    detail: str = ''          # tracker detail (PIVOT, GAP, ...)


@dataclass
class ScenarioResult:
    layout: str
    part: str
    completed: bool                       # mission 'done': both rovers back home
    done_t: float                         # sim time of 'done' (nan if not completed)
    sim_time: float                       # sim time the run lasted
    delivered: int
    loaded: int
    collisions: list                      # (t, kind, a, b)
    min_robot_gap: float
    mission_state: str
    nav_states: dict                      # rover -> tracker state at the end
    blocked: dict                         # rover -> {safety stop reason: count}
    path_commands: dict                   # rover -> paths issued ('path_issued' events)
    front_lost_stops: dict  # rover -> depth-camera front stops begun with the depth unknown
    events: list = field(repr=False, default_factory=list)
    # Recorded truth, read by lap_checks (never by the run itself):
    series: dict = field(repr=False, default_factory=dict)       # rover -> [Sample] per control tick
    truth: list = field(repr=False, default_factory=list)        # (t, leader, follower) true tracked points
    host_cmds: dict = field(repr=False, default_factory=dict)    # (rover, epoch, cmd_id) -> host PathCmd
    boxes: list = field(repr=False, default_factory=list)        # true obstacle rects (host_sim Rect)
    robot_settings: dict = field(repr=False, default_factory=dict)   # rover -> RobotSettings of the lap
    mapped_log: list = field(repr=False, default_factory=list)   # LayoutLoop.mapped_log
    true_boxes: list = field(repr=False, default_factory=list)   # true boxes of the layout (rect dicts)
    shelves: list = field(repr=False, default_factory=list)

    def reason(self):
        """Why the lap did not complete (empty when it did)."""
        if self.completed:
            return ''
        kinds = [e[1] for e in self.events]
        parts = [f'mission {self.mission_state}', f'loaded {self.loaded}',
                 f'delivered {self.delivered}', f'plan_failed {kinds.count("plan_failed")}']
        if self.collisions:
            parts.append(f'collision {self.collisions[0]}')
        return ', '.join(parts)

    def row(self):
        done = f'{self.done_t:.1f}s' if self.completed else 'no'
        return (f'{self.layout:5s} {self.part:8s} done={done:7s} delivered={self.delivered} '
                f'collisions={len(self.collisions)} min_robot_gap={self.min_robot_gap:.3f} '
                f'paths={self.path_commands} nav={self.nav_states} {self.reason()}')


class _Finished(Exception):
    """Raised from a controller to end the run early (mission done)."""


# One rover at one host tick (LayoutLoop.trace): true tracked point (x, y, theta), the pose
# the host holds (None before the first sample), its current command (id, mode, first pose
# (x, y) or None) and the tracker state.
RoverTrace = namedtuple('RoverTrace', 'true host cmd_id mode first nav')


def layout_objects(name):
    """(true boxes, shelves) of one layout as rect dicts (map_quality's truth)."""
    layouts = yaml.safe_load((DATA / 'layouts.yaml').read_text())
    if name.startswith(SITE):            # 'site' (no boxes) or 'site_<set>' (site boxes_<set>)
        boxes = 'boxes' + name[len(SITE):]
        return layouts[SITE][boxes], layouts[SITE]['shelves']
    return layouts['boxes'][name], layouts['shelves']


def load_layout(name):
    """(sim World config, mission fixture dicts) of one box layout (``site``: the rovers
    start at the site's home poses)."""
    boxes, shelves = layout_objects(name)
    config = load_config(str(DATA / 'sim_world.yaml'))
    config['obstacles'] = shelves + boxes
    if name.startswith(SITE):
        starts = yaml.safe_load((DATA / 'layouts.yaml').read_text())[SITE]['start']
        for robot, start in starts.items():
            config['robots'][robot]['start'] = list(start)
    return config, [{k: v for k, v in s.items() if k != 'type'} for s in shelves]


def goals_file(layout):
    """Taught goals of a layout: the production goals.yaml at the site, else the test copy."""
    return (SITE_CONFIG if layout.startswith(SITE) else DATA) / 'goals.yaml'


def localization_config(config, localization=None, profile=None):
    """World config with a host_sim profile overlaid, then the localization mode of every
    rover forced (None keeps what the world / profile says)."""
    if profile:
        config = apply_profile(config, profile)
    if localization:
        config = with_localization_mode(config, localization)
    return config


def front_path_ab():
    """The A/B configuration from the environment (FRONT_PATH_AB, default off)."""
    value = os.environ.get('FRONT_PATH_AB') or 'off'
    if value not in FRONT_PATH_AB_CHOICES:
        raise ValueError(f'FRONT_PATH_AB={value!r}: one of {FRONT_PATH_AB_CHOICES}')
    return value


def production_settings_raw():
    """Shipped mission.yaml merged with the arena and rover overlays."""
    return read_settings_yaml(MISSION_CONFIG / 'mission.yaml',
                              [SITE_CONFIG / 'arena.yaml', HARDWARE_CONFIG / 'rovers.yaml'])


def production_mapper_raw():
    """Shipped obstacle_mapper_node parameters: package file, site keys, sensor mounts."""
    merged = {}
    for path in (MISSION_CONFIG / 'obstacle_mapper.yaml', SITE_CONFIG / 'obstacle_mapper.yaml',
                 HARDWARE_CONFIG / 'sensor_mounts.yaml'):
        merged = merge_dicts(
            merged, yaml.safe_load(path.read_text())['obstacle_mapper_node']['ros__parameters'])
    return merged


def load_mission_settings(fixtures):
    """Production mission settings plus the layout's shelves as fixtures (FRONT_PATH_AB
    rover_host: without the front-stop zone)."""
    raw = production_settings_raw()
    raw['fixtures'] = fixtures
    if front_path_ab() == 'rover_host':
        for robot in raw['robots'].values():
            robot['plan']['front_zone'] = None
    return settings_from_dict(raw)


# Node-level keys of obstacle_mapper_node besides the MapperParams fields (the node declares them
# by hand); settings_file and output_topic are node wiring and play no part in a scenario.
_MAPPER_NODE_KEYS = {'output_topic', 'settings_file', 'ignore_zones', 'robots', 'robot_radius',
                     'sources', 'heading_accuracy'}


def _unknown_mapper_keys(raw, param_fields, source_fields):
    """Keys of the mapper yaml no mapper field accepts (per-source ones dotted)."""
    sources = set(raw.get('sources', ()))
    unknown = [k for k in raw if k not in param_fields | _MAPPER_NODE_KEYS | sources]
    for name in sources:
        unknown += [f'{name}.{k}' for k in raw[name] if k not in source_fields | {'scan_topic'}]
    return unknown


def make_mapper(settings):
    raw = yaml.safe_load((DATA / 'obstacle_mapper.yaml').read_text())['obstacle_mapper_node']['ros__parameters']
    param_fields = {f.name for f in dataclasses.fields(MapperParams)}
    source_fields = {f.name for f in dataclasses.fields(ScanSource)} - {'name'}
    unknown = _unknown_mapper_keys(raw, param_fields, source_fields)
    if unknown:
        raise ValueError(f'obstacle_mapper.yaml: unknown keys {sorted(unknown)} (a misspelled key '
                         f'would silently leave the mapper default)')
    params = MapperParams(**{k: v for k, v in raw.items() if k in param_fields})
    sources = [ScanSource(name=n, **{k: v for k, v in raw[n].items() if k in source_fields})
               for n in raw['sources']]
    radii = {r: raw['robot_radius'][r] for r in raw['robots']}
    heading = raw['heading_accuracy']
    unknown = sorted(set(heading) - {s.robot for s in sources})
    if unknown:
        raise ValueError(f'obstacle_mapper.yaml: heading_accuracy of rovers carrying no '
                         f'source {unknown}')
    b = settings.bounds
    return ObstacleMapper(params, Region(b.xmin, b.xmax, b.ymin, b.ymax),
                          parse_zones(raw['ignore_zones']), sources, radii,
                          fixtures=settings.fixtures, heading_accuracy=heading)


def preset_overrides(robot, names):
    """Keys of config/presets/<robot>_<name>.yaml merged in order, later wins (as the launch
    file applies them after config/<robot>.yaml). An unknown name raises."""
    values = {}
    for name in names or ():
        path = PRESET_DIR / f'{robot}_{name}.yaml'
        if not path.is_file():
            raise ValueError(f'unknown {robot} preset {name!r}: no {path}')
        values.update(yaml.safe_load(path.read_text())['/**']['ros__parameters'])
    return values


def _fields_of(cls, values):
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in values.items() if k in names}


def rover_stack(robot, presets=(), **kwargs):
    """Shipped config/<robot>.yaml with the presets on top, as tracker + gate + depth node."""
    values = preset_overrides(robot, presets)
    idle = st.PathCmd(epoch=0, cmd_id=0, mode='STOP')
    stack = RoverStack(drive_config(robot, **_fields_of(DriveConfig, values)),
                       gate_config(robot, **_fields_of(GateConfig, values)), idle, **kwargs)
    if stack.depth_cfg is not None:
        stack.depth_cfg = dataclasses.replace(stack.depth_cfg,
                                              **_fields_of(DepthFrontConfig, values))
    return stack


def to_rover_cmd(cmd):
    return st.PathCmd(epoch=cmd.epoch, cmd_id=cmd.cmd_id, mode=cmd.mode,
                      poses=tuple((p.x, p.y, p.theta) for p in cmd.poses),
                      speed_cap=cmd.speed_cap, goal_pos_tol=cmd.goal_pos_tol,
                      goal_yaw_tol=cmd.goal_yaw_tol, pick_at_goal=cmd.pick_at_goal,
                      lead_robot=cmd.lead_robot, gap=cmd.gap,
                      pivot_first=cmd.pivot_first, pivot_heading=cmd.pivot_heading,
                      fixture_ahead=cmd.fixture_ahead)


def scan_data(ranges, scanner):
    """A host_sim scanner's ranges as the mapper's ScanData (LaserScan layout)."""
    return ScanData(np.asarray(ranges, dtype=float), float(scanner.angle_min),
                    float(scanner.angle_increment), scanner.params.range_min,
                    scanner.params.range_max)


class CommLink:
    """Host <-> rover Wi-Fi link as release-time queues (``LayoutLoop(comm=...)``).

    A message sent at ``t`` is delivered at ``t + delay``. It is lost when either instant lies
    inside an outage window (t_start <= t < t_end), in either direction. Loss is the faithful
    model: every stream is periodic or superseded within 0.5 s."""

    def __init__(self, delay=0.0, outages=()):
        self.delay = float(delay)
        self.outages = [(float(a), float(b)) for a, b in outages]
        self.to_rover = {}         # rover -> [(release t, payload)] in release order
        self.to_host = []
        self.dropped = 0
        self.cmd_rx = {}           # rover -> delivery times of path commands
        self.pose_rx = {}          # rover -> delivery times of the rover's own pose

    def down(self, t):
        return any(a <= t < b for a, b in self.outages)

    def _put(self, queue, t, payload):
        if self.down(t) or self.down(t + self.delay):
            self.dropped += 1
            return
        queue.append((t + self.delay, payload))

    def send_to_rover(self, rover, t, payload):
        self._put(self.to_rover.setdefault(rover, []), t, payload)

    def send_to_host(self, t, payload):
        self._put(self.to_host, t, payload)

    @staticmethod
    def release(queue, t):
        """Pop and return the payloads due at ``t``."""
        due = [p for r, p in queue if r <= t + 1e-9]
        del queue[:len(due)]
        return due


class LayoutLoop:
    """Host side (engine + mapper + picker) and the two rover stacks, as run_scenario controllers."""

    def __init__(self, layout, seed=1, localization=None, profile=None, plan_delay=0.0,
                 presets=None, comm=None):
        sim_config, fixtures = load_layout(layout)
        sim_config = localization_config(sim_config, localization, profile)
        self.world = World.from_config(sim_config, seed=seed)
        settings = load_mission_settings(fixtures)
        store = GoalStore.load(str(goals_file(layout)))
        store.path = None                      # never write the teach file back
        scenario = dataclasses.replace(load_scenario(SITE_CONFIG / 'mission_scenario.yaml'),
                                       pick_cooldown=0.0)
        self._now = 0.0
        # plan_delay > 0: every path is there that many seconds after it was asked for,
        # as on the node, where paths are computed beside the tick (plan_jobs)
        runner = plan_jobs.Delayed(lambda: self._now, plan_delay) if plan_delay > 0.0 else None
        self.engine = MissionEngine(settings, scenario, store, 1_700_000_000,
                                    plan_runner=runner)
        self.picker = FakePicker(self.engine, duration=PICK_DURATION)
        self.mapper = make_mapper(settings)
        self.boxes, self.shelves = layout_objects(layout)
        self.mapped_log = []       # (t, shapes, {rover: true (x, y, radius)}) per publication
        presets = presets or {}
        if front_path_ab() != 'off':
            presets = {r: ['front_path', *(presets.get(r) or ())] for r in ('leader', 'follower')}
        self.stacks = {
            'leader': rover_stack('leader', presets.get('leader'), truncate_quantum=LEADER_QUANTUM),
            'follower': rover_stack('follower', presets.get('follower')),
        }
        self.events = []
        self.blocked = {name: {} for name in self.stacks}
        self.front_lost_stops = {name: 0 for name in self.stacks}
        self._prev_reasons = {name: set() for name in self.stacks}
        self._last_row = {}
        self.series = {name: [] for name in self.stacks}
        self.truth = []
        self.host_cmds = {}
        self._settings = settings
        self._next_tick = 0.0
        self._done_at = None
        self._stuck_since = None
        # Operator inputs: [(t, fn(engine, now))], each applied once, before the first host
        # tick at or after t (as ROS callbacks run between two timer ticks).
        self.operator = []
        # fn(loop) -> True ends the run AFTER_DONE later (a lap that never reaches 'done').
        self.finish_when = None
        self._stop_at = None
        self.trace = []                        # (t, system state, {rover: RoverTrace}) per tick
        # comm = {'delay': s, 'outages': [(t0, t1)]}: the host <-> rover link (None = ideal, as
        # before). What crosses host -> rover: path commands (on change + 2 Hz keep-alive), each
        # rover's own pose / quality / heading_aligned and the lead's pose. Rover -> host:
        # nav_feedback + safety_state, and the scans the obstacle mapper uses (outage only).
        self.comm = comm
        self.link = CommLink(**comm) if comm is not None else None
        if self.link is not None:
            self._rx_pose = {n: {r: None for r in self.stacks} for n in self.stacks}
            self._pose_seen = {}               # (rover, source) -> last sample put on the link
            self._sent_cmd = {}
            self._next_republish = 0.0
            self._fb_sent = {}                 # rover -> (key, t) of the last feedback sent
            self._host_leader_state = None     # leader tracker state as the host last heard it
            for stack in self.stacks.values():
                stack._feed_command = lambda t: None   # commands arrive only through the link

    def controllers(self):
        return {name: (lambda obs, n=name: self._control(obs, n)) for name in self.stacks}

    def _control(self, obs, name):
        if name == 'leader':
            self._host_step(obs)
        self._feed_mapper(obs, name)
        stack = self.stacks[name]
        v, w = stack(self._rover_obs(name, obs) if self.link is not None else obs)
        self._last_row[name] = row = stack.log[-1]
        if self.link is not None:
            self._send_feedback(name, obs.t, row)
        reasons = set(row[4])
        for reason in reasons - self._prev_reasons[name]:
            self.blocked[name][reason] = self.blocked[name].get(reason, 0) + 1
            if reason == 'LIDAR_STOP' and stack.depth_cfg and math.isnan(stack.front):
                self.front_lost_stops[name] += 1
        self._prev_reasons[name] = reasons
        self._record_truth(obs, name, stack, row, v, w)
        return v, w

    def _record_truth(self, obs, name, stack, row, v, w):
        tx, ty, tth = self.world.robots[name].tracked_point()
        cmd = stack.ctrl._cmd
        self.series[name].append(Sample(
            obs.t, tx, ty, tth, v, w, row[1], obs.pose.quality if obs.pose else '',
            cmd.epoch if cmd else 0, cmd.cmd_id if cmd else 0, cmd.lead_robot if cmd else '',
            row[2]))
        if name == 'leader':        # the follower is called at the same instant, same world state
            self.truth.append((obs.t, (tx, ty, tth), self.world.robots['follower'].tracked_point()))

    def _feed_mapper(self, obs, name):
        m, t = self.mapper, obs.t
        if obs.pose_is_new:
            for rover, sample in obs.poses.items():
                if sample is not None and rover in (name, 'leader', 'follower'):
                    m.update_pose(rover, sample.x, sample.y, sample.theta, t)
                    m.update_status(rover, aligned=sample.heading_aligned, quality=sample.quality)
        if self.link is not None and self.link.down(t):
            return                 # the scan cannot reach the host (delay is not modelled)
        if name == 'leader':       # Astra Mini depth scan (0.0 = no depth), /leader/depth_scan
            depth_cfg = self.stacks[name].depth_cfg
            if not obs.depth_scan_is_new or not (depth_cfg and depth_cfg.depth_scan_enabled):
                return
            scan, source = scan_data(obs.depth_scan, self.world.depth_scans[name]), 'leader_depth'
        else:
            if not obs.scan_is_new:
                return
            scan, source = scan_data(obs.scan, self.world.lidars[name]), 'follower_lidar'
        if m.process_scan(source, scan, t).changed:
            shapes = m.shapes()
            self.engine.set_mapped_obstacles(shapes)
            self.mapped_log.append((t, shapes, {n: (r.x, r.y, r.params.radius)
                                                for n, r in self.world.robots.items()}))

    def _host_step(self, obs):
        now, world = obs.t, self.engine.world
        for name, sample in obs.poses.items():
            if sample is not None:
                world.update_pose(name, sample.x, sample.y, sample.theta, now)
                world.update_heading_aligned(name, sample.heading_aligned)
                world.update_quality(name, sample.quality)
        if self.link is not None:
            self._host_receive(now)
            self._republish(now)
            if self._host_leader_state is not None:
                self.picker.step(self._host_leader_state, now)
        else:
            for name, row in self._last_row.items():
                cmd = self.stacks[name].ctrl._cmd
                world.update_nav(name, row[1], row[2], cmd.epoch if cmd else 0,
                                 cmd.cmd_id if cmd else 0, now)
                world.update_safety(name, row[3], row[4], 1.0, row[5], row[6], now)
            if 'leader' in self._last_row:
                self.picker.step(self._last_row['leader'][1], now)
        if now >= self._next_tick:
            self._next_tick = now + HOST_TICK
            self._tick(now)

    def _tick(self, now):
        while self.operator and self.operator[0][0] <= now:
            self.operator.pop(0)[1](self.engine, now)
        if self.engine.seq.system_state() == 'UNLOADING':
            self.engine.unload_done()          # operator's unload-done button
        self._now = now
        for event in self.engine.tick(now):
            self.events.append((round(now, 2),) + tuple(event))
            if event[0] == 'done' and self._done_at is None:
                self._done_at = now
                self._nav_at_done = {n: row[1] for n, row in self._last_row.items()}
        for name, cmd in self.engine.commands().items():
            if self.link is None:
                self.stacks[name].cmd = to_rover_cmd(cmd)
            else:
                self._publish_command(name, cmd, now, keepalive=False)
            self.host_cmds.setdefault((name, cmd.epoch, cmd.cmd_id), cmd)
        self._record(now)
        if self._done_at is not None and now - self._done_at >= AFTER_DONE:
            raise _Finished
        if self.finish_when is not None and self._stop_at is None and self.finish_when(self):
            self._stop_at = now + AFTER_DONE
        if self._stop_at is not None and now >= self._stop_at:
            raise _Finished
        if self._abandoned(now):
            raise _Finished

    # ---------------- host <-> rover link (comm) ----------------
    def _publish_command(self, name, cmd, now, keepalive):
        """Put a path command on the link: when it changed, or as the 2 Hz keep-alive."""
        rover_cmd = to_rover_cmd(cmd)
        if keepalive or self._sent_cmd.get(name) != rover_cmd:
            self._sent_cmd[name] = rover_cmd
            self.link.send_to_rover(name, now, ('cmd', rover_cmd))

    def _republish(self, now):
        """path_coordinator_node's 2 Hz timer (independent of the mission tick)."""
        if now + 1e-9 < self._next_republish:
            return
        self._next_republish = now + CMD_PERIOD
        for name, cmd in self.engine.commands().items():
            self._publish_command(name, cmd, now, keepalive=True)

    def _rover_obs(self, name, obs):
        """What rover ``name`` has received over the link at this tick (poses and commands)."""
        link, t, stack = self.link, obs.t, self.stacks[name]
        for source, sample in obs.poses.items():       # the host's fusion output goes out
            if sample is not None and self._pose_seen.get((name, source)) is not sample:
                self._pose_seen[(name, source)] = sample
                link.send_to_rover(name, t, ('pose', source, sample))
        fresh = False
        for message in link.release(link.to_rover.setdefault(name, []), t):
            if message[0] == 'cmd':
                stack.cmd = message[1]
                stack.ctrl.on_command(message[1], t)
                stack.gate.on_cmd(t, message[1].lead_robot, message[1])
                link.cmd_rx.setdefault(name, []).append(t)
            else:
                self._rx_pose[name][message[1]] = message[2]
                if message[1] == name:
                    fresh = True
                    link.pose_rx.setdefault(name, []).append(t)
        rx = self._rx_pose[name]
        return dataclasses.replace(obs, pose=rx[name], pose_is_new=fresh, poses=dict(rx))

    def _send_feedback(self, name, now, row):
        """nav_feedback on a change and at feedback_rate_hz, with the gate's safety_state."""
        cmd = self.stacks[name].ctrl._cmd
        epoch, cmd_id = (cmd.epoch, cmd.cmd_id) if cmd else (0, 0)
        key = (row[1], row[2], row[3], tuple(row[4]), epoch, cmd_id)
        last = self._fb_sent.get(name)
        period = 1.0 / self.stacks[name].ctrl._cfg.feedback_rate_hz
        if last is None or key != last[0] or now - last[1] >= period - 1e-9:
            self._fb_sent[name] = (key, now)
            self.link.send_to_host(now, ('fb', name, row, epoch, cmd_id))

    def _host_receive(self, now):
        world = self.engine.world
        for _, name, row, epoch, cmd_id in self.link.release(self.link.to_host, now):
            world.update_nav(name, row[1], row[2], epoch, cmd_id, now)
            world.update_safety(name, row[3], row[4], 1.0, row[5], row[6], now)
            if name == 'leader':
                self._host_leader_state = row[1]

    def _record(self, now):
        rovers = {}
        for name, cmd in self.engine.commands().items():
            first = (cmd.poses[0].x, cmd.poses[0].y) if cmd.poses else None
            row = self._last_row.get(name)
            rovers[name] = RoverTrace(self.world.robots[name].tracked_point(),
                                      self.engine.world.robot(name).pose, cmd.cmd_id, cmd.mode,
                                      first, row[1] if row else '')
        self.trace.append((now, self.engine.seq.system_state(), rovers))

    def _abandoned(self, now):
        """True once the mission has been paused or faulted for STUCK_AFTER seconds (a
        pause that recovers on its own, e.g. a one-off 'health_lost', resets the timer)."""
        if self.engine.seq.system_state() not in STUCK_STATES:
            self._stuck_since = None
            return False
        if self._stuck_since is None:
            self._stuck_since = now
        return now - self._stuck_since >= STUCK_AFTER

    def result(self, layout, part, until_done=True):
        """The lap's result. ``until_done``: judged at 'done' (events, recorded truth and the
        rovers' tracker states up to it); the idle turn after it is not judged."""
        if until_done and self._done_at is not None:
            return self._cut_at_done(self._result(layout, part))
        return self._result(layout, part)

    def _cut_at_done(self, result):
        # the idle turn's first commands go out in the tick of 'done': cut before it
        done = self._done_at - 1e-9
        return dataclasses.replace(
            result, events=[e for e in result.events if e[0] < done or e[1] == 'done'],
            series={n: [s for s in rows if s.t < done] for n, rows in result.series.items()},
            truth=[t for t in result.truth if t[0] < done], nav_states=dict(self._nav_at_done),
            path_commands=self._path_commands(done))

    def _result(self, layout, part):
        kinds = [e[1] for e in self.events]
        record = self.world.record
        return ScenarioResult(
            layout=layout, part=part, completed=self._done_at is not None,
            done_t=self._done_at if self._done_at is not None else math.nan,
            sim_time=self.world.t, delivered=kinds.count('delivered'),
            loaded=kinds.count('loaded'),
            collisions=[(round(c.t, 1), c.kind, c.robot, c.other) for c in record.collisions],
            min_robot_gap=record.min_robot_gap,
            mission_state=self.engine.seq.system_state(),
            nav_states={n: row[1] for n, row in self._last_row.items()},
            blocked=self.blocked, path_commands=self._path_commands(),
            front_lost_stops=self.front_lost_stops, events=self.events,
            series=self.series, truth=self.truth, host_cmds=self.host_cmds,
            boxes=[o for o in self.world.obstacles if hasattr(o, 'width')],
            robot_settings=self._settings.robots,
            mapped_log=self.mapped_log, true_boxes=self.boxes, shelves=self.shelves)

    def _path_commands(self, until=math.inf):
        """Paths issued per rover (up to ``until``): each one is a new command the tracker
        re-loads."""
        counts = {name: 0 for name in self.stacks}
        for event in self.events:
            if event[1] == 'path_issued' and event[0] <= until:
                counts[event[2].split(' ', 1)[0]] += 1
        return counts


def run_layout(layout, part, duration=240.0, seed=1, localization=None, profile=None,
               plan_delay=0.0, presets=None, comm=None):
    """Dispatch ``part`` in ``layout`` and run until the mission is done or ``duration`` sim seconds.

    ``localization`` (ideal | fused) forces the localization mode of both rovers;
    ``profile`` overlays a named host_sim profile. Both None = the world as in
    data/sim_world.yaml. ``plan_delay``: seconds between asking for a path and having it.
    ``presets`` = {'leader': [names], 'follower': [names]}: rover_drive config/presets files
    on top of the shipped rover configs. ``comm`` = {'delay': s, 'outages': [(t0, t1)]}: a
    slow / interrupted host <-> rover link (None = ideal)."""
    loop = LayoutLoop(layout, seed, localization, profile, plan_delay, presets, comm)
    for name in part.split('+'):          # 'gear_01+bolt_01': one trip with both parts
        if not loop.engine.dispatch(name):
            raise ValueError(f'part {name!r} was not accepted by the mission engine')
    try:
        run_scenario(loop.world, loop.controllers(), duration=duration, stop_on_arrival=False)
    except _Finished:
        pass
    return loop.result(layout, part)


if __name__ == '__main__':
    layout_arg, part_arg = sys.argv[1], sys.argv[2]
    result = run_layout(layout_arg, part_arg, float(sys.argv[3]) if len(sys.argv) > 3 else 240.0,
                        localization=os.environ.get('LOCALIZATION'), profile=os.environ.get('PROFILE'))
    if os.environ.get('VERBOSE'):
        for event in result.events:
            print('  ', event)
    print(result.row())
