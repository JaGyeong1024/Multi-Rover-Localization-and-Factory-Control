"""Mission settings and scenario (ROS-free), loaded from YAML (``mission.yaml`` + overlays,
``mission_scenario.yaml``); fields are documented there. Units: m, rad, s."""
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import yaml

from host_planning.types import Bounds, Box, PlanParams


@dataclass(frozen=True)
class RobotSettings:
    """Per-rover command values and planner parameters."""

    speed_cap: float = 0.0          # 0 = rover default
    goal_pos_tol: float = 0.05
    goal_yaw_tol: float = 0.10      # 0 = heading not checked
    arrive_pos_tol: float = 0.08    # host arrival check
    arrive_yaw_tol: float = 0.15
    plan: PlanParams = field(default_factory=PlanParams)
    # cargo area (x, y, length, width), tracked-point frame (x fwd, y left); None = no cargo
    cargo: Optional[Tuple[float, float, float, float]] = None
    front_exempt_reach: float = 0.0


@dataclass(frozen=True)
class PlatoonSettings:

    enabled: bool = True
    lead_robot: str = 'leader'
    follower: str = 'follower'
    gap: float = 0.60
    conflict_dist: float = 0.50
    avoid_parked_peer: bool = True
    peer_radius: float = 0.20
    dock_turn: bool = False
    hold_ride: bool = True


@dataclass(frozen=True)
class FreshnessSettings:

    pose_timeout: float = 0.5
    feedback_timeout: float = 2.0
    safety_timeout: float = 2.0


@dataclass(frozen=True)
class SupervisorSettings:
    """Exception-handling thresholds and retry budgets (s, m)."""

    hold_pause_s: float = 5.0
    off_path_max_replans: int = 3
    stuck_max_replans: int = 1
    goal_miss_max_reentries: int = 2
    goal_miss_confirm_s: float = 2.0
    reentry_approach_len: float = 0.60
    lidar_block_s: float = 3.0
    assumed_obstacle_depth: float = 0.50   # along the rover heading
    assumed_obstacle_width: float = 0.50   # across it
    lidar_obstacle_ttl: float = 30.0
    fixture_match_dist: float = 0.10
    plan_retry_period: float = 2.0
    max_consecutive_pick_failures: int = 2


@dataclass(frozen=True)
class PickSettings:
    """Pick completion handling."""

    picker: str = 'leader'
    gate_on_phase: bool = True
    picking_states: Tuple[str, ...] = ('PICKING',)
    error_states: Tuple[str, ...] = ('ERROR',)
    # where the arm drops a part, picker's pick-goal frame (forward, left), m; None = off
    drop_point: Optional[Tuple[float, float]] = None
    drop_margin: float = 0.03


@dataclass(frozen=True)
class MissionSettings:

    robots: Dict[str, RobotSettings]
    bounds: Optional[Bounds] = None
    fixtures: Tuple[Box, ...] = ()  # known static obstacles (shelves)
    platoon: PlatoonSettings = field(default_factory=PlatoonSettings)
    freshness: FreshnessSettings = field(default_factory=FreshnessSettings)
    supervisor: SupervisorSettings = field(default_factory=SupervisorSettings)
    pick: PickSettings = field(default_factory=PickSettings)
    tick_rate: float = 5.0
    republish_rate: float = 2.0     # Hz
    start_grace: float = 2.0
    idle_go_home: bool = False
    estop_resume_window: float = 1.0
    idle_heading: Optional[float] = None  # rad; None = off
    idle_turn_timeout: float = 30.0
    rotation_centre_error: float = 0.08

    @property
    def robot_names(self):
        """Rover names; the platoon lead first so its path exists before the follower's."""
        lead = self.platoon.lead_robot
        return sorted(self.robots, key=lambda name: name != lead)


@dataclass(frozen=True)
class PartSpec:

    id: str
    x: float
    y: float
    leader_goal: str
    follower_goal: str


@dataclass(frozen=True)
class Scenario:

    leader_standby: str
    follower_unload: str
    parts: Dict[str, PartSpec]
    follower_home: Optional[str] = None
    capacity: int = 1
    pick_cooldown: float = 0.0
    pick_dwell: float = 80.0
    pick_timeout_is_failure: bool = True
    phase_timeout: float = 120.0

    def goal_names(self, part=None):
        names = [self.leader_standby, self.follower_unload]
        if self.follower_home:
            names.append(self.follower_home)
        if part is not None:
            names += [part.leader_goal, part.follower_goal]
        return names


def _build(cls, data, label):
    """Dataclass from a mapping, rejecting unknown keys (typos must not pass silently)."""
    data = dict(data or {})
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f'{label}: unknown keys {unknown}')
    for name, value in data.items():
        if isinstance(value, list):
            data[name] = tuple(value)
    return cls(**data)


def _robot_settings(name, data):
    data = dict(data or {})
    plan = _build(PlanParams, data.pop('plan', None), f'robots.{name}.plan')
    return dataclasses.replace(_build(RobotSettings, data, f'robots.{name}'), plan=plan)


def _fixture(index, data):
    data = dict(data or {})
    label = f'fixtures[{index}] {data.pop("name", "")}'.strip()
    return _build(Box, data, label)


def settings_from_dict(data):
    """MissionSettings from the parsed ``mission.yaml`` mapping."""
    data = dict(data or {})
    robots = {name: _robot_settings(name, entry)
              for name, entry in (data.pop('robots', None) or {}).items()}
    if not robots:
        raise ValueError('settings: "robots" must list at least one rover')
    bounds = data.pop('bounds', None)
    sections = {
        'bounds': _build(Bounds, bounds, 'bounds') if bounds else None,
        'fixtures': tuple(_fixture(i, f) for i, f in enumerate(data.pop('fixtures', None) or [])),
        'platoon': _build(PlatoonSettings, data.pop('platoon', None), 'platoon'),
        'freshness': _build(FreshnessSettings, data.pop('freshness', None), 'freshness'),
        'supervisor': _build(SupervisorSettings, data.pop('supervisor', None), 'supervisor'),
        'pick': _build(PickSettings, data.pop('pick', None), 'pick'),
    }
    return _build(MissionSettings, {'robots': robots, **sections, **data}, 'settings')


def scenario_from_dict(data):
    """Scenario from the parsed ``mission_scenario.yaml`` mapping."""
    data = dict(data or {})
    parts = {p['id']: _build(PartSpec, p, f'parts.{p.get("id")}')
             for p in data.pop('parts', None) or []}
    return _build(Scenario, {'parts': parts, **data}, 'scenario')


def merge_dicts(base, overlay):
    """``base`` with ``overlay`` merged in: dicts merge recursively, any other value replaces."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_settings_yaml(path, overlays=()):
    """Raw ``mission.yaml`` mapping with the overlay files merged over it (later wins)."""
    merged = {}
    for file in (path, *overlays):
        with open(file) as stream:
            merged = merge_dicts(merged, yaml.safe_load(stream) or {})
    return merged


def load_settings(path, overlays=()):
    return settings_from_dict(read_settings_yaml(path, overlays))


def load_scenario(path):
    with open(path) as stream:
        return scenario_from_dict(yaml.safe_load(stream))
