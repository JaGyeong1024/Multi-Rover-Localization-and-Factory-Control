"""Plain data types of the planning library (no ROS dependency).

Every pose is in the ``uwb_global`` frame; units are metres and radians, yaw is
positive counter-clockwise from +x, in (-pi, pi].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    r: float


@dataclass(frozen=True)
class Box:
    """Oriented rectangle obstacle; ``width`` along ``yaw``, ``height`` across, ``pad`` grows it
    uniformly (rounded rectangle). Clearance is measured to this exact shape."""

    x: float
    y: float
    width: float
    height: float
    yaw: float = 0.0
    pad: float = 0.0


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned area the path points (tracked point) must stay inside."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass(frozen=True)
class PlanParams:
    """Every tunable of the planner (metres, radians); see mission.yaml."""

    spacing: float = 0.05            # max distance between path points
    dense_factor: int = 5            # internal sampling = spacing / dense_factor
    min_turn_radius: float = 0.30
    robot_radius: float = 0.20       # when no footprint is given
    # body rectangle (front, rear, half_width) from the tracked point; None = disc
    footprint: Optional[Tuple[float, float, float]] = None
    clearance: float = 0.05          # gap between body and obstacle surface
    bounds_margin: float = 0.0
    # rover front-stop zone ahead of the tracked point (start, length, half_width); None = off
    front_zone: Optional[Tuple[float, float, float]] = None
    # poses on the goal line this close before the goal skip the zone (0 = none)
    front_zone_runin: float = 0.0
    approach_len: float = 0.40       # straight run-in before the goal
    min_approach_len: Optional[float] = None  # None = never shorten it
    approach_shrink_steps: int = 3
    on_line_tol: float = 0.01        # lateral, "already on the run-in line"
    heading_tol: float = 0.02        # rad, "already aligned"
    direct_skew_tol: float = 0.0     # rad; line-to-goal within this of goal heading: go straight
    pivot_step: float = 0.0          # rad; 0 = never pivot
    max_pivot: float = 2.88          # largest turn (rad; < pi: direction stays unambiguous)
    pivot_sweep_step: float = 0.05   # rad
    pivot_cost: float = 0.5          # m per rad
    pivot_offset: float = 0.0        # rotation centre behind the tracked point
    tangent_scales: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.33, 0.45)
    curv_weight: float = 0.05        # cost = length + curv_weight * max|k|
    curv_ramp_len: float = 0.10
    dubins_radius_scales: Tuple[float, ...] = (1.0, 1.5)
    relax_iterations: int = 12       # Gauss-Newton steps
    relax_tol: float = 1e-5
    grid_res: float = 0.05           # A* cell size
    plan_buffer: float = 0.05        # extra inflation of the first attempt
    inflate_step: float = 0.05       # added per retry
    max_retries: int = 2
    snap_radius: float = 0.30        # max start/goal snap to a free cell
    auto_bounds_pad: float = 1.0     # when no Bounds is given
    guard_scale: float = 0.9         # end-guard radius / min_turn_radius
    platoon_search_step: float = 0.10
    platoon_detour_ratio: float = 1.3   # max connector length / chord
    platoon_min_shared: float = 0.30

    @property
    def max_curvature(self) -> float:
        return 1.0 / self.min_turn_radius

    @property
    def dense_step(self) -> float:
        return self.spacing / self.dense_factor

    @property
    def required_clearance(self) -> float:
        """Disc body: minimum path-point distance to an obstacle surface."""
        return self.robot_radius + self.clearance

    @property
    def lateral_radius(self) -> float:
        """Body half width across the heading (grid search inflation)."""
        return self.robot_radius if self.footprint is None else self.footprint[2]


@dataclass(frozen=True)
class PlatoonPlan:
    """Follower path riding on the leader path. ``join_s`` / ``branch_s``: arc lengths where the
    shared stretch starts / ends (``leader_*_s``: along the leader path); None if not shared."""

    path: List[Pose2D]
    shared: bool
    join_s: Optional[float] = None
    branch_s: Optional[float] = None
    leader_join_s: Optional[float] = None
    leader_branch_s: Optional[float] = None
