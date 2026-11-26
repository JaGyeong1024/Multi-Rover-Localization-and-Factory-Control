"""Safety gate logic: the last, independent check before the base."""
import math
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from . import state as st
from .config import GateConfig
from .corridor import PathCorridor
from .depth_front import band_slopes
from .front_memory import CELL_M, FrontMemory
from .lidar import (front_obstacle, lead_ahead, lead_gap, obstacle_points, scan_front,
                    scan_points)
from .limits import stopping_speed

OK = 'OK'
LIMITED = 'LIMITED'
STOPPED = 'STOPPED'

_V_EPS = 1e-6  # below this an input speed is treated as zero
# Reasons only the tracker can detect; relayed so the host sees one complete list.
TRACKER_REASONS = ('STALE_LEAD', 'POSE_JUMP', 'OFF_PATH', 'STUCK', 'QUALITY_LOW')
# front_path: spacing of the points a band distance is spread into across the band [m].
_BAND_SPACING_M = 0.02
# Extra look-ahead on top of _path_reach so a border point is not lost to rounding [m].
_REACH_MARGIN_M = 0.05


class GateOutput(NamedTuple):
    v: float
    w: float           # after angular_sign
    state: str         # OK | LIMITED | STOPPED
    reasons: tuple
    speed_scale: float
    blocked_x: float   # LIDAR_STOP: estimated global position of the blocker, else NaN
    blocked_y: float


def front_distance(ranges, angle_min: float, angle_increment: float, half_width: float,
                   min_range: float, yaw_offset: float = 0.0) -> float:
    """Distance to the nearest scan point inside the forward corridor (inf if none)."""
    cfg = GateConfig(lidar_half_width=half_width, lidar_min_range=min_range,
                     lidar_yaw_offset=yaw_offset, lidar_front_half_angle=math.pi / 2.0,
                     lidar_min_valid_beams=0, lidar_ignore_sectors=())
    return scan_front(ranges, angle_min, angle_increment, 0.0, cfg).dist


def _to_body(pose: tuple, x: float, y: float) -> tuple:
    dx, dy = x - pose[0], y - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return c * dx + s * dy, -s * dx + c * dy


def _to_global(pose: tuple, bx: float, by: float) -> tuple:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pose[0] + c * bx - s * by, pose[1] + s * bx + c * by


def _expired(stamp: Optional[float], timeout: float, now: float) -> bool:
    return stamp is None or now - stamp > timeout


def _path_reach(cfg: GateConfig) -> float:
    v = cfg.max_linear_speed
    stop = (cfg.lidar_to_front + cfg.lidar_stop_base + v * cfg.lidar_reaction_s
            + v * v / (2.0 * cfg.lidar_brake_decel))
    return max(stop, cfg.front_lost_near_m) + _REACH_MARGIN_M


class _Reading:
    """One evaluation of the front measurements: distance, its body-frame point, and the
    last numeric distance with its global point (for "vanished while close")."""

    def __init__(self):
        self.dist = math.inf                    # see module docstring for inf / NaN
        self.block = (math.nan, math.nan)       # body-frame point behind the last finite dist
        self.last_valid = math.inf
        self.last_valid_xy: Optional[tuple] = None   # global (None = pose unknown)
        self.stamp: Optional[float] = None      # receive time of the measurement
        self.cmd: Optional[st.PathCmd] = None   # on the path: the command it was taken on

    def record(self, dist: float, point: Optional[tuple], pose: Optional[tuple], now: float,
               cmd: Optional[st.PathCmd] = None) -> None:
        """``pose`` = the aligned pose to place the point with (None = unknown)."""
        self.dist, self.stamp, self.cmd = dist, now, cmd
        if math.isnan(dist):
            return
        self.last_valid = dist
        self.block = point if point is not None else (dist, 0.0)
        known = pose is not None and math.isfinite(dist)
        self.last_valid_xy = _to_global(pose, *self.block) if known else None


class SafetyGate:
    """Stateful only in "latest input + receive time"; no filtering."""

    def __init__(self, cfg: GateConfig):
        self._cfg = cfg
        self._input = (0.0, 0.0)
        self._input_time: Optional[float] = None
        self._pose: Optional[tuple] = None
        self._pose_time: Optional[float] = None
        self._cmd_time: Optional[float] = None
        self._estop = False
        self._aligned = False
        self._straight = _Reading()            # the front straight ahead (every measurement)
        self._on_path = _Reading()             # the same measurements on the path (front_path)
        self._stop_block = (math.nan, math.nan)  # the point that caused the last front stop
        self._stop_xy: Optional[tuple] = None    # the same point, global, when a measurement gave it
        self._front_stop_time: Optional[float] = None  # last time the front said "stop"
        self._path_cmd: Optional[st.PathCmd] = None  # front_path: current command (tracker rule)
        self._path_cmd_time: Optional[float] = None  # when it was last accepted
        self._corridor: Optional[PathCorridor] = None  # its stop zone (None = not a path)
        self._memory_cover: Optional[tuple] = None   # cache, see _remembered_cover
        self._reach = _path_reach(cfg)
        self._band = band_slopes(cfg.depth_col_min, cfg.depth_col_max, cfg.depth_hfov)
        self._lead_name = ''
        self._lead: Optional[tuple] = None     # (x, y, receive time) of the lead rover
        self._lead_gap = (math.nan, False, None)  # (free gap, lead trusted, scan receive time)
        self._last_gap = math.inf              # last measured free gap to the current lead
        self._gap_stop_time: Optional[float] = None  # last time the gap said "stop"
        self._approach = (False, None)         # (active, receive time) from the tracker
        self._tracker = ('', '', None)  # (state, detail, receive time) from nav_feedback
        self._memory = FrontMemory(cfg.front_lost_near_m) if cfg.front_memory else None
        self._memory_time: Optional[float] = None   # last time the memory was moved
        self._memory_pose: Optional[tuple] = None   # aligned pose it was moved to (None = DR)
        self._sent = (0.0, 0.0)                     # last output Twist (REP-103) for DR

    @property
    def lead_name(self) -> str:
        """Lead rover named by the latest path command ('' = none)."""
        return self._lead_name

    @property
    def lead_gap(self) -> float:
        """Free gap to the lead from the latest scan [m]; NaN = not measured."""
        return self._lead_gap[0]

    # ------------------------------------------------------------- inputs
    def on_input(self, v: float, w: float, now: float) -> None:
        self._input, self._input_time = (v, w), now

    def on_pose(self, x: float, y: float, yaw: float, now: float) -> None:
        """A non-finite sample is not a position: it does not refresh (-> STALE_POSE)."""
        if all(math.isfinite(c) for c in (x, y, yaw)):
            self._pose, self._pose_time = (x, y, yaw), now
            if self._corridor is not None:
                self._corridor.update(x, y)

    def on_cmd(self, now: float, lead_robot: str = '', cmd: Optional[st.PathCmd] = None) -> None:
        """Path command received; with ``front_path`` ``cmd`` also gives the stop zone."""
        self._cmd_time = now
        if lead_robot != self._lead_name:
            self._lead_name, self._lead = lead_robot, None
            self._lead_gap, self._last_gap = (math.nan, False, None), math.inf
        if self._cfg.front_path and cmd is not None:
            self._take_path(cmd, now)

    def _take_path(self, cmd: st.PathCmd, now: float) -> None:
        kind = st.classify_command(self._path_cmd, cmd)
        if kind == st.REJECT:
            return
        self._path_cmd_time = now
        if kind != st.NEW:
            return
        self._path_cmd, self._corridor = cmd, None
        cfg = self._cfg
        if cmd.mode != st.MODE_FOLLOW or not cmd.poses or not st.command_is_finite(cmd) \
                or cfg.lidar_half_width <= 0.0:
            return
        self._corridor = PathCorridor(cmd.poses, cfg.lidar_half_width, cfg.lidar_to_front,
                                      cfg.projection_window, cfg.init_tie_margin)
        if self._pose is not None:
            self._corridor.update(self._pose[0], self._pose[1])

    def on_lead_pose(self, x: float, y: float, now: float) -> None:
        self._lead = (x, y, now)

    def on_approach(self, active: bool, now: float) -> None:
        """Tracker says the rover is on the final approach to a pick goal."""
        self._approach = (active, now)

    def set_estop(self, active: bool) -> None:
        self._estop = active

    def set_aligned(self, aligned: bool) -> None:
        self._aligned = aligned

    def on_front_distance(self, dist: float, now: float, point: Optional[tuple] = None) -> None:
        """Front distance; ``point`` = blocker (body frame). Without it (band distance)
        the path evaluation uses points across the band at that depth."""
        self._straight.record(dist, point, self._pose if self._aligned else None, now)
        corridor = self._loaded_corridor(now)
        if point is None and corridor is not None:
            self._record_on_path(corridor, dist, now, *self._band_points(dist))

    def _band_points(self, depth: float) -> tuple:
        if not math.isfinite(depth):
            return np.zeros(0), np.zeros(0)
        right, left = self._band
        count = int(math.ceil(depth * (left - right) / _BAND_SPACING_M)) + 1
        by = depth * np.linspace(right, left, max(count, 2))
        return np.full(by.shape, depth), by

    def _record_on_path(self, corridor: PathCorridor, dist: float, now: float, bx, by) -> None:
        point = None
        if not math.isnan(dist):
            hit = corridor.nearest(self._pose, bx, by, self._reach)
            dist, point = ((math.inf, (math.nan, math.nan)) if hit is None
                           else (hit[0], (float(bx[hit[1]]), float(by[hit[1]]))))
        self._on_path.record(dist, point, self._pose, now, self._path_cmd)

    def on_scan(self, ranges, angle_min: float, angle_increment: float, range_min: float,
                now: float) -> None:
        """LaserScan -> front distance (lead left out) and free gap to the lead."""
        pts = scan_points(ranges, angle_min, angle_increment, range_min, self._cfg)
        lead_xy = self._lead_in_body(now)
        if lead_xy is not None and not lead_ahead(lead_xy, self._cfg):
            lead_xy = None      # beside / behind me: an ordinary obstacle, no gap to keep
        front = front_obstacle(pts, self._cfg, lead_xy)
        self.on_front_distance(front.dist, now, (front.x, front.y))
        corridor = self._loaded_corridor(now)
        if corridor is not None:
            candidates = obstacle_points(pts, self._cfg, lead_xy)
            self._record_on_path(corridor, front.dist, now, pts.x[candidates],
                                 pts.y[candidates])
        gap = lead_gap(pts, self._cfg, lead_xy).gap
        self._lead_gap = (gap, lead_xy is not None, now)
        if not math.isnan(gap):
            self._last_gap = gap

    def on_depth_scan(self, ranges, angle_min: float, angle_increment: float, range_min: float,
                      now: float) -> None:
        """Depth scan (camera frame taken as body frame) -> front memory; no-op if off."""
        if self._memory is None:
            return
        self._move_memory(now)
        self._memory.update(ranges, angle_min, angle_increment, range_min)
        self._memory_cover = None

    def _move_memory(self, now: float) -> None:
        cfg = self._cfg
        pose_ok = (self._pose is not None and self._aligned
                   and not _expired(self._pose_time, cfg.pose_timeout_s, now))
        if pose_ok:
            if self._memory_pose is not None and self._memory_pose != self._pose:
                dx, dy = _to_body(self._memory_pose, self._pose[0], self._pose[1])
                dyaw = math.atan2(math.sin(self._pose[2] - self._memory_pose[2]),
                                  math.cos(self._pose[2] - self._memory_pose[2]))
                self._memory.move(dx, dy, dyaw)
            self._memory_pose = self._pose
        else:
            dt = 0.0 if self._memory_time is None else max(now - self._memory_time, 0.0)
            v, w = self._sent
            dyaw = w * dt
            self._memory.move(v * dt * math.cos(dyaw / 2.0), v * dt * math.sin(dyaw / 2.0), dyaw)
            self._memory_pose = self._memory_cover = None
        self._memory_time = now

    def _remembered_ahead(self, corridor: Optional[PathCorridor]) -> Optional[tuple]:
        if self._memory is None:
            return None
        if corridor is None:
            point = self._memory.nearest_ahead(self._cfg.lidar_half_width,
                                               self._cfg.lidar_front_half_angle)
            return None if point is None else (point[0], point)
        pts = self._memory.points()
        if not len(pts):
            return None
        along = self._remembered_cover(corridor, pts) - corridor.progress
        along[along <= 0.0] = math.inf               # passed: behind the cross-section
        i = int(np.argmin(along))
        if not math.isfinite(along[i]):
            return None
        return float(along[i]), (float(pts[i, 0]), float(pts[i, 1]))

    def _remembered_cover(self, corridor: PathCorridor, pts: np.ndarray) -> np.ndarray:
        cached = self._memory_cover
        progress = corridor.progress
        if cached is None or cached[2] is not self._path_cmd or len(cached[0]) != len(pts) \
                or progress - cached[1] > _REACH_MARGIN_M:
            cover = corridor.first_cover(self._pose, pts[:, 0], pts[:, 1], self._reach)
            self._memory_cover = cached = (cover, progress, self._path_cmd)
        return cached[0]

    def _loaded_corridor(self, now: float) -> Optional[PathCorridor]:
        cfg = self._cfg
        if self._corridor is None or _expired(self._path_cmd_time, cfg.cmd_timeout_s, now):
            return None
        if not self._aligned or _expired(self._pose_time, cfg.pose_timeout_s, now):
            return None
        return self._corridor

    def _front_mode(self, now: float) -> Tuple[_Reading, Optional[PathCorridor]]:
        corridor = self._loaded_corridor(now)
        on_path = self._on_path
        if corridor is None or not corridor.follows(self._pose[2], self._cfg.heading_free_rad):
            return self._straight, None
        if on_path.stamp != self._straight.stamp or on_path.cmd is not self._path_cmd:
            return self._straight, None
        return on_path, corridor

    def _lead_in_body(self, now: float) -> Optional[tuple]:
        cfg = self._cfg
        if not (cfg.lidar_exclude_lead and self._lead_name and self._aligned):
            return None
        if self._lead is None or _expired(self._lead[2], cfg.lead_timeout_s, now):
            return None
        if _expired(self._pose_time, cfg.pose_timeout_s, now):
            return None
        return _to_body(self._pose, self._lead[0], self._lead[1])

    def on_tracker(self, state: str, detail: str, now: float) -> None:
        """Tracker feedback, for reporting only (never changes the output Twist)."""
        self._tracker = (state, detail, now)

    # --------------------------------------------------------------- step
    def step(self, now: float) -> GateOutput:
        if self._memory is not None:
            self._move_memory(now)
        stops = self._stale_reasons(now)
        limits: List[str] = []
        self._relay_tracker(now, stops, limits)
        v_in, w_in = self._usable_input(stops)
        v, w = self._clamp(v_in, w_in, limits)
        v, w = self._align_gate(v, w, stops, limits)
        v, w = self._front_gate(v, w, now, stops, limits)
        v, w = self._gap_gate(v, w, now, stops, limits)
        v, w = self._bounded(v, w, stops)
        self._sent = (v, w)
        state = STOPPED if stops else (LIMITED if limits else OK)
        bx, by = self._blocked_position(stops)
        return GateOutput(v, self._cfg.angular_sign * w, state, tuple(stops + limits),
                          self._scale(v_in, v, bool(stops)), bx, by)

    def _stale_reasons(self, now: float) -> List[str]:
        cfg = self._cfg
        checks = (
            (self._estop, 'ESTOP'),
            (_expired(self._input_time, cfg.input_timeout_s, now), 'INPUT_TIMEOUT'),
            (_expired(self._pose_time, cfg.pose_timeout_s, now), 'STALE_POSE'),
            (_expired(self._cmd_time, cfg.cmd_timeout_s, now), 'STALE_CMD'),
        )
        return [reason for active, reason in checks if active]

    def _relay_tracker(self, now: float, stops: List[str], limits: List[str]) -> None:
        state, detail, stamp = self._tracker
        if detail not in TRACKER_REASONS or _expired(stamp, self._cfg.input_timeout_s, now):
            return
        (stops if state == STOPPED else limits).append(detail)

    def _usable_input(self, stops: List[str]) -> tuple:
        if 'INPUT_TIMEOUT' in stops:
            return 0.0, 0.0
        v, w = self._input
        if math.isfinite(v) and math.isfinite(w):
            return v, w
        stops.append('BAD_INPUT')
        return 0.0, 0.0

    def _bounded(self, v: float, w: float, stops: List[str]) -> tuple:
        cfg = self._cfg
        if not (math.isfinite(v) and math.isfinite(w)) and 'BAD_INPUT' not in stops:
            stops.append('BAD_INPUT')
        if stops:
            return 0.0, 0.0
        return (min(max(v, -cfg.max_reverse_speed), cfg.max_linear_speed),
                min(max(w, -cfg.max_angular_speed), cfg.max_angular_speed))

    def _clamp(self, v: float, w: float, limits: List[str]) -> tuple:
        cfg = self._cfg
        v_c = min(max(v, -cfg.max_reverse_speed), cfg.max_linear_speed)
        w_c = w * (v_c / v) if abs(v) > _V_EPS else w
        w_c = min(max(w_c, -cfg.max_angular_speed), cfg.max_angular_speed)
        if v_c != v or w_c != w:
            limits.append('SPEED_CLAMP')
        return v_c, w_c

    def _align_gate(self, v: float, w: float, stops: List[str], limits: List[str]) -> tuple:
        cfg = self._cfg
        if self._aligned:
            return v, w
        if 0.0 <= v <= cfg.align_pass_speed and abs(w) <= cfg.align_pass_omega:
            limits.append('NOT_ALIGNED')
            return v, 0.0
        stops.append('NOT_ALIGNED')
        return 0.0, 0.0

    def _front_enabled(self) -> bool:
        return self._cfg.lidar_enabled or bool(self._cfg.front_distance_topic)

    def _approach_exempt(self, now: float) -> bool:
        cfg = self._cfg
        active, stamp = self._approach
        return cfg.front_pick_exempt and active and not _expired(stamp, cfg.approach_timeout_s,
                                                                 now)

    def _front_gate(self, v: float, w: float, now: float, stops: List[str],
                    limits: List[str]) -> tuple:
        cfg = self._cfg
        if not self._front_enabled() or self._approach_exempt(now):
            return v, w
        if _expired(self._straight.stamp, cfg.scan_timeout_s, now):
            stops.append('STALE_SCAN')
            return 0.0, 0.0
        reading, corridor = self._front_mode(now)
        allowed, block = self._front_allowed(reading, corridor), reading.block
        remembered = self._remembered_ahead(corridor)
        memory_limits = False
        if remembered is not None:
            # A remembered point is not re-measured, so slowing alone never crosses the
            # stop line; within one memory cell of it, report a stop.
            dist, point = remembered
            at_line = self._stopping_speed(dist - CELL_M) <= 0.0
            from_memory = 0.0 if at_line else self._stopping_speed(dist)
            if from_memory <= allowed:      # the remembered point is what limits
                allowed, block, memory_limits = from_memory, point, True
        if allowed <= 0.0:
            self._front_stop_time, self._stop_block = now, block
            self._stop_xy = None if memory_limits else reading.last_valid_xy
        if not _expired(self._front_stop_time, cfg.lidar_resume_s, now):
            stops.append('LIDAR_STOP')  # held: an object at the edge must not cause stop/go
            return 0.0, 0.0
        if math.isnan(reading.dist):
            v, w = self._blind_gate(v, w, stops, limits)
            if remembered is None:
                return v, w
        return self._cap(v, w, allowed, 'LIDAR_SLOW', limits)

    def _front_allowed(self, reading: _Reading, corridor: Optional[PathCorridor]) -> float:
        cfg = self._cfg
        if math.isnan(reading.dist):
            lost_close = cfg.front_lost_stops and reading.last_valid < cfg.front_lost_near_m
            return 0.0 if lost_close and self._lost_object_ahead(reading, corridor) else math.inf
        return self._stopping_speed(reading.dist)

    def _stopping_speed(self, dist: float) -> float:
        cfg = self._cfg
        return stopping_speed(dist - cfg.lidar_to_front, cfg.lidar_stop_base,
                              cfg.lidar_brake_decel, cfg.lidar_reaction_s)

    def _lost_object_ahead(self, reading: _Reading, corridor: Optional[PathCorridor]) -> bool:
        cfg = self._cfg
        if reading.last_valid_xy is None or self._pose is None or not self._aligned:
            return True
        bx, by = _to_body(self._pose, *reading.last_valid_xy)
        if corridor is not None:
            return corridor.nearest(self._pose, [bx], [by], self._reach) is not None
        return bx > 0.0 and abs(by) <= cfg.lidar_half_width \
            and abs(math.atan2(by, bx)) <= cfg.lidar_front_half_angle

    def _blind_gate(self, v: float, w: float, stops: List[str], limits: List[str]) -> tuple:
        cfg = self._cfg
        if cfg.lidar_blind_speed <= 0.0:
            stops.append('LIDAR_BLIND')
            return 0.0, 0.0
        limits.append('LIDAR_BLIND')
        return self._cap(v, w, cfg.lidar_blind_speed, '', limits)

    def _gap_gate(self, v: float, w: float, now: float, stops: List[str],
                  limits: List[str]) -> tuple:
        allowed = self._gap_allowed(now)
        if allowed <= 0.0:
            self._gap_stop_time = now
        if not _expired(self._gap_stop_time, self._cfg.lidar_resume_s, now):
            stops.append('GAP_MIN')
            return 0.0, 0.0
        return self._cap(v, w, allowed, 'GAP_SLOW', limits)

    def _gap_allowed(self, now: float) -> float:
        cfg = self._cfg
        gap, trusted, stamp = self._lead_gap
        if not (self._lead_name and trusted) or _expired(stamp, cfg.scan_timeout_s, now):
            return math.inf  # untrusted: the lead is an ordinary obstacle for the front stop
        if math.isnan(gap):
            return 0.0 if self._last_gap < cfg.gap_hard_min else math.inf
        return stopping_speed(gap, cfg.gap_hard_min, cfg.lidar_brake_decel, cfg.lidar_reaction_s)

    @staticmethod
    def _cap(v: float, w: float, allowed: float, reason: str, limits: List[str]) -> tuple:
        if v <= allowed:
            return v, w
        if reason:
            limits.append(reason)
        return allowed, w * allowed / v

    def _blocked_position(self, stops: List[str]) -> tuple:
        if 'LIDAR_STOP' not in stops or self._pose is None or not self._aligned:
            return math.nan, math.nan
        if self._stop_xy is not None:
            return self._stop_xy
        if math.isnan(self._stop_block[0]):
            return math.nan, math.nan
        return _to_global(self._pose, *self._stop_block)

    @staticmethod
    def _scale(v_in: float, v_out: float, stopped: bool) -> float:
        if abs(v_in) <= _V_EPS:
            return 0.0 if stopped else 1.0
        return min(max(v_out / v_in, 0.0), 1.0)
