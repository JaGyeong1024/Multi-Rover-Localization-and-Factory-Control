"""DriveController: one control step = project, steer, limit, ramp, monitor."""
import math
from typing import List, NamedTuple, Optional, Sequence

from . import state as st
from .config import DriveConfig
from .depth_front import DepthScan
from .face_align import FEW_POINTS, TOO_SKEWED, AlignRun, GyroYaw, RememberedFace, fit_face
from .limits import (GapLimiter, Limit, NO_LIMIT, OmegaQuantizer, VelocityRamp, arrival_limit,
                     curvature_limit, goal_limit, heading_limit, lidar_limit, quality_limit,
                     strictest)
from .monitor import (ArrivalWindow, PivotMonitor, PoseJumpFilter, StuckMonitor, is_off_path,
                      judge_arrival)
from .path import Path, PathProgress, Projection, wrap_angle
from .tracker import CrossTrackIntegral, steer

# Limit reasons that mean "stopped because of a fault" rather than "waiting".
_FAULT_LIMITS = ('STALE_LEAD', 'QUALITY_LOW', 'LIDAR_STOP')
# Safety gate stop reasons the tracker cannot see by itself.
_GATE_BLOCKS = ('ESTOP', 'LIDAR_STOP', 'LIDAR_BLIND', 'STALE_SCAN')


_SAME_GOAL_EPS = 1e-3  # m / rad: two commands end at the same goal pose
_STRAIGHT_KAPPA = 1e-6  # 1/m: numerically straight (collinear path points, float rounding)

# Squaring up to the fixture face (face_align); details while TRACKING / on arrival:
FACE_ALIGN = 'FACE_ALIGN'
FACE_ALIGNED = 'FACE_ALIGNED'            # arrived, square to the face (verified by a fit)
FACE_ALIGN_FAILED = 'FACE_ALIGN_FAILED'  # arrived, squareness not reached / not measurable
_FACE_MAX_TURNS = 3
_FACE_SCAN_WAIT_S = 1.0  # no new depth scan this long (10 Hz expected): no measurement


class _GapBridge(NamedTuple):
    """Path gap error minus LiDAR gap error at the last fresh LiDAR gap (bumpless switch)."""
    offset: float    # d [m]
    ranged: bool     # path gap was the straight-line range (lead left my path), not along it
    s: float         # my arc length then [m]
    stamp: float     # receive time of that LiDAR gap


class DriveOutput(NamedTuple):
    v: float         # linear speed command [m/s]
    w: float         # angular speed command [rad/s], REP-103
    state: str
    detail: str
    cmd_id: int
    progress: float  # 0..1 along the path
    epoch: int = 0   # epoch of the command being answered
    # Front-stop exemption for the gate: final approach to a pick/fixture goal, leading
    # pivot, squaring up to the face (the obstacle ahead is expected or v = 0).
    approach_active: bool = False


def _stationary(poses) -> bool:
    last = poses[-1] if poses else None
    return last is not None and all(abs(a - b) <= _SAME_GOAL_EPS
                                    for p in poses for a, b in zip(p[:2], last[:2]))


def _expired(stamp: Optional[float], timeout: float, now: float) -> bool:
    return stamp is None or now - stamp > timeout


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


class DriveController:
    """Follows the commanded path and stops by itself."""

    def __init__(self, cfg: DriveConfig):
        self._cfg = cfg
        self._state = st.DriveState()
        self._ramp = VelocityRamp(cfg)
        self._quantizer = OmegaQuantizer(cfg.omega_quantum, cfg.omega_quantize,
                                         cfg.omega_truncate_margin, cfg.max_angular_speed)
        self._gap = GapLimiter(cfg)
        self._jump = PoseJumpFilter(cfg)
        self._stuck = StuckMonitor(cfg)
        self._align = st.AlignManeuver(cfg.align_max_dist)
        self._cmd: Optional[st.PathCmd] = None
        self._cmd_time: Optional[float] = None
        self._path: Optional[Path] = None
        self._progress: Optional[PathProgress] = None
        self._pose: Optional[tuple] = None
        self._pose_time: Optional[float] = None
        self._lead: Optional[tuple] = None
        self._lead_gap: tuple = (math.nan, None)  # (LiDAR free gap, receive time)
        self._lead_joined = False                 # the lead has been on this command's path
        self._bridge: Optional[_GapBridge] = None  # d of the last LiDAR sighting (GAP_BRIDGED)
        self._bridge_floor: Optional[float] = None  # LiDAR stamp held at the command change
        self._aligned = False
        self._quality = cfg.quality_default
        self._lidar_front: Optional[float] = None
        self._gate = ('OK', (), 1.0, None)
        self._last_step: Optional[float] = None
        self._arrival = ArrivalWindow()
        self._pivot: Optional[float] = None      # pending leading pivot: target heading
        self._pivot_start: Optional[float] = None
        self._pivot_w = 0.0
        self._pivot_way = 0.0             # +1 / -1: the way round the current pivot turns
        self._pivot_check = PivotMonitor(cfg)
        self._xtrack = CrossTrackIntegral(cfg)
        self._gyro = GyroYaw()
        self._scan: Optional[tuple] = None        # (DepthScan, receive time)
        self._face: Optional[AlignRun] = None     # squaring up to the face (face_align)
        self._face_memory: Optional[RememberedFace] = None
        self._notes: List[str] = []               # lines for the node's log

    # ------------------------------------------------------------- inputs
    def on_command(self, cmd: st.PathCmd, now: float) -> str:
        """Apply the acceptance rule. Returns NEW, KEEPALIVE or REJECT."""
        kind = st.classify_command(self._cmd, cmd)
        if kind == st.REJECT:
            return kind
        self._cmd_time = now
        if kind == st.NEW:
            self._load(cmd)
        return kind

    def on_pose(self, x: float, y: float, yaw: float, now: float) -> bool:
        """Own pose sample. False if discarded (not a number, or a jump); discarded
        samples do not refresh the pose, so a stream of them ends in STALE_POSE."""
        if not _finite(x, y, yaw) or not self._jump.accept(x, y, now):
            return False
        self._pose, self._pose_time = (x, y, yaw), now
        self._arrival.add(self._pose)
        return True

    def on_lead_pose(self, x: float, y: float, now: float) -> None:
        if _finite(x, y):                       # else not a refresh: STALE_LEAD
            self._lead = (x, y, now)

    def on_lead_gap(self, gap: float, now: float) -> None:
        """Free gap to the lead from the gate's LiDAR (NaN = not seen)."""
        if math.isfinite(gap) or not math.isfinite(self._lead_gap[0]):
            self._lead_gap = (gap, now)

    def on_gyro(self, rate: float, stamp: float, now: float) -> None:
        """Own yaw rate [rad/s, REP-103] stamped ``stamp`` [s] (face_align)."""
        self._gyro.add(rate, stamp, now)

    def on_depth_scan(self, scan: DepthScan, now: float) -> None:
        """Depth scan (body frame). On the final run-in a good face fit is remembered for
        the goal, where the face may be nearer than the camera's minimum range."""
        self._scan = (scan, now)
        if self._face is not None or not self._face_wanted() or not self._gyro.fresh(now):
            return
        if self._path is None or self._pose is None or not self._on_final_run_in():
            return
        fit = fit_face(scan, self._cfg)
        if fit.ok:
            self._face_memory = RememberedFace(fit, self._gyro.yaw, self._gyro.gaps, now)

    def take_notes(self) -> List[str]:
        """Log lines since the last call (face_align measurements and verdicts)."""
        notes, self._notes = self._notes, []
        return notes

    def set_heading_aligned(self, aligned: bool) -> None:
        self._aligned = aligned

    def set_quality(self, quality: str) -> None:
        self._quality = quality

    def set_lidar_front(self, dist: Optional[float]) -> None:
        self._lidar_front = dist

    def set_gate(self, state: str, reasons: Sequence[str], scale: float, now: float) -> None:
        """Latest SafetyState (advisory: lets the tracker wait instead of pushing)."""
        self._gate = (state, tuple(reasons), scale, now)

    # --------------------------------------------------------------- step
    def step(self, now: float) -> DriveOutput:
        dt = self._tick(now)
        halt = self._halt_reason(now)
        if halt is not None:
            return self._halt(*halt, dt)
        return self._drive(now, dt)

    def _tick(self, now: float) -> float:
        period = 1.0 / self._cfg.control_rate_hz
        dt = period if self._last_step is None else now - self._last_step
        self._last_step = now
        return min(max(dt, 0.0), 2.0 * period)

    def _load(self, cmd: st.PathCmd) -> None:
        still_arrived = self._arrived_at_goal_of(cmd)
        arrived_detail = self._state.detail      # e.g. FACE_ALIGN_FAILED, kept on a re-send
        if not self._same_goal(cmd):
            self._face_memory = None             # the run-in's face fit belongs to its goal
        self._cmd = cmd
        self._gap.reset()
        self._state.reset()
        self._path = self._progress = None
        self._arrival.close()
        self._lead_joined = False
        # bridge offset belongs to the old path: relearn from a LiDAR gap newer than this command
        self._bridge, self._bridge_floor = None, self._lead_gap[1]
        self._stuck.reset()
        self._align.reset()
        self._xtrack.reset()
        self._pivot, self._pivot_start, self._pivot_w = None, None, 0.0
        self._pivot_check.reset()
        self._pivot_way = 0.0
        self._face = None
        if cmd.mode != st.MODE_FOLLOW:
            return
        if not cmd.poses:
            self._state.fail('EMPTY_PATH')
            return
        if not st.command_is_finite(cmd):
            self._state.fail('BAD_CMD')
            return
        self._pivot = cmd.pivot_heading if cmd.pivot_first and not still_arrived else None
        self._path = Path([(p[0], p[1]) for p in cmd.poses])
        self._progress = PathProgress(self._path, self._cfg.projection_window,
                                      self._cfg.init_tie_margin)
        if still_arrived:
            self._state.arrive(cmd.pick_at_goal, arrived_detail)

    def _arrived_at_goal_of(self, cmd: st.PathCmd) -> bool:
        if self._state.state not in (st.HOLDING, st.PARKED):
            return False
        if cmd.mode != st.MODE_FOLLOW or not _stationary(cmd.poses):
            return False
        return self._same_goal(cmd)

    def _same_goal(self, cmd: st.PathCmd) -> bool:
        old = self._cmd
        if old is None or not old.poses or not cmd.poses:
            return False
        return all(abs(a - b) <= _SAME_GOAL_EPS for a, b in zip(old.poses[-1], cmd.poses[-1]))

    def _halt_reason(self, now: float) -> Optional[tuple]:
        cfg = self._cfg
        if self._cmd is None or self._state.latched:
            return '', True
        if _expired(self._cmd_time, cfg.cmd_timeout_s, now):
            return 'STALE_CMD', True
        if self._cmd.mode != st.MODE_FOLLOW:
            return 'CMD_STOP', False
        if _expired(self._pose_time, cfg.pose_timeout_s, now):
            return ('POSE_JUMP' if self._jump.rejecting else 'STALE_POSE'), True
        gate = self._gate_block(now)
        return (gate, True) if gate else None

    def _gate_fresh(self, now: float) -> bool:
        return not _expired(self._gate[3], self._cfg.gate_timeout_s, now)

    def _gate_block(self, now: float) -> str:
        if not self._gate_fresh(now) or self._gate[0] != 'STOPPED':
            return ''
        return next((r for r in _GATE_BLOCKS if r in self._gate[1]), '')

    def _gate_scale(self, now: float) -> float:
        return self._gate[2] if self._gate_fresh(now) else 1.0

    def _halt(self, detail: str, hard: bool, dt: float) -> DriveOutput:
        if detail:
            self._state.stop(detail)
        self._stuck.reset()
        self._arrival.close()
        self._pivot_w = 0.0
        if hard:
            self._ramp.reset()
            self._quantizer.reset()
            return self._output(0.0, 0.0)
        return self._output(*self._ramp.step(0.0, 0.0, dt))

    def _drive(self, now: float, dt: float) -> DriveOutput:
        if self._face is not None:
            return self._face_step(now, dt)
        if self._pivot is not None:
            return self._pivot_step(now, dt)
        x, y, _ = self._pose
        if self._path.length <= self._cfg.goal_stop_margin:
            return self._judge_in_place(now)
        proj = self._progress.update(x, y)
        if self._aligned and is_off_path(self._cfg, proj.dist):
            # only with a heading: while unaligned the rover creeps straight (align_max_dist)
            return self._halt('OFF_PATH', True, dt)
        align = self._align.update(self._aligned, x, y)
        if align == st.ALIGN_FAILED:
            self._state.fail('ALIGN_FAILED')
            return self._halt('', True, dt)
        curvature, limits = self._steer(proj, align == st.ALIGNED, dt)
        limit = strictest(limits + self._common_limits(proj.s, now))
        v, w = self._ramp.step(limit.speed, curvature, dt)
        w = self._quantizer.apply(w)
        if self._stuck.update(v * self._gate_scale(now), x, y, dt):
            return self._halt('STUCK', True, dt)
        self._report(limit, align)
        self._check_arrival(proj.s, v, now)
        return self._output(v, w)

    def _pivot_step(self, now: float, dt: float) -> DriveOutput:
        cfg = self._cfg
        if self._pivot_start is None:
            x0, y0 = self._cmd.poses[0][:2]
            if math.hypot(self._pose[0] - x0, self._pose[1] - y0) > cfg.pivot_pos_tol:
                self._pivot = None              # not where the turn was planned: just track
                return self._drive(now, dt)
            self._pivot_start = now
        if now - self._pivot_start > cfg.pivot_timeout_s:
            return self._pivot_stuck(dt)
        error = wrap_angle(self._pivot - self._pose[2])
        if not self._aligned or abs(error) <= cfg.pivot_tol:
            return self._pivot_hold()
        error = self._same_way_round(error)
        passed = self._pivot_w * self._gate_scale(now)  # what the gate let through (as STUCK)
        if self._pivot_check.update(error, self._pose[2], passed, dt):
            return self._pivot_stuck(dt)
        self._state.track('PIVOT')
        return self._output(0.0, self._turn_rate(error, dt))

    def _turn_rate(self, error: float, dt: float) -> float:
        cfg = self._cfg
        rate = min(max(cfg.pivot_gain * abs(error), cfg.pivot_min_omega), cfg.pivot_omega)
        step = cfg.max_angular_accel * dt
        self._pivot_w += min(max(math.copysign(rate, error) - self._pivot_w, -step), step)
        return self._quantizer.apply(self._pivot_w)

    def _same_way_round(self, error: float) -> float:
        if not self._pivot_way:
            self._pivot_way = math.copysign(1.0, error)
        if error * self._pivot_way < 0.0 and abs(error) > math.pi / 2.0:
            error += self._pivot_way * 2.0 * math.pi
        return error

    def _pivot_hold(self) -> DriveOutput:
        self._pivot_w = 0.0
        self._ramp.reset()
        self._quantizer.reset()
        self._pivot_check.pause()
        if self._aligned:
            self._pivot = None                  # done: track the path from the next step
        self._state.track('PIVOT' if self._aligned else 'ALIGNING')
        return self._output(0.0, 0.0)

    def _pivot_stuck(self, dt: float) -> DriveOutput:
        self._pivot = None
        return self._halt('STUCK', True, dt)

    # ------------------------------------------------ squaring up to the face
    def _face_wanted(self) -> bool:
        cmd = self._cmd
        return self._cfg.face_align and cmd is not None and cmd.fixture_ahead

    def _start_face_align(self, now: float) -> None:
        self._face = AlignRun(start=now, measure_after=now, origin=self._gyro.yaw)
        self._state.track(FACE_ALIGN)

    def _face_step(self, now: float, dt: float) -> DriveOutput:
        run = self._face
        if now - run.start > self._cfg.face_align_timeout_s:
            return self._face_done(False, f'timeout after {run.turns} turn(s)')
        if run.target is None:
            return self._face_measure(now)
        return self._face_turn(now, dt)

    def _face_measure(self, now: float) -> DriveOutput:
        cfg, run = self._cfg, self._face
        if self._scan is None or self._scan[1] <= run.measure_after:
            if now - run.measure_after > _FACE_SCAN_WAIT_S:
                return self._face_done(False, 'no depth scan')
            return self._face_hold()
        fit, source = self._face_fit(self._scan[0], now)
        tag = f'after {run.turns} turn(s), {source}: {fit.describe()}'
        if not fit.ok:
            return self._face_done(False, f'{tag}: {fit.reason}')
        if abs(fit.phi) <= cfg.face_align_tol_rad:
            return self._face_done(True, tag)
        if run.turns >= _FACE_MAX_TURNS:
            return self._face_done(False, f'{tag}: still off')
        if not self._gyro.fresh(now):
            return self._face_done(False, f'{tag}: no gyro')
        run.target, run.turn_from = self._gyro.yaw - fit.phi, self._gyro.yaw
        run.turns += 1
        self._pivot_check.reset()
        self._notes.append(f'face_align {tag}: turning {math.degrees(-fit.phi):+.1f} deg')
        return self._face_hold()

    def _face_fit(self, scan: DepthScan, now: float) -> tuple:
        cfg = self._cfg
        fit = fit_face(scan, cfg)
        memory = self._face_memory
        if fit.reason != FEW_POINTS or memory is None:
            return fit, 'scan'
        carried = memory.carried(self._gyro, now, cfg.face_align_memory_s)
        if carried is None:
            return fit, 'scan'
        if abs(carried.phi) > cfg.face_align_max_rad:
            carried = carried._replace(reason=TOO_SKEWED)
        return carried, f'run-in fit {now - memory.stamp:.1f} s old'

    def _face_turn(self, now: float, dt: float) -> DriveOutput:
        cfg, run, gyro = self._cfg, self._face, self._gyro
        if not gyro.fresh(now):
            return self._face_done(False, 'gyro lost during the turn')
        tol = cfg.face_align_tol_rad
        turned = math.copysign(1.0, run.target - run.turn_from) * (gyro.yaw - run.turn_from)
        if turned < -tol:   # wrong way (reversed gyro/sign): never swing into the shelf
            return self._face_done(False, f'turned {math.degrees(turned):+.1f} deg the wrong way')
        if abs(gyro.yaw - run.origin) > cfg.face_align_max_rad + tol:
            return self._face_done(False, 'turned more than face_align_max_rad')
        error = run.target - gyro.yaw
        if abs(error) <= tol:
            run.target, run.measure_after = None, now + cfg.arrival_settle_s
            return self._face_hold()
        if not self._aligned:               # the gate passes no rotation without heading
            return self._face_hold()
        passed = self._pivot_w * self._gate_scale(now)
        if self._pivot_check.update(error, gyro.yaw, passed, dt):
            return self._face_done(False, 'STUCK: the gyro does not follow the turn')
        self._state.track(FACE_ALIGN)
        return self._output(0.0, self._turn_rate(error, dt))

    def _face_hold(self) -> DriveOutput:
        self._pivot_w = 0.0
        self._ramp.reset()
        self._quantizer.reset()
        self._pivot_check.pause()
        self._state.track(FACE_ALIGN)
        return self._output(0.0, 0.0)

    def _face_done(self, square: bool, why: str) -> DriveOutput:
        run, self._face = self._face, None
        detail = FACE_ALIGNED if square else FACE_ALIGN_FAILED
        total = math.degrees(self._gyro.yaw - run.origin)
        self._notes.append(f'{detail} {why}; turned {total:+.1f} deg in all')
        self._state.arrive(self._cmd.pick_at_goal, detail)
        self._pivot_w = 0.0
        self._ramp.reset()
        self._quantizer.reset()
        return self._output(0.0, 0.0)

    def _speed_cap(self) -> float:
        cap = self._cmd.speed_cap
        return min(self._cfg.max_linear_speed, cap) if cap > 0.0 else self._cfg.max_linear_speed

    def _steer(self, proj: Projection, aligned: bool, dt: float) -> tuple:
        if not aligned:
            return 0.0, [Limit(min(self._cfg.align_speed, self._speed_cap()), 'ALIGNING')]
        s, (x, y, yaw) = proj.s, self._pose
        steering = steer(self._cfg, self._path, s, x, y, yaw, self._ramp.v)
        self._xtrack.update(proj.lateral, self._ramp.v, dt)
        curvature = min(max(steering.curvature + self._xtrack.curvature,
                            -self._cfg.max_curvature), self._cfg.max_curvature)
        preview = steering.lookahead + self._cfg.curvature_preview_m
        v_max = self._speed_cap()
        return curvature, [
            Limit(v_max, ''),
            curvature_limit(self._cfg, self._path.max_abs_curvature(s, s + preview)),
            heading_limit(self._cfg, steering.heading_error, v_max),
        ]

    def _common_limits(self, s: float, now: float) -> List[Limit]:
        return [
            goal_limit(self._cfg, self._path.length - s),
            arrival_limit(self._cfg, self._path.length - s),
            quality_limit(self._cfg, self._quality),
            lidar_limit(self._cfg, self._lidar_front),
            self._gap_limit(s, now),
        ]

    def _gap_limit(self, s: float, now: float) -> Limit:
        cfg = self._cfg
        if not self._cmd.lead_robot:
            return Limit(NO_LIMIT, 'GAP')
        if self._lead is None or _expired(self._lead[2], cfg.lead_timeout_s, now):
            self._bridge = None
            return Limit(0.0, 'STALE_LEAD')
        free_gap, stamp = self._lead_gap
        lidar = math.isfinite(free_gap) and not _expired(stamp, cfg.lead_gap_timeout_s, now)
        if not cfg.gap_bridge:
            if lidar:
                return self._gap.limit(free_gap - cfg.gap_free_target, 'GAP_LIDAR')
            return self._gap.limit(self._lead_ahead(s)[0] - self._cmd.gap)
        ahead, ranged = self._lead_ahead(s)
        path_error = ahead - self._cmd.gap
        if lidar:
            error = free_gap - cfg.gap_free_target
            fresh = self._bridge is None or stamp != self._bridge.stamp
            if fresh and stamp != self._bridge_floor:
                self._bridge = _GapBridge(path_error - error, ranged, s, stamp)
            return self._gap.limit(error, 'GAP_LIDAR')
        bridge = self._bridge
        if bridge is not None and bridge.ranged == ranged \
                and abs(s - bridge.s) <= self._cmd.gap:
            return self._gap.limit(path_error - bridge.offset, 'GAP_BRIDGED')
        self._bridge = None                     # trust ended: the conservative path gap
        return self._gap.limit(path_error)

    def _lead_ahead(self, s: float) -> tuple:
        cfg = self._cfg
        lead = self._path.project(self._lead[0], self._lead[1], s - cfg.lead_back_window)
        if lead.dist <= cfg.lead_off_path_tol:
            self._lead_joined = True
        elif self._lead_joined:
            return math.hypot(self._lead[0] - self._pose[0],
                              self._lead[1] - self._pose[1]), True
        return lead.s - s, False

    def _report(self, limit: Limit, align: str) -> None:
        if limit.speed <= 0.0 and limit.reason in _FAULT_LIMITS:
            self._state.stop(limit.reason)
        else:
            self._state.track('ALIGNING' if align != st.ALIGNED else limit.reason)

    def _check_arrival(self, s: float, v: float, now: float) -> None:
        if self._path.length - s > self._cfg.goal_stop_margin or v > 0.0:
            self._arrival.close()
            return
        self._arrival.open(self._pose, now)
        if self._arrival.elapsed(now) >= self._cfg.arrival_settle_s:
            self._judge_goal(now)

    def _judge_goal(self, now: float) -> None:
        cfg, cmd = self._cfg, self._cmd
        pos_tol = cmd.goal_pos_tol if cmd.goal_pos_tol > 0.0 else cfg.default_goal_pos_tol
        arrival = judge_arrival(self._arrival.estimate(), cmd.poses[-1], pos_tol,
                                cmd.goal_yaw_tol, self._aligned)
        if arrival.ok and self._face_wanted():
            self._start_face_align(now)
        elif arrival.ok:
            self._state.arrive(cmd.pick_at_goal)
        else:
            self._state.stop('GOAL_MISSED')  # host re-enters from a backed-off pose

    def _judge_in_place(self, now: float) -> DriveOutput:
        self._progress.update(self._pose[0], self._pose[1])
        self._ramp.reset()
        self._quantizer.reset()
        self._state.track('GOAL')
        self._check_arrival(0.0, 0.0, now)
        return self._output(0.0, 0.0)

    def _approach_active(self) -> bool:
        cmd = self._cmd
        if cmd is None or self._path is None or self._pose is None:
            return False
        if self._pivot is not None or self._face is not None:
            return True
        return (cmd.pick_at_goal or cmd.fixture_ahead) and self._on_final_run_in()

    def _on_final_run_in(self) -> bool:
        cfg, path, cmd = self._cfg, self._path, self._cmd
        (x, y, yaw), goal = self._pose, cmd.poses[-1]
        s, end = self._progress.s, path.length
        dist = cfg.pick_approach_dist
        if math.hypot(goal[0] - x, goal[1] - y) > dist or end - s > dist:
            return False
        if not path.straight_after(s, _STRAIGHT_KAPPA):
            return False
        line = path.tangent(end) if end > 0.0 else goal[2]
        return self._aligned and abs(wrap_angle(yaw - line)) <= cfg.heading_free_rad

    def _output(self, v: float, w: float) -> DriveOutput:
        cmd_id, epoch = (self._cmd.cmd_id, self._cmd.epoch) if self._cmd else (0, 0)
        return DriveOutput(v, w, self._state.state, self._state.detail, cmd_id,
                           self._progress_ratio(), epoch, self._approach_active())

    def _progress_ratio(self) -> float:
        if self._path is None or self._progress is None:
            return 0.0
        if self._path.length <= 0.0:
            return 1.0
        return min(self._progress.s / self._path.length, 1.0)
