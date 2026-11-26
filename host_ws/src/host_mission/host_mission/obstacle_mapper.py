"""Static-obstacle mapper (ROS-free): range scans of any rover -> obstacle rectangles.

All planar range sensors feed ONE evidence grid (``uwb_global``), so a box seen by either
rover is avoided by both. Per scan: gate on the carrying rover's pose (fresh, GOOD, aligned,
not turning) -> hit/miss evidence per source layer -> occupied components -> face-fitted
rectangles -> assumed unseen depth (``box_depth``) -> published grown by ``inflation`` after
``confirm_scans`` sightings (dropped after ``drop_scans`` misses). Mapping error of a return at
range r: max(``inflation``, r * tan(heading accuracy of the carrying rover))."""
import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

QUALITY_GOOD = 'GOOD'

NO_POSE = 'NO_POSE'
STALE_POSE = 'STALE_POSE'
NOT_ALIGNED = 'NOT_ALIGNED'
BAD_QUALITY = 'BAD_QUALITY'
TURNING = 'TURNING'
TURNED = 'TURNED'           # heading not yet re-established after a turn


@dataclass(frozen=True)
class MapperParams:
    """Tunables shared by every source (m, rad, s); see mission.yaml."""

    cell_size: float = 0.05
    wall_margin: float = 0.05            # arena (mission bounds) shrunk by this
    hit_increment: float = 1.0
    miss_decrement: float = 1.0
    max_evidence: float = 5.0
    occupied_threshold: float = 3.0      # evidence at which a cell becomes occupied
    release_threshold: float = 0.0       # ... and at or below which it is free again
    clear_stop_short: float = 0.07       # ray clearing ends this far before the return
    # motion weighting: weight = clip(weight_ref_error / error(r), min_weight, 1),
    # error(r) = pose_sigma + |v|*pose_lag_s + r*(heading_sigma + |w|*pose_lag_s)
    weight_ref_error: float = 0.0  # m; 0 = every return counts fully
    pose_sigma: float = 0.01             # m, position error of the scanning rover at rest
    heading_sigma: float = 0.01          # rad, its heading error at rest
    pose_lag_s: float = 0.2              # s, how far the fused pose trails the motion
    min_weight: float = 0.1
    speed_window: float = 0.5
    decay_per_scan: float = 0.0  # evidence lost per scan by cells not hit (0 = off)
    pose_timeout: float = 0.25           # carrying rover's pose older than this: scan skipped
    robot_pose_timeout: float = 0.5      # another rover's pose older: no hits in the scan
    turn_rate_window: float = 0.30
    heading_settle_dist: float = 0.0  # m to travel after a fast turn before scans count (0 = off)
    min_cells: int = 4
    link_cells: int = 2
    view_side_min: float = 0.3           # |cos| needed to tell the viewing side
    box_depth: float = 0.50  # largest box side (assumed unseen depth)
    min_circle_segment: float = 0.20
    inflation: float = 0.10  # mapping error (pose + heading error x range); grows rectangles
    confirm_scans: int = 3
    drop_scans: int = 5
    match_distance: float = 0.30         # rectangle centre distance for "same obstacle"
    change_threshold: float = 0.10       # sighting outside the published rect: re-publish
    shrink_threshold: float = 0.10       # published rect outside the sighting: re-publish


@dataclass(frozen=True)
class ScanSource:
    """One range sensor: carrying rover, mount pose, trusted ranges."""

    name: str = 'follower_lidar'
    robot: str = 'follower'
    mount_x: float = 0.0
    mount_y: float = 0.0
    mount_yaw: float = 0.0
    min_range: float = 0.15  # below = own structure (scan range_min if larger)
    max_use_range: float = 4.0           # returns beyond are not mapped, only clear up to here
    max_turn_rate: float = 0.30          # rad/s: carrying rover turning faster = scan skipped
    settle_still_s: float = 0.0  # s still after a turn that also settles heading (0 = travel only)


FOLLOWER_LIDAR = ScanSource()
# Exclusion radius per rover (m): returns this close are the rover itself.
DEFAULT_ROBOT_RADII = {'leader': 0.35, 'follower': 0.25}
# Heading accuracy of each rover's localization (rad).
DEFAULT_HEADING_ACCURACY = {'leader': math.radians(8.0), 'follower': math.radians(2.5)}


@dataclass(frozen=True)
class Region:
    """Axis-aligned area (uwb_global)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def shrunk(self, margin):
        return Region(self.xmin + margin, self.xmax - margin,
                      self.ymin + margin, self.ymax - margin)

    def contains(self, xy):
        """Boolean mask of the (N, 2) points inside the area."""
        x, y = xy[:, 0], xy[:, 1]
        return (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)


@dataclass(frozen=True)
class ScanData:
    """The parts of a LaserScan the mapper needs (angles CCW from scan +x)."""

    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float = 0.0
    range_max: float = math.inf


@dataclass(frozen=True)
class ScanVerdict:
    used: bool
    reason: str = ''
    changed: bool = False


@dataclass(frozen=True)
class Rect:
    """Oriented rectangle; half_len >= half_wid; ``error`` = mapping error of its returns (m)."""

    cx: float
    cy: float
    angle: float
    half_len: float
    half_wid: float
    error: float = 0.0

    @property
    def axes(self):
        u = np.array([math.cos(self.angle), math.sin(self.angle)])
        return u, np.array([-u[1], u[0]])

    def corners(self):
        u, v = self.axes
        c = np.array([self.cx, self.cy])
        return np.array([c + su * self.half_len * u + sv * self.half_wid * v
                         for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def fit_rect(points, angle_step=math.radians(1.0), weights=None):
    """Rectangle around (N, 2) points oriented by L-shape fitting (Zhang et al., IV 2017)."""
    pts = np.asarray(points, dtype=float)
    angles = np.arange(0.0, math.pi / 2.0, angle_step)
    u = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    c1, c2 = pts @ u.T, pts @ np.stack([-u[:, 1], u[:, 0]], axis=1).T
    d1 = np.minimum(c1.max(axis=0) - c1, c1 - c1.min(axis=0))
    d2 = np.minimum(c2.max(axis=0) - c2, c2 - c2.min(axis=0))
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=float)
    best = int(np.argmin(w @ np.minimum(d1, d2)))
    return _rect_at(pts, float(angles[best]))


def _rect_at(points, angle):
    """Bounding rectangle of the points with one side along ``angle``."""
    u = np.array([math.cos(angle), math.sin(angle)])
    v = np.array([-u[1], u[0]])
    pu, pv = points @ u, points @ v
    half_u, half_v = np.ptp(pu) / 2.0, np.ptp(pv) / 2.0
    centre = u * (pu.min() + half_u) + v * (pv.min() + half_v)
    if half_u < half_v:
        angle, half_u, half_v = angle + math.pi / 2.0, half_v, half_u
    return Rect(float(centre[0]), float(centre[1]), _wrap(angle), float(half_u), float(half_v))


def face_rect(points, weights, angle, band, cell):
    """Rectangle along ``angle`` with sides on the sampled faces (hit-weighted mean of the cells
    within ``band`` of the outermost one, since pose error smears the hits)."""
    pts = np.asarray(points, dtype=float)
    w = np.asarray(weights, dtype=float)
    u = np.array([math.cos(angle), math.sin(angle)])
    v = np.array([-u[1], u[0]])
    sides = []
    for axis, across in ((u, v), (v, u)):
        proj, along = pts @ axis, pts @ across
        ends = []
        for sign in (-1.0, 1.0):
            outer = float((sign * proj).max())
            near = sign * proj >= outer - band
            if np.ptp(along[near]) > np.ptp(proj[near]):
                # a face runs along this side: its smear is across it, take its centre
                ends.append(float(np.average(proj[near], weights=w[near])))
            else:
                # the end of a face (cells run across this side): outermost cell + half a cell
                ends.append(sign * (outer + cell / 2.0))
        sides.append((ends[0], max(ends[1], ends[0])))
    (u0, u1), (v0, v1) = sides
    centre = u * (u0 + u1) / 2.0 + v * (v0 + v1) / 2.0
    half_u, half_v = (u1 - u0) / 2.0, (v1 - v0) / 2.0
    if half_u < half_v:
        angle, half_u, half_v = angle + math.pi / 2.0, half_v, half_u
    return Rect(float(centre[0]), float(centre[1]), _wrap(angle), float(half_u), float(half_v))


def extend_unseen_depth(rect, view, depth, view_side_min, is_free, step):
    """Grow the rectangle away from the sensors up to ``depth`` (assumed box size), stopping at
    the first cross-section seen free (``is_free``: points -> mask). ``view``: summed
    sensor->hit directions; axes seen side-on are not extended."""
    norm = float(np.hypot(*view))
    if norm <= 0.0:
        return rect
    centre = np.array([rect.cx, rect.cy])
    u, v = rect.axes
    half = [rect.half_len, rect.half_wid]
    for axis, other in ((0, 1), (1, 0)):
        a, b = (u, v) if axis == 0 else (v, u)
        side = float(a @ view) / norm
        missing = depth - 2.0 * half[axis]
        if abs(side) < view_side_min or missing <= 0.0:
            continue
        away = a if side > 0.0 else -a
        # samples stay 2 cells inside the sides: a grazing ray frees up to 1 cell into the box
        inner = max(half[other] - 2.0 * step, 0.0)
        across = np.linspace(-inner, inner, max(2, int(math.ceil(2.0 * inner / (step / 2.0))) + 1))
        grown = 0.0
        while grown < missing - 1e-9:
            nxt = min(grown + step / 2.0, missing)
            far = centre + away * (half[axis] + nxt)
            if is_free(far[None, :] + across[:, None] * b[None, :]).mean() >= 0.5:
                break
            grown = nxt
        centre = centre + away * grown / 2.0
        half[axis] += grown / 2.0
    half_len, half_wid, angle = half[0], half[1], rect.angle
    if half_wid > half_len:
        half_len, half_wid, angle = half_wid, half_len, _wrap(angle + math.pi / 2.0)
    return Rect(float(centre[0]), float(centre[1]), angle, half_len, half_wid)


def front_stop_rect(point, heading, depth, width):
    """Box ``depth`` x ``width`` assumed behind a face point seen straight ahead."""
    u = np.array([math.cos(heading), math.sin(heading)])
    cx, cy = np.asarray(point, dtype=float) + u * depth / 2.0
    if depth >= width:
        return Rect(float(cx), float(cy), _wrap(heading), depth / 2.0, width / 2.0)
    return Rect(float(cx), float(cy), _wrap(heading + math.pi / 2.0), width / 2.0, depth / 2.0)


def rect_shape(rect, pad):
    """Planner shape (x, y, pad, width, height, yaw) (see ``world_model.Obstacle``)."""
    return (rect.cx, rect.cy, float(pad), 2.0 * rect.half_len, 2.0 * rect.half_wid,
            rect.angle)


def rect_to_circles(rect, inflation, min_segment):
    """Cover the rectangle with circumscribed circles of near-square pieces (display only)."""
    length, width = 2.0 * rect.half_len, 2.0 * rect.half_wid
    pieces = max(1, int(round(length / max(width, min_segment, 1e-6))))
    seg = length / pieces
    radius = math.hypot(seg, width) / 2.0 + inflation
    u, _v = rect.axes
    offsets = -rect.half_len + seg * (np.arange(pieces) + 0.5)
    return [(rect.cx + float(o) * u[0], rect.cy + float(o) * u[1], radius) for o in offsets]


def rects_overlap(a, b):
    """Separating-axis overlap test."""
    ca, cb = a.corners(), b.corners()
    for axis in (*a.axes, *b.axes):
        pa, pb = ca @ axis, cb @ axis
        if pa.max() <= pb.min() or pb.max() <= pa.min():
            return False
    return True


def rect_excess(outer, inner):
    """How far ``inner``'s corners stick out of ``outer`` (m)."""
    u, v = outer.axes
    rel = inner.corners() - np.array([outer.cx, outer.cy])
    out_u = np.maximum(np.abs(rel @ u) - outer.half_len, 0.0)
    out_v = np.maximum(np.abs(rel @ v) - outer.half_wid, 0.0)
    return float(np.hypot(out_u, out_v).max())


def connected_components(cells, link):
    """Group (ix, iy) cells within ``link`` (Chebyshev) of each other."""
    remaining = set(map(tuple, cells))
    offsets = [(dx, dy) for dx in range(-link, link + 1) for dy in range(-link, link + 1)
               if dx or dy]
    groups = []
    while remaining:
        stack = [remaining.pop()]
        group = []
        while stack:
            cx, cy = stack.pop()
            group.append((cx, cy))
            for dx, dy in offsets:
                nb = (cx + dx, cy + dy)
                if nb in remaining:
                    remaining.remove(nb)
                    stack.append(nb)
        groups.append(group)
    return groups


class EvidenceGrid:
    """Hit/miss evidence per cell, with mean hit point and viewing direction."""

    def __init__(self, region, params):
        self.region = region
        self.p = params
        self.nx = max(1, int(math.ceil((region.xmax - region.xmin) / params.cell_size)))
        self.ny = max(1, int(math.ceil((region.ymax - region.ymin) / params.cell_size)))
        self.evidence = np.zeros(self.nx * self.ny)
        self.view = np.zeros((self.nx * self.ny, 2))
        self.point_sum = np.zeros((self.nx * self.ny, 2))
        self.point_count = np.zeros(self.nx * self.ny)
        self.error_sum = np.zeros(self.nx * self.ny)
        # latest look at the cell saw it free (ray passed, no return since)
        self.seen_free = np.zeros(self.nx * self.ny, dtype=bool)

    @classmethod
    def merged(cls, layers):
        """Combine layers: evidence = max, hit points and views pooled."""
        first = layers[0]
        out = cls(first.region, first.p)
        out.evidence = np.max([g.evidence for g in layers], axis=0)
        out.view = np.sum([g.view for g in layers], axis=0)
        out.point_sum = np.sum([g.point_sum for g in layers], axis=0)
        out.point_count = np.sum([g.point_count for g in layers], axis=0)
        out.error_sum = np.sum([g.error_sum for g in layers], axis=0)
        return out

    def flat_index(self, xy):
        """Flat cell index of each point, and the mask of points inside the grid."""
        ix = np.floor((xy[:, 0] - self.region.xmin) / self.p.cell_size).astype(int)
        iy = np.floor((xy[:, 1] - self.region.ymin) / self.p.cell_size).astype(int)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix * self.ny + iy, inside

    def integrate(self, origin, hit_xy, hit_dirs, hit_added, clear_dirs, clear_len,
                  hit_error=None, hit_weight=None, miss_weight=1.0):
        """One scan: +hit on added hit cells, -miss on traversed cells; cells within a
        return's ``hit_error`` are shielded from this scan's rays."""
        hit_idx, hit_in = self.flat_index(hit_xy)
        shield = self._shielded(hit_idx[hit_in], None if hit_error is None
                                else np.asarray(hit_error)[hit_in])
        add = hit_in & hit_added
        missed = np.setdiff1d(self._ray_cells(origin, clear_dirs, clear_len), shield)
        p = self.p
        if hit_weight is None:
            self.evidence[np.unique(hit_idx[add])] += p.hit_increment
        else:                                   # a cell counts its best-placed return
            gain = np.zeros(self.evidence.shape)
            np.maximum.at(gain, hit_idx[add], np.asarray(hit_weight, dtype=float)[add])
            self.evidence += p.hit_increment * gain
        self.evidence[missed] -= p.miss_decrement * miss_weight
        if p.decay_per_scan > 0.0:
            stale = np.ones(self.evidence.shape, dtype=bool)
            stale[np.unique(hit_idx[add])] = False
            self.evidence[stale] -= p.decay_per_scan
        np.clip(self.evidence, 0.0, p.max_evidence, out=self.evidence)
        np.add.at(self.view, hit_idx[add], hit_dirs[add])
        np.add.at(self.point_sum, hit_idx[add], hit_xy[add])
        np.add.at(self.point_count, hit_idx[add], 1.0)
        if hit_error is not None:
            np.add.at(self.error_sum, hit_idx[add], np.asarray(hit_error, dtype=float)[add])
        self.seen_free[missed] = True
        self.seen_free[np.unique(hit_idx[hit_in])] = False
        empty = self.evidence <= 0.0
        self.view[empty], self.point_sum[empty], self.point_count[empty] = 0.0, 0.0, 0.0
        self.error_sum[empty] = 0.0

    def _shielded(self, hit_idx, hit_error):
        """Hit cells grown by each return's error (whole cells, rounded down)."""
        cells = np.unique(hit_idx)
        if hit_error is None or not len(hit_idx):
            return cells
        reach = np.floor(hit_error / self.p.cell_size).astype(int)
        out = [cells]
        for k in np.unique(reach[reach > 0]):
            idx = np.unique(hit_idx[reach == k])
            ix, iy = idx // self.ny, idx % self.ny
            for dx in range(-k, k + 1):
                for dy in range(-k, k + 1):
                    jx, jy = ix + dx, iy + dy
                    ok = (jx >= 0) & (jx < self.nx) & (jy >= 0) & (jy < self.ny)
                    out.append(jx[ok] * self.ny + jy[ok])
        return np.unique(np.concatenate(out))

    def _ray_cells(self, origin, dirs, lengths):
        """Unique flat indices of the cells along each ray."""
        keep = lengths > 0.0
        if not np.any(keep):
            return np.zeros(0, dtype=int)
        dirs, lengths = dirs[keep], lengths[keep]
        step = self.p.cell_size * 0.5
        t = np.arange(0.0, float(lengths.max()), step)
        along = t[None, :] < lengths[:, None]
        pts = origin[None, None, :] + dirs[:, None, :] * t[None, :, None]
        idx, inside = self.flat_index(pts[along])
        return np.unique(idx[inside])

    def occupied_mask(self, previous=None):
        """Occupancy with hysteresis: set at ``occupied_threshold``, kept (``previous``) until
        evidence falls to ``release_threshold``."""
        occupied = self.evidence >= self.p.occupied_threshold
        if previous is not None:
            occupied |= previous & (self.evidence > self.p.release_threshold)
        return occupied

    def occupied_cells(self, mask=None):
        flat = np.flatnonzero(self.occupied_mask() if mask is None else mask)
        return np.stack([flat // self.ny, flat % self.ny], axis=1)

    def mean_points(self, cells):
        flat = self._flat(cells)
        return self.point_sum[flat] / np.maximum(self.point_count[flat], 1.0)[:, None]

    def view_sum(self, cells):
        return self.view[self._flat(cells)].sum(axis=0)

    def _flat(self, cells):
        cells = np.asarray(cells, dtype=int)
        return cells[:, 0] * self.ny + cells[:, 1]


@dataclass
class _Track:
    rect: Rect
    seen: int = 1
    missed: int = 0
    grows: int = 0
    shrinks: int = 0
    published: Optional[Rect] = None


class ObstacleTracker:
    """Confirmation, drop hysteresis and change thresholds over per-scan rectangles."""

    def __init__(self, params):
        self.p = params
        self.tracks: List[_Track] = []

    def update(self, rects):
        """Feed one scan's rectangles; True when the published set changed."""
        matched_tracks, matched_rects = self._match(rects)
        changed = False
        for ti, ri in zip(matched_tracks, matched_rects):
            changed |= self._sighted(self.tracks[ti], rects[ri])
        kept = []
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.missed += 1
                if track.published is None:
                    continue                        # unconfirmed: one miss ends it
                if track.missed >= self.p.drop_scans:
                    changed = True
                    continue
            kept.append(track)
        new = [_Track(rects[ri]) for ri in range(len(rects)) if ri not in matched_rects]
        for track in new:
            changed |= self._try_confirm(track)
        self.tracks = kept + new
        return changed

    def published_rects(self):
        return [t.published for t in self.tracks if t.published is not None]

    def _match(self, rects):
        pairs = sorted(
            (math.hypot(t.rect.cx - r.cx, t.rect.cy - r.cy), ti, ri)
            for ti, t in enumerate(self.tracks) for ri, r in enumerate(rects))
        used_t, used_r = [], []
        for dist, ti, ri in pairs:
            if dist > self.p.match_distance:
                break
            if ti not in used_t and ri not in used_r:
                used_t.append(ti)
                used_r.append(ri)
        return used_t, used_r

    def _sighted(self, track, rect):
        track.rect, track.missed = rect, 0
        if track.published is None:
            track.seen += 1
            return self._try_confirm(track)
        p = self.p
        grown = rect_excess(track.published, rect) > p.change_threshold
        shrunk = rect_excess(rect, track.published) > p.shrink_threshold
        track.grows = track.grows + 1 if grown else 0
        track.shrinks = track.shrinks + 1 if shrunk and not grown else 0
        if track.grows < p.confirm_scans and track.shrinks < p.drop_scans:
            return False
        track.published, track.grows, track.shrinks = rect, 0, 0
        return True

    def _try_confirm(self, track):
        if track.seen >= self.p.confirm_scans:
            track.published = track.rect
            return True
        return False


@dataclass
class _RobotState:
    pose: Optional[Tuple[float, float, float]] = None
    time: float = -math.inf
    aligned: bool = False
    quality: str = 'UNKNOWN'
    yaw_history: deque = field(default_factory=deque)   # (t, yaw)
    pos_history: deque = field(default_factory=deque)   # (t, x, y)
    turned_at: Optional[Tuple[float, float]] = None
    turning_t: float = -math.inf


MISSION_IDLE = 'IDLE'
STILL_SPEED = 0.02          # m/s: below this the scanning rover counts as standing


def integrates_scans(mission_state, freeze_while_mission):
    """With ``freeze_while_mission`` scans change the map only while IDLE or unknown
    (a refit mid-mission would re-plan rovers)."""
    return not freeze_while_mission or mission_state in (None, MISSION_IDLE)


class ObstacleMapper:
    """Scans of N sources + rover poses in, stable obstacle rectangles out."""

    def __init__(self, params, arena, ignore_zones: Sequence[Region] = (),
                 sources: Sequence[ScanSource] = (FOLLOWER_LIDAR,),
                 robot_radii: Mapping[str, float] = DEFAULT_ROBOT_RADII,
                 fixtures: Sequence[object] = (),
                 heading_accuracy: Mapping[str, float] = DEFAULT_HEADING_ACCURACY):
        """``fixtures``: known static boxes; returns within ``inflation`` of one are not mapped."""
        self.p = params
        self.region = arena.shrunk(params.wall_margin)
        self.ignore_zones = list(ignore_zones)
        self.fixtures = [(f.x, f.y, f.yaw, f.width / 2.0, f.height / 2.0)
                         for f in fixtures]
        self._occupied = None
        self.sources: Dict[str, ScanSource] = {s.name: s for s in sources}
        if len(self.sources) != len(sources):
            raise ValueError('scan source names must be unique')
        self.layers = {name: EvidenceGrid(self.region, params) for name in self.sources}
        self.radii = dict(robot_radii)
        missing = {s.robot for s in sources} - set(heading_accuracy)
        if missing:
            raise ValueError(f'no heading accuracy for the scanning rover(s) {sorted(missing)}')
        self.heading_accuracy = dict(heading_accuracy)
        names = set(self.radii) | {s.robot for s in sources}
        self.robots: Dict[str, _RobotState] = {name: _RobotState() for name in names}
        self.tracker = ObstacleTracker(params)

    def update_pose(self, robot, x, y, yaw, now):
        state = self.robots.get(robot)
        if state is None:
            return
        state.pose, state.time = (float(x), float(y), float(yaw)), float(now)
        state.yaw_history.append((float(now), float(yaw)))
        state.pos_history.append((float(now), float(x), float(y)))
        while state.pos_history and state.pos_history[0][0] < now - self.p.speed_window:
            state.pos_history.popleft()
        while state.yaw_history and state.yaw_history[0][0] < now - self.p.turn_rate_window:
            state.yaw_history.popleft()

    def update_status(self, robot, aligned=None, quality=None):
        state = self.robots.get(robot)
        if state is None:
            return
        if aligned is not None:
            state.aligned = bool(aligned)
        if quality is not None:
            state.quality = str(quality).strip().upper()

    def shapes(self):
        """Published obstacles as planner shapes, grown by ``inflation``."""
        return [rect_shape(rect, self.p.inflation) for rect in self.tracker.published_rects()]

    def circles(self):
        """Published obstacles as circles [(x, y, r)] (display only)."""
        out = []
        for rect in self.tracker.published_rects():
            out += rect_to_circles(rect, self.p.inflation, self.p.min_circle_segment)
        return out

    def process_scan(self, source, scan, now):
        """Integrate one scan of source ``source`` (name) if the gate passes."""
        src = self.sources[source]
        reason = self._gate(src, now)
        if reason:
            return ScanVerdict(False, reason)
        self._integrate(src, scan, self._sensor_pose(src), now)
        changed = self.tracker.update(self._extract_rects())
        return ScanVerdict(True, '', changed)

    def _gate(self, src, now):
        state = self.robots[src.robot]
        if state.pose is None:
            return NO_POSE
        if now - state.time > self.p.pose_timeout:
            return STALE_POSE
        if state.quality != QUALITY_GOOD:
            return BAD_QUALITY
        if not state.aligned:
            return NOT_ALIGNED
        if _turn_rate(state.yaw_history) > src.max_turn_rate:
            state.turned_at = state.pose[:2]
            state.turning_t = now
            return TURNING
        if state.turned_at is not None and src.settle_still_s > 0.0 \
                and now - state.turning_t >= src.settle_still_s \
                and _speed(state.pos_history) < STILL_SPEED:
            state.turned_at = None              # stood still: the absolute heading caught up
        if state.turned_at is not None:
            # heading after a turn is off by gyro scale error x angle (~6 deg per 180) until travel
            moved = math.hypot(state.pose[0] - state.turned_at[0],
                               state.pose[1] - state.turned_at[1])
            if moved < self.p.heading_settle_dist:
                return TURNED
            state.turned_at = None
        return ''

    def _sensor_pose(self, src):
        x, y, yaw = self.robots[src.robot].pose
        c, s = math.cos(yaw), math.sin(yaw)
        return (x + c * src.mount_x - s * src.mount_y, y + s * src.mount_x + c * src.mount_y,
                yaw + src.mount_yaw)

    def _integrate(self, src, scan, sensor, now):
        ranges = np.asarray(scan.ranges, dtype=float)
        angles = sensor[2] + scan.angle_min + scan.angle_increment * np.arange(len(ranges))
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        origin = np.array(sensor[:2])
        min_range = max(src.min_range, scan.range_min)
        returned = np.isfinite(ranges) & (ranges >= min_range) & (ranges > 0.0)
        hit = returned & (ranges <= src.max_use_range)
        clear_len = self._clear_lengths(src, ranges, returned, hit, scan.range_max)
        hit_xy = origin + dirs[hit] * ranges[hit][:, None]
        added = self._hit_allowed(src, origin, hit_xy, ranges[hit], now)
        hit_w, miss_w = self.motion_weights(src.robot, ranges[hit])
        self.layers[src.name].integrate(origin, hit_xy, dirs[hit], added, dirs, clear_len,
                                        self.mapping_error(ranges[hit], src.robot),
                                        hit_w, miss_w)

    def motion_weights(self, robot, hit_ranges):
        """(per-return weights or None, ray-clearing weight = that of a return at 1 m)."""
        p = self.p
        if p.weight_ref_error <= 0.0:
            return None, 1.0
        state = self.robots[robot]
        speed, turn = _speed(state.pos_history), _turn_rate(state.yaw_history)
        base = p.pose_sigma + speed * p.pose_lag_s
        per_m = p.heading_sigma + turn * p.pose_lag_s
        def weight(r):
            return np.clip(p.weight_ref_error / (base + np.asarray(r, dtype=float) * per_m),
                           p.min_weight, 1.0)
        return weight(hit_ranges), float(weight(1.0))

    def _clear_lengths(self, src, ranges, returned, hit, range_max):
        """Free length per beam; 0/NaN = no return (no info), +inf = free to sensor range."""
        out = np.zeros(len(ranges))
        out[hit] = np.maximum(ranges[hit] - self.p.clear_stop_short, 0.0)
        out[returned & ~hit] = src.max_use_range
        sensor_max = range_max if math.isfinite(range_max) and range_max > 0.0 \
            else src.max_use_range
        out[np.isposinf(ranges)] = min(src.max_use_range, sensor_max)
        return out

    def mapping_error(self, ranges, robot):
        """Max(``inflation``, range * tan(heading accuracy)): how far a return may lie from its surface."""
        return np.maximum(self.p.inflation, np.asarray(ranges, dtype=float)
                          * math.tan(self.heading_accuracy[robot]))

    def _hit_allowed(self, src, origin, hit_xy, hit_ranges, now):
        """Mask of returns that may count: not at the edge, a fixture, the other rover or a
        more precise published rectangle."""
        error = self.mapping_error(hit_ranges, src.robot)
        allowed = _depth_inside(hit_xy, self.region) > error
        for zone in self.ignore_zones:
            allowed &= ~zone.contains(hit_xy)
        for fixture in self.fixtures:
            allowed &= _rect_gap(hit_xy, fixture) > error
        allowed &= ~explained_by_map(hit_xy, error, self.tracker.published_rects(),
                                     self.p.inflation)
        heading = self.heading_accuracy[src.robot]
        for robot, radius in self.radii.items():
            if robot == src.robot:
                continue
            state = self.robots[robot]
            if state.pose is None or now - state.time > self.p.robot_pose_timeout:
                return np.zeros(len(hit_xy), dtype=bool)
            allowed &= ~explained_by_rover(origin, hit_xy, hit_ranges, state.pose[:2],
                                           radius, heading)
        return allowed

    def _extract_rects(self):
        """Occupied components -> rects; overlapping ones are merged and refitted."""
        p = self.p
        grid = EvidenceGrid.merged(list(self.layers.values()))
        self._occupied = grid.occupied_mask(self._occupied)
        groups = [g for g in connected_components(grid.occupied_cells(self._occupied),
                                                  p.link_cells) if len(g) >= p.min_cells]
        rects = [self._component_rect(grid, g) for g in groups]
        merged = True
        while merged:
            merged = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if rects_overlap(rects[i], rects[j]):
                        groups[i] = groups[i] + groups.pop(j)
                        rects.pop(j)
                        rects[i] = self._component_rect(grid, groups[i])
                        merged = True
                        break
                if merged:
                    break
        return rects

    def _component_rect(self, grid, group):
        p = self.p
        points = grid.mean_points(group)
        weights = grid.point_count[grid._flat(group)]
        rect = face_rect(points, weights, fit_rect(points, weights=weights).angle,
                         p.inflation, p.cell_size)
        rect = extend_unseen_depth(rect, grid.view_sum(group), p.box_depth, p.view_side_min,
                                   self._free_test(group), p.cell_size)
        error = float(grid.error_sum[grid._flat(group)].sum() / max(weights.sum(), 1.0))
        return replace(rect, error=error)

    def _free_test(self, group):
        """Points -> mask of space seen free by a sensor that sees this component."""
        flat = np.asarray(group, dtype=int)
        flat = flat[:, 0] * next(iter(self.layers.values())).ny + flat[:, 1]
        layers = [g for g in self.layers.values() if g.point_count[flat].any()]
        free = np.any([g.seen_free for g in layers], axis=0)

        def is_free(points):
            idx, inside = layers[0].flat_index(np.asarray(points, dtype=float))
            out = np.ones(len(idx), dtype=bool)
            out[inside] = free[idx[inside]]
            return out
        return is_free


def explained_by_rover(origin, hit_xy, hit_ranges, rover_xy, radius, heading_accuracy):
    """Mask of returns that may be the rover at ``rover_xy``, tested in the sensor's polar frame:
    range within ``radius`` of the rover's, bearing within asin(radius / R) + heading accuracy."""
    rover = np.asarray(rover_xy, dtype=float)
    near = np.hypot(*(hit_xy - rover).T) <= radius
    rel = rover - origin
    rover_range = float(np.hypot(*rel))
    if rover_range <= radius:
        return near | (hit_ranges <= rover_range + radius)
    rel_hits = hit_xy - origin
    bearing = np.arctan2(rel_hits[:, 1], rel_hits[:, 0]) - math.atan2(rel[1], rel[0])
    off = np.abs((bearing + math.pi) % (2.0 * math.pi) - math.pi)
    half = math.asin(radius / rover_range) + heading_accuracy
    return near | ((np.abs(hit_ranges - rover_range) <= radius) & (off <= half))


def explained_by_map(hit_xy, error, rects, map_accuracy):
    """Mask of returns that are likely a published rectangle's face seen through a larger
    mapping error: outside it by no more than the summed errors. More precise returns count."""
    out = np.zeros(len(hit_xy), dtype=bool)
    error = np.asarray(error, dtype=float)
    for rect in rects:
        accuracy = max(rect.error, map_accuracy)
        loose = error > accuracy
        if not loose.any():
            continue
        gap = _rect_gap(hit_xy, (rect.cx, rect.cy, rect.angle, rect.half_len, rect.half_wid))
        out |= loose & (gap > 0.0) & (gap <= error + accuracy)
    return out


def _depth_inside(points, region):
    """Depth of each point inside the region (m, negative outside)."""
    x, y = points[:, 0], points[:, 1]
    return np.minimum(np.minimum(x - region.xmin, region.xmax - x),
                      np.minimum(y - region.ymin, region.ymax - y))


def _rect_gap(points, rect):
    """Distance of points to the rectangle (cx, cy, yaw, half_x, half_y); 0 inside."""
    cx, cy, yaw, half_x, half_y = rect
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    ex = np.maximum(np.abs(c * dx + s * dy) - half_x, 0.0)
    ey = np.maximum(np.abs(-s * dx + c * dy) - half_y, 0.0)
    return np.hypot(ex, ey)


def _speed(history):
    if len(history) < 2 or history[-1][0] <= history[0][0]:
        return 0.0
    return math.hypot(history[-1][1] - history[0][1], history[-1][2] - history[0][2]) \
        / (history[-1][0] - history[0][0])


def _turn_rate(history):
    if len(history) < 2 or history[-1][0] <= history[0][0]:
        return 0.0
    return abs(_wrap(history[-1][1] - history[0][1])) / (history[-1][0] - history[0][0])
