"""Mapped obstacles against the truth of the layout: the obstacle mapper's measuring stick.

``LayoutLoop`` logs every publication of the mapper (sim time, the planner shapes, the true
rover circles). ``map_quality`` compares each published shape with the true objects:

  coverage   how far a true box corner lies OUTSIDE every published shape + pad (m, 0 =
             covered); a corner outside is a corner the rover may cut
  excess     how far the published RECTANGLE (pad not included) reaches beyond each side of
             the true box (m, per side in the box frame)
  phantoms   published shapes that touch no true box or shelf; ``rover`` when it touches a
             true rover body instead
  pubs/yaw   publications in which the box's shape changed / its orientation turned (by more
             than YAW_TURN modulo 90 deg)
  gap        narrowest free gap between published shapes + pads on DIFFERENT true objects (m)

Pure geometry on (x, y, pad, width, height, yaw) shapes (host_planning Minkowski rectangles)."""
import math
from dataclasses import dataclass, field

import numpy as np

YAW_TURN = math.radians(5.0)


@dataclass
class BoxStats:
    name: str
    pubs: int = 0                  # publications that carried a shape on this box
    changes: int = 0               # ... in which that shape differed from the previous one
    yaw_turns: int = 0             # ... in which its orientation turned (mod 90 deg)
    first_t: float = math.nan      # first publication covering the box
    coverage_max: float = 0.0      # worst corner outside shape + pad, after first_t (m)
    coverage_t: float = math.nan
    excess_max: float = 0.0        # worst side excess of the rectangle (m)
    excess_side: str = ''
    excess_final: float = 0.0      # side excess in the last publication
    yaws: list = field(default_factory=list)


@dataclass
class MapQuality:
    boxes: dict
    publications: int
    phantoms: list                 # (t, shape, 'rover:<name>' | 'free')
    min_gap: float = math.inf      # narrowest gap between two published padded shapes (m)
    min_gap_t: float = math.nan

    def rows(self, label):
        out = []
        for b in self.boxes.values():
            out.append(f'{label:22s} {b.name:5s} pubs={b.pubs:3d} changes={b.changes:3d} '
                       f'yaw_turns={b.yaw_turns:2d} cover_out_max={b.coverage_max:.3f}'
                       f'@{b.coverage_t:5.1f}s excess_max={b.excess_max:.3f}({b.excess_side}) '
                       f'excess_final={b.excess_final:.3f}')
        rover = [p for p in self.phantoms if p[2].startswith('rover')]
        gap = f'{self.min_gap:.3f}@{self.min_gap_t:.1f}s' if math.isfinite(self.min_gap) else '-'
        out.append(f'{label:22s} total publications={self.publications} '
                   f'phantom shape-pubs={len(self.phantoms)} (on a rover {len(rover)}) '
                   f'min_gap={gap}'
                   + (f' first phantom {self.phantoms[0][0]:.1f}s {self.phantoms[0][2]}'
                      if self.phantoms else ''))
        return out


def _corners(x, y, yaw, half_len, half_wid):
    c, s = math.cos(yaw), math.sin(yaw)
    u, v = np.array([c, s]), np.array([-s, c])
    centre = np.array([x, y])
    return np.array([centre + a * half_len * u + b * half_wid * v
                     for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def _point_rect_distance(points, x, y, yaw, half_len, half_wid):
    """Distance of (N, 2) points to the rectangle (0 inside)."""
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(points, dtype=float) - np.array([x, y])
    ex = np.maximum(np.abs(d @ np.array([c, s])) - half_len, 0.0)
    ey = np.maximum(np.abs(d @ np.array([-s, c])) - half_wid, 0.0)
    return np.hypot(ex, ey)


def _segment_distance(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.hypot(*(p - (a + t[:, None] * ab)).T)


def _polygon_gap(pa, pb):
    """Distance between two convex polygons (corner arrays), 0 when they overlap."""
    def inside(points, poly):
        signs = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            signs.append((b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0]))
        signs = np.array(signs)
        return np.all(signs >= 0, axis=0) | np.all(signs <= 0, axis=0)
    if inside(pa, pb).any() or inside(pb, pa).any():
        return 0.0
    best = math.inf
    for p, q in ((pa, pb), (pb, pa)):
        for i in range(len(q)):
            best = min(best, float(_segment_distance(p, q[i], q[(i + 1) % len(q)]).min()))
    return best


def shape_corners(shape):
    x, y, _pad, width, height, yaw = shape
    return _corners(x, y, yaw, width / 2.0, height / 2.0)


def box_corners(box):
    return _corners(box['x'], box['y'], box.get('yaw', 0.0), box['width'] / 2.0,
                    box['height'] / 2.0)


def coverage_gap(shape, box):
    """How far the worst true corner lies outside the shape + pad (m, 0 = covered)."""
    x, y, pad, width, height, yaw = shape
    d = _point_rect_distance(box_corners(box), x, y, yaw, width / 2.0, height / 2.0)
    return float(max(d.max() - pad, 0.0))


def union_coverage_gap(shapes, box):
    """How far the worst true corner lies outside every one of the shapes + pads."""
    corners = box_corners(box)
    gaps = [np.maximum(_point_rect_distance(corners, x, y, yaw, w / 2.0, h / 2.0) - pad, 0.0)
            for x, y, pad, w, h, yaw in shapes]
    return float(np.min(gaps, axis=0).max())


def side_excess(shape, box):
    """{side: m} how far the shape's rectangle reaches beyond each side of the box."""
    yaw = box.get('yaw', 0.0)
    c, s = math.cos(yaw), math.sin(yaw)
    rel = shape_corners(shape) - np.array([box['x'], box['y']])
    along, across = rel @ np.array([c, s]), rel @ np.array([-s, c])
    hw, hh = box['width'] / 2.0, box['height'] / 2.0
    return {'+x': max(along.max() - hw, 0.0), '-x': max(-along.min() - hw, 0.0),
            '+y': max(across.max() - hh, 0.0), '-y': max(-across.min() - hh, 0.0)}


def shape_object_gap(shape, corners):
    """Free distance between a padded shape and a true object's corner polygon."""
    return max(_polygon_gap(shape_corners(shape), corners) - shape[2], 0.0)


def padded_gap(a, b):
    return max(_polygon_gap(shape_corners(a), shape_corners(b)) - a[2] - b[2], 0.0)


def _turned(yaw_a, yaw_b):
    d = (yaw_a - yaw_b) % (math.pi / 2.0)
    return min(d, math.pi / 2.0 - d) > YAW_TURN


def map_quality(log, boxes, shelves=()):
    """``log``: [(t, shapes, {rover: (x, y, radius)})], one entry per mapper publication.
    ``boxes`` / ``shelves``: true rect dicts (x, y, width, height, yaw, name)."""
    stats = {b['name']: BoxStats(b['name']) for b in boxes}
    last = {}
    phantoms = []
    quality = MapQuality(stats, len(log), phantoms)
    objects = [(b, box_corners(b)) for b in list(boxes) + list(shelves)]
    for t, shapes, rovers in log:
        on_box = {}
        names = []                                 # objects each shape touches
        for shape in shapes:
            touching = [b for b, corners in objects if shape_object_gap(shape, corners) <= 0.0]
            names.append({b['name'] for b in touching})
            if not touching:
                on_rover = [name for name, (rx, ry, rr) in rovers.items()
                            if _point_rect_distance([[rx, ry]], shape[0], shape[1], shape[5],
                                                    shape[3] / 2.0, shape[4] / 2.0)[0]
                            <= shape[2] + rr]
                phantoms.append((t, shape, f'rover:{on_rover[0]}' if on_rover else 'free'))
                continue
            for box in touching:
                if box['name'] not in stats:
                    continue                       # a shelf: not a box under test
                _account(stats[box['name']], last, t, shape, box)
                on_box.setdefault(box['name'], []).append(shape)
        for box in boxes:
            if box['name'] in on_box:
                b = stats[box['name']]
                cover = union_coverage_gap(on_box[box['name']], box)
                if cover > b.coverage_max:
                    b.coverage_max, b.coverage_t = cover, t
        for i, a in enumerate(shapes):
            for j in range(i + 1, len(shapes)):
                if not names[i] or not names[j] or names[i] & names[j]:
                    continue                       # pieces of one object, or a phantom
                gap = padded_gap(a, shapes[j])
                if gap < quality.min_gap:
                    quality.min_gap, quality.min_gap_t = gap, t
    return quality


def _account(b, last, t, shape, box):
    b.pubs += 1
    if math.isnan(b.first_t):
        b.first_t = t
    prev = last.get(b.name)
    if prev is not None and prev != shape:
        b.changes += 1
        if _turned(prev[5], shape[5]):
            b.yaw_turns += 1
    last[b.name] = shape
    b.yaws.append(round(shape[5], 2))
    excess = side_excess(shape, box)
    side = max(excess, key=excess.get)
    if excess[side] > b.excess_max:
        b.excess_max, b.excess_side = excess[side], side
    b.excess_final = max(excess.values())
