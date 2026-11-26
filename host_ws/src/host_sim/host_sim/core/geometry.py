"""2D shapes, vectorised ray casting and clearance queries."""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

_PARALLEL_EPS = 1e-12


@dataclass
class Circle:
    """Circular obstacle (also used for the other rover in scans)."""

    x: float
    y: float
    radius: float
    name: str = ''
    top: Optional[float] = None     # m above the floor; None = taller than any sensor


@dataclass
class Rect:
    """Rectangle obstacle given by centre, size and yaw."""

    x: float
    y: float
    width: float
    height: float
    yaw: float = 0.0
    name: str = ''
    top: Optional[float] = None     # m above the floor; None = taller than any sensor

    def corners(self):
        """Return the 4 corners (counter-clockwise) as a (4, 2) array."""
        hw, hh = self.width / 2.0, self.height / 2.0
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([self.x, self.y])

    def segments(self):
        """Return the 4 edges as a (4, 2, 2) array of (start, end) points."""
        pts = self.corners()
        return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)


def box_segments(x_min, x_max, y_min, y_max):
    """Edges of an axis-aligned box as a (4, 2, 2) array."""
    rect = Rect((x_min + x_max) / 2.0, (y_min + y_max) / 2.0,
                x_max - x_min, y_max - y_min)
    return rect.segments()


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def ray_circle_span(origin, dirs, circle):
    """(entry, exit) distance along each unit ray through a circle (inf when missed;
    entry = exit when the ray starts inside)."""
    oc = np.asarray(origin, dtype=float) - np.array([circle.x, circle.y])
    b = dirs @ oc
    disc = b * b - (oc @ oc - circle.radius ** 2)
    hit = disc >= 0.0
    root = np.sqrt(np.where(hit, disc, 0.0))
    near, far = -b - root, -b + root
    entry = np.where(near >= 0.0, near, far)
    ok = hit & (far >= 0.0)
    return np.where(ok, entry, np.inf), np.where(ok, far, np.inf)


def ray_circle(origin, dirs, circle):
    """Distance along each unit ray to a circle (inf when missed)."""
    return ray_circle_span(origin, dirs, circle)[0]


def ray_segment_matrix(origin, dirs, segments):
    """(N rays, M segments) distance along each ray to each segment (inf when missed)."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    start = segments[:, 0, :]
    edge = segments[:, 1, :] - start
    rel = start - np.asarray(origin, dtype=float)
    denom = _cross(dirs[:, None, :], edge[None, :, :])
    ok = np.abs(denom) > _PARALLEL_EPS
    safe = np.where(ok, denom, 1.0)
    t = _cross(rel, edge)[None, :] / safe
    s = _cross(rel[None, :, :], dirs[:, None, :]) / safe
    valid = ok & (t >= 0.0) & (s >= 0.0) & (s <= 1.0)
    return np.where(valid, t, np.inf)


def ray_segments(origin, dirs, segments):
    """Distance along each unit ray to the nearest of M segments (inf when missed)."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    if len(segments) == 0:
        return np.full(len(dirs), np.inf)
    return ray_segment_matrix(origin, dirs, segments).min(axis=1)


def point_segment_distance(point, start, end):
    """Euclidean distance from a point to a segment."""
    point, start, end = (np.asarray(v, dtype=float) for v in (point, start, end))
    edge = end - start
    length_sq = float(edge @ edge)
    if length_sq <= 0.0:
        return float(np.linalg.norm(point - start))
    ratio = min(1.0, max(0.0, float((point - start) @ edge) / length_sq))
    return float(np.linalg.norm(point - (start + ratio * edge)))


def point_polyline_distance(point, polyline):
    """Distance from a point to a polyline given as an (N, 2) array."""
    pts = np.asarray(polyline, dtype=float).reshape(-1, 2)
    if len(pts) == 1:
        return float(np.linalg.norm(np.asarray(point) - pts[0]))
    return min(point_segment_distance(point, a, b) for a, b in zip(pts[:-1], pts[1:]))


def point_shape_distance(point, shape):
    """Distance from a point to a shape's surface (0 inside a rect, <0 inside a circle)."""
    if isinstance(shape, Circle):
        return math.hypot(point[0] - shape.x, point[1] - shape.y) - shape.radius
    return _point_rect_distance(point, shape)


def _point_rect_distance(point, rect):
    c, s = math.cos(rect.yaw), math.sin(rect.yaw)
    dx, dy = point[0] - rect.x, point[1] - rect.y
    local_x, local_y = c * dx + s * dy, -s * dx + c * dy
    out_x = max(abs(local_x) - rect.width / 2.0, 0.0)
    out_y = max(abs(local_y) - rect.height / 2.0, 0.0)
    return math.hypot(out_x, out_y)
