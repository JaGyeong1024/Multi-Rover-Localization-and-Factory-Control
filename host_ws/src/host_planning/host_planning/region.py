"""Goal regions: every pose that serves a task, instead of one fixed goal point.

A part is dropped at a fixed point and the follower can receive it anywhere on its
cargo area. Feasible load poses are those whose cargo area (shrunk by a margin)
contains the drop point; the heading parallel to the picker and the drop point nearest
the cargo centre are preferred. A pose qualifies only when it and its straight run-in
stay inside the bounds and keep the footprint clearance. Reachability by a forward
path is left to the caller (plan the candidates in order).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import make_validator, obstacle_array, shrink, straight
from .types import Bounds, PlanParams, Pose2D


@dataclass(frozen=True)
class DropRegion:
    """Poses whose cargo area holds ``drop`` (world x, y) at least ``margin`` inside.

    ``cargo`` = (x, y, length, width): the usable area's centre in the rover's
    tracked-point frame (x forward, y left) and its size, m. ``heading`` = preferred
    rover heading (parallel to the picker).
    """

    drop: Tuple[float, float]
    heading: float
    cargo: Tuple[float, float, float, float]
    margin: float = 0.03
    step: float = 0.02                              # sampling of the offset (m)
    heading_offsets: Tuple[float, ...] = (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3)  # rad

    def holds(self, pose: Pose2D, heading_tol: float) -> bool:
        """True when ``pose`` already is a region pose (within ``heading_tol``)."""
        dx, dy = self._offset(pose)
        cx, cy, half_l, half_w = self._usable()
        turn = math.atan2(math.sin(pose.theta - self.heading),
                          math.cos(pose.theta - self.heading))
        return abs(dx - cx) <= half_l and abs(dy - cy) <= half_w \
            and abs(turn) <= max(abs(o) for o in self.heading_offsets) + heading_tol

    def candidates(self) -> List[Pose2D]:
        """Region poses, best first: heading offset, then drop distance to the centre."""
        cx, cy, half_l, half_w = self._usable()
        if half_l < 0.0 or half_w < 0.0:
            return []
        us = _symmetric(half_l, self.step)
        vs = _symmetric(half_w, self.step)
        du, dv = np.meshgrid(us, vs, indexing='ij')
        du, dv = du.ravel(), dv.ravel()
        order = np.argsort(np.hypot(du, dv), kind='stable')
        out = []
        for offset in self.heading_offsets:
            theta = self.heading + offset
            c, s = math.cos(theta), math.sin(theta)
            for k in order:                     # drop at (cx + du, cy + dv) in my frame
                fx, fy = cx + du[k], cy + dv[k]
                out.append(Pose2D(self.drop[0] - c * fx + s * fy,
                                  self.drop[1] - s * fx - c * fy,
                                  math.atan2(s, c)))
        return out

    def _usable(self):
        cx, cy, length, width = self.cargo
        return cx, cy, length / 2.0 - self.margin, width / 2.0 - self.margin

    def _offset(self, pose):
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        rx, ry = self.drop[0] - pose.x, self.drop[1] - pose.y
        return c * rx + s * ry, -s * rx + c * ry


def _symmetric(half, step):
    count = int(math.floor(half / step + 1e-9))
    return np.arange(-count, count + 1) * step


def feasible_goals(region: DropRegion, obstacles: Sequence[object] = (),
                   bounds: Optional[Bounds] = None,
                   params: Optional[PlanParams] = None) -> List[Pose2D]:
    """Region poses (best first) whose pose and straight run-in are valid: inside the
    bounds, with footprint clearance and a free front-stop zone."""
    params = params or PlanParams()
    obs, area = obstacle_array(obstacles), shrink(bounds, params.bounds_margin)
    valid = make_validator(obs, area, params)
    out = []
    for goal in region.candidates():
        ux, uy = math.cos(goal.theta), math.sin(goal.theta)
        entry = (goal.x - params.approach_len * ux, goal.y - params.approach_len * uy)
        check = valid if params.front_zone_runin <= 0.0 else make_validator(
            obs, area, params, goal=(goal.x, goal.y, goal.theta))
        if check(straight(entry, (goal.x, goal.y), goal.theta, params.dense_step)):
            out.append(goal)
    return out
