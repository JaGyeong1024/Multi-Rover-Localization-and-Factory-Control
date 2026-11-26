"""Follower path laid on top of the leader path (platooning).

``follower path = join connector + shared leader stretch + exit to the goal``

* Join point: the earliest leader-path point, from the one nearest the follower
  onwards, that lies ahead and can be reached by a gentle polynomial connector (no
  Dubins, length at most ``platoon_detour_ratio`` x chord).
* Branch point: the latest leader-path point from which the follower goal (with its
  run-in) is still ahead and reachable by the same kind of curve.
* The shared stretch reuses the leader's points verbatim.

When no join/branch pair exists a stand-alone path is returned with ``shared`` False.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from . import avoid, smooth
from .geometry import (ISSUED_CURVATURE_SLACK, Validator, arc_lengths, obstacle_array, join,
                       make_validator, resample, shrink, to_array, to_poses,
                       wrap_angle, zone_mask)
from .types import Bounds, Circle, PlanParams, PlatoonPlan, Pose2D

Found = Tuple[int, np.ndarray]   # (leader index, dense connector)
_SAME_POSE = 1e-6
# A goal this close to a leader point (m, rad) is that point.
_ON_PATH_POS = 0.01
_ON_PATH_YAW = 0.05


def _pose(row: np.ndarray) -> Pose2D:
    return Pose2D(float(row[0]), float(row[1]), float(row[2]))


def _ahead(origin: Pose2D, target: Pose2D) -> bool:
    dx, dy = target.x - origin.x, target.y - origin.y
    forward = dx * math.cos(origin.theta) + dy * math.sin(origin.theta)
    arriving = dx * math.cos(target.theta) + dy * math.sin(target.theta)
    return forward > 0.0 and arriving > 0.0


def _max_len(a: Pose2D, b: Pose2D, params: PlanParams) -> float:
    chord = math.hypot(b.x - a.x, b.y - a.y)
    return params.platoon_detour_ratio * chord + params.spacing


def _index_step(leader: np.ndarray, params: PlanParams) -> int:
    mean = arc_lengths(leader)[-1] / max(1, len(leader) - 1)
    return max(1, int(round(params.platoon_search_step / max(mean, 1e-6))))


def _already_on(follower: Pose2D, node: Pose2D) -> bool:
    close = math.hypot(node.x - follower.x, node.y - follower.y) < _SAME_POSE
    return close and abs(wrap_angle(node.theta - follower.theta)) < _SAME_POSE


def find_join(follower: Pose2D, leader: np.ndarray, params: PlanParams,
              valid: Validator) -> Optional[Found]:
    nearest = int(np.argmin(np.hypot(leader[:, 0] - follower.x,
                                     leader[:, 1] - follower.y)))
    if _already_on(follower, _pose(leader[nearest])):
        return nearest, leader[nearest:nearest + 1]
    for i in range(nearest, len(leader), _index_step(leader, params)):
        node = _pose(leader[i])
        if not _ahead(follower, node):
            continue
        link = smooth.connect(follower, node, params, valid, allow_dubins=False,
                              max_len=_max_len(follower, node, params))
        if link is not None:
            return i, link
    return None


def _on_path(leader: np.ndarray, first: int, goal: Pose2D) -> Optional[int]:
    """Index (>= ``first``) of the leader point the goal is (position and heading), or None.

    A goal on the leader path is reached by sharing the path up to it: no exit, no
    run-in (on a curved stretch a straight run-in could never end on it)."""
    if first >= len(leader):
        return None
    rest = leader[first:]
    d = np.hypot(rest[:, 0] - goal.x, rest[:, 1] - goal.y)
    i = int(np.argmin(d))
    heading = abs(wrap_angle(float(rest[i, 2]) - goal.theta))
    return first + i if d[i] <= _ON_PATH_POS and heading <= _ON_PATH_YAW else None


def find_branch(leader: np.ndarray, first: int, goal: Pose2D, params: PlanParams,
                valid: Validator) -> Optional[Found]:
    on = _on_path(leader, first, goal)
    if on is not None:
        return on, leader[on:on + 1]
    entry = smooth.approach_pose(goal, params.approach_len)
    for i in range(len(leader) - 1, first - 1, -_index_step(leader, params)):
        node = _pose(leader[i])
        if not _ahead(node, entry):
            continue
        tail = smooth.dense_path(node, goal, params, valid, allow_dubins=False,
                                 max_len=_max_len(node, entry, params))
        if tail is not None:
            return i, tail
    return None


def _shared_plan(follower, leader, goal, params, valid, valid_issued) -> Optional[PlatoonPlan]:
    """Join + shared + exit plan, or None when the leader path can't be shared.

    ``valid`` judges the new connectors; ``valid_issued`` the leader's own stretch
    (bounds and clearance for the follower's body)."""
    joined = find_join(follower, leader, params, valid)
    if joined is None:
        return None
    s = arc_lengths(leader)
    j, link = joined
    reach = s[j] + params.platoon_min_shared
    first = int(np.searchsorted(s, reach))
    branched = find_branch(leader, first, goal, params, valid) if first < len(leader) else None
    if branched is None or not valid_issued(leader[j:branched[0] + 1]):
        return None
    b, tail = branched
    head = resample(link, params.spacing)
    exit_part = resample(tail, params.spacing) if len(tail) > 1 else tail   # on the path: none
    path = join([head, leader[j + 1:b + 1], exit_part[1:]])
    join_s = float(arc_lengths(head)[-1])
    return PlatoonPlan(to_poses(path), True, join_s, join_s + float(s[b] - s[j]),
                       float(s[j]), float(s[b]))


def plan_platoon(follower: Pose2D, leader_path: Sequence[Pose2D], goal: Pose2D,
                 obstacles: Sequence[Circle] = (), bounds: Optional[Bounds] = None,
                 params: Optional[PlanParams] = None,
                 zone_ignore: Sequence[object] = ()) -> Optional[PlatoonPlan]:
    """Follower plan sharing the leader path, else stand-alone, else None.

    ``zone_ignore``: obstacles (equal to entries of ``obstacles``) the front-stop zone
    leaves out on a shared plan (the lead rover). A stand-alone plan counts all."""
    params = params or PlanParams()
    leader = to_array(leader_path)
    if len(leader) >= 2:
        area = shrink(bounds, params.bounds_margin)
        start = (follower.x, follower.y)
        end = (goal.x, goal.y, goal.theta)
        obs, seen = obstacle_array(obstacles), zone_mask(obstacles, zone_ignore)
        valid = make_validator(obs, area, params, start=start, goal=end, seen=seen)
        valid_issued = make_validator(obs, area, params, start=start,
                                      curvature_slack=ISSUED_CURVATURE_SLACK, goal=end,
                                      seen=seen)
        plan = _shared_plan(follower, leader, goal, params, valid, valid_issued)
        if plan is not None:
            return plan
    solo = avoid.plan_around(follower, goal, obstacles, bounds, params)
    return None if solo is None else PlatoonPlan(to_poses(solo), False)
