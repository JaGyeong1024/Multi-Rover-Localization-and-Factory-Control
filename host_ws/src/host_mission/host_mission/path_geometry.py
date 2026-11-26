"""Pure geometry for path commands: footprints, fixtures, path conflicts (no commander state)."""
import dataclasses
import math

import numpy as np
from host_planning.geometry import obstacle_array, shape_gaps
from host_planning.types import Box, Circle

_REGION_SPREAD = (0.05, 0.09)   # m, rad: tried poses differ by at least one of these


def faces_fixture(robot_cfg, fixtures, goal):
    """True when a known fixture lies within ``front_exempt_reach`` ahead of the footprint front at
    ``goal``."""
    if robot_cfg.front_exempt_reach <= 0.0 or not fixtures:
        return False
    footprint = robot_cfg.plan.footprint
    if footprint is None:
        radius = robot_cfg.plan.robot_radius
        footprint = (radius, radius, radius)
    front, _rear, half_wid = footprint
    ahead = footprint_box((goal.x, goal.y, goal.theta),
                          (front + robot_cfg.front_exempt_reach, -front, half_wid))
    gaps = shape_gaps(obstacle_array([ahead]), obstacle_array(fixtures))
    return bool(np.min(gaps) <= 0.0)


def pick_goals_without_fixture(settings, scenario, goals):
    """Names of the picker's pick goals with no registered fixture ahead (untaught skipped)."""
    robot_cfg = settings.robots[settings.pick.picker]
    names = []
    for part in scenario.parts.values():
        goal = goals.get(part.leader_goal)
        if goal is not None and part.leader_goal not in names \
                and not faces_fixture(robot_cfg, settings.fixtures, goal):
            names.append(part.leader_goal)
    return names


def footprint_box(pose, footprint):
    x, y, yaw = pose
    front, rear, half_wid = footprint
    shift = (front - rear) / 2.0
    return Box(x + shift * math.cos(yaw), y + shift * math.sin(yaw), front + rear,
               2.0 * half_wid, yaw)


def planner_shape(obstacle):
    if obstacle.width > 0.0 or obstacle.height > 0.0:
        return Box(obstacle.x, obstacle.y, obstacle.width, obstacle.height, obstacle.yaw,
                   obstacle.r)
    return Circle(obstacle.x, obstacle.y, obstacle.r)


def spread_goals(goals, count):
    min_dist, min_turn = _REGION_SPREAD
    kept = []
    for goal in goals:
        if all(math.hypot(goal.x - k.x, goal.y - k.y) >= min_dist
               or abs(math.atan2(math.sin(goal.theta - k.theta),
                                 math.cos(goal.theta - k.theta))) >= min_turn
               for k in kept):
            kept.append(goal)
            if len(kept) == count:
                break
    return kept


def turn_in_point(lead_path, goal, min_back):
    """The pose of ``lead_path`` where it last crosses the straight line into ``goal``
    (through the goal, along its heading) at least ``min_back`` behind the goal, or None.
    Of the two poses around the crossing, the one nearer to the line."""
    cos, sin = math.cos(goal.theta), math.sin(goal.theta)
    found, before = None, None
    for pose in lead_path:
        side = (pose.y - goal.y) * cos - (pose.x - goal.x) * sin
        back = (goal.x - pose.x) * cos + (goal.y - pose.y) * sin
        if before is not None and before[1] * side <= 0.0 and back >= min_back:
            found = pose if abs(side) <= abs(before[1]) else before[0]
        before = (pose, side)
    return found


def paths_conflict(path, lead_path, min_dist):
    """True when ``path`` comes within ``min_dist`` of any pose of the lead's remaining path
    (spatial on purpose: the lead can be held anywhere)."""
    if len(path) < 2 or not lead_path:
        return False
    lead = [(p.x, p.y) for p in lead_path]
    return any(math.hypot(pose.x - x, pose.y - y) < min_dist
               for pose in path for x, y in lead)


def grown(shape, by):
    if isinstance(shape, Box):
        return dataclasses.replace(shape, pad=shape.pad + by)
    return dataclasses.replace(shape, r=shape.r + by)


def rotation_centre(pose, params):
    off = params.pivot_offset
    return pose.x - off * math.cos(pose.theta), pose.y - off * math.sin(pose.theta)


def farthest_point(params, radius):
    off = params.pivot_offset
    if params.footprint is None:
        return radius + abs(off)
    front, rear, half_wid = params.footprint
    return math.hypot(max(front + off, rear - off), half_wid)


def nearest_index(poses, pose):
    return min(range(len(poses)),
               key=lambda i: math.hypot(poses[i].x - pose[0], poses[i].y - pose[1]))
