"""Platoon rules (mixin of ``PathCommander``): when a rover yields to the other, and the ride
along the lead path to a turn-in or hold point."""
import dataclasses
import math

import numpy as np
from host_planning import api as planning_api
from host_planning.pivot import pivot_clear, search_departure
from host_planning.types import Pose2D

from host_mission.path_geometry import planner_shape, turn_in_point
from host_mission.path_types import YIELD, Plan, serves

_HOLD_STEP = 0.10            # m along the lead path between the hold points tried
_HOLD_MAX_BACK = 2.5         # m: no hold point farther back than this from the lead's goal


class PlatoonRules:
    """Yield and ride-to-point rules; reads ``settings``, ``cmds``, ``planner`` and the wanted goals."""

    def _waits_for_lead_turn(self, robot, want, world):
        """True while the follower should not start a new leg because the lead is still making its
        leading pivot."""
        platoon = self.settings.platoon
        if not platoon.enabled or robot != platoon.follower \
                or serves(self.cmds.get(robot), want):
            return False
        lead = world.robots.get(platoon.lead_robot)
        return lead is not None and lead.pose is not None \
            and self._turn_left(platoon.lead_robot, lead) is not None

    def _blocked_only_by_peer(self, robot, start, goal, provisional, world, params):
        """True when ``robot`` has no path only because the other rover stands in the way and moves
        first, so it waits (YIELD) rather than report NO_PATH (both stopped). The follower waits
        for a lead that is going to move; the lead only for a follower that has to leave and has
        a path."""
        platoon = self.settings.platoon
        if not platoon.avoid_parked_peer:
            return False
        if robot == platoon.follower:
            if not self._peer_will_move(platoon.lead_robot, world):
                return False
        elif not (self._peer_will_move(platoon.follower, world)
                  and self._peer_can_leave(platoon.follower, world)):
            return False
        hold = self._holds.get(robot)
        if hold is not None and hold.reason == YIELD \
                and hold.obstacle_version == world.obstacle_version:
            # Already waiting on this map from where it stands: asking again would let the pose
            # noise of the standing rover flip the answer and re-issue both paths.
            return True
        peer = platoon.lead_robot if robot == platoon.follower else platoon.follower
        others = [planner_shape(o) for o in world.obstacles()] + list(self.settings.fixtures)
        # Once the peer has left it stands at its goal: a path that exists only with the peer
        # gone altogether is not worth waiting for.
        spot = self._wanted_pose(peer, others)
        if spot is not None:
            others.append(self._footprint_shape(peer, (spot.x, spot.y, spot.theta)))
        return self._free_path(start, goal, others, provisional, params)

    def _free_path(self, start, goal, others, provisional, params):
        """True when a path (forward, else pivot-first) from ``start`` to ``goal`` keeps clear of
        ``others``."""
        bounds = self.settings.bounds

        def plan_from(pose):
            return self.planner.plan_path(pose, goal, others, bounds, params)

        path = plan_from(start)
        if path is None and params.pivot_step > 0.0 and not provisional:
            path, _heading = search_departure(start, (goal.x, goal.y), plan_from,
                                              planning_api.path_length, others, bounds, params,
                                              goal_heading=goal.theta)
        return path is not None

    def _peer_can_leave(self, peer, world):
        state = world.robots.get(peer)
        others = [planner_shape(o) for o in world.obstacles()] + list(self.settings.fixtures)
        others += self._peers_now(peer, world)
        goal = self._wanted_pose(peer, others)
        return goal is not None and self._free_path(
            Pose2D(*state.pose), goal, others, False,
            self._goal_params(peer, self._plan_params(peer), goal))

    def _wanted_pose(self, robot, obstacles):
        want = self._desired.get(robot)
        if want is None or want.goal is None:
            return None
        if want.region is None:
            return Pose2D(want.goal.x, want.goal.y, want.goal.theta)
        goals = self.planner_region(want.region, obstacles, self._plan_params(robot))
        return goals[0] if goals else None

    def _peer_will_move(self, peer, world):
        want, state = self._desired.get(peer), world.robots.get(peer)
        if want is None or want.goal is None or state is None or state.pose is None \
                or peer in self._holds:
            return False
        pose = Pose2D(*state.pose)
        cfg = self.settings.robots[peer]
        if want.region is not None:
            return not want.region.holds(pose, cfg.goal_yaw_tol)
        return math.hypot(pose.x - want.goal.x, pose.y - want.goal.y) > cfg.goal_pos_tol

    def _ride_to_turn_in(self, robot, start, goal, obstacles, params, world, arrived, now):
        """A shared path to the turn-in point of ``goal`` (module docstring), or None. The last
        rung: only when every ordinary path has to wait and none is a path behind the lead."""
        platoon = self.settings.platoon
        if not platoon.dock_turn or self._lead_path(robot, world, arrived, now) is None:
            return None
        lead_path = list(self.cmds[platoon.lead_robot].poses)
        point = turn_in_point(lead_path, goal, params.approach_len)
        if point is None \
                or math.hypot(point.x - start.x, point.y - start.y) < params.approach_len:
            return None
        bounds = self.settings.bounds
        turn = math.atan2(math.sin(goal.theta - point.theta),
                          math.cos(goal.theta - point.theta))
        turned = Pose2D(point.x, point.y, goal.theta)
        if not pivot_clear(point, turn, obstacles, bounds, params) \
                or self.planner.plan_path(turned, goal, obstacles, bounds, params) is None:
            return None
        return self._ride_to(robot, start, point, goal, lead_path, obstacles, params, world,
                             arrived)

    def _ride_to_hold(self, robot, start, goal, obstacles, params, world, arrived):
        platoon = self.settings.platoon
        if not platoon.hold_ride:
            return None
        lead_path = list(self.cmds[platoon.lead_robot].poses)
        point = self._hold_point(lead_path, goal, obstacles, params)
        if point is None \
                or math.hypot(point.x - start.x, point.y - start.y) < params.approach_len:
            return None
        plan = self._ride_to(robot, start, point, goal, lead_path, obstacles, params, world,
                             arrived)
        return plan if plan is None else dataclasses.replace(plan, hold=True)

    def _hold_point(self, lead_path, goal, obstacles, params):
        """The pose of ``lead_path`` nearest its end where the follower may stand while the lead
        finishes and from where it can reach ``goal`` (``obstacles`` hold the lead parked at its
        goal), or None."""
        bounds = self.settings.bounds
        spots = self._hold_spots(lead_path, obstacles, params)
        for pose in spots:
            if self.planner.plan_path(pose, goal, obstacles, bounds, params) is not None:
                return pose
        return next((pose for pose in spots
                     if self._free_path(pose, goal, obstacles, False, params)), None)

    def _hold_spots(self, lead_path, obstacles, params):
        """Poses of ``lead_path`` where the follower may stand while the lead finishes, nearest the
        end first, between max(``gap``, ``approach_len``) and ``_HOLD_MAX_BACK`` back, clear of
        obstacles."""
        platoon = self.settings.platoon
        lead = platoon.lead_robot
        lead_params = self._issued_params(lead, self.cmds.get(lead))
        standing = dataclasses.replace(params, front_zone_runin=0.0)
        arc = np.concatenate(([0.0], np.cumsum(
            [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(lead_path, lead_path[1:])])))
        spots = []
        back = max(params.approach_len, platoon.gap)
        while back <= _HOLD_MAX_BACK:
            i = int(np.searchsorted(arc, arc[-1] - back, side='right')) - 1
            if i < 0:
                break
            back = arc[-1] - arc[i] + _HOLD_STEP
            pose = lead_path[i]
            ahead = [p for p, s in zip(lead_path, arc) if s >= arc[i] + platoon.gap]
            body = self._footprint_shape(platoon.follower, (pose.x, pose.y, pose.theta))
            if planning_api.keeps_clearance(ahead, [body], lead_params) \
                    and planning_api.keeps_clearance([pose], obstacles, standing):
                spots.append(pose)
        return spots

    def _ride_to(self, robot, start, point, goal, lead_path, obstacles, params, world,
                 arrived):
        platoon = self.settings.platoon
        bounds = self.settings.bounds
        ignore = self._lead_zone_ignore(params, world, arrived)
        extra = {'zone_ignore': ignore} if ignore else {}
        on_the_way = dataclasses.replace(params, front_zone_runin=0.0)

        def ride_from(pose):
            ride = self.planner.plan_follower_path(pose, lead_path, point, obstacles,
                                                   bounds, on_the_way, **extra)
            if ride is None or not ride.shared:
                return None
            return Plan(tuple(ride.path), platoon.lead_robot, False, goal, via=True)

        plan, heading = search_departure(
            start, (point.x, point.y), ride_from,
            lambda p: planning_api.path_length(p.poses),
            obstacles + self._peers_beside_a_turn(robot, world), bounds, params,
            ride_from(start), point.theta)
        if plan is not None and heading is not None:
            plan = dataclasses.replace(plan, pivot_heading=heading)
        return plan

    def _parks_in_lead_way(self, robot, goal, world, arrived, now):
        lead_path = self._lead_path(robot, world, arrived, now)
        return lead_path is not None and not self._parks_clear_of_lead(goal, lead_path)

    def _parks_clear_of_lead(self, goal, lead_path):
        """True when the follower parked at ``goal`` keeps the lead's planning clearance from the
        lead's remaining path."""
        follower, lead = self.settings.platoon.follower, self.settings.platoon.lead_robot
        spot = self._footprint_shape(follower, (goal.x, goal.y, goal.theta))
        return planning_api.keeps_clearance(lead_path, [spot],
                                            self._issued_params(lead, self.cmds.get(lead)))
