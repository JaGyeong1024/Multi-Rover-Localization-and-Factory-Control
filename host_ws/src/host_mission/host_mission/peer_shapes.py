"""Where the other rover is, or will be, as planner obstacles (mixin of ``PathCommander``)."""
import math

from host_planning.pivot import swept_poses
from host_planning.types import Circle, Pose2D

from host_mission.path_geometry import farthest_point, footprint_box, grown
from host_mission.path_types import FOLLOW, past_the_turn


class PeerShapes:
    """Footprints of the other rover and its remaining turn; reads ``settings`` and ``cmds``."""

    def _parked_peers(self, robot, world, arrived):
        """Footprint per other rover: where it stands, or where it is going to park."""
        platoon = self.settings.platoon
        if not platoon.avoid_parked_peer:
            return []
        shapes = []
        for peer in self.settings.robots:
            spot = None if peer == robot else self._parking_spot(peer, world, arrived)
            if spot is not None:
                shapes.append(self._footprint_shape(peer, spot))
        if robot == platoon.lead_robot:
            # The lead never waits for a follower on its way, so that follower occupies where
            # it stands now as well as its goal (the depth camera misses a body inside 0.6 m).
            # Not for the follower: its shared path runs through the lead's place (gap control).
            shapes += self._peers_now(robot, world)
        return shapes

    def _peers_now(self, robot, world):
        shapes = []
        for peer in self.settings.robots:
            state = world.robots.get(peer)
            if peer != robot and state is not None and state.pose is not None:
                shapes.append(self._footprint_shape(peer, state.pose))
                shapes += self._turn_rest(peer, state)
        return shapes

    def peer_footprints(self, robot, world):
        return [self._footprint_shape(peer, state.pose)
                for peer, state in world.robots.items()
                if peer != robot and peer in self.settings.robots and state.pose is not None]

    def _peers_beside_a_turn(self, robot, world):
        """``_peers_now`` for a turn in place, with a peer's remaining turn grown by that peer's
        ``clearance`` (two rovers turning at once add two localization errors)."""
        shapes = []
        for peer in self.settings.robots:
            state = world.robots.get(peer)
            if peer != robot and state is not None and state.pose is not None:
                here = self._footprint_shape(peer, state.pose)
                rest = self._turn_rest(peer, state)
                if rest:
                    margin = self.settings.robots[peer].plan.clearance
                    shapes += [grown(shape, margin) for shape in [here] + rest]
                else:
                    shapes.append(here)
        return shapes

    def _turn_rest(self, peer, state):
        rest = self._turn_left(peer, state)
        if rest is None:
            return []
        x, y, yaw = state.pose
        params = self.settings.robots[peer].plan
        swept = swept_poses(Pose2D(x, y, yaw), rest, params)
        step = abs(rest) / (len(swept) - 1)
        shapes = [self._footprint_shape(peer, tuple(p)) for p in swept[1:]]
        pad = 2.0 * farthest_point(params, self.settings.platoon.peer_radius) \
            * math.sin(step / 4.0)
        return [grown(shape, pad) for shape in shapes]

    def _turn_left(self, peer, state):
        """The rest of ``peer``'s leading turn (rad) while it is still making it, else None. Still
        turning: heading not within goal_yaw_tol of pivot_heading and no report that the rover
        is past the turn (TRACKING beyond PIVOT_DETAILS, or arrived); no report yet counts as
        turning."""
        cmd = self.cmds.get(peer)
        if cmd is None or cmd.mode != FOLLOW or not cmd.pivot_first:
            return None
        yaw = state.pose[2]
        rest = math.atan2(math.sin(cmd.pivot_heading - yaw), math.cos(cmd.pivot_heading - yaw))
        if abs(rest) <= self.settings.robots[peer].goal_yaw_tol:
            return None
        return None if past_the_turn(state.nav, cmd) else rest

    def _footprint_shape(self, robot, pose):
        x, y, yaw = pose
        footprint = self.settings.robots[robot].plan.footprint
        if footprint is None:
            return Circle(x, y, self.settings.platoon.peer_radius)
        return footprint_box((x, y, yaw), footprint)

    def _lead_zone_ignore(self, params, world, arrived):
        lead = self.settings.platoon.lead_robot
        if params.front_zone is None or not self.settings.platoon.avoid_parked_peer:
            return []
        spot = self._parking_spot(lead, world, arrived)
        return [] if spot is None else [self._footprint_shape(lead, spot)]

    def _parking_spot(self, peer, world, arrived):
        cmd = self.cmds.get(peer)
        # A rover riding to a hold point counts where it stands: the hold point lies on the lead's
        # own path and the rover only reaches it behind the lead (gap control).
        if cmd is not None and cmd.mode == FOLLOW and not arrived.get(peer) and not cmd.hold:
            last = cmd.goal if cmd.via else cmd.poses[-1]
            return last.x, last.y, last.theta
        state = world.robots.get(peer)
        return None if state is None or state.pose is None else tuple(state.pose)
