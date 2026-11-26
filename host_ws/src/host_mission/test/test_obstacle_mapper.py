"""Static-obstacle mapper against host_sim's world and ray-cast LiDAR (ROS-free).

The follower LiDAR is modelled on a 10 Hz, 1830-beam scanner with 1 cm range noise; poses
get the simulator's default localization noise (2 cm, 0.02 rad). Tolerances pinned here:
  coverage   every point of a box footprint lies inside a published circle
  free space a circle reaches at most 0.30 m beyond a box seen on two faces
             (0.10 m inflation + the circumscribed-circle bulge of a 0.5 x 0.4 box)"""
import math

import numpy as np
import pytest
from helpers import engine, settings

from host_mission.obstacle_mapper import (BAD_QUALITY, FOLLOWER_LIDAR, NOT_ALIGNED,
                                          STALE_POSE, TURNED, TURNING, MapperParams, ObstacleMapper,
                                          Region, ScanData, ScanSource)
from host_mission.world_model import Obstacle
from host_planning.types import Box
from host_sim.core import (Arena, LidarParams, LocalizationParams, Rect, Robot,
                           RobotParams, World)
from host_sim.core.geometry import point_shape_distance

SCAN_DT = 0.1
G6 = LidarParams(rate_hz=10.0, num_beams=1830, range_min=0.12, range_max=12.0,
                 noise_std=0.01)
FREE_SPACE_MARGIN = 0.30
LEADER_PARKED = (2.0, 4.2, 0.0)
FOLLOWER = FOLLOWER_LIDAR.name


def arena_region():
    b = settings().bounds
    return Region(b.xmin, b.xmax, b.ymin, b.ymax)


def make_world(follower_pose, boxes=(), leader_pose=LEADER_PARKED, seed=0):
    world = World(Arena(), seed=seed)
    world.add_robot('follower', Robot(RobotParams(radius=0.13), *follower_pose),
                    LocalizationParams(), G6)
    world.add_robot('leader', Robot(RobotParams(radius=0.18), *leader_pose),
                    LocalizationParams(), LidarParams())
    for box in boxes:
        world.add_obstacle(box)
    return world


class Feeder:
    """Feeds scans of the follower plus noisy poses into a mapper, 10 Hz."""

    def __init__(self, world, mapper=None, seed=1, heading_bias=0.0, noise=True):
        self.world = world
        self.mapper = mapper or ObstacleMapper(MapperParams(), arena_region())
        self.rng = np.random.default_rng(seed)
        self.heading_bias = heading_bias
        self.pos_std, self.yaw_std = (0.02, 0.02) if noise else (0.0, 0.0)
        self.t = 0.0
        self.leader_fresh = True

    def scans(self, count):
        """Run ``count`` scans; returns how many changed the published set."""
        changes = 0
        for _ in range(count):
            changes += self.step().changed
        return changes

    def step(self):
        self.t += SCAN_DT
        f, lead = self.world.robots['follower'], self.world.robots['leader']
        noise = self.rng.normal(0.0, [self.pos_std, self.pos_std, self.yaw_std])
        self.mapper.update_pose('follower', f.x + noise[0], f.y + noise[1],
                                f.theta + self.heading_bias + noise[2], self.t)
        self.mapper.update_status('follower', aligned=True, quality='GOOD')
        if self.leader_fresh:
            self.mapper.update_pose('leader', lead.x, lead.y, lead.theta, self.t)
        lidar = self.world.lidars['follower']
        scan = ScanData(self.world.scan('follower'), lidar.angle_min, lidar.angle_increment,
                        lidar.params.range_min, lidar.params.range_max)
        return self.mapper.process_scan(FOLLOWER, scan, self.t)


def footprint_points(box, n=11):
    c = box.corners()
    grid = np.linspace(0.0, 1.0, n)
    return [c[0] + a * (c[1] - c[0]) + b * (c[3] - c[0]) for a in grid for b in grid]


def worst_uncovered(circles, box):
    """Largest distance a footprint point lies outside every circle (<= 0: covered)."""
    return max(min(math.hypot(px - x, py - y) - r for x, y, r in circles)
               for px, py in footprint_points(box))


def free_space_reach(circles, boxes):
    """How far any circle reaches beyond the nearest box surface."""
    reach = 0.0
    for x, y, r in circles:
        for a in np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False):
            edge = (x + r * math.cos(a), y + r * math.sin(a))
            reach = max(reach, min(point_shape_distance(edge, b) for b in boxes))
    return reach


def covers(circles, box):
    return bool(circles) and worst_uncovered(circles, box) <= 0.0


def circles_near(circles, x, y, dist=0.5):
    return [c for c in circles if math.hypot(c[0] - x, c[1] - y) < dist]


# The planner consumes mapper.shapes() (exact rectangles grown by the inflation, tested
# through the world model's Obstacle.contains), not the display circles.
def shape_obstacles(shapes, slack=0.0):
    return [Obstacle.from_shape((s[0], s[1], s[2] + slack, *s[3:])) for s in shapes]


def shapes_cover(shapes, box, slack=0.0):
    """Every footprint point of ``box`` lies inside a published planner shape (grown by
    ``slack``: the planner's own clearance, where a test relies on it)."""
    obstacles = shape_obstacles(shapes, slack)
    return bool(obstacles) and all(any(o.contains(px, py) for o in obstacles)
                                   for px, py in footprint_points(box))


def shapes_near(shapes, x, y, dist=0.5):
    return [s for s in shapes if math.hypot(s[0] - x, s[1] - y) < dist]


def shapes_free_space_reach(shapes, boxes, around, half=1.5, step=0.05):
    """Largest distance from the boxes of any point inside a published shape."""
    obstacles = shape_obstacles(shapes)
    reach = 0.0
    for px in np.arange(around[0] - half, around[0] + half, step):
        for py in np.arange(around[1] - half, around[1] + half, step):
            if any(o.contains(px, py) for o in obstacles):
                reach = max(reach, min(point_shape_distance((px, py), b) for b in boxes))
    return reach


def test_two_face_box_covered_not_free_space():
    box = Rect(4.0, 3.3, 0.5, 0.4, yaw=0.5)
    feeder = Feeder(make_world((2.2, 2.6, 0.3), [box]))
    feeder.scans(10)
    circles = feeder.mapper.circles()
    assert covers(circles, box)
    assert free_space_reach(circles, [box]) <= FREE_SPACE_MARGIN
    shapes = feeder.mapper.shapes()
    assert shapes_cover(shapes, box)
    assert shapes_free_space_reach(shapes, [box], (box.x, box.y)) <= FREE_SPACE_MARGIN


def test_single_face_extended_to_box_depth():
    """Only the front face is visible; the box (0.3 deep) is still covered because
    the mapper assumes box_depth (0.5) behind a single face."""
    box = Rect(4.0, 3.3, 0.3, 0.5)
    feeder = Feeder(make_world((2.2, 3.3, 0.0), [box]))
    feeder.scans(10)
    circles = feeder.mapper.circles()
    assert covers(circles, box)
    assert all(x > 3.95 for x, _y, _r in circles)
    shapes = feeder.mapper.shapes()
    assert shapes_cover(shapes, box)
    assert all(shape[0] > 3.95 for shape in shapes)


def test_walls_and_the_leader_are_not_mapped():
    world = make_world((1.8, 2.5, 0.8), leader_pose=(3.0, 3.4, 0.0))
    feeder = Feeder(world)
    feeder.scans(20)
    assert feeder.mapper.circles() == []
    assert feeder.mapper.shapes() == []


def test_known_fixture_never_mapped():
    """A layout shelf 3.4 m from the follower standby: a fixture's returns within the mapping
    error (inflation) stay unmapped; a box beside it still is."""
    shelf = Box(5.386, 2.693, 0.40, 0.40)
    box = Rect(4.4, 3.2, 0.4, 0.4)
    world = make_world((1.82, 3.79, 0.0), [Rect(shelf.x, shelf.y, 0.40, 0.40), box],
                       leader_pose=(1.82, 3.29, 0.0))
    mapper = ObstacleMapper(MapperParams(), arena_region(), fixtures=[shelf])
    feeder = Feeder(world, mapper, seed=3)
    for _ in range(150):
        feeder.step()
        assert circles_near(mapper.circles(), shelf.x, shelf.y, 0.6) == []
        assert shapes_near(mapper.shapes(), shelf.x, shelf.y, 0.6) == []
    assert covers(mapper.circles(), box)
    assert shapes_cover(mapper.shapes(), box)


def test_no_hits_while_leader_pose_stale():
    box = Rect(4.0, 3.3, 0.4, 0.4)
    feeder = Feeder(make_world((2.2, 3.3, 0.0), [box], leader_pose=(3.0, 2.6, 0.0)))
    feeder.leader_fresh = False
    feeder.scans(10)
    assert feeder.mapper.circles() == []
    assert feeder.mapper.shapes() == []
    assert not feeder.mapper.layers[FOLLOWER].evidence.any()
    feeder.leader_fresh = True
    feeder.scans(10)
    assert covers(feeder.mapper.circles(), box)
    assert shapes_cover(feeder.mapper.shapes(), box)


def test_moved_box_leaves_old_appears_new():
    old, new = Rect(4.0, 3.3, 0.4, 0.4, name='box'), Rect(3.6, 2.6, 0.4, 0.4, name='box')
    world = make_world((2.2, 3.3, 0.0), [old])
    feeder = Feeder(world)
    feeder.scans(10)
    assert circles_near(feeder.mapper.circles(), old.x, old.y)
    assert shapes_near(feeder.mapper.shapes(), old.x, old.y)
    world.remove_obstacle('box')
    world.add_obstacle(new)
    feeder.scans(20)
    circles = feeder.mapper.circles()
    assert circles_near(circles, old.x, old.y, dist=0.3) == []
    assert covers(circles, new)
    assert shapes_near(feeder.mapper.shapes(), old.x, old.y, dist=0.3) == []
    assert shapes_cover(feeder.mapper.shapes(), new)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_static_noisy_scene_set_stable(seed):
    boxes = [Rect(4.0, 3.3, 0.5, 0.4, yaw=0.5), Rect(3.3, 4.0, 0.4, 0.4)]
    feeder = Feeder(make_world((2.2, 2.6, 0.3), boxes, seed=seed), seed=seed)
    # 40 scans to settle and at most two changes in the next 60, the boxes covered at every
    # scan: the yawed box's far side, seen at a ~7 deg grazing angle with 1 deg pose noise,
    # gathers evidence slowly and its width estimate still moves.
    feeder.scans(40)
    assert all(shapes_cover(feeder.mapper.shapes(), b) for b in boxes)
    changes = 0
    for _ in range(60):
        changes += feeder.step().changed
        assert all(shapes_cover(feeder.mapper.shapes(), b) for b in boxes)
    assert changes <= 2
    assert all(covers(feeder.mapper.circles(), b) for b in boxes)


@pytest.mark.parametrize('bias_deg', [2.0, -2.0])
def test_two_degree_error_at_3m_covered(bias_deg):
    """2 deg at 3 m shifts the box ~0.10 m; inflation (0.10) + the planner clearance
    keep the true footprint covered."""
    box = Rect(4.85, 3.3, 0.5, 0.5)            # front face 3.0 m ahead
    feeder = Feeder(make_world((1.6, 3.3, 0.0), [box]), heading_bias=math.radians(bias_deg))
    feeder.scans(15)
    # The planner's exact shapes do NOT cover the footprint on their own at +2 deg (~0.03 m of
    # the box edge lies outside the inflated rectangle); the planner's clearance is the spare
    # the docstring relies on. Display circles are not checked here.
    clearance = settings().robots['follower'].plan.clearance
    assert shapes_cover(feeder.mapper.shapes(), box, slack=clearance)


def test_shifted_wall_return_not_mapped():
    """A wall right at the arena edge, seen along its length with the stated 2 deg heading
    error: a return 3 m away lands ~0.10 m inside the wall. It lies within its mapping error,
    max(inflation, r * tan 2 deg), of the edge, so it is the wall, not a box."""
    b = settings().bounds
    world = World(Arena(b.xmin, b.xmax, b.ymin, b.ymax), seed=0)
    world.add_robot('follower', Robot(RobotParams(radius=0.13), b.xmax - 0.4, b.ymax - 0.25,
                                      math.pi), LocalizationParams(), G6)
    world.add_robot('leader', Robot(RobotParams(radius=0.18), 2.5, 2.6, 0.0),
                    LocalizationParams(), LidarParams())
    feeder = Feeder(world, heading_bias=math.radians(2.0), noise=False)
    assert feeder.scans(20) == 0
    assert feeder.mapper.circles() == []


def test_scans_resume_after_travel_post_turn():
    """A turn leaves a heading error (gyro scale x angle) above the stated accuracy until
    travel corrects it: the scans of a rover that turned fast and stands, or has moved less
    than heading_settle_dist since, are skipped (reason TURNED)."""
    world = make_world((2.2, 3.3, 0.0), [Rect(4.0, 3.3, 0.4, 0.4)])
    mapper = ObstacleMapper(MapperParams(heading_settle_dist=0.30), arena_region())
    mapper.update_pose('leader', *LEADER_PARKED, 1.0)
    mapper.update_status('follower', aligned=True, quality='GOOD')
    lidar = world.lidars['follower']
    scan = ScanData(world.scan('follower'), 0.0, lidar.angle_increment)

    def verdict(x, yaw, now):
        mapper.update_pose('leader', *LEADER_PARKED, now)
        mapper.update_pose('follower', x, 3.3, yaw, now)
        return mapper.process_scan('follower_lidar', scan, now).reason

    assert verdict(2.2, 0.0, 1.0) == ''
    assert verdict(2.2, 0.5, 1.2) == TURNING                # 2.5 rad/s
    assert verdict(2.2, 0.5, 2.0) == TURNED
    assert verdict(2.4, 0.5, 3.0) == TURNED                 # 0.20 m since
    assert verdict(2.55, 0.5, 4.0) == ''                    # 0.35 m since: counts again
    assert verdict(2.56, 0.5, 5.0) == ''
    off = ObstacleMapper(MapperParams(), arena_region())    # default 0: as before
    off.update_status('follower', aligned=True, quality='GOOD')
    off.update_pose('leader', *LEADER_PARKED, 1.2)
    off.update_pose('follower', 2.2, 3.3, 0.0, 1.0)
    off.update_pose('follower', 2.2, 3.3, 0.5, 1.2)
    assert off.process_scan('follower_lidar', scan, 1.2).reason == TURNING
    off.update_pose('leader', *LEADER_PARKED, 2.0)
    off.update_pose('follower', 2.2, 3.3, 0.5, 2.0)
    assert off.process_scan('follower_lidar', scan, 2.0).reason == ''


def test_absolute_heading_settles_by_standing():
    """After the alignment the follower turns east at home and barely drives; with the travel
    rule alone the boxes ahead stayed unmapped. Its heading is held by the camera box axis, so
    standing settle_still_s after the turn counts as settled too. Driving still settles it as
    before; a source without the setting keeps the travel rule."""
    import dataclasses
    world = make_world((2.2, 3.3, 0.0), [Rect(4.0, 3.3, 0.4, 0.4)])
    src = dataclasses.replace(FOLLOWER_LIDAR, settle_still_s=2.0)
    mapper = ObstacleMapper(MapperParams(heading_settle_dist=0.30), arena_region(), sources=(src,))
    mapper.update_status('follower', aligned=True, quality='GOOD')
    lidar = world.lidars['follower']
    scan = ScanData(world.scan('follower'), 0.0, lidar.angle_increment)

    def verdict(yaw, now):
        mapper.update_pose('leader', *LEADER_PARKED, now)
        mapper.update_pose('follower', 2.2, 3.3, yaw, now)
        return mapper.process_scan('follower_lidar', scan, now).reason

    assert verdict(0.0, 1.0) == ''
    assert verdict(0.5, 1.2) == TURNING
    assert verdict(0.5, 2.0) == TURNED                      # 0.8 s standing
    assert verdict(0.5, 2.8) == TURNED                      # 1.6 s
    assert verdict(0.5, 3.4) == ''                          # 2.2 s standing: settled


@pytest.mark.parametrize('case, reason', [('stale', STALE_POSE), ('quality', BAD_QUALITY),
                                          ('aligned', NOT_ALIGNED), ('turning', TURNING)])
def test_scans_skipped_unusable_follower_pose(case, reason):
    world = make_world((2.2, 3.3, 0.0), [Rect(4.0, 3.3, 0.4, 0.4)])
    mapper = ObstacleMapper(MapperParams(), arena_region())
    mapper.update_pose('leader', *LEADER_PARKED, 1.0)
    mapper.update_status('follower', aligned=case != 'aligned',
                         quality='DEGRADED' if case == 'quality' else 'GOOD')
    mapper.update_pose('follower', 2.2, 3.3, 0.0, 0.8)
    mapper.update_pose('follower', 2.2, 3.3, 0.5 if case == 'turning' else 0.0, 1.0)
    lidar = world.lidars['follower']
    scan = ScanData(world.scan('follower'), 0.0, lidar.angle_increment)
    now = 1.5 if case == 'stale' else 1.0
    verdict = mapper.process_scan(FOLLOWER, scan, now)
    assert (verdict.used, verdict.reason) == (False, reason)
    assert not mapper.layers[FOLLOWER].evidence.any()


ASTRA = LidarParams(rate_hz=10.0, num_beams=80, fov=math.radians(58.4), range_min=0.6,
                    range_max=5.0, out_of_range_value=0.0, noise_std=0.01)
DEPTH = ScanSource(name='leader_depth', robot='leader', min_range=0.6)


def two_source_world(leader_pose, follower_pose, boxes=()):
    world = World(Arena(), seed=0)
    world.add_robot('follower', Robot(RobotParams(radius=0.13), *follower_pose),
                    LocalizationParams(), G6)
    world.add_robot('leader', Robot(RobotParams(radius=0.18), *leader_pose),
                    LocalizationParams(), LidarParams(), ASTRA)
    for box in boxes:
        world.add_obstacle(box)
    return world


class TwoFeeder:
    """Both rovers' true poses + the follower LiDAR and the leader depth scan, 10 Hz."""

    def __init__(self, world, stale=(), heading_bias=None):
        self.world, self.t, self.stale = world, 0.0, set(stale)
        self.bias = dict(heading_bias or {})       # rover -> error added to its reported yaw
        self.mapper = ObstacleMapper(MapperParams(), arena_region(),
                                     sources=(FOLLOWER_LIDAR, DEPTH))

    def step(self, sources=(FOLLOWER, DEPTH.name)):
        self.t += SCAN_DT
        for name, robot in self.world.robots.items():
            if name not in self.stale:
                self.mapper.update_pose(name, robot.x, robot.y,
                                        robot.theta + self.bias.get(name, 0.0), self.t)
            self.mapper.update_status(name, aligned=True, quality='GOOD')
        verdicts = []
        for source in sources:
            lidar = (self.world.lidars['follower'] if source == FOLLOWER
                     else self.world.depth_scans['leader'])
            ranges = (self.world.scan('follower') if source == FOLLOWER
                      else self.world.depth_scan('leader'))
            scan = ScanData(ranges, lidar.angle_min, lidar.angle_increment,
                            lidar.params.range_min, lidar.params.range_max)
            verdicts.append(self.mapper.process_scan(source, scan, self.t))
        return verdicts

    def scans(self, count, **kw):
        for _ in range(count):
            self.step(**kw)


def test_leader_depth_maps_box_occluded_for_follower():
    """sim_box1 layout: follower right behind the leader, box ahead of the leader."""
    box = Rect(3.75, 3.15, 0.4, 0.4)
    feeder = TwoFeeder(two_source_world((2.6, 3.3, 0.0), (2.1, 3.3, 0.0), [box]))
    feeder.scans(5)
    assert not feeder.mapper.layers[FOLLOWER].evidence.any()
    assert covers(feeder.mapper.circles(), box)
    assert shapes_cover(feeder.mapper.shapes(), box)
    assert feeder.mapper.layers[DEPTH.name].evidence.any()


def test_box_survives_other_sources_rays():
    """A box under the LiDAR plane: the camera sees it, the LiDAR beams pass over it.
    The LiDAR's rays clear only its own layer, so the box stays mapped; only the
    camera itself looking at the empty place removes it."""
    box = Rect(4.0, 3.3, 0.4, 0.4, name='low_box')
    world = two_source_world((2.6, 3.3, 0.0), (2.6, 2.5, 0.6), [box])
    feeder = TwoFeeder(world)
    feeder.scans(5, sources=(DEPTH.name,))
    assert covers(feeder.mapper.circles(), box)
    assert shapes_cover(feeder.mapper.shapes(), box)
    world.remove_obstacle('low_box')            # = invisible at the LiDAR's height
    feeder.scans(20, sources=(FOLLOWER,))
    assert covers(feeder.mapper.circles(), box)
    assert shapes_cover(feeder.mapper.shapes(), box)
    feeder.scans(20, sources=(DEPTH.name,))     # the camera sees the place empty
    assert circles_near(feeder.mapper.circles(), box.x, box.y) == []
    assert shapes_near(feeder.mapper.shapes(), box.x, box.y) == []


def test_source_excludes_peer_needs_fresh_pose():
    # follower 1.2 m ahead of the leader's camera; leader 1.0 m north of the follower
    world = two_source_world((2.6, 3.3, 0.0), (3.8, 3.3, 1.57))
    feeder = TwoFeeder(world)
    feeder.scans(20)
    assert feeder.mapper.circles() == []
    assert feeder.mapper.shapes() == []
    assert feeder.mapper.layers[DEPTH.name].point_count.sum() == 0
    world.add_obstacle(Rect(3.8, 4.2, 0.3, 0.3))
    stale = TwoFeeder(world, stale=('follower',))
    stale.mapper.update_pose('follower', 3.8, 3.3, 1.57, 0.0)
    verdicts = [v for _ in range(10) for v in stale.step()]
    assert not stale.mapper.layers[DEPTH.name].evidence.any()
    assert {v.reason for v in verdicts} == {'', 'STALE_POSE'}
    assert stale.mapper.circles() == []
    assert stale.mapper.shapes() == []


# The leader's fused heading can be 6-8 deg off after a 180 deg pivot and its depth camera
# then sees the follower 1.4 m away. Turned by that error about the camera, the
# follower's returns land ~0.28 m from its pose, outside its 0.25 m exclusion radius:
# mapped as a box on its own path.
LEADER_HEADING_ERROR = math.radians(8.0)  # the leader's stated heading accuracy


@pytest.mark.parametrize('distance', [1.4, 2.0])
@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_peer_with_heading_error_not_mapped(distance, sign):
    """The follower (radius 0.13) straight ahead of the leader's camera, the leader's
    reported heading off by its stated accuracy: no part of the follower is mapped."""
    world = two_source_world((2.0, 3.3, 0.0), (2.0 + distance, 3.3, math.pi / 2.0))
    feeder = TwoFeeder(world, heading_bias={'leader': sign * LEADER_HEADING_ERROR})
    feeder.scans(20, sources=(DEPTH.name,))
    assert feeder.mapper.layers[DEPTH.name].point_count.sum() == 0
    assert feeder.mapper.shapes() == []


def test_box_in_front_of_peer_mapped():
    """Cost check of the heading allowance: it widens the rover's BEARING extent only. A
    box 0.45 m in front of the follower (same bearing) is mapped; a radius grown by
    range x heading error (2.0 m x tan 8 deg = 0.28 m, 0.53 m in all) would have hidden it."""
    box = Rect(3.70, 3.3, 0.30, 0.30)     # near face 1.55 m from the camera
    world = two_source_world((2.0, 3.3, 0.0), (4.0, 3.55, 0.0), [box])
    feeder = TwoFeeder(world)
    feeder.scans(10, sources=(DEPTH.name,))
    assert shapes_cover(feeder.mapper.shapes(), box)


def test_box_beside_peer_mapped():
    """A box 0.7 m beside the follower, both 1.4 m from the leader's camera: its near edge
    lies 21 deg off the follower's bearing, outside the follower's 10 deg half-extent + the
    leader's 8 deg heading accuracy. It is mapped while the follower stands there."""
    box = Rect(3.4, 4.0, 0.30, 0.30)
    world = two_source_world((2.0, 3.3, 0.0), (3.4, 3.3, 0.0), [box])
    feeder = TwoFeeder(world)
    feeder.scans(10, sources=(DEPTH.name,))
    assert shapes_cover(feeder.mapper.shapes(), box)


def rect_reach_outside(shapes, box):
    """How far any published RECTANGLE corner (pad not included) lies outside the box."""
    reach = 0.0
    for x, y, _pad, length, width, yaw in shapes:
        u = np.array([math.cos(yaw), math.sin(yaw)])
        v = np.array([-u[1], u[0]])
        for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            corner = np.array([x, y]) + a * length / 2.0 * u + b * width / 2.0 * v
            reach = max(reach, point_shape_distance(tuple(corner), box))
    return reach


@pytest.mark.parametrize('error_deg', [7.0, 8.0])
def test_mapped_face_stays_under_larger_error(error_deg):
    """The follower's LiDAR has mapped the box from close by (error 0.10 m). Later the leader's
    camera sees its north face from 1.5 m with its heading 7-8 deg off (within its stated 8
    deg): the face lands 0.15-0.2 m north of the box. Those returns are less precise than the
    mapped face and within the two errors of it: they are not evidence, the box stays where
    the precise look put it."""
    box = Rect(3.2, 3.0, 0.50, 0.50)
    world = two_source_world((4.4, 4.1, math.atan2(-0.9, -1.2)), (2.6, 3.7, 0.0), [box])
    feeder = TwoFeeder(world, heading_bias={'leader': -math.radians(error_deg)})
    feeder.scans(10, sources=(FOLLOWER,))
    assert shapes_cover(feeder.mapper.shapes(), box)
    assert rect_reach_outside(feeder.mapper.shapes(), box) <= 0.10
    world.robots['follower'].reset_pose(2.0, 2.5, 0.0)
    feeder.scans(30, sources=(DEPTH.name,))
    assert feeder.mapper.layers[DEPTH.name].point_count.sum() > 0
    assert len(feeder.mapper.shapes()) == 1
    assert shapes_cover(feeder.mapper.shapes(), box)
    assert rect_reach_outside(feeder.mapper.shapes(), box) <= 0.10


def test_far_mapped_box_corrected_by_closer_look():
    """Cost check of that rule: it holds back returns LESS precise than the map only. A
    box first mapped by the leader's camera from 3 m with a 6 deg heading error (0.3 m off)
    is moved onto the box once the same camera looks from 1 m with the heading right."""
    box = Rect(3.2, 3.0, 0.40, 0.40)
    world = two_source_world((5.6, 3.0, math.pi), (2.0, 2.5, 0.0), [box])
    feeder = TwoFeeder(world, heading_bias={'leader': math.radians(6.0)})
    feeder.scans(10, sources=(DEPTH.name,))
    assert feeder.mapper.shapes() and not shapes_cover(feeder.mapper.shapes(), box)
    world.robots['leader'].reset_pose(4.4, 3.0, math.pi)
    feeder.bias = {}
    feeder.scans(40, sources=(DEPTH.name,))
    assert shapes_cover(feeder.mapper.shapes(), box, slack=0.03)


# These tests write evidence straight into one layer and read the rectangle the mapper
# extracts, so the rule under test is isolated from ray casting. The box is 0.40 x 0.40
# with its south-west corner at (3.0, 3.0).
BOX = Rect(3.2, 3.2, 0.40, 0.40)


def seeded_mapper():
    mapper = ObstacleMapper(MapperParams(), arena_region())
    return mapper, mapper.layers[FOLLOWER]


def seed_cells(layer, points, view, hits=10):
    """Occupied cells whose mean hit point is each of ``points``, seen along ``view``."""
    for point, count in points:
        idx, inside = layer.flat_index(np.array([point], dtype=float))
        assert inside[0]
        i = idx[0]
        layer.evidence[i] = layer.p.max_evidence
        layer.point_sum[i] += np.array(point) * count
        layer.point_count[i] += count
        layer.view[i] += np.array(view, dtype=float) * count


def face(x0, y0, x1, y1, step=0.05, hits=10):
    n = int(round(math.hypot(x1 - x0, y1 - y0) / step)) + 1
    return [((x0 + (x1 - x0) * k / (n - 1), y0 + (y1 - y0) * k / (n - 1)), hits)
            for k in range(n)]


def extracted(mapper):
    rects = mapper._extract_rects()
    assert len(rects) == 1, rects
    r = rects[0]
    return (r.cx, r.cy, mapper.p.inflation, 2.0 * r.half_len, 2.0 * r.half_wid, r.angle)


def test_partial_second_face_completed_to_depth():
    """The west face is whole, the south face only 0.25 of 0.40 (the camera's narrow view). The
    box is assumed to go on to box_depth away from the viewer: covered."""
    mapper, layer = seeded_mapper()
    seed_cells(layer, face(3.0, 3.0, 3.0, 3.4) + face(3.05, 3.0, 3.25, 3.0), view=(1.0, 1.0))
    assert shapes_cover([extracted(mapper)], BOX)


def test_assumed_depth_stops_at_seen_free_space():
    """A box only 0.25 deep (x 3.0-3.25), its west face whole and its south face seen over
    0.15 m; a sensor has seen the space beyond its east face (x >= 3.27) free. The assumed
    depth stops there (within one sampling cell), short of box_depth (x 3.5)."""
    box = Rect(3.125, 3.2, 0.25, 0.40)
    mapper, layer = seeded_mapper()
    seed_cells(layer, face(3.0, 3.0, 3.0, 3.4) + face(3.05, 3.0, 3.15, 3.0), view=(1.0, 1.0))
    for x in np.arange(3.27, 3.9, 0.05):
        for y in np.arange(2.7, 3.8, 0.05):
            idx, _ = layer.flat_index(np.array([[x, y]]))
            layer.seen_free[idx[0]] = True
    shape = extracted(mapper)
    assert shapes_cover([shape], box)
    x, y, _pad, length, width, yaw = shape
    east = x + abs(math.cos(yaw)) * length / 2.0 + abs(math.sin(yaw)) * width / 2.0
    assert east <= 3.27 + 0.05              # free edge + one sampling cell


def test_rectangle_sides_on_faces_not_smear_edge():
    """Each face's hits smear over ~0.1 m from the scanning rovers' pose error. The sides lie
    on the hit-weighted face (within 0.03 m); the inflation alone covers the error."""
    mapper, layer = seeded_mapper()
    ring = (face(3.0, 3.0, 3.0, 3.4) + face(3.0, 3.4, 3.4, 3.4) + face(3.4, 3.4, 3.4, 3.0)
            + face(3.4, 3.0, 3.0, 3.0))
    smear = [((x, y + 0.06), 2) for (x, y), _ in face(3.05, 3.4, 3.35, 3.4)]   # north, outside
    seed_cells(layer, ring + smear, view=(1.0, -1.0))
    x, y, _pad, length, width, yaw = extracted(mapper)
    north = y + (width if abs(math.sin(yaw)) < 0.5 else length) / 2.0
    assert abs(north - 3.4) <= 0.03
    assert shapes_cover([(x, y, mapper.p.inflation, length, width, yaw)], BOX)


def test_grazing_ray_does_not_erode_face():
    """A face (row of cells at y 3.0, x 3.0-3.4) left by earlier scans. Each new scan hits it
    every 0.1 m, and one beam runs through the face between two returns and on behind it.
    Every cell within the returns' mapping error is shielded, so the face does not fall apart."""
    mapper, layer = seeded_mapper()
    face_cells = face(3.0, 3.0, 3.4, 3.0)
    seed_cells(layer, face_cells, view=(0.0, 1.0))
    sensor = np.array([2.0, 2.6])
    inc = 0.0005
    ranges = np.zeros(int(round(1.2 / inc)))
    targets = [((x, 3.0), None) for x in (3.0, 3.1, 3.2, 3.3, 3.4)] + [((3.265, 3.0), 3.5)]
    for (tx, ty), far in targets:
        d = np.array([tx, ty]) - sensor
        k = int(round(math.atan2(d[1], d[0]) / inc))
        ranges[k] = far or float(np.hypot(*d))
    scan = ScanData(ranges, 0.0, inc, 0.15, 12.0)
    for t in np.arange(0.1, 1.0, 0.1):
        mapper.update_pose('leader', 5.0, 4.2, 0.0, t)
        mapper.update_status('follower', aligned=True, quality='GOOD')
        mapper.update_pose('follower', sensor[0], sensor[1], 0.0, t)
        mapper.process_scan(FOLLOWER, scan, t)
    for (px, py), _ in face_cells:
        idx, _ = layer.flat_index(np.array([[px, py]]))
        assert layer.evidence[idx[0]] >= mapper.p.occupied_threshold, (px, py)


def test_mapped_obstacles_replaced_as_whole():
    eng = engine()
    world = eng.world
    eng.set_obstacles([(2.0, 3.0, 0.1)])
    version = world.obstacle_version
    eng.set_mapped_obstacles([(4.0, 3.3, 0.4), (3.3, 4.0, 0.3)])
    assert world.obstacle_version == version + 1
    eng.set_mapped_obstacles([(4.0, 3.3, 0.4), (3.3, 4.0, 0.3)])
    assert world.obstacle_version == version + 1
    eng.set_mapped_obstacles([(3.6, 2.6, 0.4)])
    assert [(o.x, o.y, o.source) for o in world.obstacles()] == [
        (2.0, 3.0, 'manual'), (3.6, 2.6, 'mapped')]


def _scan_once(grid, hit):
    """One scan from the origin side: one return at ``hit`` (or none), no ray clearing."""
    from host_mission.obstacle_mapper import EvidenceGrid  # noqa: F401 (type only)
    xy = np.zeros((0, 2)) if hit is None else np.asarray([hit], dtype=float)
    dirs = np.zeros((len(xy), 2))
    grid.integrate(np.array([0.0, 0.0]), xy, dirs, np.ones(len(xy), dtype=bool),
                   np.zeros((0, 2)), np.zeros(0))


@pytest.mark.parametrize('decay, expect_free', [(0.0, False), (0.06, True)])
def test_unreturned_cell_expires_only_with_decay(decay, expect_free):
    from host_mission.obstacle_mapper import EvidenceGrid, MapperParams, Region
    grid = EvidenceGrid(Region(0.0, 2.0, 0.0, 2.0), MapperParams(decay_per_scan=decay))
    for _ in range(6):
        _scan_once(grid, (1.0, 1.0))
    assert grid.occupied_mask().any()
    for _ in range(90):                       # ~9 s at 10 Hz, no ray passes through the cell
        _scan_once(grid, None)
    assert (not grid.occupied_mask().any()) == expect_free


def test_intermittent_box_stays_with_decay():
    from host_mission.obstacle_mapper import EvidenceGrid, MapperParams, Region
    grid = EvidenceGrid(Region(0.0, 2.0, 0.0, 2.0), MapperParams(decay_per_scan=0.06))
    for i in range(200):                      # hit in one scan of every 8 (12 %)
        _scan_once(grid, (1.0, 1.0) if i % 8 == 0 else None)
    assert grid.occupied_mask().any()


def _weights(speed, turn, ranges, **kw):
    from host_mission.obstacle_mapper import MapperParams, ObstacleMapper
    from collections import deque
    mapper = ObstacleMapper.__new__(ObstacleMapper)
    mapper.p = MapperParams(weight_ref_error=0.025, **kw)
    class _S:
        pass
    st = _S()
    st.pos_history = deque([(0.0, 0.0, 0.0), (1.0, speed, 0.0)])
    st.yaw_history = deque([(0.0, 0.0), (1.0, turn)])
    mapper.robots = {'follower': st}
    return mapper.motion_weights('follower', ranges)


def test_standing_counts_close_returns_fully():
    stand, stand_miss = _weights(0.0, 0.0, [0.5, 1.0, 2.0])
    move, move_miss = _weights(0.12, 0.0, [0.5, 1.0, 2.0])
    turn, _ = _weights(0.0, 0.3, [2.0])
    assert stand[0] == 1.0 and stand[1] > move[1] and stand[2] > move[2]
    assert move_miss < stand_miss and turn[0] < move[2]
    assert np.all(np.diff(stand) <= 0.0)          # farther = less sure


def test_weighting_off_keeps_full_increments():
    from host_mission.obstacle_mapper import MapperParams, ObstacleMapper
    mapper = ObstacleMapper.__new__(ObstacleMapper)
    mapper.p = MapperParams()
    assert mapper.motion_weights('follower', [1.0]) == (None, 1.0)
