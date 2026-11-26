"""Localization model (rate, delay, noise, alignment) and LiDAR ray casting."""
import math

import numpy as np
import pytest
from conftest import instant_params, make_world

from host_sim.core import Circle, LidarParams, LocalizationParams, Rect

DT = 0.001


def collect(world, duration, command=None, dt=DT):
    """Step the world, returning [(emit time, measurement)] for 'leader'."""
    samples = []
    for _ in range(int(round(duration / dt))):
        if command is not None:
            world.set_command('leader', *command)
        world.step(dt)
        sample = world.poll_pose('leader')
        if sample is not None:
            samples.append((world.t, sample))
    return samples


def test_output_rate_is_30_hz(world):
    samples = collect(world, 10.0)
    assert len(samples) == pytest.approx(300, abs=4)
    gaps = np.diff([t for t, _ in samples])
    assert gaps.mean() == pytest.approx(1.0 / 30.0, abs=1e-3)


def test_measurement_lags_truth_by_the_delay(world):
    collect(world, 2.0, command=(0.2, 0.0))
    sample = None
    while sample is None:  # stop exactly at the next emission
        world.set_command('leader', 0.2, 0.0)
        world.step(DT)
        sample = world.poll_pose('leader')
    assert world.t - sample.stamp == pytest.approx(0.08, abs=2 * DT)
    true_x = world.robots['leader'].x
    assert true_x - sample.x == pytest.approx(0.2 * 0.08, abs=0.2 * 2 * DT)


def test_no_output_before_delay(world):
    samples = collect(world, 0.07)
    assert samples == []


def test_position_noise_statistics():
    loc = LocalizationParams(pos_noise_std=0.02, heading_noise_std=0.03, heading_bias=0.1)
    world = make_world(seed=3, loc=loc, start=(3.0, 3.0, 0.5))
    samples = [s for _, s in collect(world, 100.0, dt=0.01)]
    xs, ys = np.array([s.x for s in samples]), np.array([s.y for s in samples])
    thetas = np.array([s.theta for s in samples])
    assert len(samples) > 2500
    assert xs.mean() == pytest.approx(3.0, abs=0.002)
    assert ys.mean() == pytest.approx(3.0, abs=0.002)
    assert xs.std() == pytest.approx(0.02, rel=0.08)
    assert ys.std() == pytest.approx(0.02, rel=0.08)
    assert thetas.mean() == pytest.approx(0.6, abs=0.004)  # truth 0.5 + bias 0.1
    assert thetas.std() == pytest.approx(0.03, rel=0.08)


def test_slow_bias_time_correlated():
    """Optional Gauss-Markov term: large spread, tiny sample-to-sample change."""
    loc = LocalizationParams(pos_noise_std=0.0, heading_noise_std=0.0, pos_bias_std=0.02,
                             pos_bias_tau=2.0, heading_drift_std=0.03, heading_drift_tau=2.0)
    world = make_world(seed=5, loc=loc)
    samples = [s for _, s in collect(world, 400.0, dt=0.01)]
    xs, thetas = np.array([s.x for s in samples]), np.array([s.theta for s in samples])
    assert xs.std() == pytest.approx(0.02, rel=0.25)
    assert thetas.std() == pytest.approx(0.03, rel=0.25)
    assert np.corrcoef(xs[:-1], xs[1:])[0, 1] > 0.95
    assert np.abs(np.diff(xs)).mean() < 0.005


def test_same_seed_reproduces_the_noise():
    loc = LocalizationParams()
    first = [s.x for _, s in collect(make_world(seed=7, loc=loc), 1.0)]
    second = [s.x for _, s in collect(make_world(seed=7, loc=loc), 1.0)]
    assert first == second


def test_degraded_quality_uses_larger_noise():
    loc = LocalizationParams(pos_noise_std=0.02, degraded_pos_noise_std=0.08,
                             quality='DEGRADED')
    world = make_world(seed=1, loc=loc)
    samples = [s for _, s in collect(world, 60.0, dt=0.01)]
    assert samples[0].quality == 'DEGRADED'
    assert np.std([s.x for s in samples]) == pytest.approx(0.08, rel=0.1)


def test_immediate_alignment_reports_heading():
    world = make_world(start=(3.0, 3.0, 1.0))
    sample = collect(world, 0.5)[-1][1]
    assert sample.heading_aligned
    assert sample.theta == pytest.approx(1.0)


def test_alignment_needs_straight_driving():
    loc = LocalizationParams(pos_noise_std=0.0, heading_noise_std=0.0,
                             align_mode='after_distance', align_distance=0.3)
    world = make_world(loc=loc, start=(2.0, 3.0, 1.0))
    sample = collect(world, 0.5)[-1][1]
    assert not sample.heading_aligned and sample.theta == 0.0
    collect(world, 1.0, command=(0.2, 0.0))  # 0.2 m: not enough yet
    assert world.pose_status('leader')[0] is False
    sample = collect(world, 0.8, command=(0.2, 0.0))[-1][1]  # 0.36 m in total
    assert sample.heading_aligned
    assert sample.theta == pytest.approx(1.0)


def test_turning_restarts_alignment_distance():
    loc = LocalizationParams(align_mode='after_distance', align_distance=0.3)
    world = make_world(loc=loc, start=(2.0, 3.0, 0.0))
    collect(world, 1.0, command=(0.2, 0.0))
    collect(world, 0.2, command=(0.2, 1.0))  # turn faster than align_max_omega
    collect(world, 1.0, command=(0.2, 0.0))
    assert world.pose_status('leader')[0] is False
    collect(world, 0.6, command=(0.2, 0.0))
    assert world.pose_status('leader')[0] is True


def test_measurement_reports_tracked_point():
    world = make_world(base=instant_params(ref_offset_x=0.1), start=(3.0, 3.0, math.pi / 2))
    sample = collect(world, 0.5)[-1][1]
    assert (sample.x, sample.y) == pytest.approx((3.0, 3.1))


# ---- LiDAR ---------------------------------------------------------------

def test_scan_360_beams_hits_walls(world):
    ranges = world.scan('leader')  # rover at (3, 3), heading +x
    arena = world.arena
    assert ranges.shape == (360,)
    assert ranges[0] == pytest.approx(arena.x_max - 3.0)
    assert ranges[90] == pytest.approx(arena.y_max - 3.0)
    assert ranges[180] == pytest.approx(3.0 - arena.x_min)
    assert ranges[270] == pytest.approx(3.0 - arena.y_min)
    assert ranges[45] == pytest.approx((arena.y_max - 3.0) * math.sqrt(2.0))


def test_scan_rotates_with_the_rover():
    world = make_world(start=(3.0, 3.0, math.pi / 2.0))
    ranges = world.scan('leader')
    assert ranges[0] == pytest.approx(world.arena.y_max - 3.0)
    assert ranges[90] == pytest.approx(3.0 - world.arena.x_min)


def test_scan_without_walls_sees_nothing():
    world = make_world(walls=False)
    assert np.all(np.isinf(world.scan('leader')))


def test_circle_obstacle_distance():
    world = make_world(walls=False)
    world.add_obstacle(Circle(4.0, 3.0, 0.2))
    ranges = world.scan('leader')
    assert ranges[0] == pytest.approx(0.8)
    half_width = math.degrees(math.asin(0.2 / 1.0))  # 11.5 deg
    assert np.isfinite(ranges[11]) and np.isinf(ranges[12])
    assert np.isfinite(ranges[360 - 11]) and np.isinf(ranges[360 - 12])
    assert half_width == pytest.approx(11.54, abs=0.01)


def test_rect_obstacle_distance_rotated():
    world = make_world(walls=False)
    world.add_obstacle(Rect(3.0, 4.0, 0.4, 0.2))
    assert world.scan('leader')[90] == pytest.approx(0.9)
    world.clear_obstacles()
    world.add_obstacle(Rect(2.0, 3.0, 0.2, 0.2, yaw=math.pi / 4.0))
    assert world.scan('leader')[180] == pytest.approx(1.0 - 0.1 * math.sqrt(2.0))


def test_other_rover_is_seen_as_a_circle():
    from host_sim.core import Robot
    world = make_world(walls=False)
    world.add_robot('follower', Robot(instant_params(radius=0.13), 3.0, 2.0, 0.0),
                    lidar=LidarParams(noise_std=0.0))
    assert world.scan('leader')[270] == pytest.approx(1.0 - 0.13)
    assert world.scan('follower')[90] == pytest.approx(1.0 - 0.15)


def test_range_limits():
    lidar = LidarParams(noise_std=0.0, range_min=0.12, range_max=1.5)
    world = make_world(walls=False, lidar=lidar)
    world.add_obstacle(Circle(5.0, 3.0, 0.2, name='far'))
    world.add_obstacle(Circle(3.0, 3.25, 0.2, name='near'))
    ranges = world.scan('leader')
    assert np.isinf(ranges[0])   # 1.8 m > range_max
    assert ranges[90] == 0.0     # 0.05 m < range_min -> invalid value


def test_blind_sector_masks_with_wraparound():
    lidar = LidarParams(noise_std=0.0, blind_sectors=[[170.0, 190.0], [350.0, 5.0]])
    ranges = make_world(lidar=lidar).scan('leader')
    assert np.all(ranges[170:190] == 0.0)
    assert ranges[169] > 0.0 and ranges[190] > 0.0
    assert np.all(ranges[350:] == 0.0) and np.all(ranges[:5] == 0.0)
    assert ranges[5] > 0.0 and ranges[349] > 0.0


def test_lidar_mount_offset_moves_origin():
    lidar = LidarParams(noise_std=0.0, mount_x=0.1)
    world = make_world(lidar=lidar)
    assert world.scan('leader')[0] == pytest.approx(world.arena.x_max - 3.1)


def test_lidar_noise_statistics():
    world = make_world(seed=5, lidar=LidarParams(noise_std=0.01))
    front = np.array([world.scan('leader')[0] for _ in range(2000)])
    assert front.mean() == pytest.approx(world.arena.x_max - 3.0, abs=0.001)
    assert front.std() == pytest.approx(0.01, rel=0.08)


def test_camera_scanner_covers_only_fov():
    """fov < 360 deg: beams centred on +x; beyond range -> out_of_range_value."""
    from host_sim.core import Robot
    camera = LidarParams(noise_std=0.0, num_beams=80, fov=math.radians(60.0), range_min=0.6,
                         range_max=5.0, out_of_range_value=0.0, rate_hz=10.0)
    world = make_world(walls=False)
    world.add_robot('cam', Robot(instant_params(), 1.0, 1.0, 0.0), depth_scan=camera)
    world.add_obstacle(Circle(3.0, 1.0, 0.2, name='ahead'))
    world.add_obstacle(Circle(1.0, 2.0, 0.2, name='left'))      # 90 deg: outside the FOV
    lidar = world.depth_scans['cam']
    assert lidar.angle_min == pytest.approx(-math.radians(30.0) + lidar.angle_increment / 2)
    assert lidar.angle_increment * 80 == pytest.approx(math.radians(60.0))
    ranges = world.depth_scan('cam')
    assert len(ranges) == 80
    assert ranges[40] == pytest.approx(1.8, abs=0.01)
    seen = np.flatnonzero(ranges)                               # only the circle ahead
    assert len(seen) <= 17 and seen.min() >= 31 and seen.max() <= 48   # (+-5.7 deg)
    assert world.depth_scan_due('cam') and not world.depth_scan_due('leader')
    assert world.scan('cam').shape == (360,)                    # its LiDAR is unchanged


def test_lidar_rate(world):
    scans = 0
    for _ in range(2000):
        world.step(0.001)
        scans += world.lidar_due('leader')
    assert scans == pytest.approx(10, abs=1)  # 5 Hz for 2 s


# ---- vertical band of a depth camera: low obstacles (pipes on the floor) ----
BAND = dict(sensor_height=0.12, band_lower_angle=math.atan(0.0419))   # sim_default leader
NO_SEGMENTS = np.zeros((0, 2, 2))


def _forward_range(params, low):
    """Range of one beam along +x from the origin with only the ``low`` obstacles."""
    from host_sim.core import Lidar
    beam = Lidar(LidarParams(noise_std=0.0, num_beams=1, fov=0.01, range_max=5.0, **params))
    return float(beam.scan((0.0, 0.0, 0.0), [], NO_SEGMENTS, low)[0])


def _pipe_at(x, top):
    return [(np.array([[[x, -1.0], [x, 1.0]]]), top)]


@pytest.mark.parametrize('distance, top, seen', [
    (2.0, 0.05, True),     # the lower edge reaches 0.05 m from (0.12 - 0.05) / 0.0419 = 1.67 m
    (0.6, 0.05, False),    # nearer: it passes over the pipe
    (0.6, 0.10, True),     # 0.10 m is reached from 0.48 m on
    (0.6, 0.12, True),     # as high as the camera: always
])
def test_band_sees_pipe_at_lower_edge(distance, top, seen):
    result = _forward_range(BAND, _pipe_at(distance, top))
    assert result == (pytest.approx(distance) if seen else math.inf)


def test_band_sees_past_low_pipe():
    assert _forward_range(BAND, _pipe_at(0.6, 0.05) + _pipe_at(3.0, 1.0)) == pytest.approx(3.0)


def test_band_hits_low_solid_top_when_near():
    box = [(Rect(1.75, 0.0, 1.5, 0.4, top=0.05), 0.05)]           # x from 1.0 to 2.5
    assert _forward_range(BAND, box) == pytest.approx(0.07 / 0.0419, rel=1e-3)
    assert _forward_range(BAND, [(Circle(1.2, 0.0, 0.1, top=0.05), 0.05)]) == math.inf


def test_no_band_sees_every_low_obstacle():
    assert _forward_range({}, _pipe_at(0.6, 0.01)) == pytest.approx(0.6)


def test_arena_border_variants_and_room_wall():
    from host_sim.core import Arena, Robot, World
    camera = LidarParams(noise_std=0.0, num_beams=1, fov=0.01, range_min=0.6, range_max=5.0,
                         out_of_range_value=0.0, **BAND)

    def front(**arena):
        world = World(Arena(x_min=0.0, x_max=4.0, y_min=0.0, y_max=4.0, **arena))
        world.add_robot('cam', Robot(instant_params(), 3.4, 2.0, 0.0), depth_scan=camera)
        return float(world.depth_scan('cam')[0])
    assert front() == pytest.approx(0.6)                            # wall (None): always seen
    assert front(border_top=0.05) == 0.0                            # pipe passed over: no depth
    assert front(border_top=0.10) == pytest.approx(0.6)
    assert front(border_top=0.0) == 0.0                             # nothing on the border
    assert front(border_top=0.05, room_wall_offset=0.8) == pytest.approx(1.4)
    low_box = World(Arena(walls_enabled=False))
    low_box.add_robot('cam', Robot(instant_params(), 3.0, 3.0, 0.0), depth_scan=camera)
    low_box.add_obstacle(Rect(3.9, 3.0, 0.2, 0.2, top=0.05))
    assert float(low_box.depth_scan('cam')[0]) == 0.0              # 0.8 m ahead, passed over
