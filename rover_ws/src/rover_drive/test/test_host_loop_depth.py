"""The simulated depth camera's beams reduced to the depth node's front distance
(``host_loop.depth_band_distance``). Needs ``host_ws/src/host_sim`` on PYTHONPATH (skipped otherwise)."""
import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip('host_sim')

from host_sim.config import load_config                  # noqa: E402
from host_sim.core import Lidar, LidarParams             # noqa: E402

from rover_drive.core.config import DepthFrontConfig     # noqa: E402

from host_loop import depth_band_distance                # noqa: E402
from sim import load_config as rover_config              # noqa: E402

CFG = rover_config(DepthFrontConfig, 'leader')


def _camera():
    """host_sim's leader depth scanner (sim_default.yaml), noise-free, at the origin facing +x."""
    params = LidarParams(**load_config()['robots']['leader']['depth_scan'])
    return Lidar(dataclasses.replace(params, noise_std=0.0, mount_x=0.0))


def _distance(*segments):
    """Front distance with flat faces ((x0, y0), (x1, y1)) in front of the camera."""
    camera = _camera()
    ranges = camera.scan((0.0, 0.0, 0.0), [], np.array(segments, dtype=float).reshape(-1, 2, 2))
    return depth_band_distance(ranges, camera.angles, CFG)


def _face(x, half_width):
    return ((x, -half_width), (x, half_width))


def test_free_space_is_unknown():
    assert math.isnan(_distance())


def test_wall_at_one_metre():
    assert _distance(_face(1.0, 2.0)) == pytest.approx(1.0, abs=0.005)


def test_object_below_sensor_minimum_unknown():
    assert math.isnan(_distance(_face(0.4, 1.0)))


def test_object_must_cover_valid_band_share():
    # band 0.30-0.70 of the width = +-12.6 deg = 34 of 80 beams; 30 % of it = 11 beams
    assert math.isnan(_distance(_face(1.0, 0.06)))
    assert _distance(_face(1.0, 0.12)) == pytest.approx(1.0, abs=0.005)


def test_sim_band_edge_follows_depth_rows():
    """host_sim's band_lower_angle = lower edge of depth_row_max below the optical axis."""
    aspect = 480.0 / 640.0
    tan_half_v = aspect * math.tan(CFG.depth_hfov / 2.0)
    edge = math.atan((CFG.depth_row_max - 0.5) * 2.0 * tan_half_v)
    assert _camera().params.band_lower_angle == pytest.approx(edge, abs=2e-4)
