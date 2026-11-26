"""sim_stand_in_node pure parts: front distance (LiDAR and depth band), pick stand-in, unload edge."""
import math

import pytest

from host_mission.sim_stand_in_node import (DEPTH_BAND, EdgeTrigger, PickerStandIn,
                                            depth_band_distance, front_distance)


def test_front_distance_nearest_ahead_only():
    # beams at -0.2, -0.1, 0.0, 0.1, 0.2 rad; the 0.3 m return at 0.2 rad is outside +-0.15
    ranges = [0.3, 1.5, 0.0, 1.2, 0.3]
    assert front_distance(ranges, -0.2, 0.1, 0.15) == pytest.approx(1.2)


def test_front_distance_wraps_scan_at_zero():
    # 0..2pi scan: the beam just below 2pi is ahead too
    n = 360
    ranges = [5.0] * n
    ranges[n - 1] = 0.8
    assert front_distance(ranges, 0.0, 2 * math.pi / n, 0.15) == pytest.approx(0.8)


def test_front_distance_nan_without_return():
    assert math.isnan(front_distance([0.0, math.inf, 0.0], -0.1, 0.1, 0.15))


# beams at -0.4 .. 0.4 rad (0.1 apart): with the leader's hfov 1.019 and columns 0.3..0.7
# the band is |bearing| <= atan(0.2 * 2 tan(0.5095)) = 0.22 rad -> the 5 beams -0.2 .. 0.2
_ANGLE_MIN, _STEP = -0.4, 0.1


def test_depth_band_ignores_outside_beams():
    band = {**DEPTH_BAND, 'depth_percentile': 0.0}             # percentile 0 = nearest
    ranges = [0.4, 0.4, 2.0, 2.0, 2.0, 1.0, 2.0, 0.4, 0.4]     # 0.4 m returns outside the band
    expected = 1.0 * math.cos(0.1)                             # the 1.0 m beam at +0.1 rad
    assert depth_band_distance(ranges, _ANGLE_MIN, _STEP, band) == pytest.approx(expected)


def test_depth_band_nan_with_few_depths():
    no_depth = [0.0] * 9                                       # 0.0 = pixel without depth
    assert math.isnan(depth_band_distance(no_depth, _ANGLE_MIN, _STEP, DEPTH_BAND))
    one_of_five = [0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0]  # 1/5 < min ratio 0.3
    assert math.isnan(depth_band_distance(one_of_five, _ANGLE_MIN, _STEP, DEPTH_BAND))
    two_of_five = [0.0, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0]
    assert depth_band_distance(two_of_five, _ANGLE_MIN, _STEP, DEPTH_BAND) == pytest.approx(
        1.5 * math.cos(0.1), abs=0.01)


def test_picker_holds_then_reports_done_once():
    picker = PickerStandIn(duration=5.0, repeat_delay=2.0)
    assert picker.step('TRACKING', 0.0) == []
    assert picker.step('HOLDING', 1.0) == [('PICKING', None)]
    assert picker.step('HOLDING', 5.9) == []
    assert picker.step('HOLDING', 6.0) == [('DONE', True)]
    assert picker.step('HOLDING', 7.0) == []
    assert picker.step('HOLDING', 8.0) == [('PICKING', None)]


def test_edge_trigger_fires_once_per_entry():
    edge = EdgeTrigger('UNLOADING')
    assert [edge(s) for s in ('RETURNING', 'UNLOADING', 'UNLOADING', 'IDLE', 'UNLOADING')] == \
        [False, True, False, False, True]
