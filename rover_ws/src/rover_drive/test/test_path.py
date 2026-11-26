"""Path: arc length, interpolation, curvature and (monotonic) projection."""
import math

import pytest

from rover_drive.core.path import Path, PathProgress, wrap_angle

from sim import s_path, straight_path


def _xy(poses):
    return [(p[0], p[1]) for p in poses]


def _circle(radius=0.5, sweep=math.pi, n=60):
    return [(radius * math.sin(a), radius * (1 - math.cos(a)))
            for a in (sweep * i / n for i in range(n + 1))]


def test_arc_length_and_interpolation():
    path = Path([(0, 0), (1, 0), (1, 2)])
    assert path.length == pytest.approx(3.0)
    assert path.point(0.5) == pytest.approx((0.5, 0.0))
    assert path.point(2.0) == pytest.approx((1.0, 1.0))
    assert path.tangent(0.5) == pytest.approx(0.0)
    assert path.tangent(2.0) == pytest.approx(math.pi / 2)


def test_point_clamps_or_extrapolates_past_end():
    path = Path([(0, 0), (1, 0)])
    assert path.point(1.5) == pytest.approx((1.0, 0.0))
    assert path.point(1.5, extrapolate=True) == pytest.approx((1.5, 0.0))


def test_duplicate_points_are_merged():
    path = Path([(0, 0), (0, 0), (1, 0), (1, 0)])
    assert path.length == pytest.approx(1.0)


def test_single_point_path():
    path = Path([(1.0, 2.0)])
    assert path.length == 0.0
    assert path.point(0.3, extrapolate=True) == (1.0, 2.0)
    assert path.project(1.0, 2.5).dist == pytest.approx(0.5)


def test_curvature_of_a_circle_is_signed():
    left = Path(_circle(0.5))
    right = Path([(x, -y) for x, y in _circle(0.5)])
    mid = left.length / 2
    assert left.curvature(mid) == pytest.approx(2.0, rel=0.02)
    assert right.curvature(mid) == pytest.approx(-2.0, rel=0.02)
    assert left.max_abs_curvature(0.2, 0.6) == pytest.approx(2.0, rel=0.02)
    assert Path(_xy(straight_path(2.0))).max_abs_curvature(0.0, 2.0) == pytest.approx(0.0)


def test_projection_signed_lateral_error():
    path = Path(_xy(straight_path(2.0)))
    left = path.project(1.0, 0.1)
    assert (left.s, left.lateral, left.dist) == pytest.approx((1.0, 0.1, 0.1))
    assert path.project(1.0, -0.2).lateral == pytest.approx(-0.2)


def test_projection_past_end_clamps_reports():
    proj = Path(_xy(straight_path(2.0))).project(2.5, 0.0)
    assert proj.s == pytest.approx(2.0)
    assert proj.dist == pytest.approx(0.5)


def test_projection_window_restricts_search():
    # Hairpin: out along y=0 and back along y=0.2. A point near both legs.
    path = Path([(0, 0), (2, 0), (2, 0.2), (0, 0.2)])
    assert path.project(1.0, 0.15).s == pytest.approx(3.2)
    assert path.project(1.0, 0.15, 0.8, 1.3).s == pytest.approx(1.0)


def test_closed_loop_projects_to_its_start():
    loop = Path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    assert loop.project(0.0, 0.01, tie_margin=0.05).s == pytest.approx(0.0, abs=0.02)


def test_progress_monotonic_on_backward_samples():
    progress = PathProgress(Path(_xy(straight_path(3.0))), window=0.5)
    seen = [progress.update(x, 0.0).s for x in (0.5, 0.48, 0.6, 0.55, 0.3, 0.7)]
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(0.7)


def test_progress_does_not_jump_to_later_leg():
    path = Path([(0, 0), (2, 0), (2, 0.2), (0, 0.2)])
    progress = PathProgress(path, window=0.5)
    progress.update(0.9, 0.0)
    # A noisy sample right on the return leg must not capture the rover.
    assert progress.update(1.0, 0.2).s == pytest.approx(1.0)


def test_progress_first_update_searches_all():
    progress = PathProgress(Path(_xy(straight_path(3.0))), window=0.5)
    assert progress.update(2.2, 0.05).s == pytest.approx(2.2)


def test_s_path_helper_is_consistent():
    poses = s_path()
    path = Path(_xy(poses))
    assert path.length == pytest.approx(0.5 + 0.6 * math.pi + 0.5, abs=0.06)
    assert abs(wrap_angle(poses[-1][2])) < 0.05


def test_segments_cut_range_out_of_path():
    path = Path([(0, 0), (1, 0), (1, 2)])
    seg = path.segments(0.5, 2.0)
    assert seg.start.ravel().tolist() == pytest.approx([0.5, 0.0, 1.0, 0.0])
    assert seg.unit.ravel().tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])
    assert seg.length.tolist() == pytest.approx([0.5, 1.0])
    assert seg.s.tolist() == pytest.approx([0.5, 1.0])
    assert path.segments(1.0, 1.0).length.size == 0
    assert path.segments(-1.0, 9.0).length.sum() == pytest.approx(3.0)
    assert Path([(1, 1)]).segments(0.0, 1.0).length.size == 0
