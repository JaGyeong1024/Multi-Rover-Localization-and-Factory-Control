"""Heading offset frozen while the wheel encoders stand still (tracker.WHEEL_STILL).

Field case: the leader stood 6 min at the shelf, sigma(h_off) grew to 77 deg (q_hoff * time),
then hands on the rover pushed its camera box around (~0.1 m/s of fake velocity) and the NHC set the
heading to -43 deg while gyro, wheels and box axis showed no rotation."""
import math

from host_localization.fusion.se2nhc import HOFF
from host_localization.fusion.tracker import KW, RobotTracker


def _parked_then_shoved(wheel_still, feed_wheels=True, wheels_turn=False):
    trk = RobotTracker(dict(KW['leader']), wheel_still=wheel_still)
    est = trk.est
    t, x, y = 0.0, 1.0, 1.0
    trk.on_bev(t, x, y)
    est._reanchor(0.0)                                # heading anchored at 0 (as after a COG run)
    est.P[HOFF, HOFF] = math.radians(6.5) ** 2
    est._cog_seen = True
    dt = 1.0 / 21.0
    wheel = 0.0
    for k in range(int(360.0 / dt)):                  # 6 min parked, then 3 s of a shoved camera box
        t = (k + 1) * dt
        shove = t > 357.0
        if shove:
            s = min(t - 357.0, 0.5) / 0.5 * 0.08      # box centre slides 8 cm in 0.5 s towards 135 deg
            x, y = 1.0 - s * math.sqrt(0.5), 1.0 + s * math.sqrt(0.5)
        trk.on_imu(t, 0.0, 0.0)
        if feed_wheels:
            wheel += 0.1 * dt if wheels_turn else 0.0
            trk.on_wheels(t, -wheel, wheel)
        trk.on_bev(t, x, y)
    return math.degrees(est.heading())


def test_parked_heading_survives_box_push():
    assert abs(_parked_then_shoved(wheel_still=True)) < 2.0


def test_old_behaviour_reproduces_failure():
    assert abs(_parked_then_shoved(wheel_still=False)) > 20.0


def test_no_wheel_data_means_old_behaviour():
    assert _parked_then_shoved(True, feed_wheels=False) == _parked_then_shoved(False)


def test_turning_wheels_are_not_still():
    assert _parked_then_shoved(True, wheels_turn=True) == _parked_then_shoved(False, wheels_turn=True)
