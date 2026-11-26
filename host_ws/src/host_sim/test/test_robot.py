"""Base model: integration accuracy, lag, timeout and skid-steer slip."""
import math

import pytest
from conftest import drive, instant_params

from host_sim.core import Robot, RobotParams


def test_straight_line_distance():
    robot = Robot(instant_params())
    drive(robot, 0.2, 0.0, 5.0)
    assert robot.x == pytest.approx(1.0, abs=1e-9)
    assert robot.y == pytest.approx(0.0, abs=1e-12)
    assert robot.odometer == pytest.approx(1.0, abs=1e-9)


def test_full_circle_returns_to_start():
    robot = Robot(instant_params())
    omega = 2.0 * math.pi / 10.0
    drive(robot, 0.2, omega, 5.0)
    assert robot.y == pytest.approx(2.0 * 0.2 / omega, abs=1e-6)  # diameter at half turn
    drive(robot, 0.2, omega, 5.0)
    assert math.hypot(robot.x, robot.y) < 1e-6
    assert abs(robot.theta) < 1e-6


@pytest.mark.parametrize('dt', [0.001, 0.02, 0.05])
def test_arc_integration_step_independent(dt):
    robot = Robot(instant_params())
    drive(robot, 0.15, 0.8, 2.0, dt=dt)
    radius = 0.15 / 0.8
    assert robot.x == pytest.approx(radius * math.sin(1.6), abs=1e-9)
    assert robot.y == pytest.approx(radius * (1.0 - math.cos(1.6)), abs=1e-9)


def test_heading_stays_wrapped():
    robot = Robot(instant_params(), theta=3.0)
    drive(robot, 0.0, 1.0, 1.0)
    assert -math.pi < robot.theta <= math.pi
    assert robot.theta == pytest.approx(4.0 - 2.0 * math.pi, abs=1e-9)


def test_lag_reaches_63_percent_at_tau():
    robot = Robot(RobotParams(linear_time_constant=0.2, max_linear_accel=1e9))
    drive(robot, 0.2, 0.0, 0.2, dt=0.001)
    assert robot.v == pytest.approx(0.2 * (1.0 - math.exp(-1.0)), rel=1e-3)


def test_accel_limit_caps_speed_change():
    robot = Robot(RobotParams(linear_time_constant=0.0, max_linear_accel=0.5))
    drive(robot, 0.2, 0.0, 0.2)
    assert robot.v == pytest.approx(0.1, abs=1e-9)


def test_command_is_clamped_to_speed_limits():
    robot = Robot(instant_params(max_linear_speed=0.2, max_angular_speed=1.0))
    drive(robot, 5.0, -9.0, 0.1)
    assert robot.v == pytest.approx(0.2)
    assert robot.w == pytest.approx(-1.0)


def test_command_timeout_stops_the_base():
    robot = Robot(RobotParams(cmd_timeout=0.5))
    robot.set_command(0.2, 0.5)
    for _ in range(45):
        robot.step(0.01)
    assert robot.v > 0.15  # still driving inside the timeout window
    for _ in range(155):
        robot.step(0.01)
    assert abs(robot.v) < 1e-3 and abs(robot.w) < 1e-3
    x_after_stop = robot.x
    for _ in range(100):
        robot.step(0.01)
    assert robot.x == pytest.approx(x_after_stop, abs=1e-4)


def test_no_command_ever_means_no_motion():
    robot = Robot(RobotParams())
    for _ in range(100):
        robot.step(0.01)
    assert (robot.x, robot.y, robot.theta) == (0.0, 0.0, 0.0)


def test_forward_efficiency_shortens_travel():
    ideal, slipping = Robot(instant_params()), Robot(instant_params(forward_efficiency=0.9))
    for robot in (ideal, slipping):
        drive(robot, 0.2, 0.0, 5.0)
    assert slipping.x == pytest.approx(0.9 * ideal.x, rel=1e-9)


def test_tread_ratio_scales_yaw_widens_arc():
    ideal, slipping = Robot(instant_params()), Robot(instant_params(tread_ratio=0.7))
    for robot in (ideal, slipping):
        drive(robot, 0.2, 0.5, 2.0)
    assert slipping.theta == pytest.approx(0.7 * ideal.theta, rel=1e-9)
    # Same speed, lower yaw rate -> turning radius grows by 1 / ratio.
    radius = slipping.x / math.sin(slipping.theta)
    assert radius == pytest.approx((0.2 / 0.5) / 0.7, rel=1e-6)
    assert slipping.y < ideal.y  # ends less far to the left than commanded


def test_icr_offset_lateral_motion_spinning():
    robot = Robot(instant_params(icr_offset_x=0.05))
    robot.set_command(0.0, 1.0)
    robot.step(0.01)
    assert robot.lateral_velocity == pytest.approx(-0.05)
    drive(robot, 0.0, 1.0, math.pi - 0.01)
    # Half a turn about a centre 0.05 m ahead moves the body origin by the diameter.
    assert robot.x == pytest.approx(0.10, abs=1e-3)
    assert abs(robot.y) < 1e-3



def test_tracked_point_circles_in_place_turn():
    """ref_offset_x is the radius of the circle the tracked point draws, measured from
    the rotation centre, not the body origin."""
    robot = Robot(instant_params(icr_offset_x=0.03, ref_offset_x=0.02))
    points = []
    for _ in range(8):
        drive(robot, 0.0, 1.0, math.pi / 4.0, dt=0.001)
        points.append(robot.tracked_point()[:2])
    centre = (0.03, 0.0)  # the rotation centre does not move
    radii = [math.hypot(x - centre[0], y - centre[1]) for x, y in points]
    assert radii == pytest.approx([0.02] * 8, abs=1e-3)

def test_omega_quantum_rounds_in_round_mode():
    robot = Robot(instant_params(omega_quantum=0.05))
    drive(robot, 0.0, 0.12, 0.1)
    assert robot.w == pytest.approx(0.10)
    drive(robot, 0.0, 0.02, 0.1)
    assert robot.w == pytest.approx(0.0)


@pytest.mark.parametrize('command, expected', [
    (0.02, 0.0), (0.06, 0.0), (0.08, 0.0697), (0.10, 0.0697),
    (0.14, 0.1394), (0.20, 0.1394), (0.30, 0.2788),
    (0.1394, 0.1394),  # exact multiple must not fall one step short
    (-0.06, 0.0), (-0.10, -0.0697), (-0.30, -0.2788),
])
def test_omega_truncate_matches_leader_table(command, expected):
    robot = Robot(instant_params(omega_quantum=0.0697, omega_quantize='truncate'))
    drive(robot, 0.0, command, 0.1)
    assert robot.w == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize('command, expected', [
    (0.105, 0.10), (0.009, 0.0), (-0.105, -0.10),
    (0.10, 0.10),  # exact multiple must not fall one step short
])
def test_linear_truncate_matches_leader(command, expected):
    """Truncation as static_cast<int32_t>(linear_velocity_x * 100)."""
    robot = Robot(instant_params(linear_quantum=0.01, linear_quantize='truncate'))
    drive(robot, command, 0.0, 0.1)
    assert robot.v == pytest.approx(expected, abs=1e-9)


def test_linear_quantum_is_off_by_default():
    robot = Robot(instant_params())
    drive(robot, 0.105, 0.0, 0.1)
    assert robot.v == pytest.approx(0.105)


@pytest.mark.parametrize('timeout', [None, 0.0])
def test_no_timeout_keeps_last_command(timeout):
    robot = Robot(RobotParams(cmd_timeout=timeout))
    robot.set_command(0.2, 0.0)
    for _ in range(1000):  # 10 s with no further command
        robot.step(0.01)
    assert robot.v == pytest.approx(0.2, abs=1e-6)


def test_omega_deadband_ignores_small_yaw():
    robot = Robot(instant_params(omega_deadband=0.1))
    drive(robot, 0.0, 0.08, 0.5)
    assert robot.theta == 0.0
    drive(robot, 0.0, 0.2, 0.5)
    assert robot.theta == pytest.approx(0.1)


def test_hw_angular_sign_needs_rover_sign():
    reversed_hw = Robot(instant_params(hw_angular_sign=-1.0))
    drive(reversed_hw, 0.0, 0.5, 1.0)
    assert reversed_hw.theta == pytest.approx(-0.5)  # REP-103 command turns the wrong way
    compensated = Robot(instant_params(hw_angular_sign=-1.0))
    rover_angular_sign = -1.0
    drive(compensated, 0.0, rover_angular_sign * 0.5, 1.0)
    assert compensated.theta == pytest.approx(0.5)


def test_tracked_point_ahead_of_rotation():
    robot = Robot(instant_params(ref_offset_x=0.1), x=1.0, y=2.0, theta=math.pi / 2.0)
    x, y, theta = robot.tracked_point()
    assert (x, y) == pytest.approx((1.0, 2.1))
    assert theta == pytest.approx(math.pi / 2.0)
    drive(robot, 0.0, 1.0, math.pi / 2.0, dt=0.001)
    assert robot.tracked_point()[:2] == pytest.approx((0.9, 2.0), abs=1e-3)
    assert (robot.x, robot.y) == pytest.approx((1.0, 2.0))  # spins about its centre


def test_zero_traction_wheels_turn_body_still():
    robot = Robot(instant_params())
    robot.traction = 0.0
    drive(robot, 0.2, 0.5, 1.0)
    assert (robot.x, robot.y, robot.theta) == (0.0, 0.0, 0.0)
    assert robot.v == pytest.approx(0.2)


def test_reset_pose_clears_motion():
    robot = Robot(instant_params())
    drive(robot, 0.2, 0.5, 1.0)
    robot.reset_pose(2.0, 3.0, 1.0)
    robot.step(0.01)
    assert (robot.x, robot.y, robot.theta) == pytest.approx((2.0, 3.0, 1.0))
