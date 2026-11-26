"""Closed loop against ``host_sim``: tracker + gate drive the simulated rovers.
Needs ``host_ws/src/host_sim`` on PYTHONPATH (skipped otherwise)."""
import math

import pytest

pytest.importorskip('host_sim')

from host_sim.core import Circle                     # noqa: E402
from host_sim.scenario import Goal, run_scenario     # noqa: E402

from rover_drive.core import state as st             # noqa: E402

from host_loop import LEADER_QUANTUM, RoverStack, make_world, shift_path   # noqa: E402
from sim import drive_config, gate_config, make_cmd, s_path                # noqa: E402

POS_TOL = 0.05
YAW_TOL = 0.15
RUNIN_HEADING_BOUND_DEG = 3.0
# Tracked point: body origin 2.6 + icr 0.03 + ref offset 0.02. y 2.9, not the world's 3.3: from 3.3
# the S's right turn points the depth camera at the arena wall (y 4.66) from ~0.55 m, below its
# 0.6 m minimum, and the front distance turns NaN after a valid one.
LEADER_START = (2.65, 2.9)
LEADER_BODY_START = [2.6, 2.9, 0.0]
LEADER_REF_OFFSET = 0.02
FOLLOWER_START = (1.9, 3.3)
PARKED_AWAY = [5.3, 2.5, 0.0]


def _path(start):
    return shift_path(s_path(radius=0.5, lead_in=0.8), *start)


def _goal(poses):
    x, y, yaw = poses[-1]
    return Goal(x, y, theta=yaw, pos_tol=POS_TOL, yaw_tol=YAW_TOL)


def _xy(poses):
    return [[x, y] for x, y, _ in poses]


def _run_leader(truncate, seed=1):
    """Leader base of host_sim with its slip; omega cut = measured hardware or none."""
    world = make_world(seed, leader={'omega_quantum': 0.0, 'omega_deadband': 0.0,
                                     'start': LEADER_BODY_START},
                       follower={'start': PARKED_AWAY})
    poses = _path(LEADER_START)
    quantum = LEADER_QUANTUM if truncate else 0.0
    stack = RoverStack(
        drive_config('leader', ref_offset_x=LEADER_REF_OFFSET, omega_quantum=quantum),
        gate_config('leader'),
        make_cmd(poses, goal_pos_tol=POS_TOL, goal_yaw_tol=YAW_TOL, pick_at_goal=True),
        truncate_quantum=quantum)
    result = run_scenario(world, {'leader': stack}, duration=90.0,
                          goals={'leader': _goal(poses)},
                          reference_paths={'leader': _xy(poses)})
    return result, stack


def test_truncating_leader_follows_and_holds():
    result, stack = _run_leader(truncate=True)
    plain, _ = _run_leader(truncate=False)
    leader, ref = result.robots['leader'], plain.robots['leader']
    print(f'leader truncating: cross-track {leader.max_cross_track:.3f} m, goal '
          f'{leader.final_pos_error:.3f} m / {leader.final_yaw_error:.3f} rad | continuous: '
          f'{ref.max_cross_track:.3f} m, {ref.final_pos_error:.3f} m / '
          f'{ref.final_yaw_error:.3f} rad')
    assert leader.arrived and not result.collided
    assert stack.log[-1][1] == st.HOLDING
    assert leader.max_cross_track < 0.06
    assert leader.final_pos_error < POS_TOL and leader.final_yaw_error < YAW_TOL
    assert leader.max_cross_track < ref.max_cross_track + 0.01
    assert st.ERROR not in stack.states()
    assert not {'OFF_PATH', 'STUCK'} & {row[2] for row in stack.log}


def _follower_stack(poses, **cmd_kw):
    return RoverStack(drive_config('follower'), gate_config('follower'),
                      make_cmd(poses, goal_pos_tol=POS_TOL, goal_yaw_tol=YAW_TOL, **cmd_kw))


def test_lidar_gate_follower_follows_and_parks():
    world = make_world(leader={'start': PARKED_AWAY})
    poses = _path(FOLLOWER_START)
    stack = _follower_stack(poses)
    result = run_scenario(world, {'follower': stack}, duration=60.0,
                          goals={'follower': _goal(poses)},
                          reference_paths={'follower': _xy(poses)})
    follower = result.robots['follower']
    print(f'follower: cross-track {follower.max_cross_track:.3f} m, goal '
          f'{follower.final_pos_error:.3f} m / {follower.final_yaw_error:.3f} rad, '
          f'gate reasons {sorted(stack.gate_reasons())}')
    assert follower.arrived and not result.collided
    assert stack.log[-1][1] == st.PARKED
    assert follower.max_cross_track < 0.05
    assert follower.final_pos_error < POS_TOL and follower.final_yaw_error < YAW_TOL
    assert 'LIDAR_STOP' not in stack.gate_reasons()


def test_follower_stops_for_obstacle_resumes():
    world = make_world(leader={'start': PARKED_AWAY})
    poses = _path(FOLLOWER_START)
    block = poses[len(poses) // 2]
    world.add_obstacle(Circle(block[0], block[1], 0.08, name='box'))
    stack = _follower_stack(poses)
    result = run_scenario(world, {'follower': stack}, duration=80.0,
                          goals={'follower': _goal(poses)},
                          events=[(35.0, lambda w: w.remove_obstacle('box'))])
    follower = result.robots['follower']
    stops = [row for row in stack.log if 'LIDAR_STOP' in row[4]]
    assert stops and not result.collided
    assert result.min_obstacle_clearance > 0.03
    t_stop = stops[0][0]
    assert math.hypot(stops[0][5] - block[0], stops[0][6] - block[1]) < 0.15
    waiting = [row for row in stack.log if t_stop + 1.0 < row[0] < 34.0]
    assert all(row[1] == st.STOPPED and row[2] == 'LIDAR_STOP' for row in waiting)
    assert follower.arrived and stack.log[-1][1] == st.PARKED


def _u_turn_path(x0, y0, radius=0.30, lead_in=0.5, run_in=0.40, spacing=0.05):
    """West, a left half turn at the planner's minimum radius, east run-in to the goal."""
    pts, x, y, yaw = [(x0, y0)], x0, y0, math.pi
    for seg_len, kappa in ((lead_in, 0.0), (radius * math.pi, 1.0 / radius), (run_in, 0.0)):
        for _ in range(int(round(seg_len / spacing))):
            yaw_mid = yaw + 0.5 * kappa * spacing
            x += spacing * math.cos(yaw_mid)
            y += spacing * math.sin(yaw_mid)
            yaw += kappa * spacing
            pts.append((x, y))
    yaws = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(pts, pts[1:])]
    return [(px, py, yw) for (px, py), yw in zip(pts, yaws + yaws[-1:])]


@pytest.mark.parametrize('offset', [0.0, 0.05])
def test_leader_meets_heading_after_u_turn(offset):
    """Standby case: U-turn at 0.30 m radius into the 0.40 m run-in, slipping leader base
    (tread_ratio 0.7, icr 0.03, omega truncation), tracker given the true offset."""
    poses = _u_turn_path(4.0, 3.6)
    origin_to_tracked = 0.03 + offset
    world = make_world(leader={'ref_offset_x': offset,
                               'start': [4.0 + origin_to_tracked, 3.6, math.pi]},
                       follower={'start': PARKED_AWAY})
    sensor = world.sensors['leader']
    sensor.params.pos_noise_std = sensor.params.heading_noise_std = 0.0
    sensor.reset()
    stack = RoverStack(drive_config('leader', ref_offset_x=offset), gate_config('leader'),
                       make_cmd(poses, goal_pos_tol=POS_TOL, goal_yaw_tol=0.10),
                       truncate_quantum=LEADER_QUANTUM)
    result = run_scenario(world, {'leader': stack}, duration=60.0,
                          goals={'leader': _goal(poses)})
    leader = result.robots['leader']
    print(f'offset {offset}: goal {leader.final_pos_error:.3f} m / '
          f'{math.degrees(leader.final_yaw_error):.2f} deg')
    assert stack.log[-1][1] == st.PARKED
    assert leader.final_pos_error < 0.03
    assert leader.final_yaw_error < math.radians(RUNIN_HEADING_BOUND_DEG)
