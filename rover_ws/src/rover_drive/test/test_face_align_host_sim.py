"""Closed loop against ``host_sim``: the leader squares up to a shelf set askew.

Tracker + gate (front stop exemption, angular_sign -1) drive host_sim's leader base (slip, lag,
omega steps) along a straight run-in to a pick goal in front of a shelf turned 0.2 rad from the goal
heading. The depth camera sees nothing nearer than 0.6 m, so at the goal the run-in's fit, carried by
the gyro, is what the rover turns by. Depth noise matches the real camera (1.5 mm rms fit).
Needs ``host_ws/src/host_sim`` on PYTHONPATH (skipped otherwise).
"""
import dataclasses
import math

import pytest

pytest.importorskip('host_sim')

from host_sim.core import Rect                                   # noqa: E402
from host_sim.scenario import run_scenario                       # noqa: E402

from rover_drive.core import state as st                         # noqa: E402
from rover_drive.core.controller import FACE_ALIGN, FACE_ALIGNED  # noqa: E402
from rover_drive.core.depth_front import DepthScan               # noqa: E402

from host_loop import LEADER_QUANTUM, RoverStack, make_world     # noqa: E402
from sim import drive_config, gate_config, make_cmd, straight_path   # noqa: E402

BODY_START = [2.6, 2.9, 0.0]
TRACKED_START = (2.65, 2.9)          # body origin + icr + ref offset
RUN_IN = 1.2
SHELF_YAW = 0.2                      # rad off square
FACE_AHEAD = 0.30                    # m, face centre ahead of the tracked point at the goal
SHELF_DEPTH, SHELF_WIDTH = 0.285, 0.365
DEPTH_NOISE = 0.003                  # m per beam


class FaceAlignStack(RoverStack):
    """RoverStack that also feeds the tracker the gyro and the depth scan (path_tracker
    with face_align)."""

    def __init__(self, world, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world = world

    def __call__(self, obs):
        t = obs.t
        self.ctrl.on_gyro(self.world.robots['leader'].w, t, t)
        if obs.depth_scan_is_new:
            angles = obs.depth_scan_angles
            self.ctrl.on_depth_scan(DepthScan(obs.depth_scan, float(angles[0]),
                                              float(angles[1] - angles[0]),
                                              self.depth_cfg.depth_valid_min_m, 6.0), t)
        return super().__call__(obs)


def _shelf():
    """Shelf whose near face is centred FACE_AHEAD beyond the goal, turned SHELF_YAW."""
    gx, gy = TRACKED_START[0] + RUN_IN, TRACKED_START[1]
    fx, fy = gx + FACE_AHEAD, gy
    half = SHELF_DEPTH / 2.0
    return Rect(fx + half * math.cos(SHELF_YAW), fy + half * math.sin(SHELF_YAW),
                SHELF_DEPTH, SHELF_WIDTH, yaw=SHELF_YAW, name='shelf')


def test_leader_squares_up_to_askew_shelf():
    world = make_world(1, leader={'omega_quantum': 0.0, 'omega_deadband': 0.0,
                                  'hw_angular_sign': -1.0, 'start': BODY_START},
                       follower={'start': [5.3, 2.5, 0.0]})
    world.add_obstacle(_shelf())
    camera = world.depth_scans['leader']
    camera.params = dataclasses.replace(camera.params, noise_std=DEPTH_NOISE)
    poses = [(x + TRACKED_START[0], y + TRACKED_START[1], yaw)
             for x, y, yaw in straight_path(RUN_IN)]
    stack = FaceAlignStack(
        world,
        drive_config('leader', ref_offset_x=0.02, omega_quantum=LEADER_QUANTUM,
                     face_align=True),
        gate_config('leader'),
        make_cmd(poses, goal_pos_tol=0.05, goal_yaw_tol=0.10, pick_at_goal=True,
                 fixture_ahead=True),
        truncate_quantum=LEADER_QUANTUM)
    result = run_scenario(world, {'leader': stack}, duration=45.0)
    notes = stack.ctrl.take_notes()
    theta = world.robots['leader'].theta
    print(f'final yaw {math.degrees(theta):+.2f} deg (shelf {math.degrees(SHELF_YAW):+.1f}),'
          f' gate reasons {sorted(stack.gate_reasons())}\n  ' + '\n  '.join(notes))
    assert not result.collided
    assert (stack.log[-1][1], stack.log[-1][2]) == (st.HOLDING, FACE_ALIGNED)
    assert abs(theta - SHELF_YAW) <= drive_config('leader').face_align_tol_rad
    aligning = [row for row in stack.log if row[2] == FACE_ALIGN]
    assert aligning and not any(st.STOPPED == row[3] for row in aligning)
    assert any('run-in fit' in n and 'turning' in n for n in notes)
