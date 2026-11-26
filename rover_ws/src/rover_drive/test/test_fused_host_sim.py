"""Closed loop with sensor-level localization: host_sim's camera / UWB / IMU models feed the real
host fusion front-end (localization mode fused); the rover stack drives on its output.

Needs host_sim and host_localization on PYTHONPATH.
"""
import math
import time

import pytest

pytest.importorskip('host_sim')

from host_sim.config import apply_profile, load_config, with_localization_mode  # noqa: E402
from host_sim.core import World                                                  # noqa: E402
from host_sim.scenario import Goal, run_scenario                                 # noqa: E402

from rover_drive.core import state as st                                         # noqa: E402

from host_loop import RoverStack                                                 # noqa: E402
from sim import drive_config, gate_config, make_cmd, straight_path               # noqa: E402

POS_TOL = 0.05
HEADING_SETTLE = 1.0        # s after alignment
YAW_TOL = 0.15
PATH_LENGTH = 3.0           # m, straight along +x; the first 0.8 m is the alignment creep
WARM_UP = 3.0               # s the host localization runs before the path is sent
PARKED_ALONG = 1.8
PARKED_LATERAL = 0.35       # m beside the path: inside the camera occlusion (0.40), clear of the footprints
LEADER_LEAVES_AT = 25.0


def _world(mode, profile=None, parked_beside=False, seed=1):
    """Default world: follower at the arena's west side facing +x, leader parked.
    Returns (world, path poses, a parking spot of the leader away from the path)."""
    config = load_config()
    arena = config['arena']
    x0, y0 = arena['x_min'] + 0.5, (arena['y_min'] + arena['y_max']) / 2.0
    config['robots']['follower']['start'] = [x0, y0, 0.0]
    away = [arena['x_max'] - 0.4, arena['y_min'] + 0.4, 0.0]
    beside = [x0 + PARKED_ALONG, y0 + PARKED_LATERAL, 0.0]
    config['robots']['leader']['start'] = beside if parked_beside else away
    if profile:
        config = apply_profile(config, profile)
    world = World.from_config(with_localization_mode(config, mode), seed=seed)
    poses = [(x + x0, y + y0, yaw) for x, y, yaw in straight_path(PATH_LENGTH)]
    return world, poses, away


class _Recorder:
    """Rover stack as the follower's controller, logging what it saw next to the truth."""

    def __init__(self, world, poses):
        self.world = world
        self.stack = RoverStack(drive_config('follower'), gate_config('follower'),
                                make_cmd(poses, goal_pos_tol=POS_TOL, goal_yaw_tol=YAW_TOL))
        self.rows = []

    def __call__(self, obs):
        command = self.stack(obs)
        _, state, detail = self.stack.log[-1][:3]
        self.rows.append((obs.t, obs.pose, self.world.robots['follower'].tracked_point(),
                          state, detail))
        return command

    def heading_errors(self, settle=0.0):
        """|fused - true heading| [rad] of the aligned samples the rover used, from ``settle``
        seconds after the first one on."""
        aligned = [row for row in self.rows if row[1] is not None and row[1].heading_aligned]
        start = aligned[0][0] + settle if aligned else math.inf
        return [abs(math.remainder(pose.theta - truth[2], 2.0 * math.pi))
                for t, pose, truth, _, _ in aligned if t >= start]

    def position_errors(self):
        return [math.hypot(pose.x - truth[0], pose.y - truth[1])
                for _, pose, truth, _, _ in self.rows if pose is not None]


def _run(world, poses, duration, events=None):
    recorder = _Recorder(world, poses)
    run_scenario(world, {}, duration=WARM_UP)
    x, y, yaw = poses[-1]
    started = time.perf_counter()
    result = run_scenario(world, {'follower': recorder}, duration=duration,
                          goals={'follower': Goal(x, y, yaw, POS_TOL, YAW_TOL)},
                          reference_paths={'follower': [[px, py] for px, py, _ in poses]},
                          events=events)
    return result, recorder, time.perf_counter() - started


@pytest.mark.parametrize('mode', ['ideal', 'fused'])
def test_follower_aligns_arrives_straight(mode):
    world, poses, _ = _world(mode)
    result, recorder, wall = _run(world, poses, duration=60.0)
    follower = result.robots['follower']
    headings = recorder.heading_errors(settle=HEADING_SETTLE)
    print(f'{mode}: arrived {follower.arrived} at {follower.arrival_time} s, wall {wall:.2f} s '
          f'for {result.duration:.1f} sim s, max heading error {math.degrees(max(headings)):.1f} '
          f'deg ({math.degrees(max(recorder.heading_errors())):.1f} incl. alignment), max position error '
          f'{max(recorder.position_errors()):.3f} m, cross-track {follower.max_cross_track:.3f} m')
    assert follower.arrived and not result.collided
    assert recorder.stack.log[-1][1] == st.PARKED
    assert max(headings) < math.radians(5.0)
    assert follower.max_cross_track < 0.05


# Estimator-failure profile: the position coasted and sat on the UWB frame, heading far off.
def test_field_follower_passes_parked_rover():
    world, poses, away = _world('fused', profile='field_site', parked_beside=True)
    leave = (LEADER_LEAVES_AT, lambda w: w.reset_pose('leader', *away))
    result, recorder, wall = _run(world, poses, duration=90.0, events=[leave])
    follower = result.robots['follower']
    lost = [row for row in recorder.rows if row[1] is not None and row[1].quality != 'GOOD']
    print(f'field: arrived {follower.arrived}, wall {wall:.2f} s, final tracker '
          f'{recorder.stack.log[-1][1:3]}, non-GOOD samples {len(lost)}, max position error '
          f'{max(recorder.position_errors()):.2f} m, max heading error '
          f'{math.degrees(max(recorder.heading_errors())):.1f} deg')
    assert follower.arrived and not result.collided
    assert recorder.stack.log[-1][1] == st.PARKED
