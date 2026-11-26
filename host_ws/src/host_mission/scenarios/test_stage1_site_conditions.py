"""Stage 1: empty arena, full laps under site-like conditions.

Every case runs the real fusion in the loop with the site profile (host_sim
sim_profiles.yaml ``field_site``: UWB coordinates offset from the camera frame, the
camera losing a rover next to the other one) on top of the simulated bases. On top of that:

  site            nothing more
  low_traction    leader turns at half and advances at 0.85 of the commanded rate
  follower_slip   follower advances and turns at 0.90
  camera_dropout  the camera loses each rover twice for 3 s while it drives: the rover
                  stops on LOST and resumes by itself
  two_laps        two parts dispatched back to back

A lap must complete without the mission pausing or faulting. The checks from the recorded
true state (lap_checks) run on every lap after that. The camera_dropout laps also check, per window, that the quality read
LOST, the output speed went to 0 and the rover drove on afterwards.

Arena edge variants (scenario world default: a 0.05 m pipe, no room wall):
  edge            no edge / pipe 0.05 m, both parts
  estop / clear / abandon   operator inputs mid-leg (operator model 'run to completion')"""
import math

import pytest
from host_sim.scenario import run_scenario

import layout_runner
from lap_checks import (MISSION_STOPS, dropout_failures, dropout_summary, lap_failures,
                        min_centre_distance, summary)
from layout_runner import LayoutLoop, _Finished

PROFILE = 'field_site'
LAP_LIMIT = 150.0  # s per part
FINAL_NAV_STATE = 'PARKED'
MISSION_STOPS = ('paused', 'phase_timeout', 'halt', 'fault', 'aborted')

LOW_TRACTION = {'leader': {'tread_ratio': 0.5, 'forward_efficiency': 0.85}}
FOLLOWER_SLIP = {'follower': {'tread_ratio': 0.90, 'forward_efficiency': 0.90}}
DROPOUT_S = 3.0
DROPOUTS = [(20.0, 'leader'), (30.0, 'follower'), (70.0, 'leader'), (80.0, 'follower')]

# Arena edge variants: what stands on the bounds (host_sim Arena.border_top; None = wall).
EDGES = {'no_edge': 0.0, 'pipe_0.05': 0.05}
# No edge, gear: slow under the blind-speed cap (nothing in range ahead), so this
# test allows more than LAP_LIMIT.
EDGE_LAP_LIMIT = 200.0

OCCLUSION_DIST = 0.34  # m, site profile camera occlusion_dist between rover centres


def _camera_dropouts():
    return [(t, lambda world, rover=rover: world.injector.camera_lost(rover, duration=DROPOUT_S))
            for t, rover in DROPOUTS]


def run_laps(parts, seed=1, base=None, events=(), border_top='default', lap_limit=LAP_LIMIT):
    """Dispatch ``parts`` in the empty layout and run until the mission is done
    (``border_top``: what stands on the arena edge, see EDGES; 'default' = the world's)."""
    loop = LayoutLoop('empty', seed, profile=PROFILE)
    if border_top != 'default':
        loop.world.arena.border_top = border_top
    for rover, values in (base or {}).items():
        for key, value in values.items():
            assert hasattr(loop.world.robots[rover].params, key), key
            setattr(loop.world.robots[rover].params, key, value)
    for part in parts:
        assert loop.engine.dispatch(part), part
    try:
        run_scenario(loop.world, loop.controllers(), duration=lap_limit * len(parts),
                     events=list(events), stop_on_arrival=False)
    except _Finished:
        pass
    return loop.result('empty', '+'.join(parts))


def assert_lap_checks(result, front_stop=True, windows=()):
    """The lap_checks of a clean lap (no_mission_stop is asserted by assert_clean_laps)."""
    fails = lap_failures(result, clean=False, front_stop=front_stop)
    fails.update(dropout_failures(result, windows, DROPOUT_S))
    assert not fails, fails


def assert_clean_laps(result, laps):
    assert result.completed, result.reason()
    assert result.delivered == laps
    assert not result.collisions, result.collisions
    assert result.nav_states == {'leader': FINAL_NAV_STATE, 'follower': FINAL_NAV_STATE}
    stops = [event for event in result.events if event[1] in MISSION_STOPS]
    assert not stops, stops


@pytest.mark.parametrize('part,seed', [pytest.param(part, seed, id=f'{part}-{seed}')
                                       for part in ('gear_01', 'bolt_01') for seed in (1, 2, 3)])
def test_lap_under_site_conditions(part, seed, record_row):
    result = run_laps([part], seed)
    record_row('site            ' + result.row())
    record_row('    ' + summary(result))
    assert_clean_laps(result, 1)
    assert_lap_checks(result)


@pytest.mark.parametrize('name,base,part', [
    pytest.param(name, base, part, id=f'{name}-{part}')
    for name, base in (('low_traction', LOW_TRACTION), ('follower_slip', FOLLOWER_SLIP))
    for part in ('gear_01', 'bolt_01')])
def test_lap_with_slipping_bases(name, base, part, record_row):
    result = run_laps([part], base=base)
    record_row(f'{name:15s} ' + result.row())
    record_row('    ' + summary(result))
    assert_clean_laps(result, 1)
    assert_lap_checks(result)


@pytest.mark.parametrize('part', [pytest.param(part, id=part)
                                  for part in ('gear_01', 'bolt_01')])
def test_lap_with_camera_dropouts(part, record_row):
    result = run_laps([part], events=_camera_dropouts())
    record_row('camera_dropout  ' + result.row())
    record_row('    ' + summary(result))
    record_row(f'    dropouts {dropout_summary(result, DROPOUTS, DROPOUT_S)}')
    # The occlusion fault (camera loses a rover within OCCLUSION_DIST of the other) fires only
    # when the rover centres come that close; they do not in these laps.
    record_row(f'    min tracked-point distance {min_centre_distance(result):.3f} m '
               f'(occlusion at {OCCLUSION_DIST} m)')
    assert_clean_laps(result, 1)
    assert_lap_checks(result, windows=DROPOUTS)


def test_two_laps_back_to_back(record_row):
    result = run_laps(['gear_01', 'bolt_01'])
    record_row('two_laps        ' + result.row())
    record_row('    ' + summary(result))
    assert_clean_laps(result, 2)
    assert_lap_checks(result)


def _edge_cases():
    for edge in EDGES:
        for part in ('gear_01', 'bolt_01'):
            yield pytest.param(edge, part, id=f'{edge}-{part}')


IDLE_TURN_LIMIT = 60.0  # s after 'done' / the abandon for both idle turns to finish
TURN_SETTLE = 1.0               # s both rovers stand turned before the lap is judged


def _turned_home(loop):
    """Mission idle, both rovers PARKED at the last step of their idle turn (idle_turn.py)."""
    engine, turns = loop.engine, loop.engine.idle_turns
    rovers = loop.trace[-1][2]
    return loop.trace[-1][1] == 'IDLE' and all(
        r in turns.turns and not turns.under_way(r, engine._at_goal_key(r))
        and rovers[r].nav == 'PARKED' for r in engine.robots)


class _EndWhenTurned:
    """finish_when: ends the lap TURN_SETTLE s after both rovers have turned at home."""

    def __init__(self):
        self.since = None

    def __call__(self, loop):
        if not _turned_home(loop):
            self.since = None
        elif self.since is None:
            self.since = loop._now
        elif loop._now - self.since >= TURN_SETTLE:
            raise _Finished
        return False


def _operator_lap(actions, lap_limit, until_turned=False):
    """gear_01 in the empty arena (site profile, seed 1) with operator inputs
    [(t, fn(engine, now))]; returns (loop, result). ``until_turned``: runs on after 'done' /
    the abandon until both rovers have turned in place at home (mission idle_heading),
    at most IDLE_TURN_LIMIT s more than ``lap_limit``; the result is still cut at 'done'."""
    loop = LayoutLoop('empty', 1, profile=PROFILE)
    loop.operator = sorted(actions, key=lambda action: action[0])
    assert loop.engine.dispatch('gear_01')
    after_done = layout_runner.AFTER_DONE
    if until_turned:
        loop.finish_when = _EndWhenTurned()
        lap_limit += IDLE_TURN_LIMIT
        layout_runner.AFTER_DONE = IDLE_TURN_LIMIT      # the run-on after 'done' is ours
    try:
        run_scenario(loop.world, loop.controllers(), duration=lap_limit, stop_on_arrival=False)
    except _Finished:
        pass
    finally:
        layout_runner.AFTER_DONE = after_done
    return loop, loop.result('empty', 'gear_01')


def _state_before(trace, t):
    return [row for row in trace if row[0] < t][-1][1]


ROVERS = ('leader', 'follower')
ENGAGE, RELEASE = 30.0, 34.0  # s: e-stop mid-leg on the way to the pick, held 4 s
STOP_SETTLE = 1.0               # s from the engage until both stand (host tick + decel ramp)
STILL_TOL = 0.01                # m a rover may creep while the e-stop is engaged
# s after the release by which each rover has a new path. The follower yields first
# (YIELD_TO_LEAD): the leader's fresh path starts at the leader, 0.6 m ahead of it, so the
# follower's path to it comes within conflict_dist of the lead's remaining path until the
# leader has moved on.
FRESH_PATH_S = {'leader': 1.0, 'follower': 10.0}
START_TOL = 0.05                # m between a fresh path's first pose and the host's pose of
                                # the rover at that tick (true pose: + localization error)


def test_lap_with_estop_on_way_to_pick(record_row):
    """Engaged -> both stand; 'Release & Resume' (resume delivered FIRST, the two topics are not ordered) -> each rover gets a path planned from where it stands; lap ends."""
    actions = [(ENGAGE, lambda eng, now: eng.estop(True, now)),
               (RELEASE, lambda eng, now: eng.control('resume', now)),
               (RELEASE + 0.1, lambda eng, now: eng.estop(False, now))]
    loop, result = _operator_lap(actions, LAP_LIMIT + RELEASE - ENGAGE)
    record_row('estop_mid_leg   ' + result.row())
    trace = loop.trace
    assert _state_before(trace, ENGAGE) == 'TO_PICKUP'
    engaged = [row for row in trace if ENGAGE + STOP_SETTLE <= row[0] < RELEASE]
    assert {row[1] for row in engaged} == {'ESTOP'}
    for robot in ROVERS:
        start = engaged[0][2][robot].true
        creep = max(math.hypot(row[2][robot].true[0] - start[0], row[2][robot].true[1] - start[1])
                    for row in engaged)
        assert creep <= STILL_TOL, (robot, creep)
        stop_id = engaged[-1][2][robot].cmd_id
        fresh = next((row for row in trace if row[0] > RELEASE and row[2][robot].mode == 'FOLLOW'
                      and row[2][robot].cmd_id > stop_id), None)
        assert fresh is not None and fresh[0] <= RELEASE + FRESH_PATH_S[robot], (robot, fresh)
        waited = [row[2][robot] for row in trace if RELEASE < row[0] < fresh[0]]
        assert all(r.mode == 'STOP' for r in waited), robot
        rover = fresh[2][robot]
        gap = math.hypot(rover.first[0] - rover.host[0], rover.first[1] - rover.host[1])
        assert gap <= START_TOL, (robot, fresh[0], rover)
    assert_clean_laps(result, 1)


CLEAR, ADD = 30.0, 25.0         # s: a second part added, then the queue cleared, mid-leg
PAUSE, ABANDON = 30.0, 32.0     # s: mission paused mid-leg, then the trip abandoned
HOMING_LIMIT = 100.0            # s after the abandon for both rovers to park at home


def _home_goals(loop):
    """robot -> (goal key, goal pose) of its home as the mission sees it (facing west)."""
    seq, goals = loop.engine.seq, loop.engine.world.goals
    return {'leader': (seq.standby, goals.get(seq.standby)),
            'follower': (seq.follower_home, goals.get(seq.follower_home))}


def _pose_error(loop, robot, goal):
    """(position error [m], heading error [rad]) of the TRUE tracked point against ``goal``."""
    x, y, theta = loop.world.robots[robot].tracked_point()
    return math.hypot(x - goal.x, y - goal.y), abs(math.remainder(theta - goal.theta, math.tau))


def _assert_turned_home(loop, result, since):
    """The operator flows end with both rovers parked at home TURNED to the idle heading
    (east): inside the arrival tolerances of the last step of their idle turn, nothing left
    TRACKING / STOPPED, no contact, no pause or halt from ``since`` (the 'done' / the abandon)
    on."""
    assert _turned_home(loop), (result.mission_state, result.nav_states)
    for robot in ROVERS:
        goal = loop.engine.idle_turns.turns[robot].steps[-1]
        cfg = loop.engine.settings.robots[robot]
        dist, turn = _pose_error(loop, robot, goal)
        assert dist <= cfg.arrive_pos_tol and turn <= cfg.arrive_yaw_tol, (robot, dist, turn)
        assert abs(math.remainder(goal.theta - loop.engine.idle_turns.heading, math.tau)) <= \
            cfg.arrive_yaw_tol
    assert {r: loop.trace[-1][2][r].nav for r in ROVERS} == {r: 'PARKED' for r in ROVERS}
    assert not result.collisions, result.collisions
    stops = [e for e in loop.events if e[0] >= since and e[1] in MISSION_STOPS]
    assert not stops, stops


def _assert_arrived_before_turning(loop):
    """Each rover's idle turn started only once it stood PARKED at its home goal, facing west."""
    turning = {e[2].split(':')[0]: e[0] for e in loop.events if e[1] == 'idle_turn'}
    assert set(turning) == set(ROVERS), turning
    for robot, (key, goal) in _home_goals(loop).items():
        row = next(r for r in loop.trace if r[0] >= turning[robot])
        x, y, theta = row[2][robot].true
        cfg = loop.engine.settings.robots[robot]
        assert row[2][robot].nav == 'PARKED', (robot, row[0], row[2][robot].nav)
        assert math.hypot(x - goal.x, y - goal.y) <= cfg.arrive_pos_tol \
            and abs(math.remainder(theta - goal.theta, math.tau)) <= cfg.arrive_yaw_tol, \
            (robot, key, (x, y, theta))


def test_queue_clear_on_way_to_pick_keeps_trip(record_row):
    """Run to completion: clear removes the waiting part only; the running trip is
    delivered and both rovers park at home."""
    actions = [(ADD, lambda eng, now: eng.dispatch('bolt_01')),
               (CLEAR, lambda eng, now: eng.control('clear_queue', now))]
    loop, result = _operator_lap(actions, LAP_LIMIT, until_turned=True)
    record_row('clear_mid_leg   ' + result.row())
    assert _state_before(loop.trace, CLEAR) == 'TO_PICKUP'
    assert ('enqueued', 'fetch_bolt_01#2') in [e[1:] for e in result.events]
    assert [it.id for it in loop.engine.queue.items] == []
    assert not [e for e in result.events if e[1] in ('loaded', 'delivered') and 'bolt' in e[2]]
    assert result.loaded == 1
    assert_clean_laps(result, 1)
    _assert_arrived_before_turning(loop)
    _assert_turned_home(loop, result, loop._done_at)


def test_trip_abandoned_after_pause_sends_home(record_row):
    """Pause mid-leg, abandon: nothing delivered, both rovers drive home, park, then turn in place
    to face east; mission idle."""
    actions = [(PAUSE, lambda eng, now: eng.control('pause', now)),
               (ABANDON, lambda eng, now: eng.control('abandon', now))]
    loop, result = _operator_lap(actions, ABANDON + HOMING_LIMIT, until_turned=True)
    record_row('abandon_mid_leg ' + result.row())
    assert _state_before(loop.trace, PAUSE) == 'TO_PICKUP'
    assert _state_before(loop.trace, ABANDON) == 'PAUSED'
    assert [e[2] for e in result.events if e[1] == 'abandoned'] == ['not delivered: fetch_gear_01#1']
    assert result.loaded == 0 and result.delivered == 0
    _assert_arrived_before_turning(loop)
    _assert_turned_home(loop, result, ABANDON)


@pytest.mark.parametrize('edge,part', _edge_cases())
def test_lap_with_arena_edge(edge, part, record_row):
    result = run_laps([part], border_top=EDGES[edge], lap_limit=EDGE_LAP_LIMIT)
    record_row(f'edge {edge:10s} ' + result.row())
    record_row('    ' + summary(result))
    assert_clean_laps(result, 1)
    assert_lap_checks(result, front_stop=edge == 'pipe_0.05')
