"""Host <-> rover link faults in a full lap (empty layout, gear_01, seed 1, site profile).

The link (layout_runner.CommLink, ``LayoutLoop(comm=...)``) carries, host -> rover: path
commands (on change + 2 Hz keep-alive), each rover's own pose / quality / heading_aligned
and the lead's pose; rover -> host: nav_feedback + safety_state (and the mapper's scans,
outage only). ``delay`` applies to every message, an outage window drops everything in both
directions. Nothing is tuned here: a lap that breaks an assertion is a recorded finding.

Per lap the table row gives: lap time against the clean lap, the mission events from the
fault on, whether each rover stood still from the rover's staleness guard to the end of the
outage, the longest true path a rover drove between two path commands that reached it, the
smallest rover-rover gap and the collisions.

Asserted: no collision; in an outage longer than the rover-side staleness guard both rovers
stand; the mission completes (delivered, both rovers PARKED) or stops with a recorded cause."""
import functools
import math

import pytest
from host_sim.scenario import run_scenario

from lap_checks import (MISSION_STOPS, arrival_times, final_goal_errors, goal_legs,
                        min_centre_distance)
from layout_runner import LayoutLoop, _Finished

PROFILE = 'field_site'
LAP_LIMIT = 150.0               # s, as test_stage1_site_conditions
ROVERS = ('leader', 'follower')
FAULT_KINDS = ('paused', 'resumed', 'resume_deferred', 'hold', 'hold_cleared', 'plan_failed',
               'reenter', 'replan', 'add_obstacle', 'fault', 'halt', 'phase_timeout', 'aborted')
STILL_TOL = 0.01                # m a rover may creep inside the window where it must stand
SETTLE = 0.3                    # s after the staleness guard (+ link delay) before it must stand
RUN_IN = 0.6                    # m before the pick pose that make up the leader's run-in
OUTAGE_S = 3.0
MID_WAY = 20.0                  # s, on the way to the pick (clean lap: pick pose reached 37 s)


def run_lap(comm=None, lap_limit=LAP_LIMIT):
    loop = LayoutLoop('empty', 1, profile=PROFILE, comm=comm)
    assert loop.engine.dispatch('gear_01')
    try:
        run_scenario(loop.world, loop.controllers(), duration=lap_limit, stop_on_arrival=False)
    except _Finished:
        pass
    return loop, loop.result('empty', 'gear_01')


@functools.lru_cache(maxsize=None)
def clean_lap():
    """(result, start of the leader's run-in to the pick) of the lap without a fault."""
    _, result = run_lap(None)
    legs, arrived = arrival_times(result, 'leader')
    goal = next(c for (r, _, _), c in result.host_cmds.items()
                if r == 'leader' and c.mode == 'FOLLOW' and c.pick_at_goal).poses[-1]
    t_arrive = arrived[legs[0]]
    t_run_in = next(s.t for s in result.series['leader']
                    if s.t <= t_arrive and math.hypot(s.x - goal.x, s.y - goal.y) <= RUN_IN)
    return result, t_run_in


def guard_s(loop):
    """Rover-side staleness guard: the shortest of the tracker's command / pose timeouts."""
    cfg = loop.stacks['leader'].ctrl._cfg
    return min(cfg.cmd_timeout_s, cfg.pose_timeout_s)


def travelled(rows, t0, t1):
    """True path length of a rover between two times."""
    pts = [(s.x, s.y) for s in rows if t0 <= s.t <= t1]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def spread(rows, t0, t1):
    """Largest distance of the true pose from its first position in t0..t1 (m)."""
    pts = [(s.x, s.y) for s in rows if t0 <= s.t <= t1]
    return max((math.hypot(x - pts[0][0], y - pts[0][1]) for x, y in pts), default=0.0)


def longest_uncommanded(loop, result):
    """rover -> (path length [m], t from, t to) of the longest true travel between two path
    commands that reached the rover (the last one up to the end of the lap)."""
    out = {}
    for rover in ROVERS:
        rx = loop.link.cmd_rx.get(rover, [])
        edges = list(zip(rx, rx[1:])) + [(rx[-1], result.series[rover][-1].t)]
        out[rover] = max(((travelled(result.series[rover], a, b), a, b) for a, b in edges))
    return out


def lap_row(name, loop, result, clean, outage, delay):
    done = result.done_t if result.completed else math.nan
    t0 = outage[0] if outage else 0.0
    events = [(e[0], e[1], e[2]) for e in result.events if e[0] >= t0 and e[1] in FAULT_KINDS]
    lines = [f'{name:14s} done={done:.1f}s (clean {clean.done_t:.1f}s, {done - clean.done_t:+.1f}s) '
             f'delivered={result.delivered} collisions={len(result.collisions)} '
             f'min_robot_gap={result.min_robot_gap:.3f} min_centre={min_centre_distance(result):.3f} '
             f'mission={result.mission_state} nav={result.nav_states} dropped={loop.link.dropped}',
             f'    mission events from t={t0:g}: {events}']
    if not result.completed:
        errs = {r: (round(p, 3), round(y, 3)) for r, (p, y) in final_goal_errors(result).items()}
        lines.append(f'    NOT DONE: blocked={result.blocked} true_err_to_last_goal={errs} '
                     f'legs={ {r: goal_legs(result, r)[0] for r in ROVERS} }')
        lines.append(f'    last events: {[tuple(e) for e in result.events[-8:]]}')
    for rover, (dist, a, b) in longest_uncommanded(loop, result).items():
        lines.append(f'    {rover}: longest drive between two commands {dist:.3f} m '
                     f'({a:.1f}..{b:.1f} s)')
    if outage:
        t1 = outage[1]
        for rover in ROVERS:
            rows = result.series[rover]
            stop_at = next((round(s.t - t0, 2) for s in rows if s.t >= t0 and s.v == 0.0), None)
            lines.append(
                f'    {rover} in outage {outage}: moved {travelled(rows, t0, t1):.3f} m, '
                f'first v=0 {stop_at} s after the start, drift in the last '
                f'{t1 - (t0 + guard_s(loop) + delay + SETTLE):.1f} s '
                f'{spread(rows, t0 + guard_s(loop) + delay + SETTLE, t1):.3f} m')
    return '\n'.join(lines)


def check_lap(loop, result, outage, delay):
    assert not result.collisions, result.collisions
    if outage and outage[1] - outage[0] > guard_s(loop):
        t_still = outage[0] + guard_s(loop) + delay + SETTLE
        for rover in ROVERS:
            creep = spread(result.series[rover], t_still, outage[1])
            assert creep <= STILL_TOL, (rover, f'moved {creep:.3f} m in {t_still:.1f}..{outage[1]:g} s')
    stops = [e for e in result.events if e[1] in MISSION_STOPS]
    if result.completed:
        assert result.delivered == 1
        assert result.nav_states == {'leader': 'PARKED', 'follower': 'PARKED'}, result.nav_states
    else:
        assert stops, f'neither done nor a recorded stop: {result.reason()}'


def case_params():
    cases = [(f'delay_{d}', d, None) for d in (0.1, 0.3, 1.0)]
    cases += [(f'outage_{s:g}s', 0.1, (MID_WAY, MID_WAY + s)) for s in (1.0, 3.0, 8.0)]
    cases.append(('outage_run_in', 0.1, 'run_in'))
    return [pytest.param(name, delay, outage, id=name) for name, delay, outage in cases]


@pytest.mark.parametrize('name,delay,outage', case_params())
def test_lap_with_link_fault(name, delay, outage, record_row):
    clean, t_run_in = clean_lap()
    if outage == 'run_in':
        outage = (round(t_run_in, 1), round(t_run_in, 1) + OUTAGE_S)
    comm = {'delay': delay, 'outages': [outage] if outage else []}
    loop, result = run_lap(comm, lap_limit=LAP_LIMIT + (outage[1] - outage[0] if outage else 0.0)
                           + 60.0)
    record_row(lap_row(name, loop, result, clean, outage, delay))
    check_lap(loop, result, outage, delay)
