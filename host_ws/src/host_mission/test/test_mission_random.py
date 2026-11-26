"""Random operator / arm / rover sequences against the mission engine's standing rules.

The mission state lives in a phase plus several flags (task_sequencer), so a defect shows
as a COMBINATION nobody wrote a test for. Each seed is one random walk over everything
that can reach the engine: dispatches, every operator control, the e-stop line, the arm's
two latched topics (also at the wrong time) and rovers arriving. After every tick the rules
below must hold; at the end a cooperative world (rovers arrive, picks succeed, the operator
presses what is offered) must bring the mission home with an empty queue.

No simulator: rovers are teleported onto their goals (helpers.Clock), time steps are 0.2 s."""
import random

import pytest

import helpers
from helpers import Clock, report

PARTS = ('gear_01', 'bolt_01')
HOME = {'leader': 'leader_standby', 'follower': 'follower_unload'}
PHASES = ('TO_PART', 'PICKING', 'DELIVERING')
STEPS = 70
SEEDS = range(8)  # ~25 s each (real planner)


class Walk:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        # short waits so that a walk also runs into the pick deadline and the phase timeout
        self.eng = helpers.engine(pick_dwell=4.0, pick_cooldown=1.0, phase_timeout=14.0)
        self.clock = Clock(self.eng, start={'leader': (0.86, 1.45, 3.14159),
                                            'follower': (0.86, 2.05, 3.14159)})
        self.trace = []
        self.driven = False     # a rover was sent somewhere (else staying put is the end)

    # ---------- what can happen ----------
    def actions(self):
        eng, clock, rng = self.eng, self.clock, self.rng
        ids = [it.id for it in eng.queue.items] or ['fetch_none#0']
        return [
            (4, 'dispatch', lambda: eng.dispatch(rng.choice(PARTS))),
            (2, 'pause', lambda: eng.control('pause', clock.now)),
            (3, 'resume', lambda: eng.control('resume', clock.now)),
            (2, 'abandon', lambda: eng.control('abandon', clock.now)),
            (1, 'clear_queue', lambda: eng.control('clear_queue', clock.now)),
            (1, 'cancel', lambda: eng.control('cancel:' + rng.choice(ids), clock.now)),
            (1, 'align', lambda: eng.control('align', clock.now)),
            (1, 'estop on', lambda: eng.estop(True, clock.now)),
            (2, 'estop off', lambda: eng.estop(False, clock.now)),
            (3, 'unload_done', eng.unload_done),
            (6, 'leader arrives', lambda: self.arrive('leader')),
            (6, 'follower arrives', lambda: self.arrive('follower')),
            (3, 'arm PICKING', lambda: eng.on_work_state('PICKING')),
            (3, 'arm DONE true', lambda: self.arm_result('DONE', True)),
            (2, 'arm ERROR false', lambda: self.arm_result('ERROR', False)),
            (1, 'arm stale true', lambda: eng.on_pick_done(True)),
            (8, 'wait', lambda: None),
        ]

    def arrive(self, robot):
        cmd = self.eng.commands().get(robot)
        if cmd is not None and cmd.mode == 'FOLLOW' and cmd.poses:
            self.clock.move_to_goal(robot)

    def arm_result(self, state, ok):
        self.eng.on_work_state(state)
        self.eng.on_pick_done(ok)

    def step(self):
        choices = self.actions()
        _weight, name, act = self.rng.choices(choices, weights=[c[0] for c in choices])[0]
        ticks = self.rng.choice((1, 1, 2, 5, 12, 30))
        self.trace.append(f'{self.clock.now:6.1f} {name} +{ticks}')
        act()
        for _ in range(ticks):
            before = self.eng.seq.phase
            events = self.clock.tick()
            self.check(before, events)

    # ---------- the standing rules ----------
    def check(self, phase_before, events):
        eng, seq = self.eng, self.eng.seq
        where = '\n'.join(self.trace[-25:])
        modes = {robot: cmd.mode for robot, cmd in eng.commands().items()}
        self.driven = self.driven or 'FOLLOW' in modes.values()
        active = [it for it in eng.queue.items if it.status == 'active']
        if seq.state == 'PAUSED':
            assert set(modes.values()) == {'STOP'}, f'a stopped mission drives {modes}\n{where}'
        if seq.state in ('IDLE', 'DONE'):
            assert seq.phase is None and not seq.batch and not active, \
                f'idle with trip leftovers: {seq.phase} {seq.batch} {active}\n{where}'
        if seq.state == 'RUNNING':
            assert seq.phase in PHASES, f'running without a phase\n{where}'
        if seq.phase in ('TO_PART', 'PICKING'):
            assert 0 <= seq.bidx < len(seq.batch), \
                f'{seq.phase} with no part to pick ({seq.bidx}/{len(seq.batch)})\n{where}'
        assert {id(it) for it in active} == {id(it) for it in seq.batch}, \
            f'trip {[it.id for it in seq.batch]} != active rows {[it.id for it in active]}\n{where}'
        if seq.system_state() == 'UNLOADING':
            assert any(it.part_state == 'LOADED' for it in seq.batch), \
                f'waiting for the unload button with nothing loaded\n{where}'
        if seq.phase != 'PICKING':
            assert not any(cmd.pick_at_goal for cmd in eng.commands().values()), \
                f'the arm may start outside the pick phase ({seq.phase})\n{where}'
        for kind, _detail in events:
            if kind in ('loaded', 'pick_failed'):
                assert phase_before == 'PICKING', f'{kind} outside the pick phase\n{where}'
        if 'abandon' in eng.offered_controls():
            assert seq.state == 'PAUSED' and seq.phase != 'PICKING', \
                f'abandon offered in {seq.state}/{seq.phase}\n{where}'

    # ---------- a cooperative world brings it home ----------
    def finish(self):
        eng, clock = self.eng, self.clock
        eng.estop(False, clock.now)
        for _ in range(400):
            state = eng.seq.system_state()
            if state in ('PAUSED', 'FAULT', 'ESTOP'):
                eng.control('resume', clock.now)
            elif state == 'LOADING':
                report(eng, 'leader', 'HOLDING', clock.now)
                eng.on_work_state('PICKING')
                clock.tick()
                self.arm_result('DONE', True)
            elif state == 'UNLOADING':
                eng.unload_done()
            for robot in helpers.ROBOTS:
                self.arrive(robot)
            before = eng.seq.phase
            self.check(before, clock.tick(3))
            home = all(eng.arrived.values()) \
                and {r: c.goal_key for r, c in eng.commands().items()} == HOME
            if state == 'IDLE' and not eng.queue.items and (home or not self.driven):
                return
        where = '\n'.join(self.trace[-25:])
        pytest.fail(f'not home with an empty queue: {eng.seq.system_state()} / {eng.seq.phase}, '
                    f'queue {[(it.id, it.status, it.part_state) for it in eng.queue.items]}, '
                    f'arrived {eng.arrived}, '
                    f'goals {[(r, c.mode, c.goal_key) for r, c in eng.commands().items()]}\n{where}')


@pytest.mark.parametrize('seed', SEEDS)
def test_random_operation_keeps_rules_ends_home(seed):
    walk = Walk(seed)
    for _ in range(STEPS):
        walk.step()
    walk.finish()
