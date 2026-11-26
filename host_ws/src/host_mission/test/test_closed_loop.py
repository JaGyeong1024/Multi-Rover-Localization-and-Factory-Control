"""Closed loop in the 2D simulator (host_sim.scenario.run_scenario, no ROS).

The rover side is the test-only ``IdealTracker`` (plain pure pursuit, see
sim_harness.py). These tests check the HOST's output: paths that, when followed,
end inside the goal tolerances without the two rovers touching.
"""
import copy

from host_sim.config import load_config
from host_sim.core.world import World
from host_sim.scenario import Goal, run_scenario

from host_mission.path_commander import Desired, PathCommander
from host_mission.world_model import WorldModel

import helpers
from sim_harness import IdealTracker, MissionLoop

SEED = 1
SLIP_FREE = dict(forward_efficiency=1.0, tread_ratio=1.0, icr_offset_x=0.0,
                 omega_quantum=0.0, omega_deadband=0.0, ref_offset_x=0.0)


def _sim_world(slip_free):
    """Default simulator world; ``slip_free`` removes the base imperfections that a
    real tracker (not the host) has to cope with. Sensor noise and delay stay."""
    config = copy.deepcopy(load_config())
    # the arena these tests' frozen settings and goals (test/data) are written in, not the
    # site layout that sim_default.yaml follows
    config['arena'].update(x_min=1.24, x_max=6.03, y_min=1.97, y_max=4.66)
    config['robots']['leader']['start'] = [2.6, 3.3, 0.0]
    config['robots']['follower']['start'] = [1.9, 3.3, 0.0]
    for robot in config['robots'].values():
        # IdealTracker commands the base directly; the real leader's tracker flips its turn
        # (rover angular_sign -1) for a base that turns the other way
        robot['base']['hw_angular_sign'] = 1.0
        if slip_free:
            robot['base'].update(SLIP_FREE)
    return World.from_config(config, seed=SEED)


class PlanOnlyHost:
    """Smallest host loop around the real PathCommander: feeds poses, tells it when
    the leader has parked (that releases the follower's gap request)."""

    def __init__(self, trackers, want):
        self.trackers, self.want = trackers, want
        self.world = WorldModel(pose_timeout=0.5, goals=helpers.goals())
        self.commander = PathCommander(helpers.settings(), helpers.EPOCH)

    def controllers(self):
        return {name: (lambda obs, n=name: self._control(obs, n)) for name in self.trackers}

    def _control(self, obs, name):
        if name == 'leader':
            self._host_step(obs)
        return self.trackers[name].control(obs)

    def _host_step(self, obs):
        if any(sample is None for sample in obs.poses.values()):
            return
        for name, sample in obs.poses.items():
            helpers.place(self.world, name, sample.x, sample.y, sample.theta, obs.t)
        for name, tracker in self.trackers.items():   # NavFeedback: the commander needs it to
            if tracker.cmd is not None:                # see a rover at its hold point (via)
                self.world.update_nav(name, tracker.state, tracker.detail, tracker.epoch,
                                      tracker.cmd_id, obs.t)
        arrived = {name: tracker.state in ('HOLDING', 'PARKED')
                   for name, tracker in self.trackers.items()}
        self.commander.update(self.want, self.world, arrived, obs.t)
        for name, cmd in self.commander.cmds.items():
            self.trackers[name].accept(cmd)


def test_planned_paths_reach_goals_clean():
    world = _sim_world(slip_free=True)
    trackers = {'leader': IdealTracker('leader'), 'follower': IdealTracker('follower')}
    store, cfg = helpers.goals(), helpers.settings()
    targets = {'leader': 'pick_gear', 'follower': 'load_gear'}
    want = {r: Desired(g, store.get(g), follow=True) for r, g in targets.items()}
    host = PlanOnlyHost(trackers, want)
    sim_goals = {r: Goal(store.get(g).x, store.get(g).y, store.get(g).theta,
                         pos_tol=cfg.robots[r].arrive_pos_tol,
                         yaw_tol=cfg.robots[r].arrive_yaw_tol)
                 for r, g in targets.items()}

    result = run_scenario(world, host.controllers(), duration=120.0, goals=sim_goals)

    cmds = host.commander.cmds
    assert cmds['follower'].cmd_id >= 2                 # shared first, released at the end
    for robot in targets:
        outcome = result.robots[robot]
        assert outcome.arrived, (robot, outcome.final_pos_error, outcome.final_yaw_error)
        assert outcome.final_pos_error <= cfg.robots[robot].arrive_pos_tol
        assert outcome.final_yaw_error <= cfg.robots[robot].arrive_yaw_tol
        assert trackers[robot].state == 'PARKED'
    assert not result.collided, [c.describe() for c in result.collisions]
    assert result.min_robot_gap > 0.05


def test_full_mission_in_the_simulator():
    """Two parts, default simulator world (slip, noise, delay): dispatch -> pick x2 ->
    deliver -> unload -> home, driven by the real MissionEngine."""
    world = _sim_world(slip_free=False)
    engine = helpers.engine(pick_cooldown=0.0)
    loop = MissionLoop(engine, {'leader': IdealTracker('leader'),
                                'follower': IdealTracker('follower')})
    assert engine.dispatch('gear_01') and engine.dispatch('bolt_01')

    result = run_scenario(world, loop.controllers(), duration=260.0, stop_on_arrival=False)

    kinds = [kind for kind, _detail in loop.events]
    assert engine.seq.state == 'DONE', (loop.states, loop.events[-5:])
    assert kinds.count('loaded') == 2 and kinds.count('delivered') == 2
    assert loop.picker.picks == 2
    assert 'halt' not in kinds and 'phase_timeout' not in kinds
    assert not result.collided, [c.describe() for c in result.collisions]
    assert result.min_robot_gap > 0.05
    assert loop.states[:3] == ['IDLE', 'TO_PICKUP', 'LOADING'] and loop.states[-1] == 'IDLE'
