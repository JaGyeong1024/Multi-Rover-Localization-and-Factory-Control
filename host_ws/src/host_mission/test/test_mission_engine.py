"""MissionEngine: the path-mode mission end to end at the logic level (no ROS, no
simulator — rovers are teleported onto their goals)."""
import dataclasses
import math

import pytest

from host_mission.goals import GoalPose, GoalStore
from host_mission.mission_engine import MissionEngine

import helpers
from helpers import Clock, report


def _started(eng=None):
    """Engine with one dispatched part, ticked until both rovers have a path."""
    eng = eng or helpers.engine()
    clock = Clock(eng)
    assert eng.dispatch('gear_01')
    clock.tick(15)                               # start_grace (2 s) + first plan
    assert eng.seq.system_state() == 'TO_PICKUP'
    return eng, clock


def _at_pick(eng=None):
    """...and both rovers parked at the pick / load goals (mission in PICKING)."""
    eng, clock = _started(eng)
    clock.move_to_goal('leader')
    clock.move_to_goal('follower')
    clock.tick(2)
    assert eng.seq.system_state() == 'LOADING'
    return eng, clock


def _pick(eng, clock, ok=True):
    report(eng, 'leader', 'HOLDING', clock.now)
    eng.on_work_state('PICKING')
    clock.tick()
    eng.on_work_state('DONE' if ok else 'ERROR')
    eng.on_pick_done(ok)
    return clock.tick()


def test_rovers_stopped_until_dispatch():
    eng = helpers.engine()
    clock = Clock(eng)
    clock.tick(5)
    assert {cmd.mode for cmd in eng.commands().values()} == {'STOP'}
    assert all(cmd.epoch == helpers.EPOCH for cmd in eng.commands().values())


def test_dispatch_needs_known_part_goals():
    eng = helpers.engine(goal_store=GoalStore({}))
    assert not eng.dispatch('gear_01')
    assert not eng.dispatch('no_such_part')
    kinds = [kind for kind, _ in eng.tick(0.2)]
    assert kinds.count('dispatch_rejected') == 2 and eng.queue.items == []


def test_mission_sends_goals_as_shared_paths():
    eng, _clock = _started()
    leader, follower = eng.commands()['leader'], eng.commands()['follower']
    assert (leader.goal_key, follower.goal_key) == ('pick_gear', 'load_gear')
    assert leader.mode == follower.mode == 'FOLLOW'
    assert follower.lead_robot == 'leader' and follower.gap > 0.0


def test_pick_requested_when_both_arrived():
    eng, clock = _started()
    clock.move_to_goal('leader')
    clock.tick(2)
    assert eng.seq.system_state() == 'TO_PICKUP'
    assert not eng.commands()['leader'].pick_at_goal      # PARKED, the leader must not pick yet
    clock.move_to_goal('follower')
    clock.tick(2)
    assert eng.seq.system_state() == 'LOADING'
    assert eng.commands()['leader'].pick_at_goal          # now HOLDING -> leader picks


def test_position_alone_is_not_arrival():
    eng, clock = _started()
    for robot in helpers.ROBOTS:
        goal = eng.commands()[robot].poses[-1]
        clock.pose[robot] = (goal.x, goal.y, goal.theta)
        report(eng, robot, 'TRACKING', clock.now)
    clock.tick(3)
    assert eng.arrived == {'leader': False, 'follower': False}
    assert eng.seq.system_state() == 'TO_PICKUP'


def test_pick_result_completes_the_phase():
    eng, clock = _at_pick()
    events = _pick(eng, clock)
    assert ('loaded', 'fetch_gear_01#1') in events
    assert eng.seq.system_state() == 'RETURNING'
    assert eng.commands()['leader'].goal_key == 'leader_standby'


def test_latched_pick_replay_not_complete():
    eng = helpers.engine()
    eng.on_work_state('DONE')                    # leftovers of an earlier pick (latched)
    eng.on_pick_done(True)
    eng, clock = _at_pick(eng)
    eng.on_pick_done(True)
    clock.tick(5)
    assert eng.seq.system_state() == 'LOADING'
    assert ('loaded', 'fetch_gear_01#1') in _pick(eng, clock)


def test_failed_pick_removes_item():
    eng, clock = _at_pick()
    events = _pick(eng, clock, ok=False)
    assert ('pick_failed', 'fetch_gear_01#1') in events
    assert eng.queue.by_id('fetch_gear_01#1') is None
    assert eng.seq.system_state() == 'RETURNING'


def test_unanswered_pick_fails_after_dwell():
    eng, clock = _at_pick(helpers.engine(pick_dwell=3.0))
    assert eng.scenario.pick_timeout_is_failure is True
    clock.tick(20)
    assert ('pick_failed', 'fetch_gear_01#1') in clock.events
    assert eng.queue.by_id('fetch_gear_01#1') is None


def test_field_fallback_unanswered_pick_loaded():
    # pick_timeout_is_failure: false = the arm publishes no result; the timer stands in.
    eng, clock = _at_pick(helpers.engine(pick_dwell=3.0, pick_timeout_is_failure=False))
    clock.tick(1)
    assert eng.seq.system_state() == 'LOADING'
    clock.tick(20)
    assert ('loaded', 'fetch_gear_01#1') in clock.events
    assert ('pick_failed', 'fetch_gear_01#1') not in clock.events
    assert eng.queue.by_id('fetch_gear_01#1').status == 'active'
    assert eng.seq.system_state() == 'RETURNING'
    assert eng.commands()['leader'].goal_key == 'leader_standby'


def test_repeated_pick_failures_to_operator():
    eng = helpers.engine(capacity=2, pick_cooldown=0.0)
    eng.dispatch('bolt_01')
    eng, clock = _at_pick(eng)
    first = eng.seq.active_id
    _pick(eng, clock, ok=False)
    assert eng.seq.system_state() == 'TO_PICKUP' and eng.seq.active_id != first
    clock.move_to_goal('leader')
    clock.tick(12)                               # the follower yielded until the leader parked
    clock.move_to_goal('follower')
    clock.tick(2)
    events = _pick(eng, clock, ok=False)
    assert ('halt', '2 consecutive pick failures') in events
    assert eng.seq.system_state() == 'FAULT'
    assert {cmd.mode for cmd in eng.commands().values()} == {'STOP'}


def test_pick_failure_drops_part_delivers_other():
    eng = helpers.engine(capacity=2, pick_cooldown=0.0)
    eng.dispatch('bolt_01')
    eng, clock = _at_pick(eng)
    failed = eng.seq.active_id
    events = _pick(eng, clock, ok=False)
    assert ('pick_failed', failed) in events
    assert eng.queue.by_id(failed) is None and len(eng.queue.items) == 1
    other = eng.queue.items[0].id
    assert eng.seq.system_state() == 'TO_PICKUP' and eng.seq.active_id == other
    clock.move_to_goal('leader')
    clock.tick(12)
    clock.move_to_goal('follower')
    clock.tick(2)
    assert ('loaded', other) in _pick(eng, clock)
    assert eng.seq.system_state() == 'RETURNING'
    clock.move_to_goal('leader')
    clock.tick(12)
    clock.move_to_goal('follower')
    clock.tick(2)
    assert eng.seq.system_state() == 'UNLOADING'
    assert eng.unload_done()
    assert ('delivered', other) in clock.tick(3)
    assert eng.seq.state == 'DONE' and eng.queue.items == []


def test_full_mission_reaches_done():
    eng, clock = _at_pick()
    _pick(eng, clock)
    clock.move_to_goal('leader')                 # leader home; the follower yielded so far
    clock.tick(12)
    clock.move_to_goal('follower')
    clock.tick(2)
    assert eng.seq.system_state() == 'UNLOADING'
    assert eng.unload_done()
    clock.tick(3)
    assert eng.seq.state == 'DONE' and eng.queue.items == []
    assert 'done' in clock.kinds()


def test_off_path_replans_then_halts():
    eng, clock = _started()
    limit = eng.settings.supervisor.off_path_max_replans
    for attempt in range(1, limit + 1):
        before = eng.commands()['leader'].cmd_id
        report(eng, 'leader', 'STOPPED', clock.now, 'OFF_PATH')
        events = clock.tick()
        assert ('replan', f'leader: OFF_PATH ({attempt}/{limit})') in events
        assert eng.commands()['leader'].cmd_id == before + 1
    report(eng, 'leader', 'STOPPED', clock.now, 'OFF_PATH')
    events = clock.tick()
    assert any(kind == 'halt' and 'OFF_PATH' in detail for kind, detail in events)
    assert eng.seq.system_state() == 'FAULT'
    assert eng.commands()['leader'].mode == 'STOP'
    eng.control('resume', clock.now)
    clock.tick(2)
    assert eng.seq.system_state() == 'TO_PICKUP' and eng.commands()['leader'].mode == 'FOLLOW'


def test_goal_miss_sends_a_reentry_path():
    eng, clock = _started()
    before = eng.commands()['leader'].cmd_id
    report(eng, 'leader', 'STOPPED', clock.now, 'GOAL_MISSED')
    events = clock.tick()
    assert ('reenter', 'leader: GOAL_MISSED (1/2)') in events
    assert eng.commands()['leader'].cmd_id == before + 1
    assert not eng.arrived['leader']


def test_quality_lost_pauses_self_resumes():
    eng, clock = _started()
    cmd_id = eng.commands()['leader'].cmd_id
    clock.tick(5)
    for _ in range(40):                          # 8 s of LOST
        eng.world.update_quality('leader', 'LOST')
        clock.now += clock.dt
        for robot, pose in clock.pose.items():
            eng.world.update_pose(robot, *pose, clock.now)
        clock.events += eng.tick(clock.now)
    assert 'paused' in clock.kinds() and eng.seq.system_state() == 'PAUSED'
    assert eng.commands()['leader'].mode == 'STOP'
    clock.tick(3)
    assert 'resumed' in clock.kinds() and eng.seq.system_state() == 'TO_PICKUP'
    resumed = eng.commands()['leader']
    assert resumed.mode == 'FOLLOW' and resumed.cmd_id > cmd_id + 1


def test_lidar_block_registers_box_region():
    eng, clock = _started()
    cfg = eng.settings.supervisor

    def stopped_at_face(seconds, face):
        for _ in range(int(seconds / clock.dt)):
            report(eng, 'leader', 'STOPPED', clock.now, 'LIDAR_STOP')
            eng.world.update_safety('leader', 'STOPPED', ['LIDAR_STOP'], 0.0, *face, clock.now)
            clock.tick()

    stopped_at_face(cfg.lidar_block_s + 0.4, (3.1, 3.3))
    assert 'add_obstacle' in clock.kinds() and 'replan' in clock.kinds()
    # the face point is the front of an assumed box: that exact 0.5 x 0.5 square
    [region] = [o.shape for o in eng.world.obstacles()]
    assert region[:5] == pytest.approx((3.1 + cfg.assumed_obstacle_depth / 2, 3.3, 0.0,
                                        cfg.assumed_obstacle_depth,
                                        cfg.assumed_obstacle_width))
    # still stopped there (a slightly different face point) past the TTL: refreshed, not
    # stacked and not expired
    stopped_at_face(cfg.lidar_obstacle_ttl + 1.0, (3.12, 3.25))
    assert [o.shape for o in eng.world.obstacles()] == [region]
    assert clock.kinds().count('add_obstacle') == 1
    clock.tick(int(cfg.lidar_obstacle_ttl / clock.dt) + 2)
    assert eng.world.obstacles() == []


def test_front_stop_at_peer_registers_no_box():
    """Abandoned right after the start, both rovers loop back to homes 0.6 m apart: the
    leader's front stop at the parked follower must not become an assumed 0.5 m box over both
    homes. The other rover is in every plan already; the stopped rover is still re-planned."""
    half_band = 0.19                                        # shipped leader front_zone
    base = helpers.settings()
    leader = dataclasses.replace(base.robots['leader'], plan=dataclasses.replace(
        base.robots['leader'].plan, front_zone=(0.045, 0.59, half_band)))
    settings = dataclasses.replace(base, robots={**base.robots, 'leader': leader})
    eng, clock = _started(MissionEngine(settings, helpers.scenario(), helpers.goals(),
                                        helpers.EPOCH))
    cfg = eng.settings.supervisor
    assert half_band > cfg.fixture_match_dist + 0.05        # else the offset below shows nothing
    # The follower stands at the edge of the leader's sensed band: the leader reports the
    # point on its sensor axis, nearly the band's half width beside the follower's body.
    reach = eng.settings.robots['follower'].plan.footprint[2]
    clock.pose['follower'] = (3.3, 3.3 + reach + half_band - 0.02, 0.0)
    for _ in range(int((cfg.lidar_block_s + 0.4) / clock.dt)):
        report(eng, 'leader', 'STOPPED', clock.now, 'LIDAR_STOP')
        eng.world.update_safety('leader', 'STOPPED', ['LIDAR_STOP'], 0.0, 3.3, 3.3, clock.now)
        clock.tick()
    assert eng.world.obstacles() == []
    assert 'add_obstacle' not in clock.kinds() and 'replan' in clock.kinds()


def test_new_obstacle_replans_path():
    eng, clock = _started()
    before = eng.commands()['leader']
    middle = before.poses[len(before.poses) // 2]
    eng.set_obstacles([(middle.x, middle.y, 0.15)])
    clock.tick()
    after = eng.commands()['leader']
    assert after.cmd_id == before.cmd_id + 1
    clearance = min(((p.x - middle.x) ** 2 + (p.y - middle.y) ** 2) ** 0.5 for p in after.poses)
    assert clearance >= 0.15 + eng.settings.robots['leader'].plan.required_clearance - 1e-6


def test_no_path_pauses_until_obstacle_gone():
    eng, clock = _started()
    goal = eng.world.goals.get('pick_gear')
    eng.set_obstacles([(goal.x, goal.y, 0.3)])
    clock.tick(3)
    assert 'plan_failed' in clock.kinds() and eng.seq.system_state() == 'PAUSED'
    assert eng.commands()['leader'].mode == 'STOP'
    eng.set_obstacles([])
    clock.tick(4)
    assert 'plan_recovered' in clock.kinds() and eng.seq.system_state() == 'TO_PICKUP'
    assert eng.commands()['leader'].mode == 'FOLLOW'


def test_mapped_obstacle_off_paths_no_reissue():
    """A box mapped away from both paths changes nothing the rovers drive (a phantom box must not re-issue both paths)."""
    eng, clock = _started()
    before = {robot: cmd.cmd_id for robot, cmd in eng.commands().items()}
    eng.set_mapped_obstacles([(2.0, 4.2, 0.10)])
    events = clock.tick(3)
    assert ('obstacles', '1 mapped') in events
    assert 'path_issued' not in [kind for kind, _ in events]
    assert {robot: cmd.cmd_id for robot, cmd in eng.commands().items()} == before


def test_mapped_obstacle_on_lead_path_reissues():
    eng, clock = _started()
    before = eng.commands()
    middle = before['leader'].poses[len(before['leader'].poses) // 2]
    eng.set_mapped_obstacles([(middle.x, middle.y, 0.15)])
    clock.tick()
    after = eng.commands()
    assert after['leader'].cmd_id > before['leader'].cmd_id
    assert after['follower'].cmd_id > before['follower'].cmd_id


def test_pathless_rover_gets_path_when_obstacle_goes():
    eng, clock = _started()
    goal = eng.world.goals.get('pick_gear')
    eng.set_mapped_obstacles([(goal.x, goal.y, 0.3)])
    clock.tick(3)
    assert 'plan_failed' in clock.kinds() and eng.commands()['leader'].mode == 'STOP'
    eng.set_mapped_obstacles([])
    clock.tick(4)
    assert 'plan_recovered' in clock.kinds() and eng.commands()['leader'].mode == 'FOLLOW'


def test_estop_stops_never_self_resumes():
    eng, clock = _started()
    eng.estop(True)
    clock.tick()
    assert {cmd.mode for cmd in eng.commands().values()} == {'STOP'}
    eng.estop(False)
    clock.tick(3)
    assert eng.seq.system_state() == 'ESTOP'
    eng.control('resume', clock.now)
    clock.tick(2)
    assert eng.commands()['leader'].mode == 'FOLLOW'


HOME = {'leader': 'leader_standby', 'follower': 'follower_unload'}
GEAR, BOLT = 'fetch_gear_01#1', 'fetch_bolt_01#2'


def _modes(eng):
    return {cmd.mode for cmd in eng.commands().values()}


def _drives_home(eng, clock):
    """The leader is sent home; once it has parked there the follower is too (it may
    yield to the leader's path first)."""
    leader = eng.commands()['leader']
    assert (leader.mode, leader.goal_key) == ('FOLLOW', HOME['leader']), leader
    clock.move_to_goal('leader')
    clock.tick(12)
    follower = eng.commands()['follower']
    assert (follower.mode, follower.goal_key) == ('FOLLOW', HOME['follower']), follower


def _both_parked_at_home(eng, clock):
    """Both rovers driven home after a trip was abandoned, parked there."""
    clock.move_to_goal('leader')
    clock.tick(12)
    clock.move_to_goal('follower')
    clock.tick(4)
    assert eng.arrived == {'leader': True, 'follower': True}


def test_later_trip_home_parking_not_at_pick():
    """Every trip after the first must not show LOADING while the leader is still driving to
    the pick: arrival belongs to the goal a command was issued for; the home commands of the
    previous trip, still reported as reached, say nothing about the new trip's goals."""
    eng, clock = _started()
    assert eng.control('pause', clock.now) and eng.control('abandon', clock.now)
    clock.tick(2)
    _both_parked_at_home(eng, clock)
    assert eng.dispatch('gear_01')
    for _ in range(40):
        clock.tick()
        assert eng.seq.system_state() != 'LOADING', clock.now
    assert eng.seq.system_state() == 'TO_PICKUP'
    assert 'pick_requested' not in clock.kinds()


def _arm_moving(eng, clock):
    report(eng, 'leader', 'HOLDING', clock.now)
    eng.on_work_state('PICKING')
    clock.tick()


def _picking(eng=None):
    """Mission in PICKING with the leader's arm reported moving."""
    eng, clock = _at_pick(eng)
    _arm_moving(eng, clock)
    return eng, clock


def _end_pick(eng, ok=True):
    eng.on_work_state('DONE' if ok else 'ERROR')
    eng.on_pick_done(ok)


def _with_waiting_part(eng=None):
    """Mission driving to the gear pick with a bolt fetch waiting in the queue."""
    eng, clock = _started(eng)
    assert eng.dispatch('bolt_01')
    clock.tick()
    assert [it.id for it in eng.queue.pending()] == [BOLT]
    return eng, clock


def _accepted(eng, clock, command):
    assert eng.control(command, clock.now), command
    return clock.tick()


def _rejected(eng, clock, command):
    """``command`` must be refused with exactly one control_rejected event; returns its
    reason (what the operator reads)."""
    assert not eng.control(command, clock.now), command
    reasons = [detail for kind, detail in clock.tick() if kind == 'control_rejected']
    assert len(reasons) == 1, reasons
    return reasons[0]


def _abandoned(events):
    return [detail for kind, detail in events if kind == 'abandoned']


def _lost(eng, clock, robot, seconds):
    """Feed poses with ``robot``'s localization quality LOST for ``seconds``."""
    for _ in range(int(seconds / clock.dt)):
        clock.now += clock.dt
        for name, pose in clock.pose.items():
            helpers.place(eng.world, name, *pose, clock.now,
                          quality='LOST' if name == robot else 'GOOD')
        clock.events += eng.tick(clock.now)


# rule 1: cancel
def test_cancel_removes_waiting_part():
    eng, clock = _with_waiting_part()
    leader = eng.commands()['leader']
    _accepted(eng, clock, f'cancel:{BOLT}')
    assert eng.queue.by_id(BOLT) is None and eng.queue.by_id(GEAR).status == 'active'
    assert eng.seq.system_state() == 'TO_PICKUP'
    assert eng.commands()['leader'].cmd_id == leader.cmd_id


def test_cancel_running_part_rejected_all_phases():
    eng, clock = _started()
    leader = eng.commands()['leader']
    assert 'running trip' in _rejected(eng, clock, f'cancel:{GEAR}')
    assert eng.commands()['leader'].cmd_id == leader.cmd_id
    clock.move_to_goal('leader')
    clock.move_to_goal('follower')
    clock.tick(2)
    assert eng.seq.system_state() == 'LOADING'
    assert 'running trip' in _rejected(eng, clock, f'cancel:{GEAR}')
    _pick(eng, clock)
    assert eng.seq.system_state() == 'RETURNING'
    assert 'running trip' in _rejected(eng, clock, f'cancel:{GEAR}')
    _accepted(eng, clock, 'pause')
    assert 'running trip' in _rejected(eng, clock, f'cancel:{GEAR}')
    assert eng.queue.by_id(GEAR).status == 'active'


def test_failed_part_not_cancellable():
    eng, clock = _at_pick()
    _pick(eng, clock, ok=False)
    assert eng.queue.by_id(GEAR) is None
    assert 'not in the queue' in _rejected(eng, clock, f'cancel:{GEAR}')


# rule 2: clear_queue
def test_clear_removes_waiting_parts():
    eng, clock = _with_waiting_part()
    leader = eng.commands()['leader']
    _accepted(eng, clock, 'clear_queue')
    assert [it.id for it in eng.queue.items] == [GEAR]
    assert eng.seq.system_state() == 'TO_PICKUP'
    assert eng.commands()['leader'].cmd_id == leader.cmd_id


def test_clear_leaves_only_running_trip():
    eng = helpers.engine(capacity=2, pick_cooldown=0.0)
    eng.dispatch('bolt_01')
    eng, clock = _at_pick(eng)
    assert eng.dispatch('gear_01')
    clock.tick()
    trip = [it.id for it in eng.queue.items if it.status == 'active']
    assert len(trip) == 2 and len(eng.queue.pending()) == 1
    _accepted(eng, clock, 'clear_queue')
    assert [it.id for it in eng.queue.items] == trip
    assert eng.seq.system_state() == 'LOADING'


def test_clear_during_pick_lets_pick_end():
    eng, clock = _with_waiting_part()
    clock.move_to_goal('leader')
    clock.move_to_goal('follower')
    clock.tick(2)
    assert eng.seq.system_state() == 'LOADING'
    _arm_moving(eng, clock)
    _accepted(eng, clock, 'clear_queue')
    clock.tick(5)
    assert eng.seq.system_state() == 'LOADING'
    assert eng.commands()['leader'].pick_at_goal
    _end_pick(eng)
    events = clock.tick(2)
    assert ('loaded', GEAR) in events
    assert eng.seq.system_state() == 'RETURNING' and eng.queue.by_id(BOLT) is None


def test_clear_without_trip_empties_queue():
    eng = helpers.engine()
    clock = Clock(eng)
    assert eng.dispatch('gear_01') and eng.dispatch('bolt_01')
    _accepted(eng, clock, 'clear_queue')
    clock.tick(15)
    assert eng.queue.items == [] and eng.seq.system_state() == 'IDLE'
    assert _modes(eng) == {'STOP'}


# rule 3: abandon
def test_abandon_rejected_while_running():
    eng, clock = _started()
    leader = eng.commands()['leader']
    assert 'stop first' in _rejected(eng, clock, 'abandon')
    assert eng.seq.system_state() == 'TO_PICKUP'
    assert eng.commands()['leader'].cmd_id == leader.cmd_id


def test_abort_all_is_no_longer_a_command():
    eng, clock = _started()
    _accepted(eng, clock, 'pause')
    assert 'unknown command' in _rejected(eng, clock, 'abort_all')
    assert eng.seq.system_state() == 'PAUSED'


def test_abandon_after_pause_sends_home():
    eng, clock = _started()
    _accepted(eng, clock, 'pause')
    events = _accepted(eng, clock, 'abandon')
    assert len(_abandoned(events)) == 1 and GEAR in _abandoned(events)[0]
    assert 'not delivered' in _abandoned(events)[0]
    assert eng.queue.by_id(GEAR) is None                   # no entry of the trip is kept
    assert eng.seq.system_state() == 'IDLE'
    _drives_home(eng, clock)


def test_abandon_plans_home_from_standing():
    eng, clock = _started()
    _midway(eng, clock, 'leader')
    clock.tick()
    _accepted(eng, clock, 'pause')
    _accepted(eng, clock, 'abandon')
    for _ in range(100):                        # the leader may yield to the follower first
        if eng.commands()['leader'].mode == 'FOLLOW':
            break
        if eng.commands()['follower'].mode == 'FOLLOW':
            clock.move_to_goal('follower')
        clock.tick()
    leader = eng.commands()['leader']
    assert (leader.mode, leader.goal_key) == ('FOLLOW', HOME['leader'])
    assert _starts_at(leader, clock.pose['leader'])


def test_abandon_keeps_waiting_parts():
    eng, clock = _with_waiting_part()
    _accepted(eng, clock, 'pause')
    _accepted(eng, clock, 'abandon')
    assert [it.id for it in eng.queue.items] == [BOLT]
    assert eng.queue.by_id(BOLT).status == 'pending'
    clock.tick(15)
    assert eng.seq.system_state() == 'TO_PICKUP'
    assert eng.commands()['leader'].goal_key == 'pick_bolt'


def test_abandon_under_estop_waits_for_resume():
    eng, clock = _started()
    eng.estop(True)
    clock.tick()
    _accepted(eng, clock, 'abandon')
    clock.tick(5)
    assert eng.seq.system_state() == 'ESTOP' and _modes(eng) == {'STOP'}
    eng.estop(False)
    clock.tick(10)
    assert eng.seq.system_state() == 'ESTOP' and _modes(eng) == {'STOP'}
    _accepted(eng, clock, 'resume')
    clock.tick(2)
    assert eng.seq.system_state() == 'IDLE'
    _drives_home(eng, clock)


# rule 6: no lockout - every stopped state has a way out, PICKING always ends
def test_pause_in_pick_ends_pick_then_abandon():
    eng, clock = _picking()
    _accepted(eng, clock, 'pause')
    assert 'pick and place in progress' in _rejected(eng, clock, 'abandon')
    _end_pick(eng)
    events = clock.tick(2)
    assert ('loaded', GEAR) in events
    assert eng.seq.system_state() == 'PAUSED' and _modes(eng) == {'STOP'}
    _accepted(eng, clock, 'abandon')
    clock.tick(2)
    _drives_home(eng, clock)                    # the follower carries the part home


def test_estop_in_pick_ends_at_dwell_then_abandon():
    eng, clock = _picking(helpers.engine(pick_dwell=3.0))
    eng.estop(True)
    clock.tick()
    assert 'pick and place in progress' in _rejected(eng, clock, 'abandon')
    clock.tick(20)
    assert ('pick_failed', GEAR) in clock.events and eng.seq.phase != 'PICKING'
    assert eng.seq.system_state() == 'ESTOP' and _modes(eng) == {'STOP'}
    _accepted(eng, clock, 'abandon')
    eng.estop(False)
    _accepted(eng, clock, 'resume')
    clock.tick(2)
    _drives_home(eng, clock)


def test_estop_in_pick_then_resume_continues():
    eng, clock = _picking()
    eng.estop(True)
    clock.tick()
    eng.estop(False)
    _accepted(eng, clock, 'resume')
    assert eng.seq.system_state() == 'LOADING'
    _end_pick(eng)
    assert ('loaded', GEAR) in clock.tick(2)
    assert eng.seq.system_state() == 'RETURNING'


def test_pick_fault_can_be_abandoned():
    eng = helpers.engine(pick_cooldown=0.0)
    eng.dispatch('bolt_01')
    eng, clock = _at_pick(eng)
    _pick(eng, clock, ok=False)
    clock.move_to_goal('leader')
    clock.tick(12)
    clock.move_to_goal('follower')
    clock.tick(2)
    _pick(eng, clock, ok=False)
    assert eng.seq.system_state() == 'FAULT'
    _accepted(eng, clock, 'abandon')
    clock.tick(2)
    assert eng.seq.system_state() == 'IDLE'
    _drives_home(eng, clock)


def test_no_path_pause_can_be_abandoned():
    eng, clock = _started()
    goal = eng.world.goals.get('pick_gear')
    eng.set_obstacles([(goal.x, goal.y, 0.3)])
    clock.tick(3)
    assert eng.seq.system_state() == 'PAUSED'
    _accepted(eng, clock, 'abandon')
    clock.tick(2)
    assert eng.seq.system_state() == 'IDLE'
    _drives_home(eng, clock)


def test_localization_lost_in_pick_then_abandon():
    eng, clock = _picking(helpers.engine(pick_dwell=10.0))
    _lost(eng, clock, 'leader', 7.0)
    assert eng.seq.system_state() == 'PAUSED' and eng.seq.phase == 'PICKING'
    assert not eng.control('abandon', clock.now)
    _lost(eng, clock, 'leader', 6.0)            # never recovers; the pick's dwell passes
    assert ('pick_failed', GEAR) in clock.events and eng.seq.phase != 'PICKING'
    assert eng.control('abandon', clock.now)
    _lost(eng, clock, 'leader', 1.0)
    assert eng.seq.system_state() == 'IDLE'
    clock.tick(2)
    _drives_home(eng, clock)


# rule 4: unchanged
def test_rovers_stay_put_at_startup():
    eng = helpers.engine()                      # idle_go_home: false
    clock = Clock(eng)
    eng.estop(True)
    eng.estop(False)
    eng.control('resume', clock.now)
    clock.tick(5)
    assert _modes(eng) == {'STOP'}              # no mission has run yet: nothing to return from


def _unaligned(eng, robots=('leader', 'follower')):
    clock = Clock(eng)
    for robot in robots:
        clock.aligned[robot] = False
    clock.tick(5)
    return clock


def test_align_sends_rovers_home_first():
    eng = helpers.engine()                      # idle_go_home: false
    # a straight stretch short of home (on its home spot, with the follower right behind,
    # this floor has no loop back for the leader)
    clock = Clock(eng, start={'leader': (1.5, 3.3, 0.0), 'follower': (1.0, 2.6, 0.0)})
    clock.aligned = {'leader': False, 'follower': False}
    clock.tick(5)
    assert _modes(eng) == {'STOP'}
    assert eng.offered_controls() == ['align']
    _accepted(eng, clock, 'align')
    clock.tick(10)
    leader = eng.commands()['leader']           # planned from a provisional heading: the
    assert (leader.mode, leader.goal_key) == ('FOLLOW', HOME['leader'])
    assert leader.provisional
    clock.aligned['leader'] = True              # the creep gave the host the heading
    clock.tick(10)
    assert not eng.commands()['leader'].provisional
    assert (eng.commands()['leader'].mode, eng.commands()['leader'].goal_key) \
        == ('FOLLOW', HOME['leader'])


def test_align_drive_one_rover_leader_first():
    eng = helpers.engine()                      # both a straight stretch short of home
    clock = Clock(eng, start={'leader': (1.5, 3.3, 0.0), 'follower': (1.0, 2.6, 0.0)})
    clock.aligned = {'leader': False, 'follower': False}
    clock.tick(5)
    _accepted(eng, clock, 'align')
    clock.tick(10)
    cmds = eng.commands()
    assert cmds['leader'].mode == 'FOLLOW' and cmds['follower'].mode == 'STOP'
    clock.aligned['leader'] = True              # its creep gave the host the heading
    clock.tick(10)
    assert eng.commands()['follower'].mode == 'STOP'        # the leader is not home yet
    clock.move_to_goal('leader')
    clock.tick(10)
    cmds = eng.commands()
    assert (cmds['follower'].mode, cmds['follower'].goal_key) == ('FOLLOW', HOME['follower'])
    assert cmds['leader'].mode == 'FOLLOW'      # parked at home, not released from it
    clock.aligned['follower'] = True
    clock.move_to_goal('follower')
    clock.tick(10)
    assert 'align' not in eng.offered_controls()
    assert {c.mode for c in eng.commands().values()} == {'FOLLOW'}


def test_align_offered_with_one_unaligned():
    eng = helpers.engine()
    clock = _unaligned(eng, robots=('follower',))
    assert eng.offered_controls() == ['align']
    _accepted(eng, clock, 'align')


def test_align_refused_when_aligned():
    eng = helpers.engine()
    clock = Clock(eng)
    clock.tick(5)
    assert 'align' not in eng.offered_controls()
    assert 'aligned' in _rejected(eng, clock, 'align')
    clock.tick(5)
    assert _modes(eng) == {'STOP'}


def test_align_refused_running_or_estop():
    eng, clock = _started()
    clock.aligned['follower'] = False
    clock.tick()
    assert 'align' not in eng.offered_controls()
    assert 'running' in _rejected(eng, clock, 'align')
    idle = helpers.engine()
    idle_clock = _unaligned(idle)
    idle.estop(True)
    assert 'align' not in idle.offered_controls()
    assert 'e-stop' in _rejected(idle, idle_clock, 'align')


def test_abandon_offered_only_if_acceptable():
    eng, clock = _started()
    assert 'abandon' not in eng.offered_controls()
    _accepted(eng, clock, 'pause')
    assert 'abandon' in eng.offered_controls()
    eng.seq.phase = 'PICKING'
    assert 'abandon' not in eng.offered_controls()
    assert not eng.control('abandon', clock.now)


def test_early_resume_applied_on_release():
    eng, clock = _started()
    eng.estop(True)
    clock.tick()
    eng.control('resume', clock.now)            # 'Release & Resume': two topics, any order
    eng.estop(False)
    clock.tick(2)
    assert eng.seq.system_state() == 'TO_PICKUP'
    assert _modes(eng) == {'FOLLOW'}


def test_long_early_resume_not_applied():
    eng, clock = _started()
    eng.estop(True)
    clock.tick()
    eng.control('resume', clock.now)
    clock.tick(10)                              # 2 s > estop_resume_window
    eng.estop(False)
    clock.tick(3)
    assert eng.seq.system_state() == 'ESTOP' and _modes(eng) == {'STOP'}


def _midway(eng, clock, robot):
    """Put ``robot`` halfway along its current path (where it stands when stopped)."""
    poses = eng.commands()[robot].poses
    mid = poses[len(poses) // 2]
    clock.pose[robot] = (mid.x, mid.y, mid.theta)
    return mid


def _starts_at(cmd, pose, tol=0.01):
    first = cmd.poses[0]
    return math.hypot(first.x - pose[0], first.y - pose[1]) <= tol


def test_resume_after_estop_replans_from_standing():
    eng, clock = _started()
    before = eng.commands()['leader']
    _midway(eng, clock, 'leader')
    eng.estop(True)
    clock.tick(3)
    eng.estop(False)
    eng.control('resume', clock.now)
    clock.tick(2)
    after = eng.commands()['leader']
    assert after.mode == 'FOLLOW' and after.cmd_id > before.cmd_id + 1
    assert after.poses != before.poses
    assert _starts_at(after, clock.pose['leader'])


# ---------- teach ----------
def test_teach_goal_stores_pose_writes_file(tmp_path):
    store = GoalStore.load(str(tmp_path / 'goals.yaml'))
    eng = helpers.engine(goal_store=store)
    helpers.place(eng.world, 'leader', 3.21, 2.84, 1.2, now=5.0)
    assert eng.teach_goal('leader', 'pick_gear', now=5.1)
    saved = GoalStore.load(str(tmp_path / 'goals.yaml')).get('pick_gear')
    assert (saved.x, saved.y, saved.theta) == (3.21, 2.84, 1.2)
    assert eng.tick(5.2)[0][0] == 'goal_saved'


def test_teach_goal_rejects_stale_unaligned(tmp_path):
    store = GoalStore.load(str(tmp_path / 'goals.yaml'))
    eng = helpers.engine(goal_store=store)
    assert not eng.teach_goal('leader', 'pick_gear', now=1.0)
    helpers.place(eng.world, 'leader', 3.2, 2.8, 0.0, now=5.0, aligned=False)
    assert not eng.teach_goal('leader', 'pick_gear', now=5.1)
    helpers.place(eng.world, 'leader', 3.2, 2.8, 1.2, now=5.0)
    assert not eng.teach_goal('leader', 'pick_gear', now=9.0)
    assert not eng.teach_goal('rover9', 'pick_gear', now=5.1)
    reasons = [detail for kind, detail in eng.tick(9.1) if kind == 'teach_rejected']
    assert [r.split(': ')[-1] for r in reasons[:3]] == ['NO_POSE', 'NOT_ALIGNED', 'STALE_POSE']
    assert store.names() == [] and not (tmp_path / 'goals.yaml').exists()


def test_follower_load_goal_from_drop_region():
    """The named load goal only seeds the mission: the follower goes to a feasible pose
    whose cargo area holds the leader's drop point (pick goal + drop_point)."""
    store = helpers.goals()
    store.set('load_gear', GoalPose(3.0, 3.0, 1.0))
    eng, _clock = _started(helpers.engine(goal_store=store))
    pick, drop = eng.world.goals.get('pick_gear'), eng.settings.pick.drop_point
    cmd = eng.commands()['follower']
    goal = cmd.poses[-1]
    assert cmd.mode == 'FOLLOW' and cmd.goal_key == 'load_gear'
    assert cmd.region is not None and cmd.region.holds(goal, 0.0)
    assert math.hypot(goal.x - pick.x - drop[0], goal.y - pick.y - drop[1]) <= 0.10
