from host_mission.mission_queue import Item, MissionQueueModel
from host_mission.task_sequencer import MissionStateMachine


def _q(*items):
    q = MissionQueueModel()
    for it in items:
        q.enqueue(it)
    return q


def _run(seq, ticks=60, health=None, auto_unload=True):
    """Tick to completion. auto_unload simulates the operator pressing 'unload done' every tick (a no-op unless the follower is actually at the unloading zone)."""
    events = []
    t = 0.0
    for _ in range(ticks):
        t += 1.0
        if auto_unload:
            seq.unload_done()
        _, _, ev = seq.tick(t, {'leader': True, 'follower': True}, health or {})
        events += ev
        if seq.state == 'DONE':
            break
    return events


def test_single_batch_collect_then_deliver():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1), Item('fetch_b', 'b', 3.4, 4.75, 2))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    assert seq.state == 'RUNNING' and seq.phase == 'TO_PART'
    assert seq.targets == {'leader': 1, 'follower': 1}     # single-node item -> both same

    events = _run(seq)
    for kind in (('loaded', 'fetch_a'), ('loaded', 'fetch_b'),
                 ('delivered', 'fetch_a'), ('delivered', 'fetch_b'), ('done', '')):
        assert kind in events, kind
    assert seq.state == 'DONE'
    assert q.items == []                     # delivered parts vanish from the queue


def test_batch_order_is_fifo_without_cost():
    # no order_cost -> insertion order (FIFO); b inserted first stays first (no node-id sort)
    q = _q(Item('fetch_b', 'b', 3.4, 4.75, 2), Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    assert seq.targets == {'leader': 2, 'follower': 2}


def test_time_optimal_order_directed_cost():
    # directed (asymmetric) cost: tour via node 1 first is far cheaper -> visit a(1) before b(2)
    cost = {(0, 1): 1, (1, 2): 1, (2, 0): 1, (0, 2): 9, (2, 1): 9, (1, 0): 9}
    oc = lambda x, y: cost.get((x, y), 5)
    q = _q(Item('fetch_b', 'b', 3.4, 4.75, 2), Item('fetch_a', 'a', 2.6, 4.75, 1))  # b inserted first
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0, order_cost=oc)
    seq.start(0.0)
    assert seq.targets == {'leader': 1, 'follower': 1}     # ATSP overrides FIFO (=b) -> a first


def test_parallel_pick_targets():
    q = _q(Item('fetch_g', 'g', 5.0, 2.7, 29, leader_node=29, follower_node=34))
    seq = MissionStateMachine(q, 32, 22, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    assert seq.targets == {'leader': 29, 'follower': 34}   # different node per robot (parallel pick)


def test_arrival_barrier_per_own_node():
    q = _q(Item('fetch_g', 'g', 5.0, 2.7, 29, leader_node=29, follower_node=34))
    seq = MissionStateMachine(q, 32, 22, capacity=4, pick_dwell=5.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': False}, {})
    assert seq.phase == 'TO_PART'
    seq.tick(2.0, {'leader': True, 'follower': True}, {})
    assert seq.phase == 'PICKING'


def test_pick_complete_drives_loading():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=999.0)   # no dwell auto-complete
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})    # -> PICKING / LOADING
    assert seq.phase == 'PICKING' and seq.system_state() == 'LOADING'
    seq.pick_complete(True)                                  # driven by /pnp/working falling edge
    _, _, ev = seq.tick(2.0, {'leader': True, 'follower': True}, {})
    assert ('loaded', 'fetch_a') in ev


def test_manual_unload_gating():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})    # arrive -> PICKING
    seq.tick(2.0, {'leader': True, 'follower': True}, {})    # pick -> DELIVERING
    assert seq.phase == 'DELIVERING'
    seq.tick(3.0, {'leader': False, 'follower': True}, {})  # follower at unloading, no unload done
    assert seq.system_state() == 'UNLOADING'
    assert q.items[0].status == 'active'
    seq.unload_done()
    _, _, ev = seq.tick(4.0, {'leader': False, 'follower': True}, {})
    assert ('delivered', 'fetch_a') in ev
    assert seq.state == 'RUNNING'
    _, _, ev = seq.tick(5.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'DONE'


def test_unload_wait_no_timeout_waits():
    # The operator's unload_done is operator-paced: that wait is excluded from phase_timeout, so
    # the mission neither auto-completes on a timer nor faults; it waits indefinitely.
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0, phase_timeout=5.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})    # -> PICKING
    seq.tick(2.0, {'leader': True, 'follower': True}, {})    # -> DELIVERING (phase_start=2.0)
    # follower waits at unloading; the operator has NOT pressed unload done; phase_timeout passed
    _, _, ev = seq.tick(100.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'RUNNING'  # not FAULT/PAUSED (timeout excluded)
    assert seq.system_state() == 'UNLOADING'
    assert ('phase_timeout', 'DELIVERING') not in ev
    assert q.items[0].status == 'active'
    # the operator presses unload done: only then it completes
    seq.unload_done()
    ev_done = seq.tick(101.0, {'leader': True, 'follower': True}, {})[2]
    assert ('delivered', 'fetch_a') in ev_done
    seq.tick(102.0, {'leader': True, 'follower': True}, {})  # both home -> DONE
    assert seq.state == 'DONE'


def test_two_batches_when_over_capacity():
    q = _q(*[Item(f'fetch_{i}', str(i), 0.0, 0.0, i) for i in (1, 2, 3)])
    seq = MissionStateMachine(q, 0, 9, capacity=2, pick_dwell=0.0)
    seq.start(0.0)
    events = _run(seq, ticks=120)
    assert len([e for e in events if e[0] == 'delivered']) == 3
    assert ('next_batch', '') in events
    assert seq.state == 'DONE'


def test_cooldown_distinct_shelf_per_trip():
    # different shelves ride together (capacity 2); a same-shelf duplicate is deferred to a later trip
    q = _q(Item('g1', 'gear', 0, 0, 1, leader_node=1, follower_node=1),
           Item('g2', 'gear', 0, 0, 1, leader_node=1, follower_node=1),
           Item('b1', 'bolt', 0, 0, 2, leader_node=2, follower_node=2))
    seq = MissionStateMachine(q, 0, 9, capacity=2, cooldown=30.0)
    seq.start(0.0)
    assert {it.id for it in seq.batch} == {'g1', 'b1'}   # not g2 (same shelf as g1)


def test_no_cooldown_same_shelf_in_batch():
    q = _q(Item('g1', 'gear', 0, 0, 1, leader_node=1, follower_node=1),
           Item('g2', 'gear', 0, 0, 1, leader_node=1, follower_node=1))
    seq = MissionStateMachine(q, 0, 9, capacity=2, cooldown=0.0)
    seq.start(0.0)
    assert len(seq.batch) == 2                            # cooldown off → same shelf may batch


def test_cooldown_blocks_until_shelf_ready():
    q = _q(Item('g1', 'gear', 0, 0, 1, leader_node=1, follower_node=1))
    seq = MissionStateMachine(q, 0, 9, capacity=2, cooldown=30.0)
    seq._shelf_ready[1] = 35.0
    assert seq._begin_batch(10.0) is False               # on cooldown → no trip yet
    assert seq._begin_batch(35.0) is True


def test_mid_mission_add_joins_current_trip():
    q = _q(Item('fetch_a', 'a', 0, 0, 1, leader_node=1, follower_node=1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})   # arrive -> PICKING
    q.enqueue(Item('fetch_b', 'b', 0, 0, 2, leader_node=2, follower_node=2))
    seq.tick(2.0, {'leader': True, 'follower': True}, {})   # A picked -> absorb B -> TO_PART(B)
    assert seq.phase == 'TO_PART'
    assert seq.targets['leader'] == 2                       # joined THIS trip (not delivering yet)
    assert q.by_id('fetch_b').status == 'active'


def test_pause_and_auto_resume_on_health():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    _, modes, ev = seq.tick(1.0, {'leader': False, 'follower': False},
                            {'leader': 'LOST', 'follower': 'GOOD'})
    assert seq.state == 'PAUSED'
    assert modes == {'leader': 'STOP', 'follower': 'STOP'}
    assert ('paused', 'health_lost') in ev
    _, modes, ev = seq.tick(2.0, {'leader': False, 'follower': False},
                            {'leader': 'GOOD', 'follower': 'GOOD'})
    assert seq.state == 'RUNNING'
    assert ('resumed', '') in ev
    assert modes == {'leader': 'GOTO', 'follower': 'GOTO'}


def test_manual_pause_does_not_auto_resume():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4)
    seq.start(0.0)
    seq.pause()
    assert seq.state == 'PAUSED'
    seq.tick(1.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'PAUSED'
    seq.resume()
    assert seq.state == 'RUNNING'


def test_estop_pauses_release_needs_resume():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    seq.estop(True)
    _, modes, _ = seq.tick(1.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'PAUSED' and seq.system_state() == 'ESTOP'
    assert modes == {'leader': 'STOP', 'follower': 'STOP'}
    seq.estop(False)
    seq.tick(2.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'PAUSED'
    seq.resume()
    assert seq.state == 'RUNNING' and seq.phase == 'TO_PART'
    _, modes, _ = seq.tick(3.0, {'leader': False, 'follower': False}, {})
    assert modes == {'leader': 'GOTO', 'follower': 'GOTO'}


def test_phase_timeout_pauses_not_hangs():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, phase_timeout=5.0)
    seq.start(0.0)
    _, modes, ev = seq.tick(6.0, {'leader': False, 'follower': False}, {})
    assert seq.state == 'PAUSED'
    assert ('phase_timeout', 'TO_PART') in ev
    assert modes == {'leader': 'STOP', 'follower': 'STOP'}


def test_deliver_follower_drops_before_leader_home():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})   # arrive part -> PICKING
    seq.tick(2.0, {'leader': True, 'follower': True}, {})   # pick done -> DELIVERING
    assert seq.phase == 'DELIVERING'
    seq.tick(3.0, {'leader': False, 'follower': True}, {})
    assert q.items[0].status == 'active'
    seq.unload_done()
    _, _, ev = seq.tick(4.0, {'leader': False, 'follower': True}, {})
    assert ('delivered', 'fetch_a') in ev
    assert q.by_id('fetch_a') is None       # delivered → removed from queue (even though leader not home)
    assert seq.state == 'RUNNING'           # mission not DONE until leader returns
    _, _, ev = seq.tick(5.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'DONE'


def test_queue_cancel_removes_pending_only():
    q = _q(Item('fetch_a', 'a', 0, 0, 1), Item('fetch_b', 'b', 0, 0, 2))
    q.items[0].status = 'active'
    assert q.cancel('fetch_a') is False
    assert q.cancel('fetch_b') is True
    assert [it.id for it in q.items] == ['fetch_a']
    assert q.cancel('missing') is False


def test_queue_has_no_finished_failed_rows():
    assert not hasattr(MissionQueueModel, 'purge_finished')


def test_requeue_same_part_after_done():
    # a finished item blocks re-enqueue of the same id until it is dropped (the coordinator
    # drops it on re-dispatch, so the SAME part can be fetched again).
    q = _q(Item('fetch_a', 'a', 0, 0, 1))
    q.items[0].status = 'done'
    assert q.enqueue(Item('fetch_a', 'a', 0, 0, 1)) is False
    assert q.cancel('fetch_a') is True
    assert q.enqueue(Item('fetch_a', 'a', 0, 0, 1)) is True


def _two_in_trip():
    """a and b in one trip, the mission driving to a; c waiting."""
    q = _q(Item('fetch_a', 'a', 0, 0, 1, leader_node=1, follower_node=1),
           Item('fetch_b', 'b', 0, 0, 2, leader_node=2, follower_node=2),
           Item('fetch_c', 'c', 0, 0, 3, leader_node=3, follower_node=3))
    seq = MissionStateMachine(q, 0, 9, capacity=2, pick_dwell=5.0, follower_home_node=8)
    seq.start(0.0)
    assert [it.id for it in seq.batch] == ['fetch_a', 'fetch_b']
    return q, seq


def test_cancel_running_trip_refused():
    q, seq = _two_in_trip()
    assert 'running trip' in seq.cancel_refusal('fetch_a')
    assert 'running trip' in seq.cancel_refusal('fetch_b')
    seq.tick(1.0, {'leader': True, 'follower': True}, {})
    assert seq.phase == 'PICKING' and 'running trip' in seq.cancel_refusal('fetch_a')
    seq.pause()
    assert 'running trip' in seq.cancel_refusal('fetch_b')
    assert seq.cancel_refusal('fetch_c') is None
    assert 'not in the queue' in seq.cancel_refusal('missing')
    assert [it.id for it in seq.batch] == ['fetch_a', 'fetch_b']


def test_clear_waiting_spares_running_trip():
    q, seq = _two_in_trip()
    q.enqueue(Item('fetch_d', 'd', 0, 0, 4))
    q.clear_waiting()
    assert [it.id for it in q.items] == ['fetch_a', 'fetch_b']
    assert [it.id for it in seq.batch] == ['fetch_a', 'fetch_b']
    assert seq.phase == 'TO_PART' and seq.targets == {'leader': 1, 'follower': 1}


def _full_two_part_trip():
    """a and b are the whole queue; one trip of two."""
    q = _q(Item('fetch_a', 'a', 0, 0, 1, leader_node=1, follower_node=1),
           Item('fetch_b', 'b', 0, 0, 2, leader_node=2, follower_node=2))
    seq = MissionStateMachine(q, 0, 9, capacity=2, pick_dwell=5.0, follower_home_node=8)
    seq.start(0.0)
    return q, seq


def test_pick_failure_drops_part_trip_goes_on():
    q, seq = _full_two_part_trip()
    both = {'leader': True, 'follower': True}
    seq.tick(1.0, both, {})
    seq.pick_complete(False)
    _, _, ev = seq.tick(2.0, both, {})
    assert ('pick_failed', 'fetch_a') in ev
    assert q.by_id('fetch_a') is None
    assert [it.id for it in seq.batch] == ['fetch_b'] and seq.bidx == 0
    assert seq.phase == 'TO_PART' and seq.targets == {'leader': 2, 'follower': 2}
    seq.tick(3.0, both, {})
    seq.pick_complete(True)
    _, _, ev = seq.tick(4.0, both, {})
    assert ('loaded', 'fetch_b') in ev and seq.phase == 'DELIVERING'
    seq.unload_done()
    _, _, ev = seq.tick(5.0, both, {})
    assert ('delivered', 'fetch_b') in ev and ('delivered', 'fetch_a') not in ev
    assert q.items == []


def test_second_pick_failure_delivers_first():
    q, seq = _full_two_part_trip()
    both = {'leader': True, 'follower': True}
    seq.tick(1.0, both, {})
    seq.pick_complete(True)
    seq.tick(2.0, both, {})
    seq.tick(3.0, both, {})
    seq.pick_complete(False)
    _, _, ev = seq.tick(4.0, both, {})
    assert ('pick_failed', 'fetch_b') in ev and seq.phase == 'DELIVERING'
    assert [it.id for it in q.items] == ['fetch_a']
    seq.unload_done()
    _, _, ev = seq.tick(5.0, both, {})
    assert ('delivered', 'fetch_a') in ev and q.items == []


def test_only_pick_failed_goes_home_no_unload():
    """Nothing is loaded: the follower at the unload goal must not wait for the operator's
    unload-done with an empty queue."""
    q = _q(Item('fetch_a', 'a', 0, 0, 1, leader_node=1, follower_node=1))
    seq = MissionStateMachine(q, 0, 9, capacity=2, pick_dwell=5.0, follower_home_node=8)
    seq.start(0.0)
    both = {'leader': True, 'follower': True}
    seq.tick(1.0, both, {})                                     # -> PICKING
    seq.pick_complete(False)
    targets, _modes, events = seq.tick(2.0, both, {})
    assert ('pick_failed', 'fetch_a') in events
    assert targets == {'leader': 0, 'follower': 8}
    assert seq.system_state() == 'RETURNING'
    seq.tick(3.0, {'leader': False, 'follower': False}, {})
    _t, _m, events = seq.tick(30.0, both, {})
    assert ('done', '') in events and seq.state == 'DONE'


def test_failed_pick_frees_slot_for_waiting_part():
    q, seq = _two_in_trip()                                    # a, b in the trip; c waiting
    both = {'leader': True, 'follower': True}
    seq.tick(1.0, both, {})
    seq.pick_complete(False)
    seq.tick(2.0, both, {})
    assert [it.id for it in seq.batch] == ['fetch_b', 'fetch_c']
    assert [it.id for it in q.items] == ['fetch_b', 'fetch_c']


def test_abandon_refused_running_or_picking():
    q, seq = _two_in_trip()
    assert 'stop first' in seq.abandon_refusal()
    seq.tick(1.0, {'leader': True, 'follower': True}, {})          # -> PICKING
    seq.pause()
    assert 'pick and place in progress' in seq.abandon_refusal()
    seq.tick(7.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'PAUSED' and seq.phase == 'TO_PART'
    assert seq.abandon_refusal() is None


def test_abandon_drops_trip_keeps_waiting():
    q, seq = _two_in_trip()
    seq.pause()
    assert seq.abandon() == ['fetch_a', 'fetch_b']
    assert seq.state == 'IDLE' and seq.phase is None and seq.batch == []
    assert [(it.id, it.status) for it in q.items] == [('fetch_c', 'pending')]
    assert seq.targets == {'leader': 0, 'follower': 8}            # standby / follower home


def test_abandon_under_estop_waits_for_resume():
    q, seq = _two_in_trip()
    seq.estop(True)
    seq.abandon()
    seq.estop(False)
    _, modes, _ = seq.tick(1.0, {'leader': True, 'follower': True}, {})
    assert seq.system_state() == 'ESTOP' and set(modes.values()) == {'STOP'}
    seq.resume()
    assert seq.state == 'IDLE'


def test_pick_result_while_paused_ends_pick():
    q = _q(Item('fetch_a', 'a', 0, 0, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=999.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})          # -> PICKING
    seq.estop(True)
    seq.pick_complete(True)
    _, modes, ev = seq.tick(2.0, {'leader': True, 'follower': True}, {})
    assert ('loaded', 'fetch_a') in ev
    assert seq.system_state() == 'ESTOP' and set(modes.values()) == {'STOP'}
    seq.estop(False)
    seq.resume()
    assert seq.state == 'RUNNING' and seq.phase == 'DELIVERING'


def test_queued_trip_starts_from_home():
    # With next_trip_from_home the second trip ends the first with 'done' (the engine's idle
    # rules run: turn east, a start waits for it) and starts as from IDLE
    q = _q(*[Item(f'fetch_{i}', str(i), 0.0, 0.0, i) for i in (1, 2, 3)])
    seq = MissionStateMachine(q, 0, 9, capacity=2, pick_dwell=0.0, next_trip_from_home=True)
    seq.start(0.0)
    events = _run(seq, ticks=120)
    assert ('next_batch', '') not in events
    assert len([e for e in events if e[0] == 'delivered']) == 2 and seq.state == 'DONE'
    assert seq.start(200.0) == [('mission_started', '')]
