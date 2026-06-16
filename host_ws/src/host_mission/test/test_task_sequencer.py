from host_mission.mission_queue import Item, MissionQueueModel
from host_mission.task_sequencer import MissionStateMachine


def _q(*items):
    q = MissionQueueModel()
    for it in items:
        q.enqueue(it)
    return q


def _run(seq, ticks=60, health=None, auto_unload=True):
    """Tick to completion. auto_unload simulates the operator pressing '하역완료' every tick
    (a no-op unless the follower is actually at the unloading zone)."""
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
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0, drop_dwell=0.0)
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
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0, drop_dwell=0.0)
    seq.start(0.0)
    assert seq.targets == {'leader': 2, 'follower': 2}     # b first, NOT lane-node sorted


def test_time_optimal_order_respects_directed_cost():
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
    seq.tick(1.0, {'leader': True, 'follower': False}, {})   # only leader at its node
    assert seq.phase == 'TO_PART'                            # barrier NOT met
    seq.tick(2.0, {'leader': True, 'follower': True}, {})    # both at own nodes
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
    seq.tick(3.0, {'leader': False, 'follower': True}, {})   # follower at unloading, no 하역완료
    assert seq.system_state() == 'UNLOADING'
    assert q.items[0].status == 'active'                     # NOT delivered (gated)
    seq.unload_done()                                        # operator presses 하역완료
    _, _, ev = seq.tick(4.0, {'leader': False, 'follower': True}, {})
    assert ('delivered', 'fetch_a') in ev
    assert seq.state == 'RUNNING'                            # leader not home yet
    _, _, ev = seq.tick(5.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'DONE'


def test_unload_wait_no_timeout_waits_for_operator():
    # 하역완료(unload_done)는 운영자 페이스 → 그 대기는 phase_timeout에서 제외(2026-06-15 정책).
    # 따라서 타이머로 자동완료(silent deliver)도, FAULT(PAUSED)도 되지 않고 무한 대기한다.
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0, phase_timeout=5.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})    # -> PICKING
    seq.tick(2.0, {'leader': True, 'follower': True}, {})    # -> DELIVERING (phase_start=2.0)
    # follower waits at unloading; operator has NOT pressed 하역완료. >> phase_timeout 경과.
    _, _, ev = seq.tick(100.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'RUNNING'                            # FAULT/PAUSED 안 됨 (타임아웃 제외)
    assert seq.system_state() == 'UNLOADING'                 # 하역 대기 상태 유지
    assert ('phase_timeout', 'DELIVERING') not in ev
    assert q.items[0].status == 'active'                     # NOT auto-delivered on a timer
    # 운영자가 '하역완료'를 누르면 그제서야 완료된다.
    seq.unload_done()
    ev_done = seq.tick(101.0, {'leader': True, 'follower': True}, {})[2]
    assert ('delivered', 'fetch_a') in ev_done
    seq.tick(102.0, {'leader': True, 'follower': True}, {})  # both home -> DONE
    assert seq.state == 'DONE'


def test_two_batches_when_over_capacity():
    q = _q(*[Item(f'fetch_{i}', str(i), 0.0, 0.0, i) for i in (1, 2, 3)])
    seq = MissionStateMachine(q, 0, 9, capacity=2, pick_dwell=0.0, drop_dwell=0.0)
    seq.start(0.0)
    events = _run(seq, ticks=120)
    assert len([e for e in events if e[0] == 'delivered']) == 3
    assert ('next_batch', '') in events
    assert seq.state == 'DONE'


def test_cooldown_distinct_shelf_one_per_trip():
    # different shelves ride together (capacity 2); a same-shelf duplicate is deferred to a later trip
    q = _q(Item('g1', 'gear', 0, 0, 1, leader_node=1, follower_node=1),
           Item('g2', 'gear', 0, 0, 1, leader_node=1, follower_node=1),
           Item('b1', 'bolt', 0, 0, 2, leader_node=2, follower_node=2))
    seq = MissionStateMachine(q, 0, 9, capacity=2, cooldown=30.0)
    seq.start(0.0)
    assert {it.id for it in seq.batch} == {'g1', 'b1'}   # not g2 (same shelf as g1)


def test_no_cooldown_allows_same_shelf_in_batch():
    q = _q(Item('g1', 'gear', 0, 0, 1, leader_node=1, follower_node=1),
           Item('g2', 'gear', 0, 0, 1, leader_node=1, follower_node=1))
    seq = MissionStateMachine(q, 0, 9, capacity=2, cooldown=0.0)
    seq.start(0.0)
    assert len(seq.batch) == 2                            # cooldown off → same shelf may batch


def test_cooldown_blocks_begin_until_shelf_ready():
    q = _q(Item('g1', 'gear', 0, 0, 1, leader_node=1, follower_node=1))
    seq = MissionStateMachine(q, 0, 9, capacity=2, cooldown=30.0)
    seq._shelf_ready[1] = 35.0                            # shelf 1 picked recently
    assert seq._begin_batch(10.0) is False               # on cooldown → no trip yet
    assert seq._begin_batch(35.0) is True                # released → trip starts


def test_mid_mission_add_joins_current_trip():
    q = _q(Item('fetch_a', 'a', 0, 0, 1, leader_node=1, follower_node=1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})   # arrive -> PICKING
    q.enqueue(Item('fetch_b', 'b', 0, 0, 2, leader_node=2, follower_node=2))  # added mid-mission
    seq.tick(2.0, {'leader': True, 'follower': True}, {})   # A picked -> absorb B -> TO_PART(B)
    assert seq.phase == 'TO_PART'
    assert seq.targets['leader'] == 2                       # joined THIS trip (not delivering yet)
    assert q.by_id('fetch_b').status == 'active'


def test_pause_and_auto_resume_on_health():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0, drop_dwell=0.0)
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
    assert modes == {'leader': 'GOTO', 'follower': 'GOTO'}   # modes restored on auto-resume


def test_manual_pause_does_not_auto_resume():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4)
    seq.start(0.0)
    seq.pause()
    assert seq.state == 'PAUSED'
    seq.tick(1.0, {'leader': True, 'follower': True}, {})   # healthy, but manual pause holds
    assert seq.state == 'PAUSED'
    seq.resume()
    assert seq.state == 'RUNNING'


def test_estop_pauses_and_release_needs_resume():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    seq.estop(True)
    _, modes, _ = seq.tick(1.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'PAUSED' and seq.system_state() == 'ESTOP'
    assert modes == {'leader': 'STOP', 'follower': 'STOP'}
    seq.estop(False)                                         # release alone does NOT resume
    seq.tick(2.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'PAUSED'
    seq.resume()                                             # deliberate resume restores the phase
    assert seq.state == 'RUNNING' and seq.phase == 'TO_PART'
    _, modes, _ = seq.tick(3.0, {'leader': False, 'follower': False}, {})
    assert modes == {'leader': 'GOTO', 'follower': 'GOTO'}   # modes restored, not stuck STOP


def test_abort_marks_active_items_failed():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4)
    seq.start(0.0)
    assert q.items[0].status == 'active'
    seq.abort()
    assert seq.state == 'ABORTED'
    assert q.items[0].status == 'failed'


def test_phase_timeout_pauses_instead_of_hanging():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, phase_timeout=5.0)
    seq.start(0.0)
    _, modes, ev = seq.tick(6.0, {'leader': False, 'follower': False}, {})  # never arrives
    assert seq.state == 'PAUSED'
    assert ('phase_timeout', 'TO_PART') in ev
    assert modes == {'leader': 'STOP', 'follower': 'STOP'}


def test_deliver_is_decoupled_follower_drops_before_leader_home():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0, drop_dwell=0.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})   # arrive part -> PICKING
    seq.tick(2.0, {'leader': True, 'follower': True}, {})   # pick done -> DELIVERING
    assert seq.phase == 'DELIVERING'
    seq.tick(3.0, {'leader': False, 'follower': True}, {})  # follower at unloading (UNLOADING)
    assert q.items[0].status == 'active'                    # awaiting 하역완료
    seq.unload_done()                                        # operator presses 하역완료
    _, _, ev = seq.tick(4.0, {'leader': False, 'follower': True}, {})
    assert ('delivered', 'fetch_a') in ev
    assert q.by_id('fetch_a') is None       # delivered → removed from queue (even though leader not home)
    assert seq.state == 'RUNNING'           # mission not DONE until leader returns
    _, _, ev = seq.tick(5.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'DONE'


def test_queue_cancel_removes_one_pending_not_active():
    q = _q(Item('fetch_a', 'a', 0, 0, 1), Item('fetch_b', 'b', 0, 0, 2))
    q.items[0].status = 'active'                 # a is in-flight
    assert q.cancel('fetch_a') is False          # active is protected
    assert q.cancel('fetch_b') is True           # pending one removed
    assert [it.id for it in q.items] == ['fetch_a']
    assert q.cancel('missing') is False


def test_purge_finished_clears_done_and_failed():
    q = _q(Item('a', 'a', 0, 0, 1), Item('b', 'b', 0, 0, 2), Item('c', 'c', 0, 0, 3))
    q.items[0].status = 'done'
    q.items[1].status = 'failed'             # c stays pending
    q.purge_finished()
    assert [it.id for it in q.items] == ['c']


def test_requeue_same_part_after_done():
    # a finished item blocks re-enqueue of the same id until it is dropped (the coordinator
    # drops it on re-dispatch, so the SAME part can be fetched again).
    q = _q(Item('fetch_a', 'a', 0, 0, 1))
    q.items[0].status = 'done'
    assert q.enqueue(Item('fetch_a', 'a', 0, 0, 1)) is False   # duplicate id blocked while present
    assert q.cancel('fetch_a') is True                          # drop the finished entry
    assert q.enqueue(Item('fetch_a', 'a', 0, 0, 1)) is True     # now re-queues fresh


def test_cancel_active_enroute_single_returns_home():
    q = _q(Item('fetch_a', 'a', 0, 0, 1, leader_node=1, follower_node=1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=5.0)
    seq.start(0.0)
    assert seq.phase == 'TO_PART'                       # en-route, not yet picked
    assert seq.cancel_item('fetch_a', 1.0) is True
    assert seq.state == 'IDLE'
    assert seq.targets == {'leader': 0, 'follower': 9}  # rovers home (standby / unloading)
    assert q.by_id('fetch_a') is None                   # removed from the queue


def test_cancel_rejected_while_picking():
    q = _q(Item('fetch_a', 'a', 0, 0, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=5.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})   # -> PICKING
    assert seq.phase == 'PICKING'
    assert seq.cancel_item('fetch_a', 2.0) is False         # being picked → protected


def test_cancel_rejected_when_loaded_returning():
    q = _q(Item('fetch_a', 'a', 0, 0, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0)
    seq.start(0.0)
    seq.tick(1.0, {'leader': True, 'follower': True}, {})   # arrive -> PICKING
    seq.tick(2.0, {'leader': True, 'follower': True}, {})   # picked -> DELIVERING (LOADED)
    assert seq.phase == 'DELIVERING'
    assert seq.cancel_item('fetch_a', 3.0) is False         # loaded / returning → protected


def test_cancel_enroute_continues_to_next_in_batch():
    q = _q(Item('fetch_a', 'a', 0, 0, 1, leader_node=1, follower_node=1),
           Item('fetch_b', 'b', 0, 0, 2, leader_node=2, follower_node=2))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=5.0)
    seq.start(0.0)
    assert seq.targets['leader'] == 1                  # FIFO: a first
    assert seq.cancel_item('fetch_a', 1.0) is True
    assert seq.phase == 'TO_PART' and seq.targets['leader'] == 2   # advanced to b
    assert q.by_id('fetch_a') is None
