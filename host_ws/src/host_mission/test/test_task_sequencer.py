from host_mission.mission_queue import Item, MissionQueueModel
from host_mission.task_sequencer import MissionStateMachine


def _q(*items):
    q = MissionQueueModel()
    for it in items:
        q.enqueue(it)
    return q


def _run(seq, ticks=60, health=None):
    events = []
    t = 0.0
    for _ in range(ticks):
        t += 1.0
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
    assert seq.targets == {'leader': 1, 'follower': 1}

    events = _run(seq)
    for kind in (('loaded', 'fetch_a'), ('loaded', 'fetch_b'),
                 ('delivered', 'fetch_a'), ('delivered', 'fetch_b'), ('done', '')):
        assert kind in events, kind
    assert seq.state == 'DONE'
    assert all(it.status == 'done' for it in q.items)


def test_route_order_is_lane_node_ascending():
    q = _q(Item('fetch_b', 'b', 3.4, 4.75, 2), Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0, drop_dwell=0.0)
    seq.start(0.0)
    assert seq.targets == {'leader': 1, 'follower': 1}   # node 1 first, not node 2


def test_two_batches_when_over_capacity():
    q = _q(*[Item(f'fetch_{i}', str(i), 0.0, 0.0, i) for i in (1, 2, 3)])
    seq = MissionStateMachine(q, 0, 9, capacity=2, pick_dwell=0.0, drop_dwell=0.0)
    seq.start(0.0)
    events = _run(seq, ticks=120)
    assert len([e for e in events if e[0] == 'delivered']) == 3
    assert ('next_batch', '') in events
    assert seq.state == 'DONE'


def test_pause_and_auto_resume_on_health():
    q = _q(Item('fetch_a', 'a', 2.6, 4.75, 1))
    seq = MissionStateMachine(q, 0, 9, capacity=4, pick_dwell=0.0, drop_dwell=0.0)
    seq.start(0.0)
    _, modes, ev = seq.tick(1.0, {'leader': False, 'follower': False},
                            {'leader': 'LOST', 'follower': 'GOOD'})
    assert seq.state == 'PAUSED'
    assert modes == {'leader': 'STOP', 'follower': 'STOP'}
    assert ('paused', 'health_lost') in ev
    # health clears -> auto resume
    _, _, ev = seq.tick(2.0, {'leader': False, 'follower': False},
                        {'leader': 'GOOD', 'follower': 'GOOD'})
    assert seq.state == 'RUNNING'
    assert ('resumed', '') in ev


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
    seq.tick(3.0, {'leader': False, 'follower': True}, {})  # follower at unloading -> start drop
    _, _, ev = seq.tick(4.0, {'leader': False, 'follower': True}, {})  # drop done
    assert ('delivered', 'fetch_a') in ev
    assert q.items[0].status == 'done'      # delivered even though leader not home
    assert seq.state == 'RUNNING'           # mission not DONE until leader returns
    _, _, ev = seq.tick(5.0, {'leader': True, 'follower': True}, {})
    assert seq.state == 'DONE'
