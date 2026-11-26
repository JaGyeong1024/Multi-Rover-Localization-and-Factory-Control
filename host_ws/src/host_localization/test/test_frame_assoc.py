"""Per-frame camera association (no ROS): class flips between the two rovers (seen on site)."""
from host_localization.fusion import quality
from host_localization.fusion.frame_assoc import CROSS, FALLBACK, OWN, Box, process_frame
from host_localization.fusion.frontend import RobotFusion
from host_localization.fusion.tracker import KW

DT = 0.05
MAX_D = 0.15
LEADER = (2.0, 2.0)
FOLLOWER = (2.0, 2.55)          # load pose: 0.55 m to the leader's right
UWB_SHIFT = (0.9, 1.2)          # old UWB frame, learned as an offset


def _fusions():
    return {'leader': RobotFusion(dict(KW['leader'])), 'follower': RobotFusion(dict(KW['follower']))}


def _frame(fus, boxes, t, participants=('leader', 'follower'), uwb=('leader', 'follower'),
           truth=None):
    truth = truth or {'leader': LEADER, 'follower': FOLLOWER}
    for name, f in fus.items():
        f.on_imu(t, 0.0, 0.0)
        if name in uwb and round(t / DT) % 2 == 0:
            x, y = truth[name]
            f.on_uwb(t, t, x + UWB_SHIFT[0], y + UWB_SHIFT[1])
    return process_frame(fus, participants, boxes, t, t, MAX_D)


def _normal(t):
    return [Box('leader', 0.90, *LEADER, None), Box('follower', 0.85, *FOLLOWER, None)]


def _settled(uwb=('leader', 'follower')):
    fus = _fusions()
    t = 0.0
    for i in range(40):
        t = i * DT
        _frame(fus, _normal(t), t, uwb=uwb)
    return fus, t


def _run(fus, t, boxes, seconds=3.0, **kw):
    results = []
    for i in range(1, int(seconds / DT)):
        t1 = t + i * DT
        results.append(_frame(fus, boxes, t1, **kw))
    return t1, results


def _near(f, p, tol=0.03):
    x, y, _ = f.est.pose
    return abs(x - p[0]) < tol and abs(y - p[1]) < tol


def test_flipped_follower_below_leader_kept():
    fus, t = _settled()
    boxes = [Box('leader', 0.90, *LEADER, None), Box('leader', 0.70, *FOLLOWER, None)]
    t, res = _run(fus, t, boxes)
    assert res[-1]['follower'][1] == CROSS and res[-1]['leader'][1] == OWN
    assert fus['follower'].quality(t) == quality.GOOD and fus['leader'].quality(t) == quality.GOOD
    assert _near(fus['follower'], FOLLOWER) and _near(fus['leader'], LEADER)


def test_flipped_follower_above_leader_refused():
    fus, t = _settled()
    boxes = [Box('leader', 0.95, *FOLLOWER, None), Box('leader', 0.90, *LEADER, None)]
    t, res = _run(fus, t, boxes)
    assert res[-1]['follower'][1] == CROSS and res[-1]['leader'][1] == OWN
    assert _near(fus['follower'], FOLLOWER) and _near(fus['leader'], LEADER)
    assert fus['leader'].est._refused == 0


def test_far_own_box_not_fed_to_refused():
    fus, t = _settled()
    boxes = [Box('follower', 0.95, *LEADER, None), Box('follower', 0.85, *FOLLOWER, None)]
    t, res = _run(fus, t, boxes)
    assert res[-1]['follower'][1] == OWN and boxes[res[-1]['follower'][0]].conf == 0.85
    assert res[-1]['leader'][1] == CROSS
    assert fus['follower'].est._refused == 0 and fus['leader'].est._refused == 0
    assert _near(fus['follower'], FOLLOWER) and _near(fus['leader'], LEADER)


def test_leader_no_uwb_refuses_flipped():
    """Real leader occluded, follower labelled 'leader': the old path re-acquired it as the
    leader after 20 fixes (no UWB = nothing contradicts it)."""
    fus, t = _settled(uwb=('follower',))
    boxes = [Box('leader', 0.80, *FOLLOWER, None)]
    t, res = _run(fus, t, boxes, uwb=('follower',))
    assert all(r['follower'][1] == CROSS and 'leader' not in r for r in res)
    assert _near(fus['leader'], LEADER, tol=0.05)
    assert fus['leader'].est._refused == 0
    assert fus['leader'].quality(t) == quality.LOST          # no fix: stops, does not jump
    assert fus['follower'].quality(t) == quality.GOOD


def test_absent_rover_on_other_loses_box():
    """Today's site: no leader in view, the leader estimate rode on the follower's
    'leader'-labelled box. The follower's UWB vouches for the box, the leader's cannot."""
    fus = _fusions()
    t = 0.0
    for i in range(40):                   # both estimates placed on the same body
        t = i * DT
        _frame(fus, [Box('leader', 0.8, *FOLLOWER, None), Box('follower', 0.85, *FOLLOWER, None)],
               t, uwb=('follower',), truth={'leader': FOLLOWER, 'follower': FOLLOWER})
    t, res = _run(fus, t, [Box('leader', 0.8, *FOLLOWER, None)], uwb=('follower',),
                  truth={'leader': FOLLOWER, 'follower': FOLLOWER})
    assert res[-1]['follower'][1] == CROSS and 'leader' not in res[-1]
    assert fus['follower'].quality(t) == quality.GOOD


def test_best_per_class_topic_keeps_box():
    """Follower-only mode: the leader still takes its best box from bev_pose, so that box is
    never offered. Works while the real leader outscores the flipped follower, not after."""
    fus, t = _settled()
    below = [Box('leader', 0.90, *LEADER, None), Box('leader', 0.70, *FOLLOWER, None)]
    res = process_frame(fus, ('follower',), below, t + DT, t + DT, MAX_D)
    assert 'leader' not in res and res['follower'][1] == CROSS
    above = [Box('leader', 0.95, *FOLLOWER, None), Box('leader', 0.90, *LEADER, None)]
    res = process_frame(fus, ('follower',), above, t + 2 * DT, t + 2 * DT, MAX_D)
    assert res == {}                       # the follower box went to the leader topic


def test_first_sight_from_own_class():
    fus = _fusions()
    res = process_frame(fus, ('leader', 'follower'), _normal(0.0), 0.0, 0.0, MAX_D)
    assert res['leader'][1] == FALLBACK and res['follower'][1] == FALLBACK
    assert _near(fus['leader'], LEADER) and _near(fus['follower'], FOLLOWER)


def test_flipped_from_first_frame_unplaced():
    """Limit: a candidate never places a rover. A follower that is 'leader' from its very
    first frame stays unplaced (no pose, LOST) until its own class shows once."""
    fus = _fusions()
    res = process_frame(fus, ('leader', 'follower'),
                        [Box('leader', 0.9, *LEADER, None), Box('leader', 0.7, *FOLLOWER, None)],
                        0.0, 0.0, MAX_D)
    assert 'follower' not in res
    assert fus['follower'].output(0.0, 0.0) is None
