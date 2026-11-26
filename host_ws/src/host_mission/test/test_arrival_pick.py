"""Arrival judgement (heading included) and pick completion judgement."""
import dataclasses

from host_planning.types import Pose2D

from host_mission.arrival import judge_arrival
from host_mission.path_commander import PathCmd
from host_mission.pick_monitor import PickMonitor
from host_mission.world_model import WorldModel

POS_TOL, YAW_TOL, POSE_TIMEOUT, FB_TIMEOUT = 0.08, 0.15, 0.5, 2.0
GOAL = Pose2D(4.9, 2.9, 0.0)


def _cmd(cmd_id=3, issued_at=10.0, mode='FOLLOW'):
    return PathCmd(robot='leader', epoch=1, cmd_id=cmd_id, mode=mode,
                   poses=(Pose2D(4.0, 2.9, 0.0), GOAL), issued_at=issued_at)


def _world(x=4.9, y=2.9, yaw=0.0, aligned=True, pose_time=20.0, state='HOLDING',
           cmd_id=3, nav_time=20.0, detail='', epoch=1):
    world = WorldModel(pose_timeout=POSE_TIMEOUT)
    world.update_pose('leader', x, y, yaw, pose_time)
    world.update_heading_aligned('leader', aligned)
    world.update_nav('leader', state, detail, epoch, cmd_id, nav_time)
    return world.robot('leader')


def _judge(state, cmd=None, now=20.1, yaw_tol=YAW_TOL):
    return judge_arrival(state, cmd or _cmd(), now, POS_TOL, yaw_tol, POSE_TIMEOUT, FB_TIMEOUT)


def test_arrival_needs_pose_heading_report():
    assert _judge(_world()).arrived
    assert _judge(_world(state='PARKED')).arrived
    assert _judge(_world(x=4.95, yaw=0.1)).arrived


def test_parked_at_turn_in_neither_arrived_nor_missed():
    """A path that ends at a turn-in point short of the goal (PathCmd.via): the rover parks at
    its end, and that is no arrival and no miss (the commander plans on)."""
    via = dataclasses.replace(_cmd(), via=True)
    for state in (_world(), _world(state='PARKED'), _world(x=4.6)):
        verdict = _judge(state, via)
        assert not verdict.arrived and not verdict.missed, verdict


def test_not_arrived_while_tracking():
    verdict = _judge(_world(state='TRACKING'))
    assert not verdict.arrived and not verdict.missed and verdict.reason == 'ROVER_TRACKING'


def test_other_command_report_ignored():
    assert _judge(_world(cmd_id=2)).reason == 'NO_FEEDBACK'
    assert _judge(_world(nav_time=9.0), now=10.5).reason == 'NO_FEEDBACK'
    assert _judge(_world(nav_time=20.0), now=23.0).reason == 'NO_FEEDBACK'


def test_other_epoch_feedback_ignored():
    """Same cmd_id but another host run (epoch): an old answer, never an arrival or a miss."""
    assert _judge(_world(epoch=0)).reason == 'NO_FEEDBACK'
    old_miss = _world(epoch=2, state='STOPPED', detail='GOAL_MISSED')
    verdict = _judge(old_miss)
    assert verdict.reason == 'NO_FEEDBACK' and not verdict.missed
    assert _judge(_world(epoch=1)).arrived


def test_stale_pose_is_never_used():
    verdict = _judge(_world(pose_time=19.0))
    assert not verdict.arrived and verdict.reason == 'STALE_POSE'


def test_unaligned_heading_is_never_used():
    verdict = _judge(_world(aligned=False))                       # yaw arrives as 0 = goal yaw
    assert not verdict.arrived and not verdict.missed and verdict.reason == 'NOT_ALIGNED'
    assert _judge(_world(aligned=False), yaw_tol=0.0).arrived


def test_heading_error_blocks_arrival():
    verdict = _judge(_world(yaw=0.3))
    assert not verdict.arrived and verdict.missed and verdict.reason == 'YAW_ERROR'
    assert _judge(_world(yaw=3.1), cmd=None).missed
    assert _judge(_world(yaw=0.3), yaw_tol=0.0).arrived


def test_position_error_is_miss_not_arrival():
    verdict = _judge(_world(x=5.05))
    assert not verdict.arrived and verdict.missed and not verdict.reported


def test_rover_reported_goal_miss():
    verdict = _judge(_world(state='STOPPED', detail='GOAL_MISSED'))
    assert verdict.missed and verdict.reported and not verdict.arrived


def test_stop_command_has_no_arrival():
    assert _judge(_world(), cmd=_cmd(mode='STOP')).reason == 'NO_COMMAND'


def test_pick_success_after_transition():
    pick = PickMonitor()
    pick.on_work_state('NAVIGATING')
    pick.begin()
    assert pick.poll() is None
    pick.on_work_state('PICKING')
    pick.on_work_state('DONE')
    assert pick.poll() is None
    pick.on_done(True)
    assert pick.poll() is True
    assert pick.poll() is None


def test_pick_failure_reported_as_false():
    pick = PickMonitor()
    pick.begin()
    pick.on_work_state('PICKING')
    pick.on_done(False)                         # result may arrive before the state
    pick.on_work_state('ERROR')
    assert pick.poll() is False


def test_latched_replay_is_rejected():
    pick = PickMonitor()
    pick.on_work_state('DONE')                  # latched leftovers of the previous pick
    pick.on_done(True)
    pick.begin()
    assert pick.poll() is None
    pick.on_done(True)                          # replay after phase entry (bridge respawn)
    pick.on_work_state('WAITING_NEXT_OBJECT')
    assert pick.poll() is None                  # no PICKING seen in this phase


def test_replayed_true_cannot_mask_failure():
    pick = PickMonitor()
    pick.begin()
    pick.on_work_state('PICKING')
    pick.on_done(True)                          # stale replay while the pick is running
    assert pick.poll() is None
    pick.on_work_state('ERROR')
    assert pick.poll() is False


def test_result_outside_picking_ignored():
    pick = PickMonitor()
    pick.on_work_state('PICKING')
    pick.on_work_state('DONE')
    pick.on_done(True)
    assert pick.poll() is None
    pick.begin()
    pick.cancel()
    pick.on_work_state('PICKING')
    pick.on_work_state('DONE')
    pick.on_done(True)
    assert pick.poll() is None


def test_arm_failure_then_retry_is_failed():
    """The first result after the picking state is left ends the pick. An arm that leaves
    PICKING with false and tries again is no longer watched. A retry must stay inside the
    picking state and publish no result before the end."""
    pick = PickMonitor()
    pick.begin()
    pick.on_work_state('PICKING')
    pick.on_work_state('ERROR')
    pick.on_done(False)
    assert pick.poll() is False
    pick.on_work_state('PICKING')
    pick.on_work_state('DONE')
    pick.on_done(True)
    assert pick.poll() is None                  # nobody is waiting for it any more


def test_result_during_pick_counts_on_leaving():
    """pick_place_done is expected once, at the end. An intermediate false (retry inside
    PICKING) is the result the host holds when the state leaves picking; only a true that
    arrives before the next poll replaces it."""
    pick = PickMonitor()
    pick.begin()
    pick.on_work_state('PICKING')
    pick.on_done(False)                         # first attempt failed, the arm goes on
    pick.on_work_state('PICKING')
    assert pick.poll() is None
    pick.on_work_state('DONE')
    assert pick.poll() is False                 # the final true has not arrived yet


def test_arm_with_other_states_never_picking():
    """PICKING / ERROR are assumed names (settings pick.picking_states / error_states). With
    other names no result is accepted and the pick ends at its deadline (pick_dwell)."""
    pick = PickMonitor()
    pick.begin()
    pick.on_work_state('GRASPING')
    pick.on_work_state('FINISHED')
    pick.on_done(True)
    assert pick.poll() is None
    named = PickMonitor(picking_states=('GRASPING',), error_states=('FAILED',))
    named.begin()
    named.on_work_state('grasping')             # case and spaces do not matter
    named.on_work_state(' finished ')
    named.on_done(True)
    assert named.poll() is True


def test_result_without_work_state_rejected():
    """Both topics are needed: pick_place_done alone (work_state not bridged, not published)
    never ends a pick."""
    pick = PickMonitor()
    pick.begin()
    pick.on_done(True)
    assert pick.poll() is None
