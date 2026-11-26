"""Arrival judgement (ROS-free). A rover has arrived at the goal of its current path command only
when ALL hold: it reports HOLDING or PARKED for that very command (same ``epoch`` and ``cmd_id``,
received after the command was issued, still fresh); the host pose is fresh and within the
position tolerance; the heading is within tolerance (only with ``heading_aligned``, skipped when
the yaw tolerance is 0). ``missed`` flags "stopped, but not at the goal" (the rover said
GOAL_MISSED, or claims arrival while the host pose disagrees); the supervisor turns it into a
bounded re-entry."""
import math
from dataclasses import dataclass

ARRIVED_STATES = ('HOLDING', 'PARKED')
STOPPED_STATE = 'STOPPED'
GOAL_MISSED = 'GOAL_MISSED'


@dataclass(frozen=True)
class ArrivalVerdict:

    arrived: bool = False
    missed: bool = False
    reason: str = ''                # why not arrived (diagnostic)
    reported: bool = False          # the rover explicitly reported the miss


def wrap_angle(angle):
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    return math.pi if wrapped == -math.pi else wrapped


def answers(nav, cmd):
    """True when the NavReport is the rover's answer to ``cmd`` (same epoch and cmd_id)."""
    return nav is not None and cmd is not None \
        and (nav.epoch, nav.cmd_id) == (cmd.epoch, cmd.cmd_id)


def feedback_for(state, cmd, now, feedback_timeout):
    """The rover's NavReport if it answers ``cmd`` and is fresh, else None."""
    nav = state.nav if state is not None else None
    if not answers(nav, cmd):
        return None
    if nav.time < cmd.issued_at or now - nav.time > feedback_timeout:
        return None
    return nav


def judge_arrival(state, cmd, now, pos_tol, yaw_tol, pose_timeout, feedback_timeout):
    """ArrivalVerdict of one rover against its current path command."""
    if cmd is None or cmd.mode != 'FOLLOW' or not cmd.poses:
        return ArrivalVerdict(reason='NO_COMMAND')
    if getattr(cmd, 'via', False):
        # the path ends at a turn-in point short of the goal: parked there is neither
        # arrived nor missed, the commander plans on from it
        return ArrivalVerdict(reason='TURN_IN_POINT')
    nav = feedback_for(state, cmd, now, feedback_timeout)
    if nav is None:
        return ArrivalVerdict(reason='NO_FEEDBACK')
    if nav.state == STOPPED_STATE and GOAL_MISSED in nav.detail.upper():
        return ArrivalVerdict(missed=True, reason=GOAL_MISSED, reported=True)
    if nav.state not in ARRIVED_STATES:
        return ArrivalVerdict(reason=f'ROVER_{nav.state}')
    if state.pose is None or now - state.pose_time > pose_timeout:
        return ArrivalVerdict(reason='STALE_POSE')
    return _check_pose(state, cmd.poses[-1], pos_tol, yaw_tol)


def _check_pose(state, goal, pos_tol, yaw_tol):
    x, y, yaw = state.pose
    if math.hypot(x - goal.x, y - goal.y) > pos_tol:
        return ArrivalVerdict(missed=True, reason='POS_ERROR')
    if yaw_tol > 0.0:
        if not state.heading_aligned:
            return ArrivalVerdict(reason='NOT_ALIGNED')
        if abs(wrap_angle(yaw - goal.theta)) > yaw_tol:
            return ArrivalVerdict(missed=True, reason='YAW_ERROR')
    return ArrivalVerdict(arrived=True)
