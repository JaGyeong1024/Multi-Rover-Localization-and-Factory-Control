"""Idle heading rule (idle_turn.py) and the frozen-map rule (obstacle_mapper.integrates_scans)."""
import math

from host_mission.goals import GoalPose
from host_mission.idle_turn import IdleTurns, turn_steps
from host_mission.obstacle_mapper import integrates_scans

MAX_PIVOT = 2.88


def test_half_turn_clockwise_in_chord_steps():
    home = GoalPose(0.86, 1.95, math.pi)
    steps = turn_steps(home, 0.0, MAX_PIVOT, -0.063, 0.15, 0.05)   # follower, centre ahead
    assert [round(s.theta, 3) for s in steps] == [round(t, 3) for t in
                                                  (0.75 * math.pi, math.pi / 2, math.pi / 4, 0.0)]
    assert math.isclose(steps[-1].x, 0.86 - 0.126, abs_tol=1e-9)
    assert math.isclose(steps[-1].y, 1.95, abs_tol=1e-9)
    assert math.isclose(steps[1].x, 0.797, abs_tol=1e-9)
    assert math.isclose(steps[1].y, 1.95 - 0.063, abs_tol=1e-9)
    chords = [math.dist((a.x, a.y), (b.x, b.y)) for a, b in zip([home] + steps, steps)]
    assert max(chords) <= 0.05


def test_leader_half_turn_two_steps():
    steps = turn_steps(GoalPose(0.86, 1.45, math.pi), 0.0, MAX_PIVOT, 0.0, 0.15, 0.05)
    assert [(s.x, s.y, round(s.theta, 3)) for s in steps] == \
        [(0.86, 1.45, round(math.pi / 2, 3)), (0.86, 1.45, 0.0)]


def test_no_steps_facing_heading_one_for_small():
    assert turn_steps(GoalPose(1.0, 1.0, 0.05), 0.0, MAX_PIVOT, 0.0, 0.15, 0.05) == []
    steps = turn_steps(GoalPose(1.0, 1.0, 1.0), 0.0, MAX_PIVOT, 0.0, 0.15, 0.05)
    assert len(steps) == 1 and steps[0] == GoalPose(1.0, 1.0, 0.0)


class _Plan:
    max_pivot, pivot_offset = MAX_PIVOT, 0.0


class _Robot:
    plan, arrive_yaw_tol, goal_pos_tol = _Plan(), 0.15, 0.05


def test_turn_starts_home_ends_not_under_way():
    turns, events = IdleTurns(0.0, {'leader': _Robot()}), []
    home = GoalPose(0.86, 1.45, math.pi)
    assert turns.wanted('leader', 'standby', home, None, events) == ('standby', home)
    key, goal = turns.wanted('leader', 'standby', home, 'standby', events)
    assert key == 'standby@turn1' and events and turns.under_way('leader', 'standby')
    key, goal = turns.wanted('leader', 'standby', home, 'standby@turn1', events)
    assert key == 'standby@turn2' and goal.theta == 0.0
    assert turns.under_way('leader', 'standby@turn1')
    assert not turns.under_way('leader', 'standby@turn2')
    assert IdleTurns(None, {}).wanted('leader', 'standby', home, 'standby', []) == \
        ('standby', home)


def test_frozen_map_integrates_only_idle():
    assert integrates_scans('TO_PICKUP', False)
    assert integrates_scans(None, True) and integrates_scans('IDLE', True)
    assert not integrates_scans('TO_PICKUP', True) and not integrates_scans('UNLOADING', True)


def test_one_rover_turns_at_a_time():
    turns = IdleTurns(0.0, {'leader': _Robot(), 'follower': _Robot()})
    home = GoalPose(1.0, 1.0, math.pi)
    assert turns.wanted('leader', 'a', home, 'a', [])[0] == 'a@turn1'
    assert turns.wanted('follower', 'b', home, 'b', [])[0] == 'b'
    turns.wanted('leader', 'a', home, 'a@turn1', [])
    turns.wanted('leader', 'a', home, 'a@turn2', [])
    assert turns.wanted('follower', 'b', home, 'b', [])[0] == 'b@turn1'


def test_each_step_turns_about_rover():
    turns, home = IdleTurns(0.0, {'leader': _Robot()}), GoalPose(0.86, 1.45, math.pi)
    _key, goal = turns.wanted('leader', 'a', home, 'a', [], pose=(0.88, 1.46, 3.1))
    assert (goal.x, goal.y) == (0.88, 1.46) and math.isclose(goal.theta, math.pi / 2)
    _key, goal = turns.wanted('leader', 'a', home, 'a@turn1', [], pose=(0.93, 1.47, 1.5))
    assert (goal.x, goal.y, goal.theta) == (0.93, 1.47, 0.0)    # from where step 1 left it


def test_step_tolerance_covers_unknown_centre():
    turns = IdleTurns(0.0, {'leader': _Robot()}, centre_error=0.08)
    turns.wanted('leader', 'a', GoalPose(0.86, 1.45, math.pi), 'a', [])
    assert math.isclose(turns.step_tolerance('leader', 0.08), 0.08 + 0.16 * math.sin(math.pi / 4))
    assert turns.step_tolerance('follower', 0.08) == 0.08
