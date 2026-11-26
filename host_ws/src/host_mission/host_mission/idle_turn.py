"""Idle heading rule (ROS-free): an idle rover at its home goal turns in place to face
``idle_heading``. A turn larger than ``max_pivot`` is split into equal steps, each a goal pose
(the tracked point circles ``pivot_offset`` behind it) moving it at most ``chord``. One rover
turns at a time; ``IdleTurns`` tracks each rover's step."""
import math
from dataclasses import dataclass

from host_mission.goals import GoalPose

STEP_KEY = '{home}@turn{step}'      # goal key of a turn step (events, dashboard)


def turn_steps(home, heading, max_pivot, pivot_offset, tolerance, chord, away=None):
    """Goal poses of the turn from ``home`` to ``heading`` ([] if already facing it). A half
    turn sweeps the front away from ``away`` (the other rover's home), else clockwise."""
    turn = math.remainder(heading - home.theta, math.tau)
    if abs(turn) <= tolerance:
        return []
    if abs(abs(turn) - math.pi) < 1e-6:
        turn = -math.pi             # a half turn: clockwise unless that faces the other rover
        if away is not None:
            # halfway it faces home.theta + turn / 2: must not face the other rover (front stop)
            mid = home.theta + turn / 2.0
            if math.cos(mid) * (away[0] - home.x) + math.sin(mid) * (away[1] - home.y) > 0.0:
                turn = math.pi
    largest = max_pivot
    if abs(pivot_offset) > chord / 2.0:
        largest = min(largest, 2.0 * math.asin(chord / (2.0 * abs(pivot_offset))))
    count = max(1, math.ceil(abs(turn) / largest - 1e-9))
    cx = home.x - pivot_offset * math.cos(home.theta)
    cy = home.y - pivot_offset * math.sin(home.theta)
    steps = []
    for i in range(1, count + 1):
        theta = math.remainder(home.theta + turn * i / count, math.tau)
        steps.append(GoalPose(cx + pivot_offset * math.cos(theta),
                              cy + pivot_offset * math.sin(theta), theta))
    return steps


def turned_pose(pose, theta, pivot_offset):
    x, y, heading = pose
    cx, cy = x - pivot_offset * math.cos(heading), y - pivot_offset * math.sin(heading)
    return GoalPose(cx + pivot_offset * math.cos(theta), cy + pivot_offset * math.sin(theta),
                    theta)


@dataclass
class _Turn:
    home: str                       # home goal key the turn starts from
    steps: list                     # GoalPose per step
    start_theta: float = 0.0        # heading the turn starts from (the home goal's)
    index: int = 0                  # step being driven

    def key(self):
        return STEP_KEY.format(home=self.home, step=self.index + 1)


class IdleTurns:
    """Per rover: the step of its idle turn."""

    def __init__(self, heading, settings, centre_error=0.0):
        self.heading = heading      # None = rule off
        self.settings = settings    # robot name -> RobotSettings
        self.centre_error = centre_error   # m, see step_tolerance
        self.turns = {}
        self._at = {}               # robot -> ``at`` of its last ``wanted`` call
        self._queued = set()        # at home, waiting for the other rover's turn

    def clear(self):
        self.turns.clear()
        self._queued.clear()

    def wanted(self, robot, home_key, home, at, events, pose=None, away=None):
        """(goal key, GoalPose) wanted while idle: home, or the current turn step once ``at`` (the
        arrived goal key) is home. Each step turns about where the rover stands when it begins."""
        if self.heading is None or home is None:
            return home_key, home
        self._at[robot] = at
        turn = self.turns.get(robot)
        if turn is not None and turn.home != home_key:
            del self.turns[robot]           # home goal changed: start again
            turn = None
        if turn is None:
            if at != home_key:
                self._queued.discard(robot)
                return home_key, home
            if any(self.under_way(r, self._at.get(r)) for r in self.turns if r != robot):
                self._queued.add(robot)
                return home_key, home
            self._queued.discard(robot)
            cfg = self.settings[robot]
            steps = turn_steps(home, self.heading, cfg.plan.max_pivot, cfg.plan.pivot_offset,
                               cfg.arrive_yaw_tol, cfg.goal_pos_tol, away)
            if not steps:
                return home_key, home
            turn = self.turns[robot] = _Turn(home_key, steps, home.theta)
            self._from_here(turn, robot, pose)
            events.append(('idle_turn', f'{robot}: turning to {self.heading:.2f} rad in '
                                        f'{len(steps)} step(s)'))
        elif at == turn.key() and turn.index < len(turn.steps) - 1:
            turn.index += 1
            self._from_here(turn, robot, pose)
        return turn.key(), turn.steps[turn.index]

    def _from_here(self, turn, robot, pose):
        if pose is not None:
            turn.steps[turn.index] = turned_pose(pose, turn.steps[turn.index].theta,
                                                 self.settings[robot].plan.pivot_offset)

    def step_tolerance(self, robot, base):
        """``base`` plus the tracked-point error when the rotation centre is ``centre_error`` off."""
        turn = self.turns.get(robot)
        if turn is None:
            return base
        before = turn.steps[turn.index - 1].theta if turn.index else turn.start_theta
        step = abs(math.remainder(turn.steps[turn.index].theta - before, math.tau))
        return base + 2.0 * self.centre_error * math.sin(step / 2.0)

    def queued(self, robot):
        return robot in self._queued

    def under_way(self, robot, at):
        turn = self.turns.get(robot)
        return turn is not None and not (turn.index == len(turn.steps) - 1 and at == turn.key())
