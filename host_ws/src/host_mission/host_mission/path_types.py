"""Value types and command predicates shared by the path-command modules."""
from dataclasses import dataclass
from typing import Hashable, Optional, Tuple

from host_planning.region import DropRegion
from host_planning.types import Pose2D

from host_mission.arrival import ARRIVED_STATES, answers
from host_mission.goals import GoalPose

# PathCmd.mode values and the reasons a plan is withheld.
FOLLOW = 'FOLLOW'
STOP = 'STOP'
NO_PATH = 'NO_PATH'
YIELD = 'YIELD_TO_LEAD'
LEAD_TURNING = 'LEAD_TURNING'   # the follower holds its new leg while the lead turns in place
TRACKING_STATE = 'TRACKING'  # NavFeedback.state while the rover follows its path
PIVOT_DETAILS = ('PIVOT', 'ALIGNING')   # NavFeedback.detail while the leading pivot is made


@dataclass(frozen=True)
class PathCmd:
    """One PathCommand message plus host-side bookkeeping."""

    robot: str
    epoch: int
    cmd_id: int
    mode: str
    poses: Tuple[Pose2D, ...] = ()
    speed_cap: float = 0.0
    goal_pos_tol: float = 0.0
    goal_yaw_tol: float = 0.0
    pick_at_goal: bool = False
    lead_robot: str = ''
    gap: float = 0.0
    pivot_first: bool = False       # rotate in place at poses[0] to pivot_heading first
    pivot_heading: float = 0.0
    fixture_ahead: bool = False     # goal faces a known fixture: run-in exempt from front stop
    goal_key: Hashable = None       # host-side: which named goal this path serves
    goal: Optional[GoalPose] = None
    issued_at: float = 0.0
    provisional: bool = False
    region: Optional[DropRegion] = None
    via: bool = False               # the path ends at a turn-in point short of ``goal``
    hold: bool = False              # ... at a hold point, where it waits for the lead to park
    margin: float = 0.0


@dataclass(frozen=True)
class Desired:
    """What the mission wants from one rover right now. With ``region`` the goal is any feasible
    pose of it (``goal`` = the named seed pose)."""

    goal_key: Hashable = None
    goal: Optional[GoalPose] = None
    follow: bool = False
    pick_at_goal: bool = False
    region: Optional[DropRegion] = None
    goal_pos_tol: float = 0.0       # sent to the rover; 0 = its configured goal_pos_tol


@dataclass(frozen=True)
class Plan:
    """A planned path and how it was made."""

    poses: Tuple[Pose2D, ...]
    lead_robot: str
    provisional: bool
    goal: Pose2D
    pivot_heading: Optional[float] = None
    via: bool = False
    hold: bool = False
    margin: float = 0.0


@dataclass(frozen=True)
class Hold:
    """Why a rover is held (no re-plan until the map changes or ``retry_at``)."""

    reason: str                     # NO_PATH (pauses the mission) | YIELD
    obstacle_version: int
    retry_at: float


@dataclass
class Job:
    """A plan computation running on the runner."""

    key: tuple                      # what was asked: the result is void once this changes
    world: object
    result: object
    asked: float = 0.0


def on_the_way(cmd, arrived):
    """True while ``cmd`` is a path the rover has not yet arrived at."""
    return cmd is not None and cmd.mode == FOLLOW and not arrived


def past_the_turn(nav, cmd):
    """True when the rover reports it is past the leading pivot of ``cmd``."""
    if not answers(nav, cmd):
        return False
    if nav.state in ARRIVED_STATES:
        return True
    return nav.state == TRACKING_STATE and nav.detail.strip().upper() not in PIVOT_DETAILS


def serves(cmd, want):
    """True when ``cmd`` is the path already issued for ``want``."""
    if cmd is None or cmd.mode != FOLLOW or cmd.goal_key != want.goal_key:
        return False
    return cmd.goal == want.goal if want.region is None else cmd.region == want.region
