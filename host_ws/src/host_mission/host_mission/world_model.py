"""Host world model (ROS-free): the single picture the mission logic reasons on.

Per rover: latest pose + host receive time, ``heading_aligned``, localization quality,
latest NavFeedback and SafetyState. Shared: obstacle list (circles / boxes, with optional
expiry) and the named goal poses. Freshness is judged on the host's own receive time;
header stamps are never used.

The lane-graph helpers (``nearest_node`` / ``arrived``) serve ``nav_mode: lane`` only."""
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from host_mission.goals import GoalStore

QUALITY_UNKNOWN = 'UNKNOWN'


@dataclass(frozen=True)
class NavReport:

    state: str
    detail: str
    epoch: int                      # echoes PathCommand.epoch (host run)
    cmd_id: int
    time: float                     # host receive time


@dataclass(frozen=True)
class SafetyReport:

    state: str
    reasons: Tuple[str, ...]
    speed_scale: float
    blocked: Optional[Tuple[float, float]]   # None when the rover sent NaN
    time: float


@dataclass(frozen=True)
class Obstacle:
    """Obstacle shape: an oriented ``width`` x ``height`` rectangle grown by ``r`` (a circle when
    the size is zero)."""

    x: float
    y: float
    r: float
    expires: Optional[float] = None
    source: str = 'manual'          # manual | lidar | mapped | fixture
    width: float = 0.0              # along yaw
    height: float = 0.0             # across yaw
    yaw: float = 0.0

    @classmethod
    def from_shape(cls, shape, expires=None, source='manual'):
        """From an (x, y, r) circle or an (x, y, r, width, height, yaw) padded box."""
        x, y, r, *box = map(float, shape)
        width, height, yaw = box if box else (0.0, 0.0, 0.0)
        return cls(x, y, r, expires, source, width, height, yaw)

    @property
    def shape(self):
        return (self.x, self.y, self.r, self.width, self.height, self.yaw)

    def contains(self, px, py):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = px - self.x, py - self.y
        ex = max(abs(c * dx + s * dy) - self.width / 2.0, 0.0)
        ey = max(abs(-s * dx + c * dy) - self.height / 2.0, 0.0)
        return math.hypot(ex, ey) <= self.r


@dataclass(frozen=True)
class DetectedRegion:
    """Area assumed blocked behind a rover's front stop, as planner shapes."""

    shapes: Tuple[Tuple[float, ...], ...]     # Obstacle.from_shape tuples
    expires: float

    def covers(self, x, y):
        return any(Obstacle.from_shape(s).contains(x, y) for s in self.shapes)


@dataclass
class RobotState:

    pose: Optional[Tuple[float, float, float]] = None   # x, y, yaw
    pose_time: float = -math.inf
    heading_aligned: bool = False
    quality: str = QUALITY_UNKNOWN
    nav: Optional[NavReport] = None
    safety: Optional[SafetyReport] = None


@dataclass
class _Obstacles:
    manual: list = field(default_factory=list)
    detected: list = field(default_factory=list)    # DetectedRegion
    mapped: list = field(default_factory=list)      # obstacle_mapper's latest set
    version: int = 0                # bumps on every change (re-plan trigger)


class WorldModel:
    def __init__(self, graph=None, position_tol=0.0, pose_timeout=0.5, goals=None):
        self.graph = graph
        self.tol = float(position_tol)
        self.pose_timeout = float(pose_timeout)
        self.goals = goals if goals is not None else GoalStore()
        self.robots = {}            # robot -> RobotState
        self._obstacles = _Obstacles()

    def robot(self, name):
        return self.robots.setdefault(name, RobotState())

    @property
    def pose(self):
        return {name: st.pose for name, st in self.robots.items() if st.pose is not None}

    def update_pose(self, robot, x, y, yaw, now=0.0):
        state = self.robot(robot)
        state.pose = (float(x), float(y), float(yaw))
        state.pose_time = float(now)

    def update_heading_aligned(self, robot, aligned):
        self.robot(robot).heading_aligned = bool(aligned)

    def update_quality(self, robot, quality):
        self.robot(robot).quality = str(quality).strip().upper() or QUALITY_UNKNOWN

    def update_nav(self, robot, state, detail, epoch, cmd_id, now):
        self.robot(robot).nav = NavReport(str(state).strip().upper(), str(detail),
                                          int(epoch), int(cmd_id), float(now))

    def update_safety(self, robot, state, reasons, speed_scale, blocked_x, blocked_y, now):
        finite = math.isfinite(blocked_x) and math.isfinite(blocked_y)
        blocked = (float(blocked_x), float(blocked_y)) if finite else None
        self.robot(robot).safety = SafetyReport(
            str(state).strip().upper(), tuple(str(r).strip().upper() for r in reasons),
            float(speed_scale), blocked, float(now))

    def pose_fresh(self, robot, now):
        """True when a pose arrived within ``pose_timeout`` (host receive time)."""
        state = self.robots.get(robot)
        return state is not None and state.pose is not None \
            and now - state.pose_time <= self.pose_timeout

    def usable_pose(self, robot, now, need_heading=True):
        """((x, y, yaw), '') when the pose may be used, else (None, reason)."""
        state = self.robots.get(robot)
        if state is None or state.pose is None:
            return None, 'NO_POSE'
        if now - state.pose_time > self.pose_timeout:
            return None, 'STALE_POSE'
        if need_heading and not state.heading_aligned:
            return None, 'NOT_ALIGNED'
        return state.pose, ''

    @property
    def obstacle_version(self):
        return self._obstacles.version

    def obstacles(self):
        obs = self._obstacles
        detected = [Obstacle.from_shape(shape, region.expires, 'lidar')
                    for region in obs.detected for shape in region.shapes]
        return list(obs.manual) + detected + list(obs.mapped)

    def set_manual_obstacles(self, shapes):
        """Replace the operator-entered list (Obstacle.from_shape tuples). True if changed."""
        new = [Obstacle.from_shape(s) for s in shapes]
        if new == self._obstacles.manual:
            return False
        self._obstacles.manual = new
        self._obstacles.version += 1
        return True

    def set_mapped_obstacles(self, shapes):
        """Replace the LiDAR-mapped list (Obstacle.from_shape tuples). True if changed."""
        new = [Obstacle.from_shape(s, source='mapped') for s in shapes]
        if new == self._obstacles.mapped:
            return False
        self._obstacles.mapped = new
        self._obstacles.version += 1
        return True

    def add_detected_region(self, shapes, point, expires):
        """Register the area behind a rover's front stop at face point ``point``."""
        x, y = point
        for i, region in enumerate(self._obstacles.detected):
            if region.covers(x, y):
                self._obstacles.detected[i] = DetectedRegion(
                    region.shapes, max(region.expires, expires))
                return False
        shapes = tuple(tuple(map(float, s)) for s in shapes)
        self._obstacles.detected.append(DetectedRegion(shapes, float(expires)))
        self._obstacles.version += 1
        return True

    def expire_obstacles(self, now):
        kept = [r for r in self._obstacles.detected if r.expires > now]
        if len(kept) == len(self._obstacles.detected):
            return False
        self._obstacles.detected = kept
        self._obstacles.version += 1
        return True

    def nearest_node(self, robot):
        state = self.robots.get(robot)
        if state is None or state.pose is None:
            return -1
        return self.graph.nearest_node(state.pose[0], state.pose[1])

    def arrived(self, robot, target_node):
        state = self.robots.get(robot)
        if state is None or state.pose is None or target_node is None or int(target_node) < 0:
            return False
        tx, ty = self.graph.xy(target_node)
        return math.hypot(state.pose[0] - tx, state.pose[1] - ty) <= self.tol
