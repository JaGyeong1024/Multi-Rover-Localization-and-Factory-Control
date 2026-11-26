"""Named goal poses (ROS-free): name -> (x, y, theta) in ``uwb_global``.

A goal is taught by parking the rover and storing its ``/global/position``: the tracked-point bias
is then in both the stored goal and the pose used while driving, so it cancels out. YAML::

    frame_id: uwb_global
    goals:
      leader_standby: {x: 2.6, y: 3.3, theta: 0.0}
      load_gear: {relative_to: pick_gear, dx: 0.0, dy: -0.55, dtheta: 0.0}

A relative goal is placed in the frame of another goal (dx forward, dy left) and follows it when
it is re-taught; teaching the relative goal itself replaces it with the taught pose."""
import math
import os
from dataclasses import dataclass

import yaml

FRAME_ID = 'uwb_global'


@dataclass(frozen=True)
class GoalPose:

    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class RelativeGoal:

    relative_to: str
    dx: float
    dy: float
    dtheta: float


class GoalStore:

    def __init__(self, goals=None, path=None):
        self._goals = dict(goals or {})
        self.path = path

    @classmethod
    def load(cls, path):
        """Read a goal file; a missing file gives an empty store bound to ``path``."""
        if not os.path.exists(path):
            return cls({}, path)
        with open(path) as stream:
            data = yaml.safe_load(stream) or {}
        goals = {str(name): _pose_from_dict(name, entry)
                 for name, entry in (data.get('goals') or {}).items()}
        store = cls(goals, path)
        store._check_relative_goals()
        return store

    def _check_relative_goals(self):
        """Raise ValueError naming a relative goal whose base is unknown or in a cycle."""
        for name, entry in self._goals.items():
            seen, current = {name}, entry
            while isinstance(current, RelativeGoal):
                base = current.relative_to
                if base not in self._goals:
                    raise ValueError(f'goal {name!r}: relative_to unknown goal {base!r}')
                if base in seen:
                    raise ValueError(f'goal {name!r}: relative_to cycle through {base!r}')
                seen.add(base)
                current = self._goals[base]

    def get(self, name):
        """Goal pose by name (relative goals resolved), or None when unknown or broken."""
        return self._resolve(name, set())

    def _resolve(self, name, seen):
        entry = self._goals.get(name)
        if not isinstance(entry, RelativeGoal):
            return entry
        if name in seen:
            return None                            # reference cycle
        base = self._resolve(entry.relative_to, seen | {name})
        if base is None:
            return None
        c, s = math.cos(base.theta), math.sin(base.theta)
        return GoalPose(base.x + c * entry.dx - s * entry.dy,
                        base.y + s * entry.dx + c * entry.dy,
                        math.atan2(math.sin(base.theta + entry.dtheta),
                                   math.cos(base.theta + entry.dtheta)))

    def set(self, name, pose):
        if not name or not isinstance(name, str):
            raise ValueError(f'goal name must be a non-empty string, got {name!r}')
        self._goals[name] = pose

    def names(self):
        return sorted(self._goals)

    def __contains__(self, name):
        return name in self._goals

    def as_dict(self):
        return {name: _entry_to_dict(p) for name, p in sorted(self._goals.items())}

    def resolved(self):
        """{name: {x, y, theta}} with relative goals resolved (display); broken ones dropped."""
        poses = {name: self.get(name) for name in sorted(self._goals)}
        return {name: _entry_to_dict(p) for name, p in poses.items() if p is not None}

    def save(self, path=None):
        """Write the table to YAML, through a symlink if ``path`` is one (symlink-install)."""
        target = os.path.realpath(path or self.path)
        temp = target + '.tmp'
        with open(temp, 'w') as stream:
            yaml.safe_dump({'frame_id': FRAME_ID, 'goals': self.as_dict()},
                           stream, sort_keys=False)
        os.replace(temp, target)
        return target


def _entry_to_dict(entry):
    if isinstance(entry, RelativeGoal):
        return {'relative_to': entry.relative_to, 'dx': round(entry.dx, 4),
                'dy': round(entry.dy, 4), 'dtheta': round(entry.dtheta, 4)}
    return {'x': round(entry.x, 4), 'y': round(entry.y, 4), 'theta': round(entry.theta, 4)}


def _pose_from_dict(name, entry):
    if isinstance(entry, dict) and 'relative_to' in entry:
        try:
            return RelativeGoal(str(entry['relative_to']), float(entry.get('dx', 0.0)),
                                float(entry.get('dy', 0.0)), float(entry.get('dtheta', 0.0)))
        except (TypeError, ValueError) as error:
            raise ValueError(f'goal {name!r}: bad relative goal ({error})') from error
    try:
        return GoalPose(float(entry['x']), float(entry['y']), float(entry['theta']))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'goal {name!r}: need numeric x, y, theta ({error})') from error
