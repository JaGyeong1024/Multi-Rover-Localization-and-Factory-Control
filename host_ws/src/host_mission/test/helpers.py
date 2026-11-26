"""Shared builders for the path-mode tests (ROS-free)."""
import dataclasses
import os

from host_mission.goals import GoalStore
from host_mission.mission_engine import MissionEngine
from host_mission.settings import load_scenario, load_settings, read_settings_yaml

PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Unit tests read FROZEN copies, never the shipped config (tuning the site must not break
# logic tests); test_field_config.py checks the shipped files.
DATA = os.path.join(PACKAGE, 'test', 'data')
FIELD_CONFIG = os.path.join(PACKAGE, 'config')
REPO = os.path.abspath(os.path.join(PACKAGE, '..', '..', '..'))
SITE_CONFIG = os.path.join(REPO, 'config', 'site')
HARDWARE_CONFIG = os.path.join(REPO, 'config', 'hardware')
# Shipped mission settings = mission.yaml + these overlays (later wins).
FIELD_OVERLAYS = (os.path.join(SITE_CONFIG, 'arena.yaml'),
                  os.path.join(HARDWARE_CONFIG, 'rovers.yaml'))
EPOCH = 1_700_000_000
ROBOTS = ('leader', 'follower')


def field_settings_raw():
    """Shipped mission settings as the merged mapping."""
    return read_settings_yaml(os.path.join(FIELD_CONFIG, 'mission.yaml'), FIELD_OVERLAYS)


def settings():
    return load_settings(os.path.join(DATA, 'mission.yaml'))


def scenario():
    return load_scenario(os.path.join(DATA, 'mission_scenario.yaml'))


def goals():
    """Example goal table, detached from the file (tests must not rewrite it)."""
    store = GoalStore.load(os.path.join(DATA, 'goals.yaml'))
    store.path = None
    return store


def engine(planner=None, goal_store=None, **scenario_overrides):
    scen = dataclasses.replace(scenario(), **scenario_overrides)
    return MissionEngine(settings(), scen, goal_store or goals(), EPOCH, planner)


def place(world, robot, x, y, yaw, now, aligned=True, quality='GOOD'):
    """Feed a fresh, usable pose of one rover."""
    world.update_pose(robot, x, y, yaw, now)
    world.update_heading_aligned(robot, aligned)
    world.update_quality(robot, quality)


def place_at_goal(eng, robot, name, now):
    goal = eng.world.goals.get(name)
    place(eng.world, robot, goal.x, goal.y, goal.theta, now)


def report(eng, robot, state, now, detail=''):
    """NavFeedback answering the rover's current command."""
    cmd = eng.commands().get(robot)
    eng.world.update_nav(robot, state, detail, cmd.epoch if cmd else 0,
                         cmd.cmd_id if cmd else 0, now)


class Clock:
    """Drives an engine with poses refreshed every tick (so nothing goes stale)."""

    def __init__(self, eng, start=None, dt=0.2):
        self.eng, self.dt, self.now = eng, dt, 0.0
        self.pose = dict(start or {'leader': (2.6, 3.3, 0.0), 'follower': (1.9, 3.3, 0.0)})
        self.aligned = {robot: True for robot in self.pose}    # heading_aligned fed per rover
        self.events = []

    def move_to_goal(self, robot):
        """Teleport a rover onto the goal of its current command and report arrival."""
        cmd = self.eng.commands()[robot]
        goal = cmd.poses[-1]
        self.pose[robot] = (goal.x, goal.y, goal.theta)
        self._feed()
        report(self.eng, robot, 'HOLDING' if cmd.pick_at_goal else 'PARKED', self.now)

    def _feed(self):
        for robot, pose in self.pose.items():
            place(self.eng.world, robot, *pose, self.now, aligned=self.aligned[robot])

    def tick(self, count=1):
        new = []
        for _ in range(count):
            self.now += self.dt
            self._feed()
            new += self.eng.tick(self.now)
        self.events += new
        return new

    def kinds(self):
        return [kind for kind, _detail in self.events]
