"""Supervisor: host share of the exception table, limits included."""
import math

from host_planning.types import Pose2D

from host_mission.arrival import ArrivalVerdict
from host_mission.path_commander import PathCmd
from host_mission.settings import FreshnessSettings, SupervisorSettings
from host_mission.supervisor import Supervisor
from host_mission.world_model import WorldModel

CFG = SupervisorSettings()
ROBOT = 'leader'
NOT_ARRIVED = ArrivalVerdict()


class Rig:
    """One rover, fresh pose every step, helpers to inject rover reports."""

    def __init__(self):
        self.world = WorldModel(pose_timeout=0.5)
        self.sup = Supervisor(CFG, FreshnessSettings())
        self.now = 100.0
        self.cmd = self._cmd(1)
        self.verdict = NOT_ARRIVED
        self.blocked = False
        self.pose_alive = True
        self.world.update_quality(ROBOT, 'GOOD')

    def _cmd(self, cmd_id, goal='pick_gear', provisional=False):
        return PathCmd(robot=ROBOT, epoch=1, cmd_id=cmd_id, mode='FOLLOW',
                       poses=(Pose2D(0, 0, 0), Pose2D(1, 0, 0)), goal_key=goal,
                       issued_at=self.now, provisional=provisional)

    def new_cmd(self, goal='pick_gear', provisional=False):
        """What the engine does on a re-plan: next cmd_id."""
        self.cmd = self._cmd(self.cmd.cmd_id + 1, goal, provisional)

    def nav(self, state, detail=''):
        self.world.update_nav(ROBOT, state, detail, self.cmd.epoch, self.cmd.cmd_id,
                              self.now)

    def safety(self, reasons, blocked=(math.nan, math.nan)):
        state = 'STOPPED' if reasons else 'OK'
        self.world.update_safety(ROBOT, state, reasons, 0.0, *blocked, self.now)

    def step(self, dt=0.2, arrived=False):
        self.now += dt
        if self.pose_alive:
            self.world.update_pose(ROBOT, 0.0, 0.0, 0.0, self.now)
        return self.sup.evaluate(self.world, {ROBOT: self.cmd}, {ROBOT: self.verdict},
                                 {ROBOT: arrived}, {ROBOT: self.blocked}, self.now)

    def collect(self, seconds):
        """Action kinds raised over ``seconds``."""
        kinds = []
        for _ in range(int(round(seconds / 0.2))):
            kinds += _kinds(self.step())
        return kinds

    def run(self, seconds, **kwargs):
        out = None
        for _ in range(int(round(seconds / 0.2))):
            out = self.step(**kwargs)
        return out


def _kinds(out):
    return [a.kind for a in out.actions]


def test_stale_pose_pauses_then_recovers():
    rig = Rig()
    assert rig.step().health[ROBOT] == 'GOOD'
    rig.pose_alive = False
    assert rig.run(CFG.hold_pause_s - 1.0).health[ROBOT] == 'GOOD'
    out = rig.run(2.0)
    assert out.health[ROBOT] == 'LOST'
    rig.pose_alive = True
    out = rig.step()
    assert out.health[ROBOT] == 'GOOD' and ('hold_cleared', 'leader: STALE_POSE') in out.events


def test_rover_reports_stale_pose_and_lead():
    for reason in ('STALE_POSE', 'STALE_LEAD'):
        rig = Rig()
        events = []
        for _ in range(30):
            rig.safety([reason])
            out = rig.step()
            events += out.events
        assert out.health[ROBOT] == 'LOST' and ('hold', f'leader: {reason}') in events
        rig.safety([])
        assert rig.step().health[ROBOT] == 'GOOD'


def test_quality_lost_pauses_only_if_lasting():
    rig = Rig()
    rig.world.update_quality(ROBOT, 'LOST')
    assert rig.run(CFG.hold_pause_s - 1.0).health[ROBOT] == 'GOOD'
    assert rig.run(1.4).health[ROBOT] == 'LOST'
    rig.world.update_quality(ROBOT, 'DEGRADED')      # degraded only slows the rover down
    assert rig.step().health[ROBOT] == 'GOOD'


def test_no_path_pauses_resumes_with_plan():
    rig = Rig()
    rig.blocked = True
    assert rig.step().health[ROBOT] == 'LOST'
    rig.blocked = False
    assert rig.step().health[ROBOT] == 'GOOD'


def test_off_path_replans_to_limit_then_halts():
    rig = Rig()
    for attempt in range(1, CFG.off_path_max_replans + 1):
        rig.nav('STOPPED', 'OFF_PATH')
        assert _kinds(rig.step()) == ['replan'], attempt
        assert _kinds(rig.step()) == []
        rig.new_cmd()
    rig.nav('STOPPED', 'OFF_PATH')
    out = rig.step()
    assert _kinds(out) == ['halt'] and 'OFF_PATH' in out.actions[0].detail
    rig.new_cmd()
    rig.nav('STOPPED', 'OFF_PATH')
    assert _kinds(rig.step()) == []


def test_off_provisional_path_ends_creep():
    """The path of an unaligned rover is planned from a guessed heading and the rover creeps
    straight ahead wherever it leads: with its heading valid it stands beside that path. An
    alignment must not spend an OFF_PATH recovery or log a path failure."""
    rig = Rig()
    for _ in range(CFG.off_path_max_replans + 2):    # more often than the budget allows
        rig.new_cmd(provisional=True)
        rig.nav('STOPPED', 'OFF_PATH')
        out = rig.step()
        assert _kinds(out) == ['replan'] and out.actions[0].detail == 'HEADING_VALID'
        assert _kinds(rig.step()) == []
    rig.new_cmd()                                    # the path from the valid heading
    rig.nav('STOPPED', 'OFF_PATH')
    out = rig.step()
    assert _kinds(out) == ['replan'] and out.actions[0].detail == 'OFF_PATH (1/3)'


def test_off_path_for_replaced_command_ignored():
    rig = Rig()
    rig.nav('STOPPED', 'OFF_PATH')
    rig.new_cmd()                                    # feedback still carries the old cmd_id
    assert _kinds(rig.step()) == []


def test_recovery_budget_resets_on_arrival_resume():
    rig = Rig()
    for _ in range(CFG.off_path_max_replans):
        rig.nav('STOPPED', 'OFF_PATH')
        rig.step()
        rig.new_cmd()
    rig.nav('PARKED')
    rig.step(arrived=True)                           # goal reached: budget is per goal
    rig.new_cmd(goal='leader_standby')
    rig.nav('STOPPED', 'OFF_PATH')
    assert _kinds(rig.step()) == ['replan']
    rig.sup.reset()
    rig.new_cmd(goal='leader_standby')
    rig.nav('STOPPED', 'OFF_PATH')
    assert _kinds(rig.step()) == ['replan']


def test_stuck_replans_once_then_halts():
    rig = Rig()
    rig.nav('STOPPED', 'STUCK')
    assert _kinds(rig.step()) == ['replan']
    rig.new_cmd()
    rig.nav('STOPPED', 'STUCK')
    assert _kinds(rig.step()) == ['halt']


def test_stuck_needs_stopped_feedback():
    rig = Rig()
    rig.safety(['STUCK'])
    rig.nav('TRACKING')
    assert _kinds(rig.step()) == []
    rig.nav('STOPPED')
    assert _kinds(rig.step()) == ['replan']


def test_lidar_stop_registers_after_delay():
    rig = Rig()
    rig.nav('STOPPED', 'LIDAR_STOP')
    actions = []
    for _ in range(int(CFG.lidar_block_s / 0.2) - 1):
        rig.safety(['LIDAR_STOP'], blocked=(3.5, 3.0))
        actions += rig.step().actions
    assert actions == []                             # a passer-by does not become an obstacle
    for _ in range(3):
        rig.safety(['LIDAR_STOP'], blocked=(3.5, 3.0))
        actions += rig.step().actions
    assert [a.kind for a in actions] == ['add_obstacle', 'replan']
    assert actions[0].point == (3.5, 3.0)


def test_lidar_stop_at_goal_not_registered():
    """A rover parked/holding at its goal that sees a wall or shelf ahead is not "blocked
    on its way": no obstacle, no re-plan, however long the front stop lasts."""
    for state in ('PARKED', 'HOLDING'):
        rig = Rig()
        actions = []
        for _ in range(int(2 * CFG.lidar_block_s / 0.2)):
            rig.nav(state)
            rig.safety(['LIDAR_STOP'], blocked=(3.5, 3.0))
            actions += rig.step().actions
        assert actions == [], state


def test_lidar_stop_after_arrival_not_registered():
    """Arrival latched by the engine counts even when no fresh feedback is at hand."""
    rig = Rig()
    actions = []
    for _ in range(int(2 * CFG.lidar_block_s / 0.2)):
        rig.safety(['LIDAR_STOP'], blocked=(3.5, 3.0))
        actions += rig.step(arrived=True).actions
    assert actions == []


def test_cleared_lidar_stop_not_registered():
    rig = Rig()
    for _ in range(8):
        rig.safety(['LIDAR_STOP'], blocked=(3.5, 3.0))
        rig.step()
    rig.safety([])
    rig.step()
    actions = []
    for _ in range(10):
        rig.safety(['LIDAR_STOP'], blocked=(3.5, 3.0))
        actions += rig.step().actions
    assert actions == []


def test_lidar_stop_no_position_not_registered():
    rig = Rig()
    actions = []
    for _ in range(30):
        rig.safety(['LIDAR_STOP'])                   # blocked_x/y = NaN
        actions += rig.step().actions
    assert actions == []


def test_goal_miss_reenters_twice_then_halts():
    rig = Rig()
    rig.verdict = ArrivalVerdict(missed=True, reason='GOAL_MISSED', reported=True)
    for _ in range(CFG.goal_miss_max_reentries):
        assert _kinds(rig.step()) == ['reenter']
        rig.new_cmd()
    assert _kinds(rig.step()) == ['halt']


def test_host_detected_miss_must_persist():
    rig = Rig()
    rig.verdict = ArrivalVerdict(missed=True, reason='POS_ERROR')
    assert rig.collect(CFG.goal_miss_confirm_s - 0.4) == []
    rig.verdict = NOT_ARRIVED
    rig.step()
    rig.verdict = ArrivalVerdict(missed=True, reason='POS_ERROR')
    assert rig.collect(CFG.goal_miss_confirm_s - 0.4) == []
    assert rig.collect(1.0) == ['reenter']


def test_drift_after_latched_arrival_no_reenter():
    """A parked rover does not move: its pose drifting out of the host tolerance is
    localization drift, not a miss to drive a loop for."""
    rig = Rig()
    rig.verdict = ArrivalVerdict(missed=True, reason='POS_ERROR')
    assert all(_kinds(rig.step(arrived=True)) == [] for _ in range(30))


def test_stop_command_not_supervised():
    rig = Rig()
    rig.cmd = PathCmd(robot=ROBOT, epoch=1, cmd_id=9, mode='STOP', issued_at=rig.now)
    rig.nav('STOPPED', 'OFF_PATH')
    assert _kinds(rig.step()) == []
