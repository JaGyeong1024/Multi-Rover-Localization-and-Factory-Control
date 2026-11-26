"""Test-only stand-ins for the rover side, used with ``host_sim.scenario.run_scenario``.

``IdealTracker`` is NOT the rover tracker (that lives in rover_drive): it is the
simplest thing that follows a PathCmd: command acceptance by epoch/cmd_id, the leading
pivot (turn in place first), pure pursuit on the measured pose, a gap hold behind the
lead rover, and the HOLDING / PARKED / STOPPED report. ``MissionLoop`` closes the loop
between a MissionEngine and two trackers inside the simulator."""
import math

import numpy as np


class IdealTracker:
    def __init__(self, name, speed=0.15, lookahead=0.12, slow_dist=0.30, min_speed=0.03,
                 max_omega=1.5, pivot_tol=0.10):
        self.name = name
        self.pivot_tol = pivot_tol
        self.speed, self.lookahead = speed, lookahead
        self.slow_dist, self.min_speed, self.max_omega = slow_dist, min_speed, max_omega
        self.cmd = None
        self.state, self.detail = 'IDLE', ''
        self._pts = np.zeros((0, 2))
        self._s = np.zeros(0)
        self._idx = None
        self._pivot_to = None       # heading still to turn to in place (PathCmd.pivot_first)

    def accept(self, cmd):
        """Take a PathCmd when its epoch / cmd_id is newer; the same id is a keep-alive."""
        cur = self.cmd
        if cur is not None and (cmd.epoch, cmd.cmd_id) <= (cur.epoch, cur.cmd_id):
            return
        self.cmd = cmd
        self._pts = np.array([[p.x, p.y] for p in cmd.poses]).reshape(-1, 2)
        steps = np.hypot(*np.diff(self._pts, axis=0).T) if len(self._pts) > 1 else np.zeros(0)
        self._s = np.concatenate([[0.0], np.cumsum(steps)])
        self._idx = None            # projected onto the path on the first control tick
        self._pivot_to = cmd.pivot_heading if cmd.pivot_first else None
        self.state, self.detail = ('TRACKING', '') if cmd.mode == 'FOLLOW' else ('IDLE', '')

    @property
    def epoch(self):
        return 0 if self.cmd is None else self.cmd.epoch

    @property
    def cmd_id(self):
        return 0 if self.cmd is None else self.cmd.cmd_id

    def control(self, obs):
        """run_scenario controller: base Twist (v, w)."""
        if self.cmd is None or self.cmd.mode != 'FOLLOW' or obs.pose is None \
                or self.state != 'TRACKING':
            return 0.0, 0.0
        pose = obs.pose
        if self._pivot_to is not None:      # the path was planned from this heading, not the current one
            turn = _wrap(self._pivot_to - pose.theta)
            if abs(turn) > self.pivot_tol:
                return 0.0, math.copysign(min(self.max_omega, 0.5 + abs(turn)), turn)
            self._pivot_to = None
        goal = self.cmd.poses[-1]
        ahead = (goal.x - pose.x) * math.cos(goal.theta) + (goal.y - pose.y) * math.sin(goal.theta)
        self._advance(pose)
        remaining = self._s[-1] - self._s[self._idx]
        if len(self._pts) < 2 or (remaining < self.lookahead and ahead <= 0.0):
            return self._finish(pose, goal)
        if self._lead_too_close(obs):
            return 0.0, 0.0
        slow = self.speed * max(ahead, remaining) / self.slow_dist
        v = min(self.speed, max(self.min_speed, slow))
        tx, ty = self._target(goal)
        alpha = _wrap(math.atan2(ty - pose.y, tx - pose.x) - pose.theta)
        dist = max(math.hypot(tx - pose.x, ty - pose.y), 1e-3)
        w = 2.0 * v * math.sin(alpha) / dist
        return v, max(-self.max_omega, min(self.max_omega, w))

    def _advance(self, pose):
        """Monotonic nearest-point progress along the path."""
        if self._idx is None:  # new command: project onto the whole path
            self._idx = int(np.argmin(np.hypot(self._pts[:, 0] - pose.x,
                                               self._pts[:, 1] - pose.y)))
        window = self._pts[self._idx:self._idx + 20]
        self._idx += int(np.argmin(np.hypot(window[:, 0] - pose.x, window[:, 1] - pose.y)))

    def _target(self, goal):
        """Point one lookahead ahead; beyond the goal it continues on the goal heading."""
        want = self._s[self._idx] + self.lookahead
        if want <= self._s[-1]:
            j = int(np.searchsorted(self._s, want))
            return self._pts[j]
        extra = want - self._s[-1]
        return goal.x + extra * math.cos(goal.theta), goal.y + extra * math.sin(goal.theta)

    def _lead_too_close(self, obs):
        lead = obs.poses.get(self.cmd.lead_robot) if self.cmd.lead_robot else None
        return lead is not None and \
            math.hypot(lead.x - obs.pose.x, lead.y - obs.pose.y) < self.cmd.gap

    def _finish(self, pose, goal):
        pos_ok = math.hypot(pose.x - goal.x, pose.y - goal.y) <= self.cmd.goal_pos_tol
        yaw_ok = self.cmd.goal_yaw_tol <= 0.0 or \
            abs(_wrap(pose.theta - goal.theta)) <= self.cmd.goal_yaw_tol
        if pos_ok and yaw_ok:
            self.state = 'HOLDING' if self.cmd.pick_at_goal else 'PARKED'
        else:
            self.state, self.detail = 'STOPPED', 'GOAL_MISSED'
        return 0.0, 0.0


class FakePicker:
    """Leader pick coordinator stand-in: HOLDING -> PICKING -> DONE + pick_place_done."""

    def __init__(self, engine, duration=2.0, ok=True, repeat_delay=2.0):
        self.engine, self.duration, self.ok = engine, duration, ok
        self.repeat_delay = repeat_delay
        self._started = None
        self._ready_at = 0.0
        self.picks = 0
        engine.on_work_state('NAVIGATING')

    def step(self, nav_state, now):
        if self._started is None:
            if nav_state == 'HOLDING' and now >= self._ready_at:
                self._started = now
                self.engine.on_work_state('PICKING')
        elif now - self._started >= self.duration:
            self._started = None
            self._ready_at = now + self.repeat_delay
            self.picks += 1
            self.engine.on_work_state('DONE' if self.ok else 'ERROR')
            self.engine.on_pick_done(self.ok)


class MissionLoop:
    """MissionEngine + trackers + fake picker, stepped from run_scenario controllers."""

    def __init__(self, engine, trackers, tick_period=0.2, auto_unload=True):
        self.engine, self.trackers = engine, trackers
        self.picker = FakePicker(engine)
        self.tick_period, self.auto_unload = tick_period, auto_unload
        self._next_tick = 0.0
        self.events = []
        self.states = []

    def controllers(self):
        first = next(iter(self.trackers))
        return {name: (lambda obs, n=name: self._control(obs, n, n == first))
                for name in self.trackers}

    def _control(self, obs, name, is_first):
        if is_first:
            self._host_step(obs)
        return self.trackers[name].control(obs)

    def _host_step(self, obs):
        now, world = obs.t, self.engine.world
        for name, sample in obs.poses.items():
            if sample is not None and name in self.trackers:
                world.update_pose(name, sample.x, sample.y, sample.theta, now)
                world.update_heading_aligned(name, True)
                world.update_quality(name, 'GOOD')
        for name, tracker in self.trackers.items():
            world.update_nav(name, tracker.state, tracker.detail, tracker.epoch,
                             tracker.cmd_id, now)
        self.picker.step(self.trackers['leader'].state, now)
        if now < self._next_tick:
            return
        self._next_tick = now + self.tick_period
        if self.auto_unload and self.engine.seq.system_state() == 'UNLOADING':
            self.engine.unload_done()
        self.events += self.engine.tick(now)
        state = self.engine.seq.system_state()
        if not self.states or self.states[-1] != state:
            self.states.append(state)
        for name, cmd in self.engine.commands().items():
            self.trackers[name].accept(cmd)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))
