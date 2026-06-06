"""Mission state machine — the core of host_mission (조율). ROS-free, unit-testable.

  IDLE ──start──> RUNNING ──all delivered & home──> DONE
                   │  ▲                                  ▲
        LOST/timeout│  │auto-resume (health) / resume()  │ (re)start on new dispatch
                   ▼  │                                   │
                 PAUSED ──abort()──> ABORTED ─────────────┘

RUNNING sub-phases (capacity-batch on a dual lane): TO_PART → PICKING → DELIVERING.

Robustness (vs a happy-path skeleton):
  • every wait is BOUNDED — per-phase timeout → PAUSED + diagnostic, never a silent hang.
  • health LOST → PAUSED (modes STOP); AUTO-resumes when health clears. manual pause()
    stays until resume(). timeout pause stays until resume() (operator looks).
  • abort()/clear are first-class transitions (no field-poking from the node).
  • DELIVER is DECOUPLED — the follower drops as soon as IT reaches UNLOADING; the leader
    returns to STANDBY independently; mission is DONE only when both are done. A slow or
    stuck leader never blocks the drop.
  • PICKING is a result-driven handoff — completes on pick_complete(ok) (teammate A,
    pick_mode='signal') or a mock dwell (pick_mode='dwell'); a failed pick marks the item
    'failed' and moves on instead of hanging.

tick(now, arrived, health) -> (targets, modes, events).
  arrived = {robot: bool} each at its current target; health = {robot: GOOD|DEGRADED|LOST|UNKNOWN}
  modes   = {robot: 'GOTO'|'STOP'}; events = list of (kind, detail) for logging/dashboard.
"""

_HOME_MODES = {'leader': 'GOTO', 'follower': 'GOTO'}


class MissionStateMachine:
    def __init__(self, queue, standby_node, unloading_node, capacity,
                 pick_dwell=8.0, drop_dwell=2.0, phase_timeout=90.0,
                 follower_home_node=None):
        self.queue = queue
        self.standby = int(standby_node)          # leader wait zone
        self.unloading = int(unloading_node)       # follower drop-off (하역)
        # follower's OWN wait zone (separate from drop-off); defaults to unloading.
        self.follower_home = int(follower_home_node) if follower_home_node is not None \
            else int(unloading_node)
        self.capacity = int(capacity)
        self.pick_dwell = float(pick_dwell)   # fallback if no pick_complete() signal arrives
        self.drop_dwell = float(drop_dwell)
        self.phase_timeout = float(phase_timeout)

        self.state = 'IDLE'                   # IDLE|RUNNING|PAUSED|ABORTED|DONE
        self.phase = None                     # TO_PART|PICKING|DELIVERING (when RUNNING)
        self.targets = {'leader': self.standby, 'follower': self.follower_home}
        self.modes = dict(_HOME_MODES)
        self.active_id = ''

        self.batch = []
        self.bidx = 0
        self._phase_start = 0.0
        self._pick_deadline = 0.0
        self._pick_done = False
        self._pick_ok = True
        self._dropped = False
        self._drop_deadline = None
        self._resume_phase = None
        self._pause_reason = None             # 'health'|'manual'|'timeout'

    # ---------- external signals ----------
    def start(self, now):
        if self.state in ('IDLE', 'DONE', 'ABORTED') and self.queue.pending():
            self._begin_batch(now)
            return [('mission_started', '')]
        return []

    def pick_complete(self, ok=True):
        """Teammate A reports pick&place result (pick_mode='signal')."""
        self._pick_done = True
        self._pick_ok = bool(ok)

    def pause(self):
        if self.state == 'RUNNING':
            self._resume_phase = self.phase
            self._pause_reason = 'manual'
            self.state = 'PAUSED'

    def resume(self):
        if self.state == 'PAUSED':
            self.state = 'RUNNING'
            self.phase = self._resume_phase
            self._resume_phase = None
            self._pause_reason = None

    def abort(self):
        for it in self.queue.items:
            if it.status == 'active':
                it.status = 'failed'
        self.batch = []
        self.active_id = ''
        self.phase = None
        self.state = 'ABORTED'
        self.targets = {'leader': self.standby, 'follower': self.unloading}
        self.modes = dict(_HOME_MODES)

    # ---------- internal transitions ----------
    def _begin_batch(self, now):
        pend = sorted(self.queue.pending(), key=lambda it: it.lane_node)  # up-lane order
        self.batch = pend[:self.capacity]
        for it in self.batch:
            it.status = 'active'
            it.part_state = 'ASSIGNED'
        self.bidx = 0
        self.state = 'RUNNING'
        self._enter_to_part(now)

    def _enter_to_part(self, now):
        self.phase = 'TO_PART'
        node = self.batch[self.bidx].lane_node
        self.targets = {'leader': node, 'follower': node}
        self.modes = dict(_HOME_MODES)
        self.active_id = self.batch[self.bidx].id
        self._phase_start = now

    def _enter_picking(self, now):
        self.phase = 'PICKING'
        self._pick_deadline = now + self.pick_dwell
        self._pick_done = False
        self._pick_ok = True
        self._phase_start = now

    def _enter_delivering(self, now):
        self.phase = 'DELIVERING'
        self.targets = {'leader': self.standby, 'follower': self.unloading}
        self.modes = dict(_HOME_MODES)
        self.active_id = ''
        self._dropped = False
        self._drop_deadline = None
        self._phase_start = now

    def _go_paused(self, reason):
        self._resume_phase = self.phase
        self._pause_reason = reason
        self.state = 'PAUSED'

    # ---------- main tick ----------
    def tick(self, now, arrived, health=None):
        events = []
        health = health or {}
        lost = any(h == 'LOST' for h in health.values())

        # health pause / auto-resume
        if self.state == 'RUNNING' and lost:
            self._go_paused('health')
            events.append(('paused', 'health_lost'))
        elif self.state == 'PAUSED' and self._pause_reason == 'health' and not lost:
            self.state = 'RUNNING'
            self.phase = self._resume_phase
            self._resume_phase = None
            self._pause_reason = None
            self._phase_start = now          # extend timeout after a pause
            events.append(('resumed', ''))

        if self.state != 'RUNNING':
            if self.state == 'PAUSED':
                self.modes = {r: 'STOP' for r in self.targets}
            return dict(self.targets), dict(self.modes), events

        # liveness: no phase waits forever
        if now - self._phase_start > self.phase_timeout:
            self._go_paused('timeout')
            self.modes = {r: 'STOP' for r in self.targets}
            events.append(('phase_timeout', self.phase or ''))
            return dict(self.targets), dict(self.modes), events

        if self.phase == 'TO_PART':
            if arrived.get('leader') and arrived.get('follower'):
                self._enter_picking(now)

        elif self.phase == 'PICKING':
            # complete on teammate A's pick_complete() signal, or a dwell-timeout fallback
            if self._pick_done or now >= self._pick_deadline:
                it = self.batch[self.bidx]
                if self._pick_done and not self._pick_ok:
                    it.status = 'failed'
                    events.append(('pick_failed', it.id))
                else:
                    it.part_state = 'LOADED'
                    events.append(('loaded', it.id))
                self.bidx += 1
                if self.bidx < len(self.batch):
                    self._enter_to_part(now)
                else:
                    self._enter_delivering(now)

        elif self.phase == 'DELIVERING':
            # follower drops at the 하역 zone, then returns to its OWN wait zone; leader returns
            # to standby — both decoupled. (Travel to pickups was platooning; only the ends split.)
            just_dropped = False
            if not self._dropped and arrived.get('follower'):
                if self._drop_deadline is None:
                    self._drop_deadline = now + self.drop_dwell
                elif now >= self._drop_deadline:
                    for it in self.batch:
                        if it.status == 'active':
                            it.status = 'done'
                            it.part_state = 'DELIVERED'
                            events.append(('delivered', it.id))
                    self._dropped = True
                    self.targets = {'leader': self.standby, 'follower': self.follower_home}
                    just_dropped = True
            if self._dropped and not just_dropped:
                if self.queue.pending():
                    self._begin_batch(now)
                    events.append(('next_batch', ''))
                elif arrived.get('leader') and arrived.get('follower'):
                    self.state = 'DONE'
                    self.phase = None
                    events.append(('done', ''))

        return dict(self.targets), dict(self.modes), events

    # ---------- 관제 projection ----------
    def system_state(self):
        """Operation state for 관제/dashboard, projected from internal phase/flags.
        IDLE | PLATOONING | PICK_PLACE | DELIVERING | UNLOADING | RETURNING |
        PAUSED | FAULT | ABORTED."""
        if self.state in ('IDLE', 'DONE'):
            return 'IDLE'
        if self.state == 'ABORTED':
            return 'ABORTED'
        if self.state == 'PAUSED':
            return 'FAULT' if self._pause_reason == 'timeout' else 'PAUSED'
        if self.phase == 'TO_PART':
            return 'PLATOONING'
        if self.phase == 'PICKING':
            return 'PICK_PLACE'
        if self.phase == 'DELIVERING':
            if self._dropped:
                return 'RETURNING'
            if self._drop_deadline is not None:
                return 'UNLOADING'
            return 'DELIVERING'
        return 'IDLE'

    def robot_state(self, robot, arrived):
        """Per-robot mission state (role-specific projection) for the dashboard —
        derived from the single mission SM (no per-robot SM fragmentation)."""
        if self.state in ('IDLE', 'DONE'):
            return 'IDLE'
        if self.state == 'ABORTED':
            return 'ABORTED'
        if self.state == 'PAUSED':
            return 'FAULT' if self._pause_reason == 'timeout' else 'PAUSED'
        if self.phase == 'TO_PART':
            return 'AT_PICKUP' if arrived else 'MOVING'
        if self.phase == 'PICKING':
            return 'PICKING' if robot == 'leader' else 'RECEIVING'
        if self.phase == 'DELIVERING':
            if robot == 'follower':
                if self._dropped:
                    return 'RETURNING'           # heading back to its own wait zone
                return 'UNLOADING' if self._drop_deadline is not None else 'DELIVERING'
            return 'IDLE' if arrived else 'RETURNING'
        return 'IDLE'
