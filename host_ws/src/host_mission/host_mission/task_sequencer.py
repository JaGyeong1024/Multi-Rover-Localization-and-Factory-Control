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
  • DELIVER is DECOUPLED — the follower's drop completes on the operator '하역완료'
    (unload_done) once it has reached UNLOADING; the leader returns to STANDBY independently;
    mission is DONE only when both are done. A slow/stuck leader never blocks the drop.
  • PICKING (관제: LOADING) completes on pick_complete() — driven by the picker's
    /pnp/working falling edge (node side) — or a pick_dwell fallback if no signal arrives.

tick(now, arrived, health) -> (targets, modes, events).
  arrived = {robot: bool} each at its current target; health = {robot: GOOD|DEGRADED|LOST|UNKNOWN}
  modes   = {robot: 'GOTO'|'STOP'}; events = list of (kind, detail) for logging/dashboard.
"""

_HOME_MODES = {'leader': 'GOTO', 'follower': 'GOTO'}


class MissionStateMachine:
    def __init__(self, queue, standby_node, unloading_node, capacity,
                 pick_dwell=8.0, drop_dwell=2.0, phase_timeout=90.0,
                 follower_home_node=None, order_cost=None, cooldown=0.0):
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
        # order_cost(a_node, b_node) -> directed travel cost; None => keep FIFO order
        self._order_cost = order_cost
        self.cooldown = float(cooldown)       # per-shelf pick cooldown (s); 0 = off
        self._shelf_ready = {}                # shelf node -> earliest next-pick time

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
        self._unload_done = False             # operator '하역완료' (manual unload gate)
        self._unloading = False               # follower at unloading awaiting done (UNLOADING proj)
        self._estop = False                   # E-stop engaged (latched safety line)
        self._resume_phase = None
        self._pause_reason = None             # 'health'|'manual'|'timeout'|'estop'
        self._just_resumed = False            # refresh phase timers on the next tick

    # ---------- external signals ----------
    def start(self, now):
        if self.state in ('IDLE', 'DONE', 'ABORTED') and self.queue.pending():
            if self._begin_batch(now):
                return [('mission_started', '')]
        return []

    def pick_complete(self, ok=True):
        """Picker reports pick&place result — driven by the /pnp/working falling edge
        (node side); ok defaults True (success/fail not modeled for now)."""
        self._pick_done = True
        self._pick_ok = bool(ok)

    def unload_done(self):
        """Operator pressed '하역완료' — release the manual unload gate."""
        self._unload_done = True

    def cancel_item(self, item_id, now):
        """Operator cancels one part. PROTECTED: a part being PICKED (LOADING) or already
        LOADED/returning. CANCELLABLE: a queued (pending) part, or one still EN-ROUTE to
        pickup (TO_PART, not yet picked). When the current pickup target is dropped the
        remaining batch is re-planned (next pickup / deliver loaded / else home). True if cancelled."""
        it = self.queue.by_id(item_id)
        if it is None:
            return False
        if it.status != 'active':
            return self.queue.remove(item_id)         # pending / done / failed → just drop
        if it.part_state == 'LOADED' or it not in self.batch:
            return False                              # already on the carrier → protect
        idx = self.batch.index(it)
        if idx == self.bidx and self.phase == 'PICKING':
            return False                              # being picked right now → protect
        # cancellable (en-route TO_PART target, or a not-yet-reached batch item)
        self.batch.pop(idx)
        self.queue.remove(item_id)
        if idx == self.bidx and self.phase == 'TO_PART':
            if self.bidx < len(self.batch):
                self._enter_to_part(now)              # advance to the next pickup
            elif any(b.part_state == 'LOADED' for b in self.batch):
                self._enter_delivering(now)           # deliver what is already loaded
            else:                                     # nothing picked, nothing left → home
                self.state = 'IDLE'
                self.phase = None
                self.batch = []
                self.active_id = ''
                self.targets = {'leader': self.standby, 'follower': self.follower_home}
                self.modes = dict(_HOME_MODES)
        return True

    def estop(self, engaged=True):
        """E-stop safety line (latched). Engaged -> immediate PAUSED (tick forces modes
        STOP), regardless of mission state. Release does NOT auto-resume — the operator
        must call resume() deliberately (safety)."""
        self._estop = bool(engaged)
        if engaged:
            if self.state == 'RUNNING':
                self._resume_phase = self.phase
            self._pause_reason = 'estop'
            self.state = 'PAUSED'
        # release: stay PAUSED (modes STOP) until an explicit resume()

    def pause(self):
        if self.state == 'RUNNING':
            self._resume_phase = self.phase
            self._pause_reason = 'manual'
            self.state = 'PAUSED'

    def resume(self):
        if self.state != 'PAUSED' or self._estop:
            return                            # E-stop must be released before resuming
        if self._resume_phase is None:
            self.state = 'IDLE'               # paused with no active phase (e.g. estop while idle)
        else:
            self.state = 'RUNNING'
            self.phase = self._resume_phase
            self._just_resumed = True         # next tick refreshes phase_start / pick_deadline
        self._resume_phase = None
        self._pause_reason = None
        self.modes = dict(_HOME_MODES)        # restore GOTO (pause/estop had forced STOP)

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
    def _batchable(self, now, exclude_shelves):
        """Pending items pickable NOW: off-cooldown, and (when cooldown is active) at most one
        per shelf per trip — so same-shelf duplicates spread across trips while different
        shelves ride together. `exclude_shelves` = shelves already committed to this trip."""
        out, used = [], set(exclude_shelves)
        for it in self.queue.pending():
            s = it.leader_node
            if now < self._shelf_ready.get(s, 0.0):
                continue                                 # shelf on cooldown
            if self.cooldown > 0 and s in used:
                continue                                 # one pick / shelf / trip
            out.append(it)
            used.add(s)
        return out

    def _begin_batch(self, now):
        """Form the next trip from pickable pending parts (capacity-bounded, cooldown-aware),
        time-optimally ordered. Returns False if nothing is pickable yet (all on cooldown) —
        the caller then waits and retries."""
        cand = self._batchable(now, set())[:self.capacity]
        self.batch = self._order_pickups(cand)
        if not self.batch:
            return False
        for it in self.batch:
            it.status = 'active'
            it.part_state = 'ASSIGNED'
        self.bidx = 0
        self.state = 'RUNNING'
        self._enter_to_part(now)
        return True

    def _absorb_pending(self, now):
        """While still COLLECTING, pull newly-queued parts into the current trip (capacity +
        cooldown + distinct-shelf aware) so a mid-mission add joins THIS batch instead of
        forcing a second trip. Once DELIVERING starts, new parts go to the next batch."""
        room = self.capacity - len(self.batch)
        if room <= 0:
            return
        used = {b.leader_node for b in self.batch}
        new = self._batchable(now, used)[:room]
        for it in new:
            it.status = 'active'
            it.part_state = 'ASSIGNED'
        self.batch.extend(new)

    def _order_pickups(self, items):
        """Visit order within a batch. Lanes are DIRECTED, so order matters (asymmetric TSP).
        With an injected directed route-cost, brute-force the order that minimises the leader
        tour standby -> pickups -> standby (uniform speed => distance == time; leader is the
        binding agent). Without a cost (unit tests), keep FIFO."""
        items = list(items)
        if self._order_cost is None or len(items) <= 1:
            return items
        if len(items) > 8:                               # N! guard; greedy/FIFO fallback for big N
            return items
        from itertools import permutations
        best, best_cost = items, float('inf')
        for perm in permutations(items):
            cost, prev = 0.0, self.standby
            for it in perm:
                cost += self._order_cost(prev, it.leader_node)
                prev = it.leader_node
            cost += self._order_cost(prev, self.standby)  # leader's return leg
            if cost < best_cost:
                best_cost, best = cost, list(perm)
        return best

    def _enter_to_part(self, now):
        self.phase = 'TO_PART'
        it = self.batch[self.bidx]
        # parallel pick: leader and follower stop at their OWN nodes (different per shelf)
        self.targets = {'leader': it.leader_node, 'follower': it.follower_node}
        self.modes = dict(_HOME_MODES)
        self.active_id = it.id
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
        self._unload_done = False
        self._unloading = False
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
            self._just_resumed = True        # refresh phase_start AND pick_deadline (below)
            self.modes = dict(_HOME_MODES)   # restore GOTO (pause had forced STOP)
            events.append(('resumed', ''))

        if self.state != 'RUNNING':
            if self.state == 'PAUSED':
                self.modes = {r: 'STOP' for r in self.targets}
            return dict(self.targets), dict(self.modes), events

        # just resumed (manual/estop/health): refresh phase timers so a long pause does NOT
        # instantly phase_timeout, nor spuriously complete a pick on a stale _pick_deadline.
        if self._just_resumed:
            self._phase_start = now
            if self.phase == 'PICKING':
                self._pick_deadline = now + self.pick_dwell
            self._just_resumed = False

        # liveness: no phase waits forever — EXCEPT the operator-paced 하역완료 wait.
        # follower가 하역존에 도착해 unload_done(운영자 '하역완료')을 기다리는 구간은 사람이
        # 누를 때까지 무한 대기해야 정상 → 타임아웃 제외(여기서 FAULT 금지). 복귀 주행 등 나머지
        # DELIVERING 구간/타 단계는 그대로 phase_timeout 적용(끼임 방지).
        waiting_unload = (self.phase == 'DELIVERING' and not self._dropped
                          and bool(arrived.get('follower')))
        if not waiting_unload and now - self._phase_start > self.phase_timeout:
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
                    self._shelf_ready[it.leader_node] = now + self.cooldown   # start shelf cooldown
                    events.append(('loaded', it.id))
                self.bidx += 1
                self._absorb_pending(now)     # mid-mission adds join THIS trip (still collecting)
                if self.bidx < len(self.batch):
                    self._enter_to_part(now)
                else:
                    self._enter_delivering(now)

        elif self.phase == 'DELIVERING':
            # follower returns to the 하역 zone, leader to standby. The drop completes ONLY on
            # the operator '하역완료' (unload_done) — never on a timer. Then mission is DONE when
            # the leader is back at standby (decoupled from the drop).
            just_dropped = False
            self._unloading = (not self._dropped) and bool(arrived.get('follower'))
            if not self._dropped and arrived.get('follower') and self._unload_done:
                for it in self.batch:
                    if it.status == 'active':
                        it.status = 'done'
                        it.part_state = 'DELIVERED'
                        events.append(('delivered', it.id))
                        self.queue.remove(it.id)   # delivered → vanish from the queue (no lingering 'done')
                self._dropped = True
                self.targets = {'leader': self.standby, 'follower': self.follower_home}
                self._phase_start = now    # 귀가 구간은 fresh 타임아웃 창 — 긴(무한) 하역 대기 시작점을 물려받지 않게
                just_dropped = True
            if self._dropped and not just_dropped:
                if self.queue.pending():
                    if self._begin_batch(now):
                        events.append(('next_batch', ''))
                    # else: pending parts still on cooldown — wait at home, retry next tick
                elif arrived.get('leader') and arrived.get('follower'):
                    self.state = 'DONE'
                    self.phase = None
                    events.append(('done', ''))

        return dict(self.targets), dict(self.modes), events

    # ---------- 관제 projection ----------
    def shelf_ready_in(self, shelf_node, now):
        """Seconds until `shelf_node` comes off cooldown (0 = ready now) — lets 관제 show a
        'waiting Ns' badge on a same-shelf duplicate that can't be fetched yet."""
        return max(0.0, self._shelf_ready.get(int(shelf_node), 0.0) - now)

    def system_state(self):
        """Operation state for 관제/dashboard, projected from internal phase/flags.
        Happy path: IDLE | TO_PICKUP | LOADING | RETURNING | UNLOADING.
        Safety/fault (kept for robustness): ESTOP | PAUSED | FAULT | ABORTED."""
        if self.state in ('IDLE', 'DONE'):
            return 'IDLE'
        if self.state == 'ABORTED':
            return 'ABORTED'
        if self.state == 'PAUSED':
            if self._pause_reason == 'estop':
                return 'ESTOP'
            return 'FAULT' if self._pause_reason == 'timeout' else 'PAUSED'
        if self.phase == 'TO_PART':
            return 'TO_PICKUP'
        if self.phase == 'PICKING':
            return 'LOADING'
        if self.phase == 'DELIVERING':
            if self._dropped:
                return 'RETURNING'
            return 'UNLOADING' if self._unloading else 'RETURNING'
        return 'IDLE'

    def robot_state(self, robot, arrived):
        """Per-robot mission state (role-specific projection) for the dashboard —
        derived from the single mission SM (no per-robot SM fragmentation)."""
        if self.state in ('IDLE', 'DONE'):
            return 'IDLE'
        if self.state == 'ABORTED':
            return 'ABORTED'
        if self.state == 'PAUSED':
            if self._pause_reason == 'estop':
                return 'ESTOP'
            return 'FAULT' if self._pause_reason == 'timeout' else 'PAUSED'
        if self.phase == 'TO_PART':
            return 'AT_PICKUP' if arrived else 'TO_PICKUP'
        if self.phase == 'PICKING':
            return 'LOADING'                 # leader picks, follower receives
        if self.phase == 'DELIVERING':
            if robot == 'follower':
                if self._dropped:
                    return 'RETURNING'           # heading back to the unloading/home zone
                return 'UNLOADING' if self._unloading else 'RETURNING'
            return 'IDLE' if arrived else 'RETURNING'
        return 'IDLE'
