"""Mission state machine (ROS-free, unit-testable).

  IDLE ──start──> RUNNING ──all delivered & home──> DONE
                   │  ▲                                  ▲
        LOST/timeout│  │auto-resume (health) / resume()  │ (re)start on new dispatch
                   ▼  │                                   │
                 PAUSED ──abandon()──> IDLE ──────────────┘
                (not while PICKING)

RUNNING sub-phases: TO_PART -> PICKING -> DELIVERING.
  * Every wait is bounded (per-phase timeout -> PAUSED). Health LOST pauses and auto-resumes;
    manual, timeout and fault pauses wait for resume().
  * A running trip is never changed by the operator: only waiting parts can be cancelled,
    only a stopped trip abandoned.
  * The follower's drop completes on the operator's unload_done(); the leader returns
    independently; DONE when both are done.
  * PICKING completes on pick_complete() or at the pick_dwell deadline, also while PAUSED
    (the host cannot stop the arm).

tick(now, arrived, health) -> (targets, modes, events): arrived = {robot: bool},
health = {robot: GOOD|DEGRADED|LOST|UNKNOWN}, modes = {robot: 'GOTO'|'STOP'},
events = [(kind, detail)]."""

_HOME_MODES = {'leader': 'GOTO', 'follower': 'GOTO'}


class MissionStateMachine:
    def __init__(self, queue, standby_node, unloading_node, capacity,
                 pick_dwell=8.0, phase_timeout=90.0,
                 follower_home_node=None, order_cost=None, cooldown=0.0,
                 dwell_is_failure=False, next_trip_from_home=False):
        # targets are opaque hashable goal keys
        self.queue = queue
        # True: a part queued mid-trip starts its trip only from home (idle rules apply between)
        self.next_trip_from_home = next_trip_from_home
        self.standby = standby_node
        self.unloading = unloading_node  # follower drop-off
        # follower's own wait zone; defaults to unloading
        self.follower_home = follower_home_node if follower_home_node is not None \
            else unloading_node
        # True: no completion within pick_dwell = failed; False: counts as loaded
        self.dwell_is_failure = bool(dwell_is_failure)
        self.capacity = int(capacity)
        self.pick_dwell = float(pick_dwell)
        self.phase_timeout = float(phase_timeout)
        # order_cost(a, b) -> travel cost; None = FIFO
        self._order_cost = order_cost
        self.cooldown = float(cooldown)
        self._shelf_ready = {}

        self.state = 'IDLE'                   # IDLE|RUNNING|PAUSED|DONE
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
        self._unload_done = False  # operator 'unload done' (manual unload gate)
        self._unloading = False
        self._estop = False
        self._resume_phase = None
        self._pause_reason = None             # 'health'|'manual'|'timeout'|'estop'|'fault'
        self._just_resumed = False            # refresh phase timers on the next tick

    def start(self, now):
        if self.state in ('IDLE', 'DONE') and self.queue.pending():
            if self._begin_batch(now):
                return [('mission_started', '')]
        return []

    def pick_complete(self, ok=True):
        self._pick_done = True
        self._pick_ok = bool(ok)

    def unload_done(self):
        self._unload_done = True

    def cancel_refusal(self, item_id):
        """Why the operator may not remove ``item_id`` from the queue (None = may)."""
        it = self.queue.by_id(item_id)
        if it is None:
            return f'{item_id}: not in the queue'
        if it.status == 'active':
            return f'{item_id} is in the running trip: stop, then abandon'
        return None

    def estop(self, engaged=True):
        """Latched e-stop: PAUSED at once; release does not auto-resume."""
        self._estop = bool(engaged)
        if engaged:
            if self.state == 'RUNNING':
                self._resume_phase = self.phase
            elif self.state != 'PAUSED':
                self._resume_phase = None
            self._pause_reason = 'estop'
            self.state = 'PAUSED'
        # release: stay PAUSED (modes STOP) until an explicit resume()

    def pause(self):
        if self.state == 'RUNNING':
            self._resume_phase = self.phase
            self._pause_reason = 'manual'
            self.state = 'PAUSED'

    def halt(self):
        if self.state == 'RUNNING':
            self._go_paused('fault')

    def resume(self):
        if self.state != 'PAUSED' or self._estop:
            return
        if self._resume_phase is None:
            self.state = 'IDLE'
        else:
            self.state = 'RUNNING'
            self.phase = self._resume_phase
            self._just_resumed = True
        self._resume_phase = None
        self._pause_reason = None
        self.modes = dict(_HOME_MODES)

    def abandon_refusal(self):
        """Why the trip may not be abandoned now (None = may): only stopped, never mid-pick."""
        if self.state != 'PAUSED':
            return 'the mission is running: stop first' if self.state == 'RUNNING' \
                else 'no trip to abandon'
        if self.phase == 'PICKING':
            return 'pick and place in progress: resume and let it end'
        return None

    def abandon(self):
        """Drop the stopped trip and go home; returns dropped item ids."""
        dropped = [it.id for it in self.batch if it.status == 'active']
        for it in list(self.queue.items):
            if it.status == 'active':
                self.queue.remove(it.id)
        self._go_home('IDLE')
        return dropped

    def _estop_held(self):

        return self.state == 'PAUSED' and self._pause_reason == 'estop'

    def _go_home(self, idle_state):

        self.phase = None
        self.batch = []
        self.active_id = ''
        self._resume_phase = None
        if not self._estop_held():
            self.state = idle_state
            self._pause_reason = None
        self.targets = {'leader': self.standby, 'follower': self.follower_home}
        self.modes = dict(_HOME_MODES)

    def _batchable(self, now, exclude_shelves):
        """Pending items pickable now: off-cooldown, one per shelf per trip."""
        out, used = [], set(exclude_shelves)
        for it in self.queue.pending():
            s = it.leader_node
            if now < self._shelf_ready.get(s, 0.0):
                continue
            if self.cooldown > 0 and s in used:
                continue
            out.append(it)
            used.add(s)
        return out

    def _begin_batch(self, now):
        """Form the next trip from pickable pending parts (time-optimal order); False if none."""
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
        """Visit order within a batch: minimal leader tour by the route cost, else FIFO."""
        items = list(items)
        if self._order_cost is None or len(items) <= 1:
            return items
        if len(items) > 8:
            return items
        from itertools import permutations
        best, best_cost = items, float('inf')
        for perm in permutations(items):
            cost, prev = 0.0, self.standby
            for it in perm:
                cost += self._order_cost(prev, it.leader_node)
                prev = it.leader_node
            cost += self._order_cost(prev, self.standby)
            if cost < best_cost:
                best_cost, best = cost, list(perm)
        return best

    def _enter_to_part(self, now):
        self.phase = 'TO_PART'
        it = self.batch[self.bidx]
        # leader and follower stop at their OWN nodes (different per shelf)
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
        """The way back; with nothing loaded there is nothing to unload, straight home."""
        self.phase = 'DELIVERING'
        self._dropped = not self.batch
        self.targets = {'leader': self.standby,
                        'follower': self.follower_home if self._dropped else self.unloading}
        self.modes = dict(_HOME_MODES)
        self.active_id = ''
        self._unload_done = False
        self._unloading = False
        self._phase_start = now

    def _end_pick(self, now, events):
        """Complete the pick on pick_complete() or the dwell deadline; go to the next pickup or
        deliver."""
        if not self._pick_done and now < self._pick_deadline:
            return
        it = self.batch[self.bidx]
        timed_out = not self._pick_done
        if (self._pick_done and not self._pick_ok) or (timed_out and self.dwell_is_failure):
            # not delivered: the event is its record. It leaves the queue and the trip's
            # batch (bidx then already points at the next pickup); the trip goes on.
            events.append(('pick_failed', it.id))
            self.queue.remove(it.id)
            self.batch.pop(self.bidx)
        else:
            it.part_state = 'LOADED'
            self._shelf_ready[it.leader_node] = now + self.cooldown
            events.append(('loaded', it.id))
            self.bidx += 1
        self._absorb_pending(now)     # mid-mission adds join THIS trip (still collecting)
        if self.bidx < len(self.batch):
            self._enter_to_part(now)
        else:
            self._enter_delivering(now)

    def _go_paused(self, reason):
        self._resume_phase = self.phase
        self._pause_reason = reason
        self.state = 'PAUSED'

    def _apply_health(self, lost, events):
        """Pause on a lost link; resume when a health pause clears."""
        if self.state == 'RUNNING' and lost:
            self._go_paused('health')
            events.append(('paused', 'health_lost'))
        elif self.state == 'PAUSED' and self._pause_reason == 'health' and not lost:
            self.state = 'RUNNING'
            self.phase = self._resume_phase
            self._resume_phase = None
            self._pause_reason = None
            self._just_resumed = True
            self.modes = dict(_HOME_MODES)
            events.append(('resumed', ''))

    def _tick_not_running(self, now, events):
        if self.state == 'PAUSED':
            if self.phase == 'PICKING':
                self._end_pick(now, events)
                self._resume_phase = self.phase
            self.modes = {r: 'STOP' for r in self.targets}

    def _tick_delivering(self, now, arrived, events):
        # follower returns to the unload zone, leader to standby. The drop completes only
        # on the operator's unload_done, never on a timer; DONE when the leader is back.
        just_dropped = False
        self._unloading = (not self._dropped) and bool(arrived.get('follower'))
        if not self._dropped and arrived.get('follower') and self._unload_done:
            for it in self.batch:
                if it.status == 'active':
                    it.status = 'done'
                    it.part_state = 'DELIVERED'
                    events.append(('delivered', it.id))
                    self.queue.remove(it.id)
            self._dropped = True
            self.targets = {'leader': self.standby, 'follower': self.follower_home}
            self._phase_start = now
            just_dropped = True
        if self._dropped and not just_dropped:
            if self.queue.pending() and not self.next_trip_from_home:
                if self._begin_batch(now):
                    events.append(('next_batch', ''))
                # else: pending parts still on cooldown — wait at home, retry next tick
            elif arrived.get('leader') and arrived.get('follower'):
                self._go_home('DONE')
                events.append(('done', ''))

    def tick(self, now, arrived, health=None):
        events = []
        health = health or {}
        lost = any(h == 'LOST' for h in health.values())

        self._apply_health(lost, events)

        if self.state != 'RUNNING':
            self._tick_not_running(now, events)
            return dict(self.targets), dict(self.modes), events

        # just resumed: refresh the phase timer so a long pause does not instantly time out.
        # The pick deadline is the arm's and keeps running.
        if self._just_resumed:
            self._phase_start = now
            self._just_resumed = False

        # liveness: no phase waits forever, except the follower waiting at the unload zone for the
        # operator's unload_done (no timeout there). The return drive and other phases keep
        # phase_timeout.
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
            self._end_pick(now, events)

        elif self.phase == 'DELIVERING':
            self._tick_delivering(now, arrived, events)

        return dict(self.targets), dict(self.modes), events

    def shelf_ready_in(self, shelf_node, now):
        """Seconds until ``shelf_node`` comes off cooldown (0 = ready now)."""
        return max(0.0, self._shelf_ready.get(shelf_node, 0.0) - now)

    def system_state(self):
        """Dashboard operation state: IDLE | TO_PICKUP | LOADING | RETURNING | UNLOADING | ESTOP |
        PAUSED | FAULT."""
        if self.state in ('IDLE', 'DONE'):
            return 'IDLE'
        if self.state == 'PAUSED':
            if self._pause_reason == 'estop':
                return 'ESTOP'
            return 'FAULT' if self._pause_reason in ('timeout', 'fault') else 'PAUSED'
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
        """Per-robot dashboard state, derived from the single mission state machine."""
        if self.state in ('IDLE', 'DONE'):
            return 'IDLE'
        if self.state == 'PAUSED':
            if self._pause_reason == 'estop':
                return 'ESTOP'
            return 'FAULT' if self._pause_reason in ('timeout', 'fault') else 'PAUSED'
        if self.phase == 'TO_PART':
            return 'AT_PICKUP' if arrived else 'TO_PICKUP'
        if self.phase == 'PICKING':
            return 'LOADING'
        if self.phase == 'DELIVERING':
            if robot == 'follower':
                if self._dropped:
                    return 'RETURNING'
                return 'UNLOADING' if self._unloading else 'RETURNING'
            return 'IDLE' if arrived else 'RETURNING'
        return 'IDLE'
