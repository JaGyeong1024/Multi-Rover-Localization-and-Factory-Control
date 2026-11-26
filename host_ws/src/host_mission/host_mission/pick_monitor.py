"""Pick completion judgement for ``nav_mode: path`` (ROS-free).

The leader's pick coordinator publishes two LATCHED topics: ``/leader/work_state`` (NAVIGATING ->
PICKING -> DONE | ERROR) and ``/leader/pick_place_done`` (true = success). Latched topics replay
their last sample to new subscribers, so a result is accepted only when the PICKING phase is
active (``begin``), the leader was seen picking, a ``pick_place_done`` arrived after that, and the
work state then left picking. The outcome is ``done`` AND the work state did not end in an error.
"""


class PickMonitor:
    def __init__(self, picking_states=('PICKING',), error_states=('ERROR',)):
        self._picking_states = tuple(picking_states)
        self._error_states = tuple(error_states)
        self._work_state = ''
        self._active = False
        self._saw_picking = False
        self._end_state = None      # work state right after picking
        self._done = None           # latest accepted pick_place_done value

    @property
    def active(self):
        return self._active

    def begin(self):
        self._active = True
        self._saw_picking = self._work_state in self._picking_states
        self._end_state = None
        self._done = None

    def cancel(self):
        self._active = False

    def on_work_state(self, state):
        state = str(state).strip().upper()
        previous, self._work_state = self._work_state, state
        if not self._active:
            return
        if state in self._picking_states:
            self._saw_picking = True
            self._end_state = None
        elif self._saw_picking and previous in self._picking_states:
            self._end_state = state

    def on_done(self, ok):
        """``/leader/pick_place_done`` sample; ignored unless a pick was seen running."""
        if self._active and self._saw_picking:
            self._done = bool(ok)

    def poll(self):
        """True / False once the pick has finished, else None. Clears on a result."""
        if not self._active or self._done is None or self._end_state is None:
            return None
        self._active = False
        return self._done and self._end_state not in self._error_states
