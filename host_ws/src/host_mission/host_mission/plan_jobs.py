"""Where a path computation runs: a runner has ``submit(fn, *args)`` returning a
``concurrent.futures``-style object (``done()``, ``result()``, ``cancel()``). ``Inline``
computes at once (tests, in-process laps, the commander's default); ``worker()`` computes on one
thread so a slow plan does not stop the mission tick. No ROS imports."""
from concurrent.futures import ThreadPoolExecutor


class _Done:
    def __init__(self, value):
        self._value = value

    def done(self):
        return True

    def result(self):
        return self._value

    def cancel(self):
        return False


class Inline:

    def submit(self, fn, *args):
        return _Done(fn(*args))


class Delayed:
    """Results appear ``delay`` seconds of ``clock()`` after they were asked for (repeatable stand-
    in for a slow worker)."""

    def __init__(self, clock, delay):
        self.clock, self.delay = clock, float(delay)

    def submit(self, fn, *args):
        return _Later(self.clock, self.clock() + self.delay, fn(*args))


class _Later(_Done):
    def __init__(self, clock, ready_at, value):
        super().__init__(value)
        self._clock, self._ready_at = clock, ready_at

    def done(self):
        return self._clock() >= self._ready_at


def worker():
    """One background thread; an error of the computation is raised by ``result()``."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix='plan')
