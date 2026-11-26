"""Command watchdog logic: is the safety gate still alive?"""
from typing import Optional


class CmdWatchdog:
    """Tracks the last heartbeat. A gate never heard from counts as silent."""

    def __init__(self, timeout_s: float):
        self._timeout = timeout_s
        self._last_alive: Optional[float] = None

    def on_alive(self, now: float) -> None:
        self._last_alive = now

    def must_stop(self, now: float) -> bool:
        """True while the watchdog has to hold the base at zero."""
        return self._last_alive is None or now - self._last_alive > self._timeout
