"""Localization health gate (ROS-free logic). Parses DiagnosticStatus levels per
robot. Policy: LOST -> block (STOP). UNKNOWN (no diagnostics published yet — the
publisher does not exist in code today) -> allow, so the demo is not blocked. Switch
to strict before real-robot integration (plan §전제 R2)."""
from diagnostic_msgs.msg import DiagnosticStatus

_LEVEL = {DiagnosticStatus.OK: 'GOOD', DiagnosticStatus.WARN: 'DEGRADED',
          DiagnosticStatus.ERROR: 'LOST'}


class HealthGate:
    def __init__(self):
        self.health = {}   # robot -> GOOD|DEGRADED|LOST

    def update(self, diag_array):
        for st in diag_array.status:
            # status name convention: "localization/<robot>"
            if '/' in st.name:
                robot = st.name.split('/')[-1]
                self.health[robot] = _LEVEL.get(st.level, 'UNKNOWN')

    def get(self, robot):
        return self.health.get(robot, 'UNKNOWN')
