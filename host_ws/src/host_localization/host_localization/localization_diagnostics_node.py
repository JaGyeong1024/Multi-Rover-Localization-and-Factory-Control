"""Health aggregation for the localization pipeline -> /localization/diagnostics at 1 Hz.

Receive liveness/rate per stage plus chrony time sync, used by the mission health_gate
(`localization/<robot>`) and the dashboard. Age/rate use monotonic receive time, not
header.stamp; subscriptions are BEST_EFFORT. Statuses: localization/<robot>, pipeline/<stage>,
align/<robot>, sync/<source>.
"""
import subprocess
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Imu
from std_msgs.msg import Bool, Float32, String

from .fusion import quality

# matches fusion's latched topics
_ALIGNED_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)

OK, WARN, ERROR = DiagnosticStatus.OK, DiagnosticStatus.WARN, DiagnosticStatus.ERROR
_LABEL = {OK: 'GOOD', WARN: 'DEGRADED', ERROR: 'LOST'}
_SEVERITY = [OK, WARN, ERROR]      # levels are bytes: order explicitly

RATE_WINDOW = 2.0          # [s] sliding window for Hz
QF_MIN = 50.0              # below: DEGRADED
CHRONY_WARN_MS = 5.0       # |offset| above: WARN
# Freshness thresholds [s]; uwb/perception share quality.py's constants.
TH = {'uwb': quality.UWB_TIMEOUT, 'camera': 0.3, 'perception': quality.BEV_TIMEOUT,
      'fusion': 1.0, 'output': 0.5}
_QUALITY_LEVEL = {quality.GOOD: OK, quality.DEGRADED: WARN, quality.LOST: ERROR}


class Monitor:
    """Topic receive liveness/rate on monotonic time."""

    def __init__(self):
        self.times = deque()
        self.last = None

    def hit(self, *_):
        t = time.monotonic()
        self.last = t
        self.times.append(t)

    def age(self):
        return None if self.last is None else time.monotonic() - self.last

    def hz(self):
        now = time.monotonic()
        while self.times and now - self.times[0] > RATE_WINDOW:
            self.times.popleft()
        return len(self.times) / RATE_WINDOW


class LocalizationDiagnosticsNode(Node):
    def __init__(self):
        super().__init__('localization_diagnostics_node')
        self.robots = self.declare_parameter('robots', ['leader', 'follower']).value
        self.pub = self.create_publisher(DiagnosticArray, '/localization/diagnostics', 10)

        q = qos_profile_sensor_data
        self.mon = {}
        self.qf = {r: None for r in self.robots}
        self.aligned = {r: None for r in self.robots}   # heading_aligned (None = not received)
        self.quality = {r: None for r in self.robots}   # /{r}/localization_quality
        self.latency_ms = {r: None for r in self.robots}  # capture->pose latency (EMA, ms)

        def m(key):
            self.mon[key] = Monitor()
            return self.mon[key]

        for r in self.robots:
            # leader corrected, follower raw
            uwb_def = f'/uwb/{r}_pose_corrected' if r == 'leader' else f'/uwb/{r}_pose'
            uwb_t = self.declare_parameter(f'{r}.uwb', uwb_def).value
            self.create_subscription(PoseStamped, uwb_t, m(f'uwb/{r}').hit, q)
            self.create_subscription(Float32, f'/uwb/{r}_qf',
                                     lambda msg, R=r: self.qf.__setitem__(R, float(msg.data)), q)
            m(f'perception/{r}')  # _on_bev hits it
            self.create_subscription(PoseStamped, f'/perception/{r}/bev_pose',
                                     lambda msg, R=r: self._on_bev(msg, R), q)
            self.create_subscription(PoseStamped, f'/{r}/pose', m(f'fusion/{r}').hit, q)
            self.create_subscription(PoseStamped, f'/{r}/global/position', m(f'output/{r}').hit, q)
            self.create_subscription(Imu, f'/{r}/imu', m(f'imu/{r}').hit, q)
            self.create_subscription(Bool, f'/{r}/heading_aligned',
                                     lambda msg, R=r: self.aligned.__setitem__(R, bool(msg.data)),
                                     _ALIGNED_QOS)
            self.create_subscription(String, f'/{r}/localization_quality',
                                     lambda msg, R=r: self.quality.__setitem__(R, msg.data),
                                     _ALIGNED_QOS)
        self.create_subscription(CompressedImage, '/global_cam/processed/compressed', m('camera').hit, q)

        self.chrony = {'ok': False, 'offset_ms': None, 'source': 'n/a'}
        self.create_timer(1.0, self._tick)
        self.get_logger().info('localization_diagnostics_node up -> /localization/diagnostics @1Hz')

    @staticmethod
    def _kv(k, v):
        return KeyValue(key=k, value=('n/a' if v is None else str(v)))

    def _age_ms(self, key):
        a = self.mon[key].age()
        return None if a is None else round(a * 1000.0, 1)

    def _hz(self, key):
        return round(self.mon[key].hz(), 1)

    def _on_bev(self, msg, r):
        # bev_pose keeps the image stamp, so now - stamp is the capture->pose delay (EMA).
        self.mon[f'perception/{r}'].hit()
        cap = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        lat = (now - cap) * 1000.0
        if 0.0 <= lat < 5000.0:   # ignore clock skew / garbage stamps
            prev = self.latency_ms[r]
            self.latency_ms[r] = round(lat if prev is None else 0.6 * prev + 0.4 * lat, 1)

    def _poll_chrony(self):
        try:
            out = subprocess.run(['chronyc', '-c', 'tracking'],
                                 capture_output=True, text=True, timeout=1.0)
            if out.returncode != 0 or not out.stdout.strip():
                self.chrony = {'ok': False, 'offset_ms': None, 'source': 'n/a'}
                return
            f = out.stdout.strip().split(',')
            # CSV: [1]=ref name, [4]=offset [s]
            self.chrony = {'ok': True, 'offset_ms': float(f[4]) * 1000.0, 'source': f[1]}
        except Exception:
            self.chrony = {'ok': False, 'offset_ms': None, 'source': 'n/a'}

    def _stage(self, name, prefix, thresh, qf=False):
        """pipeline/<stage> status; WARN if any robot is stale."""
        s = DiagnosticStatus(name=name, hardware_id='host')
        vals, worst, seen = [], OK, False
        for r in self.robots:
            key = f'{prefix}/{r}'
            a = self.mon[key].age()
            vals.append(self._kv(f'{r}_hz', self._hz(key)))
            vals.append(self._kv(f'{r}_age_ms', self._age_ms(key)))
            if qf:
                vals.append(self._kv(f'{r}_qf', self.qf[r]))
            if a is not None:
                seen = True
                if a > thresh:
                    worst = WARN
            else:
                worst = WARN
        s.level = worst
        s.message = 'ok' if worst == OK else ('absent' if not seen else 'stale')
        s.values = vals
        return s

    def _tick(self):
        self._poll_chrony()
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = []

        # per-robot roll-up (mission gate)
        for r in self.robots:
            uwb_a = self.mon[f'uwb/{r}'].age()
            fus_a = self.mon[f'fusion/{r}'].age()
            out_a = self.mon[f'output/{r}'].age()
            per_a = self.mon[f'perception/{r}'].age()
            qf = self.qf[r]
            if out_a is None or out_a > 1.0 or fus_a is None or fus_a > 1.0:
                level = ERROR
            elif (uwb_a is None or uwb_a > TH['uwb']) or (qf is not None and qf < QF_MIN) \
                    or (per_a is None or per_a > TH['perception']):
                level = WARN
            else:
                level = OK
            # Never better than the quality the rover is told (can be stricter than liveness).
            level = max(level, _QUALITY_LEVEL.get(self.quality[r], OK), key=_SEVERITY.index)
            s = DiagnosticStatus(name=f'localization/{r}', level=level,
                                 message=_LABEL[level], hardware_id='host')
            s.values = [self._kv('uwb_age_ms', self._age_ms(f'uwb/{r}')),
                        self._kv('fusion_age_ms', self._age_ms(f'fusion/{r}')),
                        self._kv('output_age_ms', self._age_ms(f'output/{r}')),
                        self._kv('output_hz', self._hz(f'output/{r}')),
                        self._kv('qf', qf),
                        self._kv('quality', self.quality[r])]
            st.append(s)

        # pipeline stages
        st.append(self._stage('pipeline/uwb', 'uwb', TH['uwb'], qf=True))
        st.append(self._stage('pipeline/perception', 'perception', TH['perception']))
        st.append(self._stage('pipeline/fusion', 'fusion', TH['fusion']))
        out_stage = self._stage('pipeline/output', 'output', TH['output'])
        lats = [v for v in (self.latency_ms[r] for r in self.robots) if v is not None]
        if lats:
            out_stage.values.append(self._kv('latency_ms', round(max(lats), 1)))
        st.append(out_stage)
        st.append(self._stage('pipeline/imu', 'imu', 1.0))

        # heading alignment: aligned=OK, else WARN
        for r in self.robots:
            al = self.aligned[r]
            a = DiagnosticStatus(
                name=f'align/{r}', hardware_id='host',
                level=OK if al else WARN,
                message=('aligned' if al else ('unaligned' if al is False else 'no data')))
            a.values = [self._kv('heading_aligned', al)]
            st.append(a)

        # camera (single)
        cam = DiagnosticStatus(name='pipeline/camera', hardware_id='host')
        ca = self.mon['camera'].age()
        cam.level = OK if (ca is not None and ca < TH['camera']) else WARN
        cam.message = 'ok' if cam.level == OK else ('absent' if ca is None else 'stale')
        cam.values = [self._kv('hz', self._hz('camera')), self._kv('age_ms', self._age_ms('camera'))]
        st.append(cam)

        # sync/chrony
        ch = DiagnosticStatus(name='sync/chrony', hardware_id='host')
        if not self.chrony['ok']:
            ch.level, ch.message, ch.values = WARN, 'chrony unavailable', []
        else:
            off = self.chrony['offset_ms']
            ch.level = OK if abs(off) < CHRONY_WARN_MS else WARN
            ch.message = f'offset {off:.2f}ms'
            ch.values = [self._kv('offset_ms', round(off, 3)), self._kv('source', self.chrony['source'])]
        st.append(ch)

        arr.status = st
        self.pub.publish(arr)


def main():
    rclpy.init()
    node = LocalizationDiagnosticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
