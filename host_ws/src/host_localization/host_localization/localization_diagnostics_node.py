"""localization_diagnostics_node — 호스트 위치추정 파이프라인 헬스 집계.

파이프라인 각 단계의 *수신 liveness/rate*(wall-clock 기준)와 시간동기(chrony)를 모아
`diagnostic_msgs/DiagnosticArray`를 `/localization/diagnostics`로 1Hz 발행.
미션 `health_gate`(status name `localization/<robot>`)와 대시보드 SystemHealthPanel이 **공유**.

설계 노트:
- age/rate는 `header.stamp`가 아니라 **노드 수신 monotonic 시각** 기준 — bag 리플레이서도
  진짜 'arrival staleness'를 재고, stamp=now()/구식 stamp(녹화시각)에 흔들리지 않는다.
- 모든 구독은 BEST_EFFORT(`qos_profile_sensor_data`) — reliable·best_effort 발행자 모두와 호환(최대 수신).
- status name 규약: `localization/<robot>`(롤업 level = 미션 게이트 계약 OK/WARN/ERROR),
  `pipeline/<stage>`, `sync/<source>`. `values`(KeyValue)에 `*_hz`·`*_age_ms`·`qf`·`offset_ms` 등
  → 패널이 prefix로 그룹핑해 타일 렌더.

Run:    ros2 run host_localization localization_diagnostics_node
        (또는 dashboard.launch.py에 포함)
Verify: ros2 topic echo /localization/diagnostics
"""
import subprocess
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Imu
from std_msgs.msg import Float32

OK, WARN, ERROR = DiagnosticStatus.OK, DiagnosticStatus.WARN, DiagnosticStatus.ERROR
_LABEL = {OK: 'GOOD', WARN: 'DEGRADED', ERROR: 'LOST'}

RATE_WINDOW = 2.0          # s — Hz 슬라이딩 윈도우
QF_MIN = 50.0              # qf < 50 = 저품질(DEGRADED)
CHRONY_WARN_MS = 5.0       # |offset| > 5ms = 동기 WARN
# freshness 임계(s) — 조사 근거(STRUCTURE/plan): uwb 2.0, cam 0.3, perception/fusion 1.0, output 0.5
TH = {'uwb': 2.0, 'camera': 0.3, 'perception': 1.0, 'fusion': 1.0, 'output': 0.5}


class Monitor:
    """토픽 수신 liveness/rate 추적 (monotonic 기준 — bag/clock 무관)."""

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

        def m(key):
            self.mon[key] = Monitor()
            return self.mon[key]

        for r in self.robots:
            # leader=corrected, follower=raw (test2 bag); override 가능
            uwb_def = f'/uwb/{r}_pose_corrected' if r == 'leader' else f'/uwb/{r}_pose'
            uwb_t = self.declare_parameter(f'{r}.uwb', uwb_def).value
            self.create_subscription(PoseStamped, uwb_t, m(f'uwb/{r}').hit, q)
            self.create_subscription(Float32, f'/uwb/{r}_qf',
                                     lambda msg, R=r: self.qf.__setitem__(R, float(msg.data)), q)
            self.create_subscription(PoseStamped, f'/perception/{r}/bev_pose', m(f'perception/{r}').hit, q)
            self.create_subscription(PoseStamped, f'/{r}/pose', m(f'fusion/{r}').hit, q)
            self.create_subscription(PoseStamped, f'/{r}/global/position', m(f'output/{r}').hit, q)
            self.create_subscription(Imu, f'/{r}/imu', m(f'imu/{r}').hit, q)
        self.create_subscription(CompressedImage, '/global_cam/processed/compressed', m('camera').hit, q)

        self.chrony = {'ok': False, 'offset_ms': None, 'source': 'n/a'}
        self.create_timer(1.0, self._tick)
        self.get_logger().info('localization_diagnostics_node up -> /localization/diagnostics @1Hz')

    # ---- helpers ----
    @staticmethod
    def _kv(k, v):
        return KeyValue(key=k, value=('n/a' if v is None else str(v)))

    def _age_ms(self, key):
        a = self.mon[key].age()
        return None if a is None else round(a * 1000.0, 1)

    def _hz(self, key):
        return round(self.mon[key].hz(), 1)

    def _poll_chrony(self):
        try:
            out = subprocess.run(['chronyc', '-c', 'tracking'],
                                 capture_output=True, text=True, timeout=1.0)
            if out.returncode != 0 or not out.stdout.strip():
                self.chrony = {'ok': False, 'offset_ms': None, 'source': 'n/a'}
                return
            f = out.stdout.strip().split(',')
            # chronyc -c tracking CSV: [1]=ref name, [4]=system time offset(s)
            self.chrony = {'ok': True, 'offset_ms': float(f[4]) * 1000.0, 'source': f[1]}
        except Exception:
            self.chrony = {'ok': False, 'offset_ms': None, 'source': 'n/a'}

    def _stage(self, name, prefix, thresh, qf=False):
        """per-robot 단계 status (pipeline/<stage>). 한 로봇이라도 stale=WARN."""
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

    # ---- main ----
    def _tick(self):
        self._poll_chrony()
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = []

        # per-robot 롤업 (미션 게이트 계약)
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
            s = DiagnosticStatus(name=f'localization/{r}', level=level,
                                 message=_LABEL[level], hardware_id='host')
            s.values = [self._kv('uwb_age_ms', self._age_ms(f'uwb/{r}')),
                        self._kv('fusion_age_ms', self._age_ms(f'fusion/{r}')),
                        self._kv('output_age_ms', self._age_ms(f'output/{r}')),
                        self._kv('output_hz', self._hz(f'output/{r}')),
                        self._kv('qf', qf)]
            st.append(s)

        # pipeline 단계
        st.append(self._stage('pipeline/uwb', 'uwb', TH['uwb'], qf=True))
        st.append(self._stage('pipeline/perception', 'perception', TH['perception']))
        st.append(self._stage('pipeline/fusion', 'fusion', TH['fusion']))
        st.append(self._stage('pipeline/output', 'output', TH['output']))
        st.append(self._stage('pipeline/imu', 'imu', 1.0))

        # camera (단일)
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
