#!/usr/bin/env python3
"""uwb_reader_node — GrowSpace Q1 listener serial → ROS 2 topics.

Tested firmware: DWM1001 fw_ver=x01030001 (Mar 27 2019 build).
The listener's INF location stream is gated by the `les` toggle (persisted
in NVM); we explicitly probe and toggle it on at startup so the node works
regardless of the prior NVM state. Native line format:

    [<uptime> INF] loc_data: <N>
     <idx>) <TAGID>[x,y,z,qf,x<flags>]

We parse the per-tag lines and route each tag to a rover role by short ID.
Topics (one set per role; <role> in {leader, follower}):
    /uwb/<role>_pose    geometry_msgs/PoseStamped  (frame_id from param)
    /uwb/<role>_qf      std_msgs/Float32           (Quality Factor 0~100)
    /uwb/<role>_marker  visualization_msgs/Marker  (QF-colored sphere)
    /uwb/status         std_msgs/String            ("ok" | "timeout"; leader freshness)

Roles map to tag short IDs via the leader_tag_id / follower_tag_id params.
Multi-tag (follower) support pulled forward from council DEF-003 on 2026-05-31
after empirically confirming the listener reports 2 tags at once (loc_data: 2).
"""

import re
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker
import serial


# Matches a per-tag line, e.g. " 0) 08BA[0.39,0.31,-0.06,67,x04]"
TAG_LINE_RE = re.compile(
    r'([0-9A-Fa-f]{4})\[\s*'
    r'(-?\d+\.\d+)\s*,\s*'
    r'(-?\d+\.\d+)\s*,\s*'
    r'(-?\d+\.\d+)\s*,\s*'
    r'(\d+)\s*,\s*x[0-9A-Fa-f]+\s*\]'
)

# Leader is declared stale (status "timeout") after this gap with no pose.
LEADER_TIMEOUT_S = 2.0

# Serial reconnect backoff bounds (seconds). The CP2102 USB bridge can
# re-enumerate in sub-second bursts, so retry quickly but cap the interval so a
# long outage does not busy-loop the open() syscall.
RECONNECT_MIN_S = 0.5
RECONNECT_MAX_S = 5.0


class _TagChannel:
    """Publishers + freshness state for one UWB tag (one rover role).

    One channel owns /uwb/<role>_pose, _qf and _marker. The marker is a
    QF-colored sphere (red=0 .. green=100); UWB has no heading, so a point is
    more honest than an arrow. Roles share the QF color scale and are told
    apart by marker namespace ('uwb_<role>'). `last_seen` is updated only by
    publish(); it backs the owning node's freshness/timeout check.
    """

    def __init__(self, node, role, frame_id, marker_scale):
        self.role = role
        self.frame_id = frame_id
        self.marker_scale = marker_scale
        self.last_seen = node.get_clock().now()
        self.seen = False   # publish()가 한 번이라도 불렸는지 — 생존 경고용
                            # (안 본 role은 의도적 미연결일 수 있어 경고 안 함).
        self._avg_buf = []  # (x,y,z) accumulated for averaged debug logging
        self._pub_pose = node.create_publisher(
            PoseStamped, f'/uwb/{role}_pose', 10)
        self._pub_qf = node.create_publisher(Float32, f'/uwb/{role}_qf', 10)
        self._pub_marker = node.create_publisher(
            Marker, f'/uwb/{role}_marker', 10)

    def publish(self, x, y, z, qf, now):
        self.last_seen = now
        self.seen = True
        stamp = now.to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        self._pub_pose.publish(pose)

        qf_msg = Float32()
        qf_msg.data = float(qf)
        self._pub_qf.publish(qf_msg)

        self._pub_marker.publish(self._make_marker(x, y, z, qf, stamp))
        self._avg_buf.append((x, y, z))

    def _make_marker(self, x, y, z, qf, stamp):
        qf_norm = max(0.0, min(1.0, qf / 100.0))
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = f'uwb_{self.role}'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.marker_scale
        marker.scale.y = self.marker_scale
        marker.scale.z = self.marker_scale
        marker.color.r = 1.0 - qf_norm
        marker.color.g = qf_norm
        marker.color.b = 0.0
        marker.color.a = 1.0
        return marker

    def drain_avg(self):
        """Mean of buffered samples since last call as (n, ax, ay, az); clears
        the buffer. Returns None when nothing was buffered."""
        buf = self._avg_buf
        self._avg_buf = []
        if not buf:
            return None
        n = len(buf)
        return (n,
                sum(p[0] for p in buf) / n,
                sum(p[1] for p in buf) / n,
                sum(p[2] for p in buf) / n)


class UWBReaderNode(Node):
    def __init__(self):
        super().__init__('uwb_reader_node')

        self.declare_parameter('port', '/dev/ttyUWB')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('frame_id', 'uwb_global')
        # Tag short IDs (4-hex) per role. leader_tag_id empty => auto-lock the
        # first observed tag as leader (single-tag convenience). follower_tag_id
        # empty => no follower channel is created (no auto-lock for follower).
        self.declare_parameter('leader_tag_id', '')
        self.declare_parameter('follower_tag_id', '')
        self.declare_parameter('marker_scale', 0.05)  # meters (sphere diameter)

        self.port = self.get_parameter('port').value
        self.baud = self.get_parameter('baud_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        leader_id = self.get_parameter('leader_tag_id').value.upper()
        follower_id = self.get_parameter('follower_tag_id').value.upper()
        self.marker_scale = float(self.get_parameter('marker_scale').value)

        self.pub_status = self.create_publisher(String, '/uwb/status', 10)

        # role -> _TagChannel. Leader always exists (it may auto-lock); follower
        # only when its tag ID is configured.
        self.channels = {
            'leader': _TagChannel(
                self, 'leader', self.frame_id, self.marker_scale)}
        self.tag_to_role = {}
        if leader_id:
            self.tag_to_role[leader_id] = 'leader'
        if follower_id:
            self.tag_to_role[follower_id] = 'follower'
            self.channels['follower'] = _TagChannel(
                self, 'follower', self.frame_id, self.marker_scale)
        # Leader auto-locks the first unrecognized tag only when no explicit
        # leader_tag_id was given. A follower tag plugged in first must not get
        # mislabeled as leader, so auto-lock skips any already-assigned role.
        self._leader_auto = not leader_id
        self._auto_locked_id = None

        self.get_logger().info(
            f'port={self.port} @ {self.baud}, frame_id={self.frame_id}, '
            f'leader={leader_id or "<auto>"}, '
            f'follower={follower_id or "<none>"}')

        # Serial connection state. The CP2102 USB bridge re-enumerates on power/
        # cable glitches (kernel: "USB disconnect" → new ttyUSB), leaving our fd
        # dead so readline() raises forever. We treat the port as a resource that
        # can drop anytime: on error, close and reconnect with capped exponential
        # backoff. A device absent at startup is likewise non-fatal — the read
        # timer keeps retrying until it appears.
        self.ser = None
        self._reconnect_delay = RECONNECT_MIN_S
        self._reconnect_at = 0.0  # monotonic deadline; 0 => attempt immediately
        self._connect()

        self.timer = self.create_timer(0.01, self._read_serial)
        self.status_timer = self.create_timer(1.0, self._publish_status)
        # 측정치 평균은 debug 전용(평소 숨김). 1s마다 버퍼를 비워 메모리 바운드 + debug 출력.
        self.avg_timer = self.create_timer(1.0, self._log_avg)

    def _open_serial(self):
        """Open the port and wake the listener prompt. Raises on failure."""
        ser = serial.Serial(self.port, self.baud, timeout=0.1)
        time.sleep(0.3)
        # Wake the dwm> prompt; harmless if already awake.
        ser.write(b'\r\r')
        time.sleep(0.2)
        ser.reset_input_buffer()
        return ser

    def _connect(self):
        """(Re)open the port and (re)start the location stream.

        Returns True on success. Any serial error during open or stream probing
        leaves us cleanly disconnected (self.ser is None) so the caller retries —
        the device may still be mid-re-enumeration."""
        try:
            self.ser = self._open_serial()
            self._ensure_location_stream()
        except Exception as e:
            self.get_logger().warn(
                f'UWB 연결 실패 ({self.port}): {e}', throttle_duration_sec=5)
            self._drop_serial()
            return False
        self.get_logger().info('UWB 시리얼 연결됨 — location stream 기동')
        return True

    def _drop_serial(self):
        """Close the (likely dead) port and mark us disconnected."""
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def _ensure_location_stream(self):
        # `les` is a NVM-persisted ON/OFF toggle. We can't query it, so probe:
        # toggle once, listen, toggle again if nothing arrived.
        for attempt in (1, 2):
            self.ser.write(b'les\r')
            deadline = time.monotonic() + 1.5
            saw_loc = False
            while time.monotonic() < deadline:
                line = self.ser.readline().decode('utf-8', errors='ignore')
                if 'loc_data' in line:
                    saw_loc = True
                    break
            if saw_loc:
                self.get_logger().debug(
                    f'Location stream ON (les toggle attempt {attempt}).')
                self.ser.reset_input_buffer()
                return
        self.get_logger().warn(
            'No loc_data after two `les` toggles — listener may be silent '
            '(tag off, anchors unpowered, or out of range).')

    def _read_serial(self):
        if self.ser is None:
            # Disconnected: retry on the backoff deadline, growing the interval on
            # repeated failure so a long outage doesn't hammer open().
            if time.monotonic() >= self._reconnect_at:
                if self._connect():
                    self._reconnect_delay = RECONNECT_MIN_S
                else:
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2, RECONNECT_MAX_S)
                    self._reconnect_at = time.monotonic() + self._reconnect_delay
            return

        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
        except Exception as e:
            # EOF on a re-enumerated CP2102 surfaces here as a SerialException.
            # Drop the dead fd and start a fresh backoff; udev needs a beat to
            # recreate the /dev/ttyUWB symlink against the new ttyUSB.
            self.get_logger().warn(
                f'Read error: {e} — 포트 닫고 재연결', throttle_duration_sec=5)
            self._drop_serial()
            self._reconnect_delay = RECONNECT_MIN_S
            self._reconnect_at = time.monotonic() + RECONNECT_MIN_S
            return

        if not line:
            return

        m = TAG_LINE_RE.search(line)
        if not m:
            return

        role = self._role_for_tag(m.group(1).upper())
        if role is None:
            return

        self.channels[role].publish(
            float(m.group(2)), float(m.group(3)), float(m.group(4)),
            int(m.group(5)), self.get_clock().now())

    def _role_for_tag(self, tag_id):
        """Map a tag short ID to a role, applying leader auto-lock.

        Returns the role name, or None for an unrecognized tag to ignore (e.g.
        a third tag with no assigned role)."""
        role = self.tag_to_role.get(tag_id)
        if role is not None:
            return role
        if self._leader_auto:
            if self._auto_locked_id is None:
                self._auto_locked_id = tag_id
                self.tag_to_role[tag_id] = 'leader'
                self.get_logger().debug(f'Auto-locked leader tag: {tag_id}')
                return 'leader'
            if tag_id == self._auto_locked_id:
                return 'leader'
        return None

    def _log_avg(self):
        for role, channel in self.channels.items():
            avg = channel.drain_avg()
            if avg is None:
                continue
            n, ax, ay, az = avg
            # 측정치 = debug 전용 (--log-level debug 때만). 평소엔 경고 위주로 조용히.
            self.get_logger().debug(
                f'UWB {role} avg ({n} samples): '
                f'x={ax:.3f} y={ay:.3f} z={az:.3f} [m]')

    def _publish_status(self):
        # /uwb/status는 기존 계약대로 leader 신선도(ok|timeout). 생존점검은 1Hz로 자주
        # 돌되, 화면엔 "한 번이라도 봤던 role이 끊겼을 때"만(=이상) WARN(throttle). 처음부터
        # 안 보인 role은 의도적 미연결일 수 있어 침묵 — 측정치 대신 경고 위주.
        now = self.get_clock().now()
        leader_gap = (now - self.channels['leader'].last_seen).nanoseconds / 1e9
        msg = String()
        msg.data = 'ok' if leader_gap < LEADER_TIMEOUT_S else 'timeout'
        self.pub_status.publish(msg)

        for role, ch in self.channels.items():
            gap = (now - ch.last_seen).nanoseconds / 1e9
            if ch.seen and gap > LEADER_TIMEOUT_S:
                self.get_logger().warn(
                    f'UWB {role} 끊김 — {gap:.1f}s 무신호',
                    throttle_duration_sec=5)

    def destroy_node(self):
        if getattr(self, 'ser', None) is not None and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UWBReaderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
