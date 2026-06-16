// Shared per-robot state: latest pose (+ wall-clock receipt time) and battery.
//
// Two access patterns from one source:
//   - poseRef : a mutable ref the DigitalTwin reads every animation frame
//               WITHOUT triggering React re-renders (pose arrives ~30Hz).
//   - battery / poseTick : throttled React state for cards/header.
//
// The staleness guard lives in the DigitalTwin and keys off poseRef[r].rxWall.

import React, {
  createContext,
  useContext,
  useRef,
  useState,
  useEffect,
} from 'react';
import { useSubscription } from './useRos';

export const ROBOTS = ['leader', 'follower'];

const RobotContext = createContext(null);

// yaw (about Z) from a geometry quaternion.
function yawFromQuat(q) {
  if (!q) return 0;
  const { x = 0, y = 0, z = 0, w = 1 } = q;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

// ---- 배터리: 전압 → SoC(%) ------------------------------------------------
// turtlebot3 = 11.1V LiPo 3S. BatteryState.percentage가 부정확(11.8V인데 20% 등)이라
// 단자전압에서 직접 환산한다. 주의: 근사다 — 주행 중 부하로 전압 sag(실제보다 낮게 나옴),
// LiPo 방전곡선이 셀당 3.8V대에서 평탄해 비선형. 데모 게이지엔 충분하나 정밀 잔량계 아님.
const BATTERY_CELLS = 3; // 3S 팩 (leader/follower 공통)
// per-cell 개방전압 → SoC(%) 보간점(내림차순). 흔히 인용되는 LiPo 방전 테이블 근사.
const LIPO_CELL_SOC = [
  [4.2, 100], [4.1, 90], [4.0, 80], [3.92, 70], [3.86, 60], [3.83, 50],
  [3.79, 40], [3.76, 30], [3.72, 20], [3.65, 10], [3.45, 5], [3.3, 0],
];

function socFromVoltage(voltage, cells = BATTERY_CELLS) {
  if (typeof voltage !== 'number' || !isFinite(voltage) || voltage <= 0) return null;
  const v = voltage / cells; // per-cell 전압으로 환산
  const t = LIPO_CELL_SOC;
  if (v >= t[0][0]) return 100;
  if (v <= t[t.length - 1][0]) return 0;
  for (let i = 0; i < t.length - 1; i++) {
    const [vh, sh] = t[i];
    const [vl, sl] = t[i + 1];
    if (v <= vh && v >= vl) return sl + ((v - vl) / (vh - vl)) * (sh - sl); // 선형 보간
  }
  return null;
}

// ---- 배터리 표시 안정화: '전압'(raw 측정값)에 래치 적용 후 %로 환산 -----------
// 핵심: %가 아니라 측정 원신호인 '전압'에 안정화를 건다. 전압→SoC가 비선형이라
// %에 deadband를 걸면 곡선 가파른 near-empty 구간에서 양자화 노이즈가 증폭돼 출렁인다.
// 전압을 먼저 래치(deadband 밖 변화가 같은 방향 N회 확인될 때만 갱신)하면 전압·% 둘 다 안정.
// (실측 bag: 전압이 11.06~11.10V로 ±0.02V 양자화 → 0.06V deadband로 흡수.)
const V_DEADBAND = 0.15; // V : 이 미만 변화 무시 (bag가 11.0~11.1V로 0.1V 진동 → 위로 덮어 흡수)
const V_DROP_CONFIRM = 2; // 하강 확인 횟수 (1=즉시, 2=sag 견딤·권장)
const V_RISE_CONFIRM = 2; // 상승 확인 횟수 (충전 등 진짜 상승만)
const SOC_DISPLAY_STEP = 10; // % : 래치로 안정화된 값을 이 단위로 반올림(경계 채터 없음). 5로 낮추면 더 세밀.

// 범용 래치(히스테리시스+디바운스). f={disp,dir,count}, raw=이번 측정. 반환=안정화 값.
function latch(f, raw, deadband, dropN, riseN) {
  if (raw == null) return f.disp; // 데이터 없으면 마지막값 유지
  if (f.disp == null) { f.disp = raw; f.dir = 0; f.count = 0; return f.disp; }
  const dropping = raw <= f.disp - deadband;
  const rising = raw >= f.disp + deadband;
  if (!dropping && !rising) { f.dir = 0; f.count = 0; return f.disp; } // deadband 내 → 흔들림 무시
  const dir = dropping ? -1 : 1;
  if (dir !== f.dir) { f.dir = dir; f.count = 1; } // 방향 전환 → 후보 재시작
  else f.count += 1; // 같은 방향 연속 누적
  if (f.count >= (dir < 0 ? dropN : riseN)) { f.disp = raw; f.dir = 0; f.count = 0; } // 확인 시 갱신
  return f.disp;
}

export function RobotStateProvider({ children }) {
  // poseRef.current.leader = { x, y, yaw, rxWall } — read by the twin via rAF.
  const poseRef = useRef({
    leader: { x: 0, y: 0, yaw: 0, rxWall: 0 },
    follower: { x: 0, y: 0, yaw: 0, rxWall: 0 },
  });

  // Throttled mirror of pose receipt times so cards/header can show age without
  // re-rendering on every 30Hz message.
  const [poseAge, setPoseAge] = useState({ leader: 0, follower: 0 });
  const [battery, setBattery] = useState({
    leader: { percentage: null, voltage: null },
    follower: { percentage: null, voltage: null },
  });
  // Real per-robot mission state from /{robot}/status (host robot_state() projection).
  const [mission, setMission] = useState({
    leader: { state: '', task: '', laneNode: -1 },
    follower: { state: '', task: '', laneNode: -1 },
  });
  // 래치 필터 상태(per-robot) — 메시지마다 갱신하되 re-render 유발 안 하게 ref에 보존.
  const batFilterRef = useRef({
    leader: { disp: null, dir: 0, count: 0 },
    follower: { disp: null, dir: 0, count: 0 },
  });

  function handlePose(robot, msg) {
    const p = msg && msg.pose;
    if (!p || !p.position) return;
    poseRef.current[robot] = {
      x: p.position.x || 0,
      y: p.position.y || 0,
      yaw: yawFromQuat(p.orientation),
      rxWall: Date.now(),
    };
  }

  function handleBattery(robot, msg) {
    if (!msg) return;
    // 전압(raw 측정)을 먼저 래치로 안정화 → SoC 환산. (%에 직접 안 걸어 비선형 증폭 회피)
    const vRaw = typeof msg.voltage === 'number' && isFinite(msg.voltage) ? msg.voltage : null;
    const voltage = latch(batFilterRef.current[robot], vRaw, V_DEADBAND, V_DROP_CONFIRM, V_RISE_CONFIRM);
    const soc = voltage != null ? socFromVoltage(voltage) : null;
    // 안정화된 SoC를 표시 단위(10%)로 반올림 — 입력이 래치돼 있어 경계 채터 없음.
    const pct = soc != null ? Math.round(soc / SOC_DISPLAY_STEP) * SOC_DISPLAY_STEP : null;
    setBattery((prev) => ({ ...prev, [robot]: { percentage: pct, voltage } }));
  }

  function handleStatus(robot, msg) {
    if (!msg) return;
    setMission((prev) => ({
      ...prev,
      [robot]: { state: msg.state || '', task: msg.current_task || '', laneNode: msg.lane_node },
    }));
  }

  useSubscription(
    '/leader/global/position',
    'geometry_msgs/msg/PoseStamped',
    (m) => handlePose('leader', m),
    []
  );
  useSubscription(
    '/follower/global/position',
    'geometry_msgs/msg/PoseStamped',
    (m) => handlePose('follower', m),
    []
  );
  useSubscription(
    '/leader/battery_state',
    'sensor_msgs/msg/BatteryState',
    (m) => handleBattery('leader', m),
    []
  );
  useSubscription(
    '/follower/battery_state',
    'sensor_msgs/msg/BatteryState',
    (m) => handleBattery('follower', m),
    []
  );
  useSubscription(
    '/leader/status',
    'host_mission_interfaces/msg/RobotStatus',
    (m) => handleStatus('leader', m),
    []
  );
  useSubscription(
    '/follower/status',
    'host_mission_interfaces/msg/RobotStatus',
    (m) => handleStatus('follower', m),
    []
  );

  // Throttle pose-age state updates to ~5Hz (cards don't need 30Hz).
  useEffect(() => {
    const id = setInterval(() => {
      setPoseAge({
        leader: poseRef.current.leader.rxWall,
        follower: poseRef.current.follower.rxWall,
      });
    }, 200);
    return () => clearInterval(id);
  }, []);

  return (
    <RobotContext.Provider value={{ poseRef, poseAge, battery, mission }}>
      {children}
    </RobotContext.Provider>
  );
}

export function useRobotState() {
  return useContext(RobotContext);
}
