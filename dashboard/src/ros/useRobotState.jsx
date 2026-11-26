// Per-robot pose and battery. poseRef is read every animation frame by the DigitalTwin
// without re-rendering (pose arrives ~30 Hz); battery / poseTick are throttled React state.

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

// ---- Battery: voltage -> SoC (%) ---------------------------------------
// 3S LiPo (11.1 V). BatteryState.percentage is unreliable, so SoC comes from the terminal
// voltage: approximate (load sag, flat curve near 3.8 V/cell), fine for a gauge.
const BATTERY_CELLS = 3; // 3S pack (both rovers)
// Per-cell open-circuit voltage -> SoC (%) points, descending (typical LiPo table).
const LIPO_CELL_SOC = [
  [4.2, 100], [4.1, 90], [4.0, 80], [3.92, 70], [3.86, 60], [3.83, 50],
  [3.79, 40], [3.76, 30], [3.72, 20], [3.65, 10], [3.45, 5], [3.3, 0],
];

function socFromVoltage(voltage, cells = BATTERY_CELLS) {
  if (typeof voltage !== 'number' || !isFinite(voltage) || voltage <= 0) return null;
  const v = voltage / cells; // per-cell voltage
  const t = LIPO_CELL_SOC;
  if (v >= t[0][0]) return 100;
  if (v <= t[t.length - 1][0]) return 0;
  for (let i = 0; i < t.length - 1; i++) {
    const [vh, sh] = t[i];
    const [vl, sl] = t[i + 1];
    if (v <= vh && v >= vl) return sl + ((v - vl) / (vh - vl)) * (sh - sl); // linear interpolation
  }
  return null;
}

// ---- Display stabilisation: latch the raw voltage, then convert to % ----
// latched on voltage, not % (SoC is nonlinear in voltage; a % deadband amplifies noise near empty)
const V_DEADBAND = 0.15; // V, changes smaller than this are ignored (voltage jitters ~0.1 V)
const V_DROP_CONFIRM = 2; // consecutive readings to accept a drop (1 = immediate; 2 rides out sag)
const V_RISE_CONFIRM = 2; // consecutive readings to accept a rise (e.g. charging)
const SOC_DISPLAY_STEP = 10; // %, the latched SoC is rounded to this step (no boundary chatter)

// Hysteresis + debounce latch. f = {disp, dir, count}, raw = this reading; returns the stable value.
function latch(f, raw, deadband, dropN, riseN) {
  if (raw == null) return f.disp; // no data: keep the last value
  if (f.disp == null) { f.disp = raw; f.dir = 0; f.count = 0; return f.disp; }
  const dropping = raw <= f.disp - deadband;
  const rising = raw >= f.disp + deadband;
  if (!dropping && !rising) { f.dir = 0; f.count = 0; return f.disp; } // inside the deadband: ignore
  const dir = dropping ? -1 : 1;
  if (dir !== f.dir) { f.dir = dir; f.count = 1; } // direction changed: restart the candidate
  else f.count += 1; // same direction: accumulate
  if (f.count >= (dir < 0 ? dropN : riseN)) { f.disp = raw; f.dir = 0; f.count = 0; } // confirmed: update
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
  // Per-robot latch state, kept in a ref so updates do not re-render.
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
    // Latch the raw voltage first, then convert to SoC.
    const vRaw = typeof msg.voltage === 'number' && isFinite(msg.voltage) ? msg.voltage : null;
    const voltage = latch(batFilterRef.current[robot], vRaw, V_DEADBAND, V_DROP_CONFIRM, V_RISE_CONFIRM);
    const soc = voltage != null ? socFromVoltage(voltage) : null;
    // Round the stable SoC to the display step.
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
