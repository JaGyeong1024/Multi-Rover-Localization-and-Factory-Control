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
    let pct = msg.percentage;
    // BatteryState.percentage may be 0..1 OR 0..100 — normalize to 0..100.
    if (typeof pct === 'number') {
      if (pct <= 1) pct = pct * 100;
    } else {
      pct = null;
    }
    const voltage = typeof msg.voltage === 'number' ? msg.voltage : null;
    setBattery((prev) => ({ ...prev, [robot]: { percentage: pct, voltage } }));
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
    <RobotContext.Provider value={{ poseRef, poseAge, battery }}>
      {children}
    </RobotContext.Provider>
  );
}

export function useRobotState() {
  return useContext(RobotContext);
}
