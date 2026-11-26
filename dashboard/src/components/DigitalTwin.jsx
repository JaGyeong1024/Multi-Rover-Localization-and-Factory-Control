import React, { useEffect, useRef, useState } from 'react';
import { useRobotState } from '../ros/useRobotState';
import { useDiagnostics } from '../ros/useDiagnostics';
import { useMission } from '../ros/useMission';
import { useMissionState } from '../ros/useMissionState';
import { createTwin } from './digitalTwin/createTwin';

export default function DigitalTwin() {
  const mountRef = useRef(null);
  const { poseRef } = useRobotState();
  const { getStatus } = useDiagnostics();
  const { dispatch, queue } = useMission();
  const { pathRef, obstacles } = useMissionState();
  const obstaclesRef = useRef(obstacles);
  obstaclesRef.current = obstacles;

  // Keep latest diagnostics-derived state accessible inside the rAF loop.
  const getStatusRef = useRef(getStatus);
  getStatusRef.current = getStatus;

  // Bridge mission state into the (run-once) three.js effect via refs:
  //   dispatchRef — click handler publishes /mission/dispatch.
  //   queuedRef   — set of part ids currently in the queue → glow their cubes.
  const dispatchRef = useRef(dispatch);
  dispatchRef.current = dispatch;
  const queuedRef = useRef(new Set());
  queuedRef.current = new Set((queue || []).map((q) => q.part));

  const resetViewRef = useRef(() => {});
  // Toggle for the goals that are not a rover's own place (pick / load goals ...); default
  // OFF. Home goals are always drawn. Read inside the rAF loop without re-running the effect.
  const [showGoals, setShowGoals] = useState(false);
  const showGoalsRef = useRef(showGoals);
  showGoalsRef.current = showGoals;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;
    const twin = createTwin(mount, { poseRef, pathRef, obstaclesRef, queuedRef, showGoalsRef, dispatchRef });
    resetViewRef.current = twin.resetView;
    return twin.dispose;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="twin">
      <div className="twin-bar">
        <span className="twin-title">Digital Twin View</span>
        <div className="twin-bar-btns">
          <button
            type="button"
            className="twin-reset"
            aria-pressed={showGoals}
            onClick={() => setShowGoals((v) => !v)}
          >
            Goals: {showGoals ? 'ON' : 'OFF'}
          </button>
          <button type="button" className="twin-reset" onClick={() => resetViewRef.current()}>
            Reset View
          </button>
        </div>
      </div>
      <div ref={mountRef} className="twin-canvas" />
    </div>
  );
}
