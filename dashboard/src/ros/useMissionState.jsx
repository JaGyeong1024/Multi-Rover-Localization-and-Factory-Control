// "Path mode" state: what the host planned and what each rover reports.
//   /{robot}/path_cmd (2 Hz), nav_feedback, safety_state, localization_quality,
//   /mission/events, /mission/obstacles, /mission/goals (latched table)
// pathRef is read every animation frame by the DigitalTwin (no re-render); pathState / events /
// obstacles are React state. Each datum is {msg, rx}; judge staleness with isFresh().

import React, { createContext, useContext, useRef, useState, useEffect, useCallback } from 'react';
import { useSubscription } from './useRos';
import { ROBOTS } from './useRobotState';
import { COLORS } from '../theme';

export const MISSION_STALE_MS = 2000;
export const isFresh = (d, now = Date.now()) => !!d && now - d.rx <= MISSION_STALE_MS;

const EMPTY = () => ({ pathCmd: null, nav: null, safety: null, quality: null });
const emptyPathState = () => Object.fromEntries(ROBOTS.map((r) => [r, EMPTY()]));

const MissionStateContext = createContext({
  pathState: emptyPathState(),
  pathRef: { current: emptyPathState() },
  events: [],
  obstacles: [],
  now: Date.now(),
});

// JSON String message -> parsed value (null on malformed data).
function parseJson(m) {
  try {
    return JSON.parse((m && m.data) || '');
  } catch (e) {
    return null;
  }
}

// Obstacle JSON ([{x,y,r}] or {obstacles:[...]}) -> clean list.
function parseObstacles(m) {
  const v = parseJson(m);
  const list = Array.isArray(v) ? v : (v && v.obstacles) || [];
  return list
    .map((o) => ({
      x: Number(o.x), y: Number(o.y), r: Number(o.r),
      // rectangle form (mapper boxes, operator rectangles); "r" only covers it
      width: Number(o.width) || 0, height: Number(o.height) || 0,
      yaw: Number(o.yaw) || 0, pad: Number(o.pad) || 0,
    }))
    .filter((o) => isFinite(o.x) && isFinite(o.y) &&
      ((isFinite(o.r) && o.r > 0) || (o.width > 0 && o.height > 0)));
}

export function MissionStateProvider({ children }) {
  const pathRef = useRef(emptyPathState());
  const [pathState, setPathState] = useState(pathRef.current);
  const [events, setEvents] = useState([]); // newest last, {seq, kind, detail}
  const [obsBySource, setObsBySource] = useState({});
  const [now, setNow] = useState(Date.now());
  const seqRef = useRef(0);

  // 0.5 s tick so staleness badges update even when no message arrives.
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);

  const put = useCallback((robot, key, msg) => {
    pathRef.current[robot][key] = { msg, rx: Date.now() };
    // fresh object per update so consumers re-render
    setPathState({ ...pathRef.current, [robot]: { ...pathRef.current[robot] } });
  }, []);

  const T = 'host_mission_interfaces/msg/';
  // /dashboard/reset (std_msgs/Empty) reloads the page: clean event timeline and cards
  // before a recorded run.
  useSubscription('/dashboard/reset', 'std_msgs/msg/Empty', () => window.location.reload(), []);
  useSubscription('/leader/path_cmd', T + 'PathCommand', (m) => put('leader', 'pathCmd', m), [put]);
  useSubscription('/leader/nav_feedback', T + 'NavFeedback', (m) => put('leader', 'nav', m), [put]);
  useSubscription('/leader/safety_state', T + 'SafetyState', (m) => put('leader', 'safety', m), [put]);
  useSubscription('/leader/localization_quality', 'std_msgs/msg/String',
    (m) => put('leader', 'quality', m), [put]);
  useSubscription('/follower/path_cmd', T + 'PathCommand', (m) => put('follower', 'pathCmd', m), [put]);
  useSubscription('/follower/nav_feedback', T + 'NavFeedback', (m) => put('follower', 'nav', m), [put]);
  useSubscription('/follower/safety_state', T + 'SafetyState', (m) => put('follower', 'safety', m), [put]);
  useSubscription('/follower/localization_quality', 'std_msgs/msg/String',
    (m) => put('follower', 'quality', m), [put]);

  // Coordinator's goal table (latched JSON {name: {x, y, theta}}); the twin draws it.
  useSubscription('/mission/goals', 'std_msgs/msg/String', (m) => {
    const g = parseJson(m);
    if (g && typeof g === 'object') pathRef.current.goals = g;
  }, []);

  useSubscription('/mission/events', 'std_msgs/msg/String', (m) => {
    const e = parseJson(m);
    if (!e) return;
    seqRef.current += 1;
    const item = { seq: seqRef.current, kind: String(e.kind || 'event'), detail: String(e.detail || ''),
      rx: Date.now() };
    setEvents((es) => [...es, item].slice(-50));
  }, []);

  // Obstacle sources: one line per topic (operator input, follower-LiDAR map).
  const useObstacleSource = (topic, key) =>
    useSubscription(topic, 'std_msgs/msg/String',
      (m) => setObsBySource((s) => ({ ...s, [key]: parseObstacles(m) })), []);
  useObstacleSource('/mission/obstacles', 'operator');
  // mapped boxes (/mission/mapped_obstacles) reach the planner but are not drawn
  // useObstacleSource('/mission/mapped_obstacles', 'mapped');

  const obstacles = Object.values(obsBySource).flat();

  return (
    <MissionStateContext.Provider value={{ pathState, pathRef, events, obstacles, now }}>
      {children}
    </MissionStateContext.Provider>
  );
}

export function useMissionState() {
  return useContext(MissionStateContext);
}

// ---- Presentation helpers (cards + twin label share these) ---------------------------
// alert = STOPPED / ERROR / LOST, warn = LIMITED / DEGRADED, grey = older than MISSION_STALE_MS
const NAV_COLOR = {
  TRACKING: COLORS.good, PARKED: COLORS.good, HOLDING: COLORS.pickup,
  IDLE: COLORS.idle, STOPPED: COLORS.lost, ERROR: COLORS.lost,
};
const SAFETY_COLOR = { OK: COLORS.good, LIMITED: COLORS.degraded, STOPPED: COLORS.lost };
const QUALITY_COLOR = { GOOD: COLORS.good, DEGRADED: COLORS.degraded, LOST: COLORS.lost };

// {text, color, alert} for each of nav / safety / quality of one robot (null when never received).
// Nav/safety are rover-published at a steady rate; a stale one is shown grey as "STALE".
export function summarizePath(d, now = Date.now()) {
  const out = {};
  const wrap = (item, state, colors, text) => {
    if (!item) return null;
    const fresh = isFresh(item, now);
    const key = state || '—';
    return {
      text: fresh ? text : `STALE ${key}`,
      color: fresh ? colors[key] || COLORS.stale : COLORS.stale,
      alert: fresh && (key === 'STOPPED' || key === 'ERROR' || key === 'LOST'),
      fresh,
    };
  };
  const nav = d.nav && d.nav.msg;
  if (nav) {
    const pct = isFinite(nav.progress) ? ` ${Math.round(nav.progress * 100)}%` : '';
    out.nav = wrap(d.nav, nav.state, NAV_COLOR, `${nav.state}${pct}${nav.detail ? ` · ${nav.detail}` : ''}`);
  }
  const sf = d.safety && d.safety.msg;
  if (sf) {
    const reasons = (sf.reasons || []).join(' ');
    const scale = isFinite(sf.speed_scale) ? ` ×${sf.speed_scale.toFixed(2)}` : '';
    out.safety = wrap(d.safety, sf.state, SAFETY_COLOR, `${sf.state}${scale}${reasons ? ` · ${reasons}` : ''}`);
  }
  const q = d.quality && d.quality.msg;
  if (q) out.quality = wrap(d.quality, q.data, QUALITY_COLOR, q.data);
  return out;
}
