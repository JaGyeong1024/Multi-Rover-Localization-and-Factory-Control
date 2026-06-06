import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useRos } from '../ros/useRos';
import { useDiagnostics } from '../ros/useDiagnostics';
import { useMission } from '../ros/useMission';

// Event / alarm timeline — "what just happened" (the missing history in an ops
// console). NOT a raw log: events are emitted only on state TRANSITIONS (edges):
// health changes, sensor drop/recover, time-sync warnings, dispatch, e-stop.
// All derived from existing topics/state (no backend changes).

const NAME_LABEL = {
  'localization/leader': 'LEADER LOC',
  'localization/follower': 'FOLLOWER LOC',
  'pipeline/uwb': 'UWB',
  'pipeline/camera': 'GLOBAL CAM',
  'pipeline/perception': 'PERCEPTION',
  'pipeline/fusion': 'FUSION',
  'pipeline/output': 'FINAL OUTPUT',
  'pipeline/imu': 'IMU',
  'sync/chrony': 'TIME SYNC',
};
const WATCH = Object.keys(NAME_LABEL);

const SEV = {
  crit: { color: 'var(--lost)', icon: '⚠' },
  warn: { color: 'var(--degraded)', icon: '▲' },
  info: { color: 'var(--good)', icon: '•' },
};

const pad = (n) => String(n).padStart(2, '0');
const hhmmss = (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;

function diagEvent(name, level) {
  const label = NAME_LABEL[name] || name;
  if (name.startsWith('localization/')) {
    if (level >= 2) return ['crit', `${label} LOST`];
    if (level === 1) return ['warn', `${label} DEGRADED`];
    return ['info', `${label} recovered`];
  }
  if (name === 'sync/chrony') {
    return level >= 1 ? ['warn', 'time sync warning (offset↑)'] : ['info', 'time sync OK'];
  }
  return level >= 1 ? ['warn', `${label} dropped`] : ['info', `${label} recovered`];
}

export default function EventTimeline({ estopActive }) {
  const { connected } = useRos();
  const { statuses } = useDiagnostics();
  const { queue } = useMission();
  const [events, setEvents] = useState([]);
  const idRef = useRef(0);

  const push = useCallback((sev, src, text) => {
    setEvents((es) => {
      idRef.current += 1;
      return [{ id: idRef.current, t: new Date(), sev, src, text }, ...es].slice(0, 100);
    });
  }, []);

  // connection transitions (skip initial)
  const prevConn = useRef(null);
  useEffect(() => {
    if (prevConn.current !== null && connected !== prevConn.current) {
      if (connected) push('info', 'rosbridge', 'rosbridge connected');
      else push('crit', 'rosbridge', 'rosbridge disconnected');
    }
    prevConn.current = connected;
  }, [connected, push]);

  // diagnostics level transitions (first receipt only primes, no event)
  const prevLevels = useRef(null);
  useEffect(() => {
    const names = Object.keys(statuses);
    if (names.length === 0) return;
    if (prevLevels.current === null) {
      const init = {};
      for (const n of names) init[n] = statuses[n].level;
      prevLevels.current = init;
      return;
    }
    for (const n of WATCH) {
      const st = statuses[n];
      if (!st) continue;
      const prev = prevLevels.current[n];
      if (prev !== undefined && prev !== st.level) {
        const [sev, text] = diagEvent(n, st.level);
        push(sev, n.split('/')[0], text);
      }
      prevLevels.current[n] = st.level;
    }
  }, [statuses, push]);

  // mission dispatches (queue growth)
  const prevQ = useRef(null);
  useEffect(() => {
    if (prevQ.current !== null && queue.length > prevQ.current) {
      const item = queue[queue.length - 1];
      push('info', 'mission', `dispatch: ${item.label || item.part}${item.zone ? ` (${item.zone})` : ''}`);
    }
    prevQ.current = queue.length;
  }, [queue, push]);

  // e-stop
  const prevEstop = useRef(false);
  useEffect(() => {
    if (estopActive !== prevEstop.current) {
      push(estopActive ? 'crit' : 'info', 'e-stop', estopActive ? 'E-STOP engaged — dispatch halted' : 'E-STOP cleared');
      prevEstop.current = estopActive;
    }
  }, [estopActive, push]);

  return (
    <div className="panel events-panel">
      <div className="panel-head">
        <span className="panel-title">Events / Alarms</span>
        <span className="panel-sub">{events.length ? `${events.length}` : 'idle'}</span>
      </div>
      {events.length === 0 ? (
        <div className="events-empty">No events — shown on state changes</div>
      ) : (
        <ul className="events-list">
          {events.map((e) => {
            const s = SEV[e.sev] || SEV.info;
            return (
              <li key={e.id} className={`event-row event-${e.sev}`}>
                <span className="event-time">{hhmmss(e.t)}</span>
                <span className="event-dot" style={{ color: s.color }} aria-hidden="true">{s.icon}</span>
                <span className="event-src">{e.src}</span>
                <span className="event-text">{e.text}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
