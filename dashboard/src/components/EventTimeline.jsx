import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useRos } from '../ros/useRos';
import { useDiagnostics } from '../ros/useDiagnostics';
import { useMission, useEstop } from '../ros/useMission';
import { useMissionState } from '../ros/useMissionState';
import { prettyId } from '../missionMap';

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
  'align/leader': 'LEADER IMU',
  'align/follower': 'FOLLOWER IMU',
  'sync/chrony': 'TIME SYNC',
};
const WATCH = Object.keys(NAME_LABEL);
const HOLD_MS = 3000;   // a sensor / localization level is listed once it has held this long

const SEV = {
  crit: { color: 'var(--lost)', icon: '⚠' },
  warn: { color: 'var(--degraded)', icon: '▲' },
  info: { color: 'var(--good)', icon: '•' },
};

// /mission/events kinds that deserve a warning / alarm colour (mirrors the coordinator's _WARN_EVENTS).
// A refused operator command is an alarm: the operator expected something to happen.
const CRIT_KINDS = new Set(['control_rejected', 'superseded']);
const WARN_KINDS = new Set(['halt', 'paused', 'phase_timeout', 'pick_failed',
  'teach_rejected', 'dispatch_rejected', 'abandoned', 'start_after_turn_timeout']);
// Planner internals / echoes that are not operator-level: never listed. An unknown kind IS shown.
// enqueued: the dispatch line below already covers it. control: echo of the operator's own click.
// estop: the e-stop line below (from /mission/state) already covers engage/clear.
// hold / plan_failed and their ends: a rover waiting for a path, which mostly clears by itself
// (the rover card shows it standing and why); one that lasts becomes 'paused', which is listed.
const HIDDEN_KINDS = new Set(['path_issued', 'obstacles', 'plan_waiting', 'replan', 'reenter',
  'add_obstacle', 'control', 'enqueued', 'resume_deferred', 'estop',
  'hold', 'hold_cleared', 'plan_failed', 'plan_recovered']);

// Queue item id ("fetch_gear_01#1") -> "Gear 01".
const itemName = (id) => prettyId(String(id).replace(/^fetch_/, '').replace(/#\d+$/, ''));

// Host event -> one plain line. Kinds without a confident mapping stay "kind: detail".
function hostEventText(kind, detail) {
  switch (kind) {
    case 'mission_started': return 'mission started';
    case 'loaded': return `loaded ${itemName(detail)}`;
    case 'delivered': return `delivered ${itemName(detail)}`;
    case 'pick_failed': return `pick failed ${itemName(detail)}`;
    case 'unload_done': return 'unload done';
    case 'idle_turn': return `${detail.split(':')[0]} turns east`;
    case 'start_waits_for_turn': return 'start waits: rovers turning east';
    case 'start_after_turn_timeout': return `started, not east: ${detail}`;
    case 'control_rejected': return `refused: ${detail}`;
    case 'superseded': return `${detail} ignores this host`;   // short: the row cuts at ~30 characters
    case 'abandoned': {
      const m = /^not delivered: (.*)$/.exec(detail);
      const names = m && m[1] !== 'none' ? m[1].split(', ').map(itemName).join(', ') : '';
      return names ? `abandoned ${names}` : 'abandoned';
    }
    default: return detail ? `${kind}: ${detail}` : kind;
  }
}

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
  if (name.startsWith('align/')) {
    return level === 0
      ? ['info', `${label} heading aligned ✓`]
      : ['warn', `${label} heading unaligned`];
  }
  return level >= 1 ? ['warn', `${label} dropped`] : ['info', `${label} recovered`];
}

export default function EventTimeline() {
  const { active: estopActive } = useEstop();
  const { connected } = useRos();
  const { statuses } = useDiagnostics();
  const { queue } = useMission();
  const { events: missionEvents } = useMissionState();
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

  // diagnostics level transitions (first receipt only primes, no event).
  // Sensor / localization levels are listed only once a level has held HOLD_MS and differs
  // from the last listed one: a status flickering for under a second listed a dropped +
  // recovered pair each time (dozens per run). Alignment stays immediate.
  const prevLevels = useRef(null);
  const pending = useRef({});         // name -> { level, since } not yet listed
  useEffect(() => {
    const timer = setInterval(() => {
      const now = Date.now();
      for (const [n, p] of Object.entries(pending.current)) {
        if (now - p.since < HOLD_MS) continue;
        const [sev, text] = diagEvent(n, p.level);
        push(sev, n.split('/')[0], text);
        prevLevels.current[n] = p.level;
        delete pending.current[n];
      }
    }, 500);
    return () => clearInterval(timer);
  }, [push]);
  useEffect(() => {
    const names = Object.keys(statuses);
    if (names.length === 0) return;
    if (prevLevels.current === null) {
      const init = {};
      for (const n of names) init[n] = statuses[n].level;
      prevLevels.current = init;
      // Alignment status is listed even at prime (misaligned is actionable); the other
      // statuses only prime, to avoid flooding the list on connect.
      for (const n of WATCH) {
        if (n.startsWith('align/') && statuses[n]) {
          const [sev, text] = diagEvent(n, statuses[n].level);
          push(sev, n.split('/')[0], text);
        }
      }
      return;
    }
    for (const n of WATCH) {
      const st = statuses[n];
      if (!st) continue;
      const prev = prevLevels.current[n];
      if (prev === undefined) { prevLevels.current[n] = st.level; continue; }
      if (n.startsWith('align/')) {
        if (prev !== st.level) {
          const [sev, text] = diagEvent(n, st.level);
          push(sev, n.split('/')[0], text);
        }
        prevLevels.current[n] = st.level;
        continue;
      }
      if (st.level === prev) delete pending.current[n];          // back before HOLD_MS: nothing
      else if (pending.current[n]?.level !== st.level) {
        pending.current[n] = { level: st.level, since: Date.now() };
      }
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

  // host /mission/events (JSON {kind, detail}); only entries newer than the last seen seq.
  const lastSeq = useRef(0);
  useEffect(() => {
    for (const e of missionEvents) {
      if (e.seq <= lastSeq.current) continue;
      lastSeq.current = e.seq;
      if (HIDDEN_KINDS.has(e.kind)) continue;
      const sev = CRIT_KINDS.has(e.kind) ? 'crit' : WARN_KINDS.has(e.kind) ? 'warn' : 'info';
      push(sev, 'host', hostEventText(e.kind, e.detail));
    }
  }, [missionEvents, push]);

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
        <div className="events-empty">No events</div>
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
