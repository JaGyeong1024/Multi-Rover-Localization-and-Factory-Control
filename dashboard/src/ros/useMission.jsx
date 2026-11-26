// Mission state mirrored from the host_mission coordinator (single source of truth):
// `/mission/queue` (latched) and `/mission/state` (5 Hz); `stateAge` (ms, Infinity = never)
// exposes a dead mission node. Outputs: `/task/add`, `/unload/done`, `/mission/control`
// (pause | resume | abandon | clear_queue | cancel:<id> | align), `/estop` (latched Bool).
// The host enforces the operator rules (a running trip completes; only waiting parts can be
// cancelled, only a stopped trip abandoned); hiding a button here is a courtesy.

import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { useSubscription, usePublisher } from './useRos';

const MissionContext = createContext({
  queue: [],
  activeId: '',
  systemState: 'IDLE',
  offered: [],
  stateAge: Infinity,
  estopSentAt: 0,
  dispatch: () => {},
  pause: () => {},
  resume: () => {},
  engageEstop: () => {},
  releaseEstop: () => {},
  clearQueue: () => {},
  cancelItem: () => {},
  abandon: () => {},
  clearAllAndReturn: () => {},
  align: () => {},
  unloadDone: () => {},
});

// "gear_01" -> "Gear 01"
function prettyLabel(id) {
  if (!id) return '';
  return id
    .split('_')
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

export function MissionProvider({ children }) {
  const [queue, setQueue] = useState([]);
  const [activeId, setActiveId] = useState('');
  const [systemState, setSystemState] = useState('IDLE');
  const [offered, setOffered] = useState([]); // /mission/offered
  const stateRx = useRef(0); // Date.now() of the last /mission/state (0 = never)
  const [stateAge, setStateAge] = useState(Infinity);
  const [estopSentAt, setEstopSentAt] = useState(0);
  useEffect(() => {
    const id = setInterval(
      () => setStateAge(stateRx.current ? Date.now() - stateRx.current : Infinity), 500);
    return () => clearInterval(id);
  }, []);

  // Inputs publish here; the coordinator owns the actual queue/state.
  const publishTask = usePublisher('/task/add', 'std_msgs/msg/String');
  const publishControl = usePublisher('/mission/control', 'std_msgs/msg/String');
  const publishUnloadDone = usePublisher('/unload/done', 'std_msgs/msg/Empty');
  const publishEstop = usePublisher('/estop', 'std_msgs/msg/Bool');

  // Mirror the coordinator's queue. Items: {id, part{id,coord,state}, lane_node, x,y, status}.
  useSubscription(
    '/mission/queue',
    'host_mission_interfaces/msg/MissionQueue',
    (msg) => {
      setActiveId(msg.active_id || '');
      setQueue(
        (msg.items || []).map((it) => {
          const partId = (it.part && it.part.id) || '';
          return {
            id: it.id,
            part: partId,
            label: prettyLabel(partId) || it.id,
            partState: (it.part && it.part.state) || '',
            laneNode: it.lane_node,
            x: it.x,
            y: it.y,
            readyIn: it.ready_in || 0, // s until this part's shelf is off cooldown
            status: it.status || 'pending', // pending | active
          };
        })
      );
    },
    []
  );

  // Operation state: IDLE | TO_PICKUP | LOADING | RETURNING | UNLOADING | ESTOP | PAUSED | FAULT.
  useSubscription(
    '/mission/state',
    'std_msgs/msg/String',
    (msg) => {
      stateRx.current = Date.now();
      setSystemState(msg.data || 'IDLE');
    },
    []
  );

  // Commands the host would accept now, for the controls whose refusal depends on what
  // only the host knows (abandon: not during a pick; align: idle + a heading not aligned).
  useSubscription(
    '/mission/offered',
    'std_msgs/msg/String',
    (msg) => {
      try {
        const list = JSON.parse(msg.data);
        setOffered(Array.isArray(list) ? list : []);
      } catch {
        setOffered([]);
      }
    },
    []
  );

  // Both inputs converge here. fire-and-forget; the queue updates via subscription.
  const dispatch = useCallback(
    (partId) => {
      if (!partId) return;
      publishTask({ data: `fetch:${partId}` });
    },
    [publishTask]
  );

  // clear_queue removes the waiting parts only; the running trip runs on.
  const clearQueue = useCallback(() => {
    publishControl({ data: 'clear_queue' });
  }, [publishControl]);

  // Cancel ONE waiting part by id (the host refuses a part of the running trip).
  const cancelItem = useCallback(
    (itemId) => {
      if (!itemId) return;
      publishControl({ data: `cancel:${itemId}` });
    },
    [publishControl]
  );

  // Operator 'unloading done' — host ignores it unless the follower is at the unloading zone.
  const unloadDone = useCallback(() => {
    publishUnloadDone({});
  }, [publishUnloadDone]);

  const pause = useCallback(() => publishControl({ data: 'pause' }), [publishControl]);
  const resume = useCallback(() => publishControl({ data: 'resume' }), [publishControl]);
  const abandon = useCallback(() => publishControl({ data: 'abandon' }), [publishControl]);
  const align = useCallback(() => publishControl({ data: 'align' }), [publishControl]);

  // One click: pause (if running), abandon the trip, clear the waiting parts; the host then
  // sends the rovers home. During a pick the host refuses the abandon and the trip stays paused.
  const clearAllAndReturn = useCallback(() => {
    if (hasRunningTrip(queue)) {
      if (!STOPPED_STATES.has(systemState)) publishControl({ data: 'pause' });
      publishControl({ data: 'abandon' });
    }
    publishControl({ data: 'clear_queue' });
  }, [publishControl, queue, systemState]);

  // E-STOP: /estop true pauses everything; release (/estop false) needs a deliberate resume.
  const engageEstop = useCallback(() => {
    publishEstop({ data: true });
    setEstopSentAt(Date.now());
  }, [publishEstop]);
  const releaseEstop = useCallback(() => {
    publishEstop({ data: false });
    publishControl({ data: 'resume' });
    setEstopSentAt(0);
  }, [publishEstop, publishControl]);

  return (
    <MissionContext.Provider
      value={{
        queue, activeId, systemState, offered, stateAge, estopSentAt, dispatch, clearQueue, cancelItem,
        abandon, clearAllAndReturn, align, unloadDone, pause, resume, engageEstop, releaseEstop,
      }}
    >
      {children}
    </MissionContext.Provider>
  );
}

export function useMission() {
  return useContext(MissionContext);
}

// Stopped mission states: the only ones in which the host accepts 'abandon' (it still
// refuses while a pick is in progress; its reason is then logged).
export const STOPPED_STATES = new Set(['PAUSED', 'FAULT', 'ESTOP']);


// True when the queue holds a part of the running trip (host status 'active').
export const hasRunningTrip = (queue) => queue.some((item) => item.status === 'active');

// /mission/state older than this = the mission node is not running (it publishes at 5 Hz).
export const STATE_STALE_MS = 2000;
// After pressing E-STOP the overlay shows at once; this long without the mission state
// confirming ESTOP, it goes away again (the stale-state warning then tells why).
const ESTOP_CONFIRM_MS = 3000;

// {active, stale, ageS}: E-STOP overlay / header state, driven by the mission state.
export function useEstop() {
  const { systemState, stateAge, estopSentAt } = useMission();
  const stale = !(stateAge <= STATE_STALE_MS);
  const pending = estopSentAt > 0 && Date.now() - estopSentAt < ESTOP_CONFIRM_MS;
  return {
    active: systemState === 'ESTOP' || pending,
    stale,
    ageS: isFinite(stateAge) ? Math.floor(stateAge / 1000) : null,
  };
}
