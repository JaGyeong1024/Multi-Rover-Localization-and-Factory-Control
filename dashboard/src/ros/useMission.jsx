// Local mock mission queue, shared between MissionDispatch and MissionQueue.
//
// MOCK ONLY — there is no real host_mission backend wired yet. Clicking a part
// publishes `/mission/dispatch` (a real String stub) AND appends to this local
// queue. When the lane/Part/Action backend lands, replace this store with the
// real `/mission/queue` + `/mission/state` subscriptions (CONTROL_DASHBOARD_PLAN §5.4).

import React, { createContext, useContext, useState, useCallback } from 'react';

const MissionContext = createContext({
  queue: [],
  enqueue: () => {},
  remove: () => {},
  clear: () => {},
});

let _seq = 0;

export function MissionProvider({ children }) {
  const [queue, setQueue] = useState([]);

  const enqueue = useCallback((part) => {
    setQueue((prev) => {
      _seq += 1;
      const item = {
        id: `${Date.now()}-${_seq}`,
        part: part.id,
        label: part.label,
        zone: part.zone,
        // First item is "active", the rest "pending".
        status: prev.length === 0 ? 'active' : 'pending',
      };
      return [...prev, item];
    });
  }, []);

  // 개별 삭제 — 제거 후 active가 없으면 다음 항목을 active로 승격.
  const remove = useCallback((id) => {
    setQueue((prev) => {
      const next = prev.filter((it) => it.id !== id);
      if (next.length && !next.some((it) => it.status === 'active')) {
        next[0] = { ...next[0], status: 'active' };
      }
      return next;
    });
  }, []);

  const clear = useCallback(() => setQueue([]), []);

  return (
    <MissionContext.Provider value={{ queue, enqueue, remove, clear }}>
      {children}
    </MissionContext.Provider>
  );
}

export function useMission() {
  return useContext(MissionContext);
}
