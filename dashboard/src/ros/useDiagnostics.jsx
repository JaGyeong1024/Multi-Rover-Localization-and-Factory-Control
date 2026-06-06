// Shared diagnostics store.
//
// `/localization/diagnostics` (diagnostic_msgs/msg/DiagnosticArray) is consumed
// by BOTH the Header (per-robot signal lights) and the SystemHealthPanel. We
// subscribe ONCE in a provider and expose:
//   - statuses : { [name]: {name, level, message, values:{key:value}} }
//   - lastRx   : wall-clock ms of last DiagnosticArray (0 if never)
//   - getStatus(name)
//
// `values` is flattened from [{key,value}] into a plain object for easy access.

import React, { createContext, useContext, useRef, useState, useCallback } from 'react';
import { useSubscription } from './useRos';

const DiagContext = createContext({
  statuses: {},
  lastRx: 0,
  getStatus: () => null,
});

function flattenValues(values) {
  const out = {};
  if (Array.isArray(values)) {
    for (const kv of values) {
      if (kv && typeof kv.key === 'string') out[kv.key] = kv.value;
    }
  }
  return out;
}

export function DiagnosticsProvider({ children }) {
  const [statuses, setStatuses] = useState({});
  const [lastRx, setLastRx] = useState(0);
  const statusesRef = useRef(statuses);
  statusesRef.current = statuses;

  useSubscription(
    '/localization/diagnostics',
    'diagnostic_msgs/msg/DiagnosticArray',
    (msg) => {
      const next = {};
      const arr = (msg && msg.status) || [];
      for (const s of arr) {
        next[s.name] = {
          name: s.name,
          level: typeof s.level === 'number' ? s.level : Number(s.level) || 0,
          message: s.message || '',
          values: flattenValues(s.values),
        };
      }
      setStatuses(next);
      setLastRx(Date.now());
    },
    []
  );

  const getStatus = useCallback((name) => statusesRef.current[name] || null, []);

  return (
    <DiagContext.Provider value={{ statuses, lastRx, getStatus }}>
      {children}
    </DiagContext.Provider>
  );
}

export function useDiagnostics() {
  return useContext(DiagContext);
}
