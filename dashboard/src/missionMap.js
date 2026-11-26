// Mission map = /mission_map.json, generated at dashboard start from the mission
// configuration by dashboard/gen_mission_map.py (run_dashboard.sh, tools/container/dash_inner.sh).
// Nothing in the dashboard keeps its own copy of coordinates: bounds, fixtures (shelves),
// parts and goal poses all come from here; the live goal table (/mission/goals) replaces
// `goals` once the coordinator's latched message arrives (DigitalTwin).
//
// The file may appear a few seconds after the page loads (the simulator writes its configs
// first), so the fetch retries until it gets a valid file.
import { useEffect, useState } from 'react';

const URL = '/mission_map.json';
const RETRY_MS = 2000;

let cached = null;
let pending = null;

function load() {
  if (cached) return Promise.resolve(cached);
  if (!pending) {
    pending = new Promise((resolve) => {
      const attempt = () =>
        fetch(URL, { cache: 'no-store' })
          .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
          .then((m) => {
            if (!m || !m.bounds) throw new Error('no bounds');
            cached = m;
            resolve(m);
          })
          .catch((e) => {
            console.warn(`[map] ${URL} not ready (${e.message}); retrying`);
            setTimeout(attempt, RETRY_MS);
          });
      attempt();
    });
  }
  return pending;
}

export function useMissionMap() {
  const [map, setMap] = useState(cached);
  useEffect(() => {
    let live = true;
    load().then((m) => live && setMap(m));
    return () => {
      live = false;
    };
  }, []);
  return map;
}

export function loadMissionMap() {
  return load();
}

// "gear_01" -> "Gear 01"
export function prettyId(id) {
  return String(id || '')
    .split('_')
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}
