// Mission state — backed by the real host_mission coordinator (domain 30).
//
// SINGLE SOURCE OF TRUTH = the coordinator. The queue is broadcast on
// `/mission/queue` (latched) and the operation state on `/mission/state`. Inputs
// (DigitalTwin part-click, MissionDispatch buttons) only PUBLISH `/task/add`
// "fetch:<part_id>"; the coordinator enqueues and re-publishes, which this
// provider mirrors. So the dashboard reflects real rover progress and survives
// reconnects (transient_local). See host_mission_design §7.
//
// Operator outputs: `/task/add` (queue a pickup), `/unload/done` (하역완료 — only
// honoured by the host while UNLOADING), `/mission/control` (clear_queue/resume).

import React, { createContext, useContext, useState, useCallback } from 'react';
import { useSubscription, usePublisher } from './useRos';

const MissionContext = createContext({
  queue: [],
  activeId: '',
  systemState: 'IDLE',
  dispatch: () => {},
  clearQueue: () => {},
  cancelItem: () => {},
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

  // Inputs publish here; the coordinator owns the actual queue/state.
  const publishTask = usePublisher('/task/add', 'std_msgs/msg/String');
  const publishControl = usePublisher('/mission/control', 'std_msgs/msg/String');
  const publishUnloadDone = usePublisher('/unload/done', 'std_msgs/msg/Empty');

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
            readyIn: it.ready_in || 0, // s until this part's shelf is off cooldown
            status: it.status || 'pending', // pending | active | done | failed
          };
        })
      );
    },
    []
  );

  // Operation state: IDLE | TO_PICKUP | LOADING | RETURNING | UNLOADING | ESTOP | PAUSED | FAULT | ABORTED.
  useSubscription(
    '/mission/state',
    'std_msgs/msg/String',
    (msg) => setSystemState(msg.data || 'IDLE'),
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

  // clear_queue aborts + empties the WHOLE queue.
  const clearQueue = useCallback(() => {
    publishControl({ data: 'clear_queue' });
  }, [publishControl]);

  // Cancel ONE part by id (host removes it unless it is the active/in-flight item).
  const cancelItem = useCallback(
    (itemId) => {
      if (!itemId) return;
      publishControl({ data: `cancel:${itemId}` });
    },
    [publishControl]
  );

  // Operator '하역완료' — host ignores it unless the follower is at the unloading zone.
  const unloadDone = useCallback(() => {
    publishUnloadDone({});
  }, [publishUnloadDone]);

  return (
    <MissionContext.Provider
      value={{ queue, activeId, systemState, dispatch, clearQueue, cancelItem, unloadDone }}
    >
      {children}
    </MissionContext.Provider>
  );
}

export function useMission() {
  return useContext(MissionContext);
}
