import React from 'react';
import { useMission, useEstop } from '../ros/useMission';
import { useMissionMap, prettyId } from '../missionMap';

// Console task buttons — the SAME parts the DigitalTwin shelves expose, both from the
// generated mission map (scenario `parts`). Clicking publishes /task/add "fetch:<id>"
// (via useMission); the coordinator enqueues.
export default function MissionDispatch() {
  const { dispatch } = useMission();
  const { active: estopActive } = useEstop(); // the E-STOP overlay covers this panel
  const map = useMissionMap();
  const parts = (map && map.parts) || [];

  return (
    <div className="dispatch">
      <div className="panel-head">
        <span className="panel-title">Task</span>
      </div>
      <ul className="part-list">
        {parts.map((p) => (
          <li key={p.id} className="part-row">
            <span className="part-name">{prettyId(p.id)}</span>
            <span className="part-zone">{p.fixture || p.leader_goal || ''}</span>
            <button type="button" className="part-add" disabled={estopActive} onClick={() => dispatch(p.id)}>
              + Queue
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
