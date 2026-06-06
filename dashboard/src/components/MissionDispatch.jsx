import React from 'react';
import { usePublisher } from '../ros/useRos';
import { useMission } from '../ros/useMission';

// Static part list (mock). When the host_mission lane/Part/Action backend is
// confirmed, replace with the real part registry + lane projection.
const PARTS = [
  { id: 'gear_01', label: 'Gear 01', zone: 'A' },
  { id: 'gear_02', label: 'Gear 02', zone: 'A' },
  { id: 'bolt_01', label: 'Bolt 01', zone: 'B' },
  { id: 'panel_01', label: 'Panel 01', zone: 'C' },
  { id: 'bracket_01', label: 'Bracket 01', zone: 'C' },
];

export default function MissionDispatch() {
  // Real stub publisher — /mission/dispatch String "fetch:<part>".
  const publishDispatch = usePublisher('/mission/dispatch', 'std_msgs/msg/String');
  const { enqueue } = useMission();

  const dispatch = (part) => {
    publishDispatch({ data: `fetch:${part.id}` });
    enqueue(part);
  };

  return (
    <div className="dispatch">
      <div className="panel-head">
        <span className="panel-title">Task</span>
      </div>
      <ul className="part-list">
        {PARTS.map((p) => (
          <li key={p.id} className="part-row">
            <span className="part-name">{p.label}</span>
            <span className="part-zone">zone {p.zone}</span>
            <button type="button" className="part-add" onClick={() => dispatch(p)}>
              + Queue
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
