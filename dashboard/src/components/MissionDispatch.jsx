import React from 'react';
import { useMission } from '../ros/useMission';

// Console task buttons — the SAME parts the DigitalTwin shelves expose. ids must
// match map.json shelves[].part and host_mission mission_scenario_map.yaml.
// Clicking publishes /task/add "fetch:<id>" (via useMission); the coordinator enqueues.
const PARTS = [
  { id: 'gear_01', label: 'Gear 01', shelf: 0 },
  { id: 'bolt_01', label: 'Bolt 01', shelf: 1 },
];

export default function MissionDispatch() {
  const { dispatch } = useMission();

  return (
    <div className="dispatch">
      <div className="panel-head">
        <span className="panel-title">Task</span>
      </div>
      <ul className="part-list">
        {PARTS.map((p) => (
          <li key={p.id} className="part-row">
            <span className="part-name">{p.label}</span>
            <span className="part-zone">shelf {p.shelf}</span>
            <button type="button" className="part-add" onClick={() => dispatch(p.id)}>
              + Queue
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
