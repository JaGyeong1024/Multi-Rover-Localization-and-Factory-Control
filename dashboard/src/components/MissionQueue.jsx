import React from 'react';
import { useMission } from '../ros/useMission';

const CHIP = {
  active: { label: 'ACTIVE', cls: 'chip-active' },
  pending: { label: 'PENDING', cls: 'chip-pending' },
  done: { label: 'DONE', cls: 'chip-done' },
};

export default function MissionQueue() {
  const { queue, remove } = useMission();

  return (
    <div className="queue">
      <div className="panel-head">
        <span className="panel-title">Task Queue</span>
      </div>

      <ol className="queue-list">
        {queue.map((item, i) => {
          const chip = CHIP[item.status] || CHIP.pending;
          return (
            <li key={item.id} className={`queue-item ${i === 0 ? 'queue-item-active' : ''}`}>
              <span className="queue-idx">{i + 1}</span>
              <div className="queue-main">
                <span className="queue-label">{item.label}</span>
                <span className="queue-zone">zone {item.zone}</span>
              </div>
              <div className="queue-right">
                <span className={`chip ${chip.cls}`}>{chip.label}</span>
                <button type="button" className="queue-cancel" onClick={() => remove(item.id)}>
                  cancel
                </button>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
