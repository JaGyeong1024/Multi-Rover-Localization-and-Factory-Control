import React from 'react';
import { useMission } from '../ros/useMission';

const CHIP = {
  active: { label: 'ACTIVE', cls: 'chip-active' },
  pending: { label: 'PENDING', cls: 'chip-pending' },
  done: { label: 'DONE', cls: 'chip-done' },
  failed: { label: 'FAILED', cls: 'chip-failed' },
};

export default function MissionQueue() {
  const { queue, systemState, clearQueue, cancelItem, unloadDone } = useMission();

  return (
    <div className="queue">
      <div className="panel-head">
        <span className="panel-title">Task Queue</span>
        <span className={`mission-state state-${(systemState || 'idle').toLowerCase()}`}>
          {systemState}
        </span>
        {queue.length > 0 && (
          <button type="button" className="queue-clear" onClick={clearQueue}>
            Clear
          </button>
        )}
      </div>

      <ol className="queue-list">
        {queue.map((item, i) => {
          // pending but its shelf is still on cooldown → "WAIT Ns" (so it reads as waiting, not stuck)
          const waiting = item.status === 'pending' && item.readyIn > 0.5;
          const chip = waiting
            ? { label: 'WAIT', cls: 'chip-wait' }
            : CHIP[item.status] || CHIP.pending;
          // ✕ cancellable: queued (pending), OR still en-route to pickup (active, not yet
          // LOADED, while TO_PICKUP). A part being picked (LOADING) or returning is protected.
          const cancellable =
            item.status === 'pending' ||
            (item.status === 'active' && item.partState !== 'LOADED' && systemState === 'TO_PICKUP');
          return (
            <li
              key={item.id}
              className={`queue-item ${item.status === 'active' ? 'queue-item-active' : ''}`}
            >
              <span className="queue-idx">{i + 1}</span>
              <div className="queue-main">
                <span className="queue-label">{item.label}</span>
                <span className="queue-zone">
                  {waiting ? `restock ${Math.ceil(item.readyIn)}s` : item.partState || `node ${item.laneNode}`}
                </span>
              </div>
              <div className="queue-right">
                <span className={`chip ${chip.cls}`}>{chip.label}</span>
                {cancellable && (
                  <button
                    type="button"
                    className="queue-cancel"
                    title="이 부품 취소"
                    aria-label={`cancel ${item.label}`}
                    onClick={() => cancelItem(item.id)}
                  >
                    ✕
                  </button>
                )}
              </div>
            </li>
          );
        })}
      </ol>

      {/* persistent at the bottom — disabled (grey) until the follower reaches UNLOADING,
          then enabled (green, pulse). No layout shift / button movement. */}
      <button
        type="button"
        className="unload-done"
        onClick={unloadDone}
        disabled={systemState !== 'UNLOADING'}
      >
        UNLOAD DONE
      </button>
    </div>
  );
}
