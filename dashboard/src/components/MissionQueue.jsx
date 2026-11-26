import React from 'react';
import { useMission, STATE_STALE_MS, STOPPED_STATES, hasRunningTrip } from '../ros/useMission';

const CHIP = {
  active: { label: 'ACTIVE', cls: 'chip-active' },
  pending: { label: 'PENDING', cls: 'chip-pending' },
  done: { label: 'DONE', cls: 'chip-done' },
};

// Path mode has no lane node (absent / negative): show the part id and shelf coordinates instead.
function locationText(item) {
  if (item.laneNode >= 0) return `node ${item.laneNode}`;
  const xy = isFinite(item.x) && isFinite(item.y) ? ` (${item.x.toFixed(2)}, ${item.y.toFixed(2)})` : '';
  return `${item.part || item.id}${xy}`;
}

// Mission states in which the operator may pause, and those a resume continues from.
const PAUSABLE = new Set(['TO_PICKUP', 'LOADING', 'RETURNING', 'UNLOADING']);
const RESUMABLE = new Set(['PAUSED', 'FAULT']); // ESTOP: release on the E-STOP overlay

export default function MissionQueue() {
  const { queue, systemState, offered, stateAge, clearQueue, cancelItem, abandon, clearAllAndReturn,
    unloadDone, pause, resume, align } = useMission();
  const stale = !(stateAge <= STATE_STALE_MS);
  const waitingCount = queue.filter((item) => item.status !== 'active').length;
  const stoppedTrip = STOPPED_STATES.has(systemState) && hasRunningTrip(queue);
  // The host says which of these it would accept now (/mission/offered): abandon only for a
  // stopped trip that is not in a pick, align only while idle with a heading not aligned.
  // ESTOP: the abandon button also sits on the E-STOP overlay (which covers this panel).
  const canAbandon = !stale && offered.includes('abandon');
  const canAlign = !stale && offered.includes('align');
  const canPause = PAUSABLE.has(systemState);
  const canResume = RESUMABLE.has(systemState);
  // One click drops the running trip and the waiting parts; never during a pick (the arm
  // cannot be stopped from here), running (LOADING) or stopped (abandon not offered).
  const picking = systemState === 'LOADING' || (stoppedTrip && !canAbandon);

  return (
    <div className="queue">
      <div className="panel-head">
        <span className="panel-title">Queue</span>
        {stale ? (
          <span className="mission-state state-nodata">
            {isFinite(stateAge) ? `no data ${Math.floor(stateAge / 1000)}s` : 'no data'}
          </span>
        ) : (
          <span className={`mission-state state-${(systemState || 'idle').toLowerCase()}`}>
            {systemState}
          </span>
        )}
        {/* Every control keeps its place; one that cannot be used now is only greyed out. */}
        <button
          type="button"
          className="queue-clear"
          disabled={stale || !(canPause || canResume)}
          onClick={canResume ? resume : pause}
        >
          {canResume ? 'Resume' : 'Pause'}
        </button>
        <button type="button" className="queue-clear" disabled={!canAlign} onClick={align}>
          Align
        </button>
      </div>
      <div className="queue-controls">
        <button
          type="button"
          className="queue-clear queue-abandon"
          disabled={!canAbandon}
          onClick={abandon}
        >
          Abandon
        </button>
        <button
          type="button"
          className="queue-clear"
          disabled={stale || waitingCount === 0}
          onClick={clearQueue}
        >
          Clear
        </button>
        <button
          type="button"
          className="queue-clear queue-abandon"
          disabled={stale || picking || queue.length === 0}
          onClick={clearAllAndReturn}
        >
          Clear all
        </button>
      </div>

      <ol className="queue-list">
        {queue.map((item, i) => {
          // pending but its shelf is still on cooldown → "WAIT Ns" (so it reads as waiting, not stuck)
          const waiting = item.status === 'pending' && item.readyIn > 0.5;
          const chip = waiting
            ? { label: 'WAIT', cls: 'chip-wait' }
            : CHIP[item.status] || CHIP.pending;
          // ✕ only on waiting parts; a part of the running trip (host status 'active') is
          // locked until the trip ends (to drop it: stop, then abandon the trip).
          const cancellable = item.status === 'pending';
          const locked = item.status === 'active';
          return (
            <li
              key={item.id}
              className={`queue-item ${item.status === 'active' ? 'queue-item-active' : ''}`}
            >
              <span className="queue-idx">{i + 1}</span>
              <div className="queue-main">
                <span className="queue-label">{item.label}</span>
                <span className="queue-zone">
                  {waiting ? `restock ${Math.ceil(item.readyIn)}s` : item.partState || locationText(item)}
                </span>
              </div>
              <div className="queue-right">
                <span className={`chip ${chip.cls}`}>{chip.label}</span>
                {cancellable && (
                  <button
                    type="button"
                    className="queue-cancel"
                    title="Cancel"
                    aria-label={`cancel ${item.label}`}
                    onClick={() => cancelItem(item.id)}
                  >
                    ✕
                  </button>
                )}
                {locked && (
                  <span
                    className="queue-lock"
                    aria-label={`${item.label} locked: running trip`}
                  >
                    🔒
                  </span>
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
        disabled={stale || systemState !== 'UNLOADING'}
      >
        UNLOAD DONE
      </button>
    </div>
  );
}
