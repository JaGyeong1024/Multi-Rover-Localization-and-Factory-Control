import React, { useEffect } from 'react';
import { usePublisher } from '../ros/useRos';

// Command-level abort ONLY. cmd_vel is permanently owned by the rovers
// (CLAUDE.md K-1), so this is NOT a hardware kill — it publishes
// /mission/control "abort_all" and raises a full-screen warning. Final stop
// authority is the rover safety filter.
export default function EmergencyStop({ active, onClear }) {
  const publishControl = usePublisher('/mission/control', 'std_msgs/msg/String');

  // When the overlay becomes active (triggered from the header button), publish
  // the abort once on the rising edge of `active`.
  useEffect(() => {
    if (active) {
      publishControl({ data: 'abort_all' });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  if (!active) return null;

  return (
    <div className="estop-overlay" role="alertdialog" aria-modal="true">
      <div className="estop-overlay-card">
        <div className="estop-overlay-icon" aria-hidden="true">⚠</div>
        <div className="estop-overlay-title">EMERGENCY STOP — halted</div>
        <div className="estop-overlay-actions">
          <button type="button" className="estop-clear" onClick={onClear}>
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
