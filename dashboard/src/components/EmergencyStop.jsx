import React, { useEffect } from 'react';
import { usePublisher } from '../ros/useRos';

// E-STOP = dedicated latched safety line (`/estop` Bool), independent of the mission
// FSM. Engaging publishes `/estop true` → the host forces every rover to STOP and
// pauses the mission. Releasing publishes `/estop false`; the host does NOT auto-resume,
// so we also send a deliberate `/mission/control resume`. cmd_vel stays owned by the
// rovers (CLAUDE.md K-1) — final stop authority is the rover safety filter.
export default function EmergencyStop({ active, onClear }) {
  const publishEstop = usePublisher('/estop', 'std_msgs/msg/Bool');
  const publishControl = usePublisher('/mission/control', 'std_msgs/msg/String');

  // Engage on the rising edge of `active`.
  useEffect(() => {
    if (active) publishEstop({ data: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  if (!active) return null;

  const releaseAndResume = () => {
    publishEstop({ data: false }); // release the latched E-stop
    publishControl({ data: 'resume' }); // deliberate resume (host won't auto-resume)
    onClear();
  };

  return (
    <div className="estop-overlay" role="alertdialog" aria-modal="true">
      <div className="estop-overlay-card">
        <div className="estop-overlay-icon" aria-hidden="true">⚠</div>
        <div className="estop-overlay-title">EMERGENCY STOP — all rovers halted</div>
        <div className="estop-overlay-actions">
          <button type="button" className="estop-clear" onClick={releaseAndResume}>
            Release &amp; Resume
          </button>
        </div>
      </div>
    </div>
  );
}
