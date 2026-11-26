import React from 'react';
import { useMission, useEstop } from '../ros/useMission';

// E-STOP = dedicated latched safety line (`/estop` Bool), independent of the mission
// FSM. Engaging (header button) publishes `/estop true` → the host forces every rover to
// STOP and its state becomes ESTOP. The overlay follows that state (not browser state), so
// a reload while stopped still offers the release. Releasing publishes `/estop false`, then
// a deliberate `/mission/control resume` (the host does not auto-resume). cmd_vel stays
// owned by the rovers — final stop authority is the rover safety filter.
// 'Abandon' drops the stopped trip; under the e-stop nothing moves until the
// operator still presses Resume (release + resume), then the rovers drive home. A refused command
// is only logged (Events), not shown here.
export default function EmergencyStop() {
  const { releaseEstop, abandon, offered } = useMission();
  const { active, stale, ageS } = useEstop();
  const trip = offered.includes('abandon');   // the host would accept it now

  if (!active) return null;

  return (
    <div className="estop-overlay" role="alertdialog" aria-modal="true">
      <div className="estop-overlay-card">
        <div className="estop-overlay-icon" aria-hidden="true">⚠</div>
        <div className="estop-overlay-title">EMERGENCY STOP</div>
        {stale && (
          <div className="estop-overlay-warn">
            {ageS == null ? 'NO DATA' : `NO DATA ${ageS} s`}
          </div>
        )}
        <div className="estop-overlay-actions">
          {trip && (
            <button type="button" className="estop-clear estop-abandon" onClick={abandon}>
              Abandon
            </button>
          )}
          <button type="button" className="estop-clear" onClick={releaseEstop}>
            Resume
          </button>
        </div>
      </div>
    </div>
  );
}
