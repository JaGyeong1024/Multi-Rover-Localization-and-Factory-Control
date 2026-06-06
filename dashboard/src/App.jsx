import React, { useState } from 'react';
import { RosProvider } from './ros/useRos';
import { DiagnosticsProvider } from './ros/useDiagnostics';
import { MissionProvider } from './ros/useMission';
import { RobotStateProvider } from './ros/useRobotState';

import Header from './components/Header.jsx';
import DigitalTwin from './components/DigitalTwin.jsx';
import CameraPanel from './components/CameraPanel.jsx';
import RoverCards from './components/RoverCards.jsx';
import SystemHealthPanel from './components/SystemHealthPanel.jsx';
import MissionDispatch from './components/MissionDispatch.jsx';
import MissionQueue from './components/MissionQueue.jsx';
import EventTimeline from './components/EventTimeline.jsx';
import EmergencyStop from './components/EmergencyStop.jsx';

// Provider tree (outermost = connection):
//   RosProvider → RobotStateProvider + DiagnosticsProvider (both need ros)
//   MissionProvider (local mock state, ros only used by dispatch publisher)
function Providers({ children }) {
  return (
    <RosProvider>
      <RobotStateProvider>
        <DiagnosticsProvider>
          <MissionProvider>{children}</MissionProvider>
        </DiagnosticsProvider>
      </RobotStateProvider>
    </RosProvider>
  );
}

export default function App() {
  // Full-screen E-STOP overlay state lives at the app root so it can tint
  // everything. The button publishes; the overlay is purely visual.
  const [estopActive, setEstopActive] = useState(false);

  return (
    <Providers>
      <div className="app-grid">
        <Header onEstop={() => setEstopActive(true)} estopActive={estopActive} />

        <main className="main-grid">
          <div className="top-row">
            <DigitalTwin />
            <CameraPanel />
          </div>

          <div className="bottom-row">
            <section className="panel task-planner">
              <div className="panel-head">
                <span className="panel-title">Task Planner</span>
              </div>
              <div className="task-planner-body">
                <div className="subpanel">
                  <MissionDispatch />
                </div>
                <div className="subpanel">
                  <MissionQueue />
                </div>
              </div>
            </section>
            <EventTimeline estopActive={estopActive} />
            <RoverCards />
            <SystemHealthPanel />
          </div>
        </main>
      </div>

      <EmergencyStop active={estopActive} onClear={() => setEstopActive(false)} />
    </Providers>
  );
}
