import React from 'react';
import { RosProvider } from './ros/useRos';
import { DiagnosticsProvider } from './ros/useDiagnostics';
import { MissionProvider } from './ros/useMission';
import { RobotStateProvider } from './ros/useRobotState';
import { MissionStateProvider } from './ros/useMissionState';

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
//   MissionProvider (mirrors the coordinator's queue/state)
function Providers({ children }) {
  return (
    <RosProvider>
      <RobotStateProvider>
        <DiagnosticsProvider>
          <MissionProvider>
            <MissionStateProvider>{children}</MissionStateProvider>
          </MissionProvider>
        </DiagnosticsProvider>
      </RobotStateProvider>
    </RosProvider>
  );
}

export default function App() {
  // E-STOP: the header button publishes /estop; the full-screen overlay follows the
  // mission state (/mission/state == ESTOP), see ros/useMission.jsx useEstop().
  return (
    <Providers>
      <div className="app-grid">
        <Header />

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
            <EventTimeline />
            <RoverCards />
            <SystemHealthPanel />
          </div>
        </main>
      </div>

      <EmergencyStop />
    </Providers>
  );
}
