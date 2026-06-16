import React, { useEffect, useState } from 'react';
import { useDiagnostics } from '../ros/useDiagnostics';
import { useRobotState, ROBOTS } from '../ros/useRobotState';
import { HEALTH, COLORS, levelToHealth, levelToColor } from '../theme';

// Battery bar — never fake a full battery when there is no data ("—").
function BatteryBar({ pct, voltage }) {
  const hasData = typeof pct === 'number';
  const clamped = hasData ? Math.max(0, Math.min(100, pct)) : 0;
  let barColor = COLORS.good; // 50~100% 초록
  if (hasData) {
    if (clamped <= 20) barColor = COLORS.lost; // 0~20% 빨강(저전압 경고)
    else if (clamped < 50) barColor = COLORS.degraded; // 20~50% 노랑 (50%부터 초록)
  }
  return (
    <div className="battery">
      <div className="battery-track">
        <div
          className="battery-fill"
          style={{ width: hasData ? `${clamped}%` : '0%', background: hasData ? barColor : 'transparent' }}
        />
      </div>
      <div className="battery-readout">
        <span className="battery-pct">{hasData ? `${Math.round(clamped)}%` : '—'}</span>
        <span className="battery-volt">{typeof voltage === 'number' ? `${voltage.toFixed(1)}V` : '—'}</span>
      </div>
    </div>
  );
}

// Rover card: name + state-machine badge (one dot, colored by health/trust) + battery.
function RoverCard({ robot }) {
  const { getStatus } = useDiagnostics();
  const { battery, mission } = useRobotState();
  const st = getStatus(`localization/${robot}`);
  const key = st ? levelToHealth(st.level) : 'WAITING';
  const h = HEALTH[key];
  // Real per-robot mission state from /{robot}/status (host robot_state() projection);
  // shows '—' until a coordinator publishes. The dot stays health-colored (loc trust).
  const ms = (mission && mission[robot]) || {};
  const stateLabel = ms.state || '—';
  const bat = battery[robot] || { percentage: null, voltage: null };
  return (
    <div className="rover-card">
      <div className="rover-card-head">
        <span className="rover-card-name">{robot}</span>
        <span className="state-badge" style={{ borderColor: h.color, color: h.color }}>
          <span className="dot" style={{ background: h.color }} aria-hidden="true" />
          {stateLabel}
        </span>
      </div>
      <div className="rover-card-msg">
        {ms.task ? `task: ${ms.task}` : st ? st.message : 'no diagnostics'}
      </div>
      <BatteryBar pct={bat.percentage} voltage={bat.voltage} />
    </div>
  );
}

// Horizontal row ([dot] name ........ value) — reuses pipe-tile styling.
function StatusRow({ color, name, value }) {
  return (
    <div className="pipe-tile">
      <span className="dot" style={{ background: color }} aria-hidden="true" />
      <div className="pipe-tile-body">
        <div className="pipe-tile-name">{name}</div>
        <span className="tile-metrics">{value}</span>
      </div>
    </div>
  );
}

export default function RoverCards() {
  const { getStatus } = useDiagnostics();
  const { poseAge } = useRobotState();
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);

  // freshest pose age (wall-clock receipt based)
  let worst = Infinity;
  for (const r of ROBOTS) {
    const rx = poseAge[r];
    const a = rx ? Math.max(0, now - rx) : Infinity; // guard: now may lag rxWall slightly
    if (a < worst) worst = a;
  }
  let ageText;
  let ageColor;
  if (!isFinite(worst)) { ageText = 'no data'; ageColor = COLORS.lost; }
  else if (worst > 1000) { ageText = `${Math.round(worst)}ms`; ageColor = COLORS.lost; }
  else if (worst > 150) { ageText = `${Math.round(worst)}ms`; ageColor = COLORS.degraded; }
  else { ageText = `${Math.round(worst)}ms`; ageColor = COLORS.good; }

  const sync = getStatus('sync/chrony');
  const syncColor = sync ? levelToColor(sync.level) : COLORS.stale;
  const syncText = sync ? sync.message || '—' : '⌧ no data';

  return (
    <div className="panel rover-cards">
      <div className="panel-head">
        <span className="panel-title">Rover Status</span>
      </div>
      <div className="rover-cards-list">
        {ROBOTS.map((r) => (
          <RoverCard key={r} robot={r} />
        ))}
      </div>
      <div className="rover-extra pipe-grid">
        <StatusRow color={ageColor} name="POSE AGE" value={ageText} />
        <StatusRow color={syncColor} name="SYNC" value={syncText} />
      </div>
    </div>
  );
}
