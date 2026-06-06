import React, { useEffect, useState } from 'react';
import { useDiagnostics } from '../ros/useDiagnostics';
import { useRos } from '../ros/useRos';
import { ROBOTS } from '../ros/useRobotState';
import { COLORS, levelToColor } from '../theme';

function val(status, key) {
  if (!status || status.values == null) return null;
  const v = status.values[key];
  return v === undefined ? null : v;
}
function num(v, digits = 0) {
  if (v == null) return null;
  const n = Number(v);
  if (Number.isNaN(n)) return v;
  return n.toFixed(digits);
}

// Horizontal row: [dot] name ........ value(s).
function Row({ color, name, children, absent }) {
  return (
    <div className={`pipe-tile ${absent ? 'pipe-tile-absent' : ''}`}>
      <span className="dot" style={{ background: color }} aria-hidden="true" />
      <div className="pipe-tile-body">
        <div className="pipe-tile-name">{name}</div>
        <span className="tile-metrics">{children}</span>
      </div>
    </div>
  );
}

function PipelineRow({ name, label, single }) {
  const { getStatus } = useDiagnostics();
  const st = getStatus(name);
  if (!st) return <Row color={COLORS.stale} name={label} absent>⌧ no data</Row>;
  const color = levelToColor(st.level);
  if (single) {
    return (
      <Row color={color} name={label}>
        {num(val(st, 'hz')) ?? '—'}Hz · {num(val(st, 'age_ms')) ?? '—'}ms
      </Row>
    );
  }
  return (
    <Row color={color} name={label}>
      {ROBOTS.map((r) => (
        <span key={r} className="tile-metric-pair">
          <b>{r[0].toUpperCase()}</b> {num(val(st, `${r}_hz`)) ?? '—'}Hz/{num(val(st, `${r}_age_ms`)) ?? '—'}ms
        </span>
      ))}
    </Row>
  );
}

function UwbRow() {
  const { getStatus } = useDiagnostics();
  const st = getStatus('pipeline/uwb');
  if (!st) return <Row color={COLORS.stale} name="UWB" absent>⌧ no data</Row>;
  return (
    <Row color={levelToColor(st.level)} name="UWB">
      {ROBOTS.map((r) => (
        <span key={r} className="tile-metric-pair">
          <b>{r[0].toUpperCase()}</b> {num(val(st, `${r}_hz`)) ?? '—'}Hz·qf{num(val(st, `${r}_qf`)) ?? '—'}
        </span>
      ))}
    </Row>
  );
}

// End-to-end latency — null (—) until computable; auto-fills if the diagnostics
// aggregator publishes `latency_ms` (needs capture-stamp propagation).
function LatencyRow() {
  const { getStatus } = useDiagnostics();
  const st = getStatus('pipeline/output');
  const lat = st ? num(val(st, 'latency_ms')) : null;
  return (
    <Row color={lat != null ? COLORS.good : COLORS.stale} name="end-to-end" absent={lat == null}>
      {lat != null ? `${lat}ms` : '—'}
    </Row>
  );
}

export default function SystemHealthPanel() {
  const { lastRx } = useDiagnostics();
  const { connected } = useRos();
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const diagDead = !lastRx || now - lastRx > 3000;

  return (
    <div className="panel health-panel">
      <div className="panel-head">
        <span className="panel-title">System</span>
      </div>

      {diagDead && (
        <div className="health-dead">⚠ no diagnostics (is localization_diagnostics_node running?)</div>
      )}

      <div className="health-group">
        <div className="health-group-title">Connection</div>
        <div className="pipe-grid">
          <Row color={connected ? COLORS.good : COLORS.lost} name="rosbridge">
            {connected ? 'connected' : 'down'}
          </Row>
        </div>
      </div>

      <div className="health-group">
        <div className="health-group-title">Sensors / Pipeline</div>
        <div className="pipe-grid">
          <UwbRow />
          <PipelineRow name="pipeline/camera" label="GLOBAL CAM" single />
          <PipelineRow name="pipeline/imu" label="IMU" />
          <PipelineRow name="pipeline/output" label="FINAL OUTPUT" />
        </div>
      </div>

      <div className="health-group">
        <div className="health-group-title">Latency</div>
        <div className="pipe-grid">
          <LatencyRow />
        </div>
      </div>
    </div>
  );
}
