// Design tokens — control-room (NOC/SCADA) dark theme. Dense, serious, numerics
// in monospace. Color carries STATE only and is ALWAYS paired with a text label
// + icon/shape (accessibility — never color alone).

export const COLORS = {
  // Dark neutral base
  background: '#666c73', // 3D scene bg, soft gray
  surface: '#161a20', // panel surface (3D only; the UI uses CSS --bg)
  inset: '#1c222a', // tiles / insets
  floor: '#9ba1a9', // 3D floor — Gazebo gray
  grid1: '#7f858d',
  grid2: '#888e96',
  roverBody: '#454e5c', // slate, contrasts with the gray floor
  shelfBoard: '#7c828c',
  shelfLeg: '#5d636c',
  partMarker: '#4aa3e0',
  text: '#d6dbe2',
  textDim: '#8a93a0',

  // State accents (mission / mode) — muted, high-contrast on dark
  idle: '#5c6470',
  pickup: '#e0a93a', // PICKUP / work
  platooning: '#2ec27e', // PLATOONING / moving
  detaching: '#4aa3e0',

  // Health
  good: '#2ec27e',
  degraded: '#e0a93a',
  lost: '#e5484d',
  stale: '#5c6470',
};

// Health states — color + label + icon (dual-coded).
export const HEALTH = {
  GOOD: { color: COLORS.good, label: 'GOOD', icon: '●' },
  DEGRADED: { color: COLORS.degraded, label: 'WARN', icon: '▲' },
  LOST: { color: COLORS.lost, label: 'LOST', icon: '⚠' },
  STALE: { color: COLORS.stale, label: 'STALE', icon: '⌧' },
  WAITING: { color: COLORS.stale, label: 'WAIT', icon: '…' },
};

// Mission / mode states — color + label + icon (dual-coded).
export const MODE = {
  IDLE: { color: COLORS.idle, label: 'IDLE', icon: '○' },
  PICKUP: { color: COLORS.pickup, label: 'PICK', icon: '✦' },
  PLATOONING: { color: COLORS.platooning, label: 'MOVE', icon: '→' },
  DETACHING: { color: COLORS.detaching, label: 'DETACH', icon: '◇' },
};

// Per-role colours (rover paths and goal markers in the digital twin).
export const ZONE_COLORS = {
  leader_standby: '#4aa3e0',
  loading: '#e0a93a',
  follower: '#2ec27e',
  default: '#8a93a0',
};
// DiagnosticStatus.level → health key. 0=OK, 1=WARN, 2=ERROR.
export function levelToHealth(level) {
  if (level === 0) return 'GOOD';
  if (level === 1) return 'DEGRADED';
  if (level === 2) return 'LOST';
  return 'WAITING';
}

// Diagnostic level → simple traffic-light color (for generic dots).
export function levelToColor(level) {
  if (level === 0) return COLORS.good;
  if (level === 1) return COLORS.degraded;
  if (level === 2) return COLORS.lost;
  return COLORS.stale; // unknown / absent
}
