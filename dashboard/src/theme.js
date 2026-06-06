// Design tokens — control-room (NOC/SCADA) dark theme. Dense, serious, numerics
// in monospace. Color carries STATE only and is ALWAYS paired with a text label
// + icon/shape (accessibility — never color alone). Mirrors CONTROL_DASHBOARD_PLAN.md §6.

export const COLORS = {
  // Dark neutral base
  background: '#666c73', // 3D scene bg — Gazebo風 soft gray (검정 대신 회색)
  surface: '#161a20', // panel surface (UI는 CSS --bg 사용; 이 값은 3D만)
  inset: '#1c222a', // tiles / insets
  floor: '#9ba1a9', // 3D floor — Gazebo gray
  grid1: '#7f858d',
  grid2: '#888e96',
  roverBody: '#454e5c', // slate — gray 바닥에서 대비
  shelfBoard: '#7c828c',
  shelfLeg: '#5d636c',
  partMarker: '#4aa3e0',
  text: '#d6dbe2',
  textDim: '#8a93a0',

  // State accents (mission / mode) — muted, high-contrast on dark
  idle: '#5c6470',
  pickup: '#e0a93a', // PICKUP / 작업
  platooning: '#2ec27e', // PLATOONING / 이동
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

// map.json zone type → 색 + 라벨 (디지털 트윈 zone 오버레이). 경로(graph)는 미표시.
export const ZONE_COLORS = {
  leader_standby: '#4aa3e0',
  loading: '#e0a93a',
  follower: '#2ec27e',
  default: '#8a93a0',
};
export const ZONE_LABEL = {
  leader_standby: 'Leader Station',
  loading: 'Discharging',
  follower: 'Follower Station',
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

// Room geometry (uwb_global frame, meters). Origin = wall corner.
export const ROOM = {
  width: 6.9, // x along this wall
  depth: 6.2, // y along this wall
};

// Active zone (the rovers' real operating area, map.json active_zone). The twin
// camera frames THIS so rovers are large/centered instead of lost in a big floor.
export const ACTIVE_ZONE = {
  x_min: 1.6295,
  x_max: 5.7295,
  y_min: 2.0995,
  y_max: 4.6395,
};
