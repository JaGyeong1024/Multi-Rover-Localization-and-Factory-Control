# 관제 콘솔 — Multi-Rover Control Dashboard

A graduation-demo control console for the host PC of a multi-rover localization
system (leader + follower). Web frontend only: **Vite + React + roslib +
three.js**. It renders measured state — it is **not** a simulator.

> Design rationale and SSOT links live in `../CONTROL_DASHBOARD_PLAN.md`. This
> README only covers running the frontend.

---

## Prerequisites

### Node / npm

```bash
node -v   # 18+ recommended
npm install
```

Pinned deps: `react@18.3.1`, `react-dom@18.3.1`, `vite@5.4.10`, `roslib@1.4.1`,
`three@0.169.0`.

### ROS backend (must be running on the host)

The dashboard talks to the backend over two channels, **directly from the
browser** (no Vite proxy):

| Channel | URL | Provided by |
|---|---|---|
| ROS topics (pub/sub) | `ws://localhost:9090` | `rosbridge_server` |
| Camera MJPEG | `http://localhost:8080` | `web_video_server` |

```bash
# rosbridge (ROS 2 Humble, domain 30)
ros2 launch rosbridge_server rosbridge_websocket_launch.xml

# MJPEG video server
ros2 run web_video_server web_video_server
```

The dashboard auto-reconnects to rosbridge every 2s if the socket drops, so the
order you start things in does not matter.

---

## Run

```bash
npm install          # once (no network in CI sandbox — run on a machine with npm registry access)
npm run dev          # Vite dev server
# open http://localhost:5173
```

Build / preview a production bundle:

```bash
npm run build
npm run preview
```

---

## Expected backend topics

### Subscribed (read by the dashboard)

| Topic | Type | Used by |
|---|---|---|
| `/leader/global/position`, `/follower/global/position` | `geometry_msgs/msg/PoseStamped` | DigitalTwin, Header age banner |
| `/leader/battery_state`, `/follower/battery_state` | `sensor_msgs/msg/BatteryState` | RoverCards (handles 0..1 **or** 0..100 percentage) |
| `/localization/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | Header lights, RoverCards, SystemHealthPanel |

`DiagnosticArray.status[]` is rendered **generically** by grouping `status.name`
on `/`. Expected status names:

- `localization/leader`, `localization/follower` — per-robot trust rollup
  (`level` → signal light; `values`: `uwb_age_ms`, `fusion_age_ms`,
  `output_age_ms`, `output_hz`, `qf`).
- `pipeline/uwb` — `values`: `{leader,follower}_{hz,age_ms,qf}`.
- `pipeline/perception`, `pipeline/fusion`, `pipeline/output`, `pipeline/imu` —
  `values`: `{robot}_hz`, `{robot}_age_ms`.
- `pipeline/camera` — `values`: `hz`, `age_ms`.
- `sync/chrony` — `message` (e.g. `"offset 0.21ms"` / `"chrony unavailable"`);
  `values`: `offset_ms`, `source`.

Missing statuses degrade gracefully (gray tile + ⌧, or "수신 대기"). If no
`DiagnosticArray` arrives for >3s the panel shows
`diagnostics 수신 안됨 (localization_diagnostics_node 미실행?)`.

### Advertised (published by the dashboard)

| Topic | Type | Trigger |
|---|---|---|
| `/mission/dispatch` | `std_msgs/msg/String` | "+큐" button → `data = "fetch:<part>"` |
| `/mission/control` | `std_msgs/msg/String` | E-STOP → `data = "abort_all"` |

### Camera (MJPEG, **not** via rosbridge)

`<img src="http://localhost:8080/stream?topic=<topic>&type=mjpeg">`

- default: `/global_cam/processed/compressed` (top-down BEV)
- toggle: `/global_cam/image_raw/compressed` (raw wide)

---

## Notes & assumptions

- **Mission backend is mocked.** Dispatch publishes a real `/mission/dispatch`
  String stub and appends to a **local** queue (no `/mission/queue` /
  `/mission/state` subscription yet). Replace `src/ros/useMission.js` when the
  `host_mission` lane/Part/Action backend lands.
- **E-STOP is command-level only.** It publishes `/mission/control: abort_all`
  and raises a full-screen warning. `cmd_vel` is owned by the rovers — this is
  **not** a hardware kill. Final stop authority is the rover safety filter.
- **Coordinate mapping** (uwb_global → three.js) is isolated in
  `src/components/DigitalTwin.jsx` (`rosToScene` / `yawToSceneRot`): `x` along
  the 6.9m wall, `y` along the 6.2m wall, `mesh.position.set(x, 0, -y)`,
  `rotation.y = -yaw`. **Sign/axis must be field-verified against the ceiling
  camera** — flip the `y` and/or `yaw` sign there if a rover drives the wrong
  way.
- **Staleness guard**: the twin freezes + dims + flags `STALE` a rover whose
  last pose is older than 150ms (wall clock), so a dead feed cannot animate
  smoothly toward a stale pose and hide death.
- **Latency card is static** (~39ms host budget, target <100ms). Live
  end-to-end latency needs capture-timestamp propagation, which is not wired
  yet — no fake live gauge is shown.

---

## File map

```
dashboard/
├─ package.json · vite.config.js · index.html · .gitignore
├─ README.md
└─ src/
   ├─ main.jsx · App.jsx · styles.css · theme.js
   ├─ ros/
   │  ├─ useRos.js          rosbridge connection + auto-reconnect, sub/pub hooks
   │  ├─ useRobotState.js   pose (ref for twin) + battery state
   │  ├─ useDiagnostics.js  DiagnosticArray store (shared)
   │  └─ useMission.js      local mock mission queue (shared)
   └─ components/
      ├─ Header.jsx · DigitalTwin.jsx · CameraPanel.jsx · RoverCards.jsx
      ├─ SystemHealthPanel.jsx · MissionDispatch.jsx · MissionQueue.jsx
      └─ EmergencyStop.jsx
```
