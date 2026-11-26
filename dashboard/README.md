# Control Dashboard

Web console for the host PC: a three.js Digital Twin of both rovers, the ceiling camera view,
the task queue, rover and sensor health, an event timeline and an E-STOP.
Vite + React + roslib + three.js (urdf-loader). It only renders measured state and sends
operator commands; all logic runs in the ROS nodes.

## Run

`bash host_ws/run_host.sh` starts everything, including this UI. To run the front end alone:

```bash
npm install
npm run dev        # http://localhost:5173
```

The browser connects directly to two backend services on the host:

| Channel | URL | Provided by |
|---|---|---|
| ROS topics | `ws://localhost:9090` | `rosbridge_server` |
| Camera (MJPEG) | `http://localhost:8080` | `web_video_server` |

The rosbridge connection reconnects automatically, so start order does not matter.

## Topics

| Direction | Topics |
|---|---|
| Subscribed | `/{robot}/global/position`, `/{robot}/localization_quality`, `/{robot}/battery_state`, `/{robot}/path_cmd`, `/{robot}/nav_feedback`, `/{robot}/safety_state`, `/{robot}/status`, `/localization/diagnostics`, `/mission/{state,queue,events,goals,offered,obstacles,mapped_obstacles}` |
| Published | `/task/add`, `/mission/control`, `/unload/done`, `/estop`, `/dashboard/reset` |
| Camera | `/global_cam/processed` (top-down BEV, via web_video_server) |

E-STOP is a command-level stop (`/estop`); the final stop authority is the rover safety gate.

## Generated assets

`host_ws/run_dashboard.sh` prepares two untracked files before starting the UI:
`public/mission_map.json` (goal poses, fixtures and bounds, from `gen_mission_map.py` and the
mission config) and the rover URDF meshes in `public/urdf/` (`fetch_urdf.sh`, first run only).
