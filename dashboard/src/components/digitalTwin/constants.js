import { ZONE_COLORS } from '../../theme';

// Pose staleness: a twin that smoothly animates toward a stale pose hides death. Past
// STALE_MS the interpolation freezes and the rover is dimmed.
export const STALE_MS = 150;
// With no pose for longer than this the rover is hidden entirely (treated as absent). The
// backend stops publishing within 0.5 s of losing its source, so only a brief dim
// (STALE_MS..HIDE_MS) is shown first; short gaps do not flicker.
export const HIDE_MS = 1500;
export const POSE_SMOOTHING = 0.18; // per-frame lerp factor (~time-constant @60fps)

export const PATH_COLOR = { leader: ZONE_COLORS.leader_standby, follower: ZONE_COLORS.follower };
export const FLOOR_Y = 0.05; // overlays sit above the floor grid (0.004)
export const FLOOR_MARGIN = 0.4; // m of floor drawn around the driving bounds

// URDF package -> served path (dashboard/public/urdf/, fetched by fetch_urdf.sh).
export const URDF_PKGS = {
  turtlebot3_description: '/urdf/turtlebot3_description',
  turtlebot3_manipulation_description: '/urdf/turtlebot3_manipulation_description',
};
export const URDF_URLS = {
  leader: '/urdf/turtlebot3_manipulation_description/urdf/turtlebot3_manipulation.urdf',
  follower: '/urdf/turtlebot3_description/urdf/turtlebot3_waffle.urdf',
};
