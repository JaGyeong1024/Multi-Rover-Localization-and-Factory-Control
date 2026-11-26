import * as THREE from 'three';

// COORDINATE MAPPING (uwb_global -> three.js, y-up)
//   ROS uwb_global: x along the 6.9m wall, y along the 6.2m wall, z up.
//   three.js is y-up, so the floor lives in the X/Z plane.
//     mesh.position.set(x, 0, -y)
//     mesh.rotation.y = yaw       (yaw = atan2(2(wz+xy), 1-2(yy+zz)))
//   NOTE: the sign/axis here MUST be field-verified against the ceiling camera.
//   If a rover appears to drive the "wrong way", flip the y sign and/or the yaw
//   sign and re-check against a known physical heading. This is intentionally
//   isolated in one place so verification is a one-line change.
export function rosToScene(x, y) {
  return new THREE.Vector3(x, 0, -y);
}
// Nose (+x local) goes to (cos t, 0, -sin t) under three's Ry(t); a ROS heading
// (cos yaw, sin yaw) maps to scene (cos yaw, 0, -sin yaw), so t = yaw.
export const yawToSceneRot = (yaw) => yaw;

// Shortest-path angle lerp (handles wrap-around at ±π).
export function lerpAngle(a, b, t) {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}
