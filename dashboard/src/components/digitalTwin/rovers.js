import * as THREE from 'three';
import URDFLoader from 'urdf-loader';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { URDF_PKGS, STALE_MS, HIDE_MS, POSE_SMOOTHING } from './constants';
import { rosToScene, yawToSceneRot, lerpAngle } from './coords';
import { makeRoverLabel, drawRoverLabel } from './labels';

const MODEL_COLOR = '#474e58';
const matteMaterial = () =>
  new THREE.MeshStandardMaterial({ color: MODEL_COLOR, roughness: 0.9, metalness: 0 });

// Rover group holds only the URDF model, added asynchronously by loadRover.
function buildRover() {
  const group = new THREE.Group();
  group.userData.dimmable = []; // URDF mesh materials are appended after load
  return group;
}

// Load a turtlebot3 URDF into group: z-up -> y-up, meshes get a flat neutral material.
function loadRover(group, url) {
  const loader = new URDFLoader(new THREE.LoadingManager());
  loader.packages = URDF_PKGS;
  loader.loadMeshCb = (path, mgr, done) => {
    if (path.toLowerCase().endsWith('.stl')) {
      new STLLoader(mgr).load(
        path,
        (geom) => {
          const mesh = new THREE.Mesh(geom, matteMaterial());
          mesh.castShadow = true;
          done(mesh);
        },
        undefined,
        () => done(null)
      );
    } else {
      done(null); // skip .dae sensor meshes (not needed for the shape)
    }
  };
  loader.load(
    url,
    (robot) => {
      robot.rotation.x = -Math.PI / 2; // URDF z-up → three y-up
      robot.traverse((o) => {
        if (o.isMesh) {
          // ignore URDF materials; one matte neutral for all
          o.material = matteMaterial();
          o.castShadow = true;
          group.userData.dimmable.push(o.material);
        }
      });
      group.add(robot);
    },
    undefined,
    (err) => console.warn('[twin] URDF 로드 실패:', url, err)
  );
}

// Create one rover per name (model loads async), each with its name tag, added to scene.
export function createRovers(scene, urdfUrls, names) {
  const rovers = {};
  const labels = {};
  for (const r of names) {
    rovers[r] = buildRover();
    loadRover(rovers[r], urdfUrls[r]);
  }
  for (const r of names) {
    scene.add(rovers[r]);
    const label = makeRoverLabel();
    rovers[r].add(label);
    labels[r] = label;
  }
  return { rovers, labels };
}

// Smoothed render state per rover, lerped toward the target pose each frame.
export function createRenderState(names) {
  const state = {};
  for (const r of names) state[r] = { x: 0, y: 0, yaw: 0, init: false };
  return state;
}

// Move one rover toward its pose target; hide / freeze / dim according to pose age.
export function updateRover(rover, label, rs, target, now, name) {
  // present = a pose has ever arrived (rxWall is 0 until then). Without it the render
  // default (the origin) would show a translucent ghost for an undetected rover.
  const present = !!target.rxWall;
  const age = present ? now - target.rxWall : Infinity;
  const stale = age > STALE_MS;
  // never received, or older than HIDE_MS: hide entirely
  const visible = present && age <= HIDE_MS;

  rover.visible = visible; // hiding also hides the label (child)
  if (!visible) return;

  // Snap on first valid pose so it doesn't slide in from the origin.
  if (!rs.init) {
    rs.x = target.x;
    rs.y = target.y;
    rs.yaw = target.yaw;
    rs.init = true;
  }

  // STALENESS GUARD: only interpolate toward target while fresh.
  if (!stale) {
    const t = POSE_SMOOTHING;
    rs.x += (target.x - rs.x) * t;
    rs.y += (target.y - rs.y) * t;
    rs.yaw = lerpAngle(rs.yaw, target.yaw, t);
  }

  const p = rosToScene(rs.x, rs.y);
  rover.position.set(p.x, 0, p.z);
  rover.rotation.y = yawToSceneRot(rs.yaw);

  // Dim on STALE (model opacity only).
  const opacity = stale ? 0.32 : 1.0;
  for (const mat of rover.userData.dimmable) {
    mat.transparent = true;
    mat.opacity = opacity;
  }

  drawRoverLabel(label, name.toUpperCase());
  label.material.opacity = stale ? 0.45 : 0.95;
}
