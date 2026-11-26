import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Renderer, camera + orbit controls, and lights for the twin view.

export function createRenderer(mount, width, height) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  mount.appendChild(renderer.domElement);
  return renderer;
}

export function createCamera(renderer, width, height) {
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  return { camera, controls };
}

// Oblique camera framing of the driving bounds (rovers large and centred): a 4.1 x 2.54 m
// area seen from 2.64 m up and 2.2 m back, scaled to the bounds' size.
const VIEW_ZOOM_OUT = 1.18; // the shelves stand outside the bounds (east): keep them in the first view
export function defaultCameraPose(b) {
  const bb = b || { xmin: -1, xmax: 1, ymin: -1, ymax: 1 };
  const cx = (bb.xmin + bb.xmax) * 0.5;
  const cy = (bb.ymin + bb.ymax) * 0.5;
  const s = Math.max((bb.xmax - bb.xmin) / 4.1, (bb.ymax - bb.ymin) / 2.54, 0.5) * VIEW_ZOOM_OUT;
  return {
    pos: new THREE.Vector3(cx, 2.64 * s, -cy + 2.2 * s),
    target: new THREE.Vector3(cx, 0, -cy),
  };
}

export function addLights(scene) {
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9a9a92, 1.0));
  const dir = new THREE.DirectionalLight(0xffffff, 0.42);
  dir.position.set(0, 7, 0); // re-centred on the bounds once the map is in
  dir.castShadow = true;
  dir.shadow.mapSize.set(2048, 2048);
  dir.shadow.camera.near = 1;
  dir.shadow.camera.far = 30;
  dir.shadow.camera.left = -6;
  dir.shadow.camera.right = 6;
  dir.shadow.camera.top = 6;
  dir.shadow.camera.bottom = -6;
  scene.add(dir);
  return dir;
}

// Re-centre the key light and its shadow target on the bounds.
export function centerLight(scene, dir, b) {
  dir.position.set((b.xmin + b.xmax) / 2 - 1.0, 7, -(b.ymin + b.ymax) / 2 + 1.0);
  dir.target.position.set((b.xmin + b.xmax) / 2, 0, -(b.ymin + b.ymax) / 2);
  scene.add(dir.target);
}

// Click-to-queue: raycast the part cubes. A click (not an OrbitControls drag) calls
// onPick(partId). Returns a cleanup function.
export function attachPartPicking(renderer, camera, partMeshes, onPick) {
  const el = renderer.domElement;
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let downX = 0;
  let downY = 0;
  let downT = 0;
  const setNdc = (ev) => {
    const rect = el.getBoundingClientRect();
    ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    ndc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  };
  const pickPart = () => {
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(partMeshes, false);
    return hits.length ? hits[0].object : null;
  };
  const onPointerDown = (ev) => {
    downX = ev.clientX;
    downY = ev.clientY;
    downT = Date.now();
  };
  const onPointerUp = (ev) => {
    // drag/rotate vs click: small move + short hold = click
    if (Math.hypot(ev.clientX - downX, ev.clientY - downY) > 6 || Date.now() - downT > 500) return;
    setNdc(ev);
    const part = pickPart();
    if (part) onPick(part.userData.partId);
  };
  const onPointerMove = (ev) => {
    setNdc(ev);
    el.style.cursor = pickPart() ? 'pointer' : '';
  };
  el.addEventListener('pointerdown', onPointerDown);
  el.addEventListener('pointerup', onPointerUp);
  el.addEventListener('pointermove', onPointerMove);
  return () => {
    el.removeEventListener('pointerdown', onPointerDown);
    el.removeEventListener('pointerup', onPointerUp);
    el.removeEventListener('pointermove', onPointerMove);
  };
}
