import * as THREE from 'three';
import { ROBOTS } from '../../ros/useRobotState';
import { COLORS } from '../../theme';
import { loadMissionMap } from '../../missionMap';
import { URDF_URLS } from './constants';
import {
  createRenderer, createCamera, defaultCameraPose, addLights, centerLight, attachPartPicking,
} from './sceneSetup';
import { addMapElements, rebuildGoals, updatePartHighlights } from './mapScene';
import { createOverlays, updateOverlays, clearGroup } from './overlays';
import { createRovers, createRenderState, updateRover } from './rovers';

// Build the whole twin scene into `mount` and start the render loop.
// `live` carries refs read every frame (no re-render): poseRef, pathRef, obstaclesRef,
// queuedRef, showGoalsRef, dispatchRef. Returns { resetView, dispose }.
export function createTwin(mount, live) {
  const width = mount.clientWidth || 800;
  const height = mount.clientHeight || 500;

  const renderer = createRenderer(mount, width, height);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.background);
  const { camera, controls } = createCamera(renderer, width, height);

  let mapBounds = null; // from /mission_map.json
  const resetView = () => {
    const d = defaultCameraPose(mapBounds);
    camera.position.copy(d.pos);
    controls.target.copy(d.target);
    controls.update();
  };
  resetView();

  const dir = addLights(scene);

  // Static map from the generated /mission_map.json (mission config = single source):
  // floor around the driving bounds, bounds outline + grid, fixtures, parts, goals.
  const floorMat = new THREE.MeshStandardMaterial({ color: COLORS.floor, roughness: 1, metalness: 0 });
  const mapGroup = new THREE.Group();
  scene.add(mapGroup);
  const goalGroup = new THREE.Group();
  scene.add(goalGroup);
  let mapData = null;
  let goalKey = '';
  const partMeshes = []; // clickable part cubes (raycaster targets)
  let disposed = false;
  loadMissionMap().then((map) => {
    if (disposed) return;
    mapData = map;
    mapBounds = map.bounds;
    partMeshes.push(...addMapElements(mapGroup, map, floorMat));
    centerLight(scene, dir, map.bounds);
    resetView();
  });

  const overlays = createOverlays(scene, ROBOTS, width, height);

  const unpick = attachPartPicking(renderer, camera, partMeshes, (partId) => {
    if (live.dispatchRef.current) live.dispatchRef.current(partId);
  });

  const { rovers, labels } = createRovers(scene, URDF_URLS, ROBOTS);
  const renderState = createRenderState(ROBOTS);

  // Goal markers: the coordinator's live table (/mission/goals, latched) when it has
  // arrived, else the generated map's (same files the coordinator loaded).
  function updateGoals() {
    if (!mapData) return;
    const goals = live.pathRef.current.goals || mapData.goals || {};
    const gk = JSON.stringify(goals);
    if (gk !== goalKey) {
      goalKey = gk;
      clearGroup(goalGroup);
      rebuildGoals(goalGroup, goals, mapData);
    }
    for (const o of goalGroup.children) o.visible = o.userData.home || live.showGoalsRef.current;
  }

  let raf;
  function animate() {
    raf = requestAnimationFrame(animate);
    const now = Date.now();
    updateGoals();
    updateOverlays(overlays, ROBOTS, live.pathRef.current, live.poseRef.current, live.obstaclesRef.current, now);
    for (const r of ROBOTS) {
      updateRover(rovers[r], labels[r], renderState[r], live.poseRef.current[r], now, r);
    }
    updatePartHighlights(partMeshes, live.queuedRef.current);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  const ro = new ResizeObserver(() => {
    const w = mount.clientWidth;
    const h = mount.clientHeight;
    if (w === 0 || h === 0) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    for (const r of ROBOTS) overlays.pathLines[r].material.resolution.set(w, h); // px-space linewidth
  });
  ro.observe(mount);

  const dispose = () => {
    disposed = true;
    cancelAnimationFrame(raf);
    ro.disconnect();
    unpick();
    controls.dispose();
    renderer.dispose();
    if (renderer.domElement.parentNode === mount) {
      mount.removeChild(renderer.domElement);
    }
  };
  return { resetView, dispose };
}
