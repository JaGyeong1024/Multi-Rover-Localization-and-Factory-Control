import * as THREE from 'three';
import { COLORS, ZONE_COLORS } from '../../theme';
import { prettyId } from '../../missionMap';
import { rosToScene, yawToSceneRot } from './coords';
import { makeTextSprite } from './labels';
import { PATH_COLOR, FLOOR_MARGIN } from './constants';

// Static map builders — from the generated /mission_map.json (uwb_global → scene (x,0,-y)).

// Axis-aligned outline of {xmin,xmax,ymin,ymax} slightly above the floor.
function buildRectOutline(b, css, y = 0.03) {
  const c = [
    new THREE.Vector3(b.xmin, y, -b.ymin), new THREE.Vector3(b.xmax, y, -b.ymin),
    new THREE.Vector3(b.xmax, y, -b.ymax), new THREE.Vector3(b.xmin, y, -b.ymax),
  ];
  const geo = new THREE.BufferGeometry().setFromPoints([...c, c[0]]);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color: new THREE.Color(css) }));
}

// Grid lines at multiples of `step` inside the bounds (carpet tiles are 0.5 m in this frame).
function buildBoundsGrid(b, step = 0.5) {
  const pts = [];
  const y = 0.004;
  for (let x = Math.ceil(b.xmin / step) * step; x <= b.xmax + 1e-6; x += step) {
    pts.push(new THREE.Vector3(x, y, -b.ymin), new THREE.Vector3(x, y, -b.ymax));
  }
  for (let yy = Math.ceil(b.ymin / step) * step; yy <= b.ymax + 1e-6; yy += step) {
    pts.push(new THREE.Vector3(b.xmin, y, -yy), new THREE.Vector3(b.xmax, y, -yy));
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({ color: new THREE.Color(COLORS.grid1), transparent: true, opacity: 0.5 })
  );
}

// Fixture (shelf) = low one-level table, oriented: width along yaw, height across.
const SHELF_TOP = 0.1;
function buildShelf(f) {
  const g = new THREE.Group();
  const top = SHELF_TOP;
  const boardMat = new THREE.MeshStandardMaterial({ color: COLORS.shelfBoard, roughness: 0.9, metalness: 0 });
  const legMat = new THREE.MeshStandardMaterial({ color: COLORS.shelfLeg, roughness: 0.9, metalness: 0 });
  const board = new THREE.Mesh(new THREE.BoxGeometry(f.width, 0.04, f.height), boardMat);
  board.position.y = top; board.castShadow = true; board.receiveShadow = true;
  g.add(board);
  const lx = Math.max(0.02, f.width / 2 - 0.03), lz = Math.max(0.02, f.height / 2 - 0.03);
  for (const [px, pz] of [[-lx, -lz], [lx, -lz], [-lx, lz], [lx, lz]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.03, top, 0.03), legMat);
    leg.position.set(px, top / 2, pz); leg.castShadow = true; g.add(leg);
  }
  const p = rosToScene(f.x, f.y);
  g.position.set(p.x, 0, p.z);
  g.rotation.y = yawToSceneRot(f.yaw || 0);
  return g;
}

// Part = clickable cube on its shelf (or on the floor ahead of the pick goal when no
// fixture is registered). userData.partId → raycaster click → /task/add "fetch:<id>".
const PART_BASE = '#a9b2bd'; // one colour for all parts (Gear/Bolt told apart by label)
export function buildPart(part) {
  const size = 0.16 * 0.8;
  const base = part.fixture ? SHELF_TOP + 0.02 : 0; // shelf board top, else the floor
  const mat = new THREE.MeshStandardMaterial({ color: PART_BASE, roughness: 0.55, metalness: 0.1 });
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(size, size, size), mat);
  const p = rosToScene(part.x, part.y);
  mesh.position.set(p.x, base + size / 2, p.z);
  mesh.castShadow = true;
  // white text, tinted via sprite material.color (green when queued)
  const label = makeTextSprite(prettyId(part.id), '#ffffff');
  label.position.set(0, size / 2 + 0.09, 0); // above the cube top
  mesh.add(label);
  mesh.userData = { partId: part.id, mat, label };
  return mesh;
}

// Goal ownership: colour = the rover that uses the goal (leader / follower, from the
// scenario roles and parts), else neutral.
function goalOwners(map) {
  const owner = {};
  const roles = (map && map.roles) || {};
  if (roles.leader_standby) owner[roles.leader_standby] = 'leader';
  for (const k of ['follower_unload', 'follower_home']) if (roles[k]) owner[roles[k]] = 'follower';
  for (const p of (map && map.parts) || []) {
    if (p.leader_goal) owner[p.leader_goal] = 'leader';
    if (p.follower_goal) owner[p.follower_goal] = 'follower';
  }
  return owner;
}

// Names of the rovers' own places (leader standby, follower unload / home), from the
// scenario roles of the generated mission map (the same file the host loaded).
function homeGoals(map) {
  const roles = (map && map.roles) || {};
  return new Set(['leader_standby', 'follower_unload', 'follower_home'].map((k) => roles[k]).filter(Boolean));
}

// Goal pose marker: ring at the position + heading tick + floor name label.
function buildGoal(name, g, css) {
  const grp = new THREE.Group();
  const col = new THREE.Color(css);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.06, 0.08, 32),
    new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide, transparent: true, opacity: 0.9 })
  );
  ring.rotation.x = -Math.PI / 2;
  grp.add(ring);
  const tick = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.16, 0, 0)]),
    new THREE.LineBasicMaterial({ color: col })
  );
  grp.add(tick); // local +x = heading after rotation.y = theta
  const p = rosToScene(g.x, g.y);
  grp.position.set(p.x, 0.025, p.z);
  grp.rotation.y = yawToSceneRot(g.theta || 0);
  const label = makeTextSprite(name, css);
  label.position.set(p.x, 0.16, p.z);
  const out = new THREE.Group();
  out.add(grp, label);
  return out;
}

// Rebuild the goal markers into `goalGroup`; `home` ones stay visible when goals are off.
export function rebuildGoals(goalGroup, goals, map) {
  const owner = goalOwners(map);
  const homes = homeGoals(map);
  for (const [name, g] of Object.entries(goals)) {
    const css = PATH_COLOR[owner[name]] || ZONE_COLORS.default;
    const marker = buildGoal(name, g, css);
    marker.userData.home = homes.has(name);
    goalGroup.add(marker);
  }
}

// Floor, bounds outline + grid, fixtures and clickable parts into `group`.
// Returns the part meshes (raycaster targets).
export function addMapElements(group, map, floorMat) {
  const b = map.bounds;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(b.xmax - b.xmin + 2 * FLOOR_MARGIN, b.ymax - b.ymin + 2 * FLOOR_MARGIN),
    floorMat
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set((b.xmin + b.xmax) / 2, 0, -(b.ymin + b.ymax) / 2);
  floor.receiveShadow = true;
  group.add(floor);
  group.add(buildRectOutline(b, '#d0d4da'));
  group.add(buildBoundsGrid(b));
  // Fixtures named outside_* only cut the arena's rounded corners off: not drawn (no shelf there).
  for (const f of map.fixtures || []) if (!String(f.name || '').startsWith('outside')) group.add(buildShelf(f));
  const parts = [];
  for (const part of map.parts || []) {
    if (part.x == null || part.y == null) continue;
    const pm = buildPart(part);
    group.add(pm);
    parts.push(pm);
  }
  return parts;
}

// Queued feedback — keep the cube's OWN hue (subtle emissive brighten) and recolor its
// name label (white → green). Reflects the real /mission/queue.
export function updatePartHighlights(partMeshes, queued) {
  for (const pm of partMeshes) {
    const on = queued.has(pm.userData.partId);
    if (on) {
      pm.userData.mat.emissive.copy(pm.userData.mat.color); // same hue, lit up
      pm.userData.mat.emissiveIntensity = 0.5;
    } else {
      pm.userData.mat.emissive.set('#000000');
      pm.userData.mat.emissiveIntensity = 0;
    }
    if (pm.userData.label) {
      pm.userData.label.material.color.set(on ? '#5fd98a' : '#dfe4ea');
    }
  }
}
