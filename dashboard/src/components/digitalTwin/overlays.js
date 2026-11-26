import * as THREE from 'three';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { COLORS } from '../../theme';
import { isFresh } from '../../ros/useMissionState';
import { PATH_COLOR, FLOOR_Y } from './constants';

// Path overlays: one polyline per rover from /{robot}/path_cmd.poses, obstacle shapes from
// /mission/obstacles, and a marker where the LiDAR says the path is blocked. Geometry is
// rebuilt only when the data signature changes.

// Empty `group`, disposing what it held (meshes, lines, sprites, nested groups).
export function clearGroup(group) {
  for (const c of [...group.children]) {
    group.remove(c);
    c.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        if (o.material.map) o.material.map.dispose();
        o.material.dispose();
      }
    });
  }
}

// Rebuild `group` to show obstacles: {x,y,width,height,yaw,pad} rectangles (mapped boxes,
// operator rectangles) as they are, {x,y,r} circles as discs. Translucent fill + edge.
function drawObstacles(group, obstacles) {
  clearGroup(group);
  const col = new THREE.Color(COLORS.lost);
  const fillMat = () =>
    new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
  for (const o of obstacles) {
    if (o.width > 0 && o.height > 0) {
      const w = o.width + 2 * (o.pad || 0);
      const h = o.height + 2 * (o.pad || 0);
      const g = new THREE.Group();
      const fill = new THREE.Mesh(new THREE.PlaneGeometry(w, h), fillMat());
      fill.rotation.x = -Math.PI / 2;
      g.add(fill);
      const c = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2], [-w / 2, -h / 2]];
      g.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(c.map(([a, b]) => new THREE.Vector3(a, 0.002, -b))),
        new THREE.LineBasicMaterial({ color: col })
      ));
      g.position.set(o.x, FLOOR_Y, -o.y);
      g.rotation.y = o.yaw || 0;
      group.add(g);
      continue;
    }
    const disc = new THREE.Mesh(new THREE.CircleGeometry(o.r, 32), fillMat());
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(Math.max(o.r - 0.015, 0.001), o.r, 32),
      new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide })
    );
    for (const m of [disc, ring]) {
      m.rotation.x = -Math.PI / 2; // lie flat on the floor
      m.position.set(o.x, FLOOR_Y, -o.y);
      group.add(m);
    }
  }
}

// Create path lines, blocker markers and the obstacle group (added to scene) for `names`.
export function createOverlays(scene, names, width, height) {
  const pathLines = {};
  const pathKeys = {};
  const creptOn = {}; // robot -> "epoch:cmd_id" of the command it crept on without a heading
  for (const r of names) {
    const mat = new LineMaterial({
      color: new THREE.Color(PATH_COLOR[r]).getHex(), linewidth: 5, transparent: true, opacity: 0.95,
    });
    mat.resolution.set(width, height); // px-space linewidth → needs canvas size
    const line = new Line2(new LineGeometry(), mat);
    line.visible = false;
    scene.add(line);
    pathLines[r] = line;
    pathKeys[r] = '';
  }
  const obstacleGroup = new THREE.Group();
  scene.add(obstacleGroup);
  const blocked = {};
  for (const r of names) {
    const m = new THREE.Mesh(
      new THREE.ConeGeometry(0.07, 0.16, 4),
      new THREE.MeshBasicMaterial({ color: COLORS.lost })
    );
    m.rotation.x = Math.PI; // point down at the blocker
    m.visible = false;
    scene.add(m);
    blocked[r] = m;
  }
  return { pathLines, pathKeys, creptOn, obstacleGroup, obstacleKey: '', blocked };
}

// Index of the path pose nearest `at`, so only the rest of the path is drawn.
function nearestPoseIndex(poses, at) {
  if (!(at && isFinite(at.x) && isFinite(at.y))) return 0;
  let first = 0;
  let best = Infinity;
  poses.forEach((q, i) => {
    const dq = (q.x - at.x) ** 2 + (q.y - at.y) ** 2;
    if (dq < best) { best = dq; first = i; }
  });
  return Math.min(first, poses.length - 2);
}

// Path polyline for one rover. Hidden when mode is not FOLLOW, poses empty, no fresh
// path_cmd, or the rover has finished it.
function updatePath(ov, r, d, at, now) {
  const pc = d.pathCmd && isFresh(d.pathCmd, now) ? d.pathCmd.msg : null;
  const poses = pc && pc.mode === 'FOLLOW' ? pc.poses || [] : [];
  // Drawn is the path the rover says it is following: its feedback names this
  // command. Never the path of an alignment creep (reported ALIGNING on it): that one
  // is planned from a guessed heading and the rover goes straight ahead instead.
  const nav = d.nav && isFresh(d.nav, now) ? d.nav.msg : null;
  const mine = pc && nav && nav.epoch === pc.epoch && nav.cmd_id === pc.cmd_id;
  const cmdKey = pc ? `${pc.epoch}:${pc.cmd_id}` : '';
  if (mine && String(nav.detail || '').includes('ALIGNING')) ov.creptOn[r] = cmdKey;
  // Not a finished one either: once the rover reports it has reached the end of this
  // path (HOLDING / PARKED) the line would only show where it came from, and it stayed
  // on the floor through a pick or until UNLOAD DONE.
  const finished = mine && (nav.state === 'HOLDING' || nav.state === 'PARKED');
  const shown = mine && poses.length >= 2 && ov.creptOn[r] !== cmdKey && !finished;
  const key = shown ? `${cmdKey}:${poses.length}` : '';
  if (key === ov.pathKeys[r]) return;
  ov.pathKeys[r] = key;
  if (key) {
    // Drawn from where the rover is: the host sends the same poses again under a new
    // command id when the lead parks (gap request dropped), and the whole path -
    // also the part already driven - appeared for the last metres.
    const first = nearestPoseIndex(poses, at);
    const pts = [];
    for (const q of poses.slice(first)) pts.push(q.x, FLOOR_Y, -q.y);
    // A new geometry per path: three.js keeps the segment count of the first
    // positions a LineGeometry was drawn with, so a longer path set on the same
    // geometry was cut off at the length of the first one.
    ov.pathLines[r].geometry.dispose();
    ov.pathLines[r].geometry = new LineGeometry();
    ov.pathLines[r].geometry.setPositions(pts);
  }
  ov.pathLines[r].visible = !!key;
}

// Blocker marker: LIDAR_* reason + finite blocked_x/y on a fresh safety_state.
function updateBlocker(ov, r, d, now) {
  const sf = d.safety && isFresh(d.safety, now) ? d.safety.msg : null;
  const lidar = sf && (sf.reasons || []).some((x) => String(x).startsWith('LIDAR_'));
  const bx = lidar ? Number(sf.blocked_x) : NaN;
  const by = lidar ? Number(sf.blocked_y) : NaN;
  const showBlocked = isFinite(bx) && isFinite(by);
  ov.blocked[r].visible = showBlocked;
  if (showBlocked) ov.blocked[r].position.set(bx, 0.2, -by);
}

// Per-frame update of paths, blocker markers and obstacles.
export function updateOverlays(ov, names, pathState, poses, obstacles, now) {
  for (const r of names) {
    updatePath(ov, r, pathState[r], poses[r], now);
    updateBlocker(ov, r, pathState[r], now);
  }
  const oKey = obstacles.map((o) => `${o.x},${o.y},${o.r},${o.width},${o.height},${o.yaw}`).join('|');
  if (oKey !== ov.obstacleKey) {
    ov.obstacleKey = oKey;
    drawObstacles(ov.obstacleGroup, obstacles);
  }
}
