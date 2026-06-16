import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { useRobotState, ROBOTS } from '../ros/useRobotState';
import { useDiagnostics } from '../ros/useDiagnostics';
import { useMission } from '../ros/useMission';
import { useSubscription } from '../ros/useRos';
import { COLORS, ROOM, ACTIVE_ZONE, MODE, ZONE_COLORS, ZONE_LABEL, levelToHealth } from '../theme';

// ---------------------------------------------------------------------------
// COORDINATE MAPPING (uwb_global → three.js, y-up)
//   ROS uwb_global: x along the 6.9m wall, y along the 6.2m wall, z up.
//   three.js is y-up, so the floor lives in the X/Z plane.
//     mesh.position.set(x, 0, -y)
//     mesh.rotation.y = yaw       (yaw = atan2(2(wz+xy), 1-2(yy+zz)))
//   NOTE: the sign/axis here MUST be field-verified against the ceiling camera.
//   If a rover appears to drive the "wrong way", flip the y sign and/or the yaw
//   sign and re-check against a known physical heading. This is intentionally
//   isolated in one place so verification is a one-line change.
// ---------------------------------------------------------------------------
function rosToScene(x, y) {
  return new THREE.Vector3(x, 0, -y);
}
// +yaw: nose(+x local)는 three Ry(θ)로 (cosθ,0,-sinθ)로 가고, ROS heading
// (cos yaw,sin yaw)는 scene (cos yaw,0,-sin yaw)로 매핑되므로 θ=yaw가 맞다(부호 검증 완료).
const yawToSceneRot = (yaw) => yaw;

// Staleness threshold: a twin that smoothly animates toward a stale pose hides
// death. If now - lastReceipt > this, we FREEZE interpolation, dim the rover,
// and flag STALE.
const STALE_MS = 150;
// /leader/route is event-driven + latched (transient_local): the leader publishes its
// planned path only when it CHANGES, not on a timer (confirmed 2026-06-14 — no republish
// for minutes). So we HOLD the last received route and keep drawing it until a new
// LeaderRoute supersedes node_ids (an empty node_ids clears the overlay). Do NOT time-out
// the overlay — the previous ~4s staleness drop assumed a continuous stream that does not exist.
// Hard-hide threshold: 이보다 오래 pose 미수신이면 로버를 통째 숨긴다(부재로 간주).
// 백엔드 global_position_node가 소스 끊김 시 0.5s 내 발행을 멈추므로(stale 가드), 그 뒤
// 잠깐의 dim(STALE_MS~HIDE_MS)만 보여주고 사라진다 — "없으면 안 뜬다". 짧은 gap엔 안 깜빡임.
const HIDE_MS = 1500;

// Mission states with NO active route → clear the /leader/route overlay (the leader may keep
// its last route latched after finishing — see route hold note below). Only IDLE/DONE (rovers
// at rest). NOT ABORTED: abort() sends leader→standby / follower→unloading in GOTO mode
// (task_sequencer), i.e. an active RETURN route the operator should still see.
const INACTIVE_STATES = new Set(['IDLE', 'DONE']);

// Shortest-path angle lerp (handles wrap-around at ±π).
function lerpAngle(a, b, t) {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}

// URDF 패키지 → 서빙 경로 (vendored: dashboard/public/urdf/, turtlebot3 Apache-2.0).
const URDF_PKGS = {
  turtlebot3_description: '/urdf/turtlebot3_description',
  turtlebot3_manipulation_description: '/urdf/turtlebot3_manipulation_description',
};

// 로버 그룹 = URDF 모델만(상태 링·노즈 제거 — 깔끔). 모델은 loadRover가 비동기로 추가.
function buildRover() {
  const group = new THREE.Group();
  group.userData.dimmable = []; // URDF 메시 머티리얼은 로드 후 append
  return group;
}

// turtlebot3 URDF 비동기 로드 → group에 추가. z-up→y-up 회전, 메시는 무광 neutral 덮어쓰기.
function loadRover(group, url) {
  const manager = new THREE.LoadingManager();
  const loader = new URDFLoader(manager);
  loader.packages = URDF_PKGS;
  loader.loadMeshCb = (path, mgr, done) => {
    if (path.toLowerCase().endsWith('.stl')) {
      new STLLoader(mgr).load(
        path,
        (geom) => {
          const mesh = new THREE.Mesh(
            geom,
            new THREE.MeshStandardMaterial({ color: '#474e58', roughness: 0.9, metalness: 0 })
          );
          mesh.castShadow = true;
          done(mesh);
        },
        undefined,
        () => done(null)
      );
    } else {
      done(null); // .dae 센서 메시 등은 스킵(형태엔 불필요)
    }
  };
  loader.load(
    url,
    (robot) => {
      robot.rotation.x = -Math.PI / 2; // URDF z-up → three y-up
      robot.traverse((o) => {
        if (o.isMesh) {
          // URDF 머티리얼 무시하고 무광 neutral로 통일(상태색은 링/노즈가 담당).
          o.material = new THREE.MeshStandardMaterial({ color: '#474e58', roughness: 0.9, metalness: 0 });
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

// A canvas-texture text sprite floating above each rover (name + task).
function makeLabel() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 80;
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.6, 0.1875, 1); // 256:80 비율 유지, 작게
  sprite.position.y = 0.46;
  sprite.userData = { canvas, ctx: canvas.getContext('2d'), tex, text: '' };
  return sprite;
}

// 미니멀 태그: 박스 없이 이름 텍스트만(대비용 외곽선).
function drawLabel(sprite, text) {
  const { canvas, ctx, tex } = sprite.userData;
  if (sprite.userData.text === text) return;
  sprite.userData.text = text;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = 'bold 36px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 7;
  ctx.strokeStyle = 'rgba(14,17,22,0.85)';
  ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
  ctx.fillStyle = '#eef1f4';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  tex.needsUpdate = true;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ---- Map (zone/shelf) builders — from /map.json, axis-aligned rects -------
// 모두 uwb_global 좌표 → scene(x,0,-y). zone은 bbox 평면 + 외곽선 + 라벨, 선반은 낮은 테이블.
function buildZoneFill(minx, miny, maxx, maxy, css) {
  const geo = new THREE.PlaneGeometry(maxx - minx, maxy - miny);
  const mat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(css), transparent: true, opacity: 0.22,
    depthWrite: false, side: THREE.DoubleSide,
  });
  const m = new THREE.Mesh(geo, mat);
  m.rotation.x = -Math.PI / 2;
  m.position.set((minx + maxx) / 2, 0.02, -(miny + maxy) / 2);
  return m;
}

function buildZoneOutline(minx, miny, maxx, maxy, css) {
  const c = [
    new THREE.Vector3(minx, 0.03, -miny), new THREE.Vector3(maxx, 0.03, -miny),
    new THREE.Vector3(maxx, 0.03, -maxy), new THREE.Vector3(minx, 0.03, -maxy),
  ];
  const geo = new THREE.BufferGeometry().setFromPoints([...c, c[0]]);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color: new THREE.Color(css) }));
}

function buildShelf(s) {
  const g = new THREE.Group();
  const top = Math.max(0.06, s.z || 0.1); // 낮은 1단 테이블
  const boardMat = new THREE.MeshStandardMaterial({ color: COLORS.shelfBoard, roughness: 0.9, metalness: 0 });
  const legMat = new THREE.MeshStandardMaterial({ color: COLORS.shelfLeg, roughness: 0.9, metalness: 0 });
  const board = new THREE.Mesh(new THREE.BoxGeometry(s.w, 0.04, s.h), boardMat);
  board.position.y = top; board.castShadow = true; board.receiveShadow = true;
  g.add(board);
  const lx = Math.max(0.02, s.w / 2 - 0.03), lz = Math.max(0.02, s.h / 2 - 0.03);
  for (const [px, pz] of [[-lx, -lz], [lx, -lz], [-lx, lz], [lx, lz]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.03, top, 0.03), legMat);
    leg.position.set(px, top / 2, pz); leg.castShadow = true; g.add(leg);
  }
  const p = rosToScene(s.x, s.y);
  g.position.set(p.x, 0, p.z);
  return g;
}

// 선반 위 부품 = 클릭 가능한 정육면체 박스 (uwb_global 좌표 = 선반 중앙). type별 색.
// userData.partId 로 raycaster 클릭 → /mission/dispatch "fetch:<id>".
const PART_BASE = '#a9b2bd'; // 모든 부품 통일색 (Gear/Bolt 구분은 라벨로)
function buildPart(shelf) {
  const size = 0.16 * 0.8; // 0.8× (사용자 요청)
  const boardTop = Math.max(0.06, shelf.z || 0.1) + 0.02; // 선반 보드 윗면(두께 0.04 중심)
  const mat = new THREE.MeshStandardMaterial({ color: PART_BASE, roughness: 0.55, metalness: 0.1 });
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(size, size, size), mat);
  const p = rosToScene(shelf.x, shelf.y);
  mesh.position.set(p.x, boardTop + size / 2, p.z); // 보드 위에 얹힘
  mesh.castShadow = true;
  // 상단 이름 라벨(Gear/Bolt) — 큐브의 자식 스프라이트(항상 카메라 향함). 클릭/글로우엔 영향 없음.
  const t = (shelf.part.type || '').trim();
  const name = t ? t[0].toUpperCase() + t.slice(1) : shelf.part.id || 'Part';
  const label = makePartLabel(name);
  label.position.set(0, size / 2 + 0.09, 0); // 큐브 윗면 위로
  mesh.add(label);
  mesh.userData = { partId: shelf.part.id, mat, label };
  return mesh;
}

// 부품 큐브 상단 이름 라벨 — 캔버스 텍스처 스프라이트(항상 카메라 향함, depthTest off).
function makePartLabel(text) {
  const FONT = 'bold 30px sans-serif';
  const H = 64;
  const PX2WORLD = 0.5 / 256;
  const canvas = document.createElement('canvas');
  let ctx = canvas.getContext('2d');
  ctx.font = FONT;
  const textW = Math.ceil(ctx.measureText(text).width);
  canvas.width = 16 + textW + 16;
  canvas.height = H;
  ctx = canvas.getContext('2d'); // width 변경 → 상태 리셋, 재취득
  ctx.font = FONT;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(8,10,14,0.92)'; // 배경 대신 외곽선으로 가독성 확보
  ctx.lineWidth = 6;
  ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
  ctx.fillStyle = '#ffffff'; // 흰 텍스트 → 스프라이트 material.color로 틴트(선택 시 초록)
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set(canvas.width * PX2WORLD, canvas.height * PX2WORLD, 1);
  return sp;
}

// 캔버스 폭을 텍스트에 맞춰 자동 산정 → 긴 라벨("Follower Station")도 안 잘림.
// 글자 크기는 픽셀→월드 배율(PX2WORLD)을 고정해 모든 라벨이 동일하게 보인다(폭만 가변).
function makeMapLabel(text, css, x, y) {
  const FONT = 'bold 30px sans-serif';
  const H = 64;
  const PX2WORLD = 0.72 / 256; // 기존 256px→0.72world 와 동일한 글자 크기 유지
  const canvas = document.createElement('canvas');
  let ctx = canvas.getContext('2d');
  ctx.font = FONT;
  const textW = Math.ceil(ctx.measureText(text).width);
  canvas.width = 16 + textW + 16; // 텍스트 + 좌우 여백 (배경/컬러바 제거)
  canvas.height = H;
  ctx = canvas.getContext('2d'); // canvas.width 변경 시 컨텍스트 상태 리셋 → 재취득
  ctx.font = FONT;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(8,10,14,0.92)'; // 배경 대신 외곽선
  ctx.lineWidth = 6;
  ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
  ctx.fillStyle = css; // 존 색으로 텍스트 (식별 유지)
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas); tex.minFilter = THREE.LinearFilter;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set(canvas.width * PX2WORLD, canvas.height * PX2WORLD, 1); // 폭만 가변, 글자 크기 일정
  const p = rosToScene(x, y); sp.position.set(p.x, 0.28, p.z);
  return sp;
}

function addMapElements(group, map) {
  const az = map && map.bev_meta && map.bev_meta.active_zone;
  if (az) group.add(buildZoneOutline(az.x_min, az.y_min, az.x_max, az.y_max, '#d0d4da'));
  for (const z of (map && map.zones) || []) {
    const xs = z.points.map((p) => p.x);
    const ys = z.points.map((p) => p.y);
    const minx = Math.min(...xs), maxx = Math.max(...xs);
    const miny = Math.min(...ys), maxy = Math.max(...ys);
    const css = ZONE_COLORS[z.type] || ZONE_COLORS.default;
    group.add(buildZoneFill(minx, miny, maxx, maxy, css));
    group.add(buildZoneOutline(minx, miny, maxx, maxy, css));
    group.add(makeMapLabel(ZONE_LABEL[z.type] || z.name, css, (minx + maxx) / 2, (miny + maxy) / 2));
  }
  for (const s of (map && map.shelves) || []) group.add(buildShelf(s));
}

// 활동영역(active zone) 안에만 그리는 그리드(0.5m). 방 전체 GridHelper 대신.
function buildActiveGrid(az, step = 0.5) {
  const pts = [];
  const y = 0.004;
  for (let x = az.x_min; x <= az.x_max + 1e-6; x += step) {
    pts.push(new THREE.Vector3(x, y, -az.y_min), new THREE.Vector3(x, y, -az.y_max));
  }
  for (let yy = az.y_min; yy <= az.y_max + 1e-6; yy += step) {
    pts.push(new THREE.Vector3(az.x_min, y, -yy), new THREE.Vector3(az.x_max, y, -yy));
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({ color: new THREE.Color(COLORS.grid1), transparent: true, opacity: 0.5 })
  );
}

// ---- Lane graph (nodes + edges) — 바닥 위 임시 디버그 오버레이 ----------------
// uwb_global 좌표를 rosToScene과 동일하게 (x,0,-y)로 매핑 → 로버·존과 같은 좌표계에
// 정확히 겹친다(노드 좌표 정합 육안 검증용). 단일 진실 = /map.json graph. Graph 토글로 on/off.
function makeNodeLabel(text) {
  const FONT = 'bold 26px sans-serif';
  const H = 40;
  const PX2WORLD = 0.34 / 256;
  const canvas = document.createElement('canvas');
  let ctx = canvas.getContext('2d');
  ctx.font = FONT;
  const textW = Math.ceil(ctx.measureText(text).width);
  canvas.width = 10 + textW + 10;
  canvas.height = H;
  ctx = canvas.getContext('2d'); // width 변경 → 상태 리셋, 재취득
  ctx.font = FONT;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(8,10,14,0.92)';
  ctx.lineWidth = 5;
  ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
  ctx.fillStyle = '#cfe0ff';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set(canvas.width * PX2WORLD, canvas.height * PX2WORLD, 1);
  return sp;
}

function buildGraph(graph) {
  const group = new THREE.Group();
  const nodes = (graph && graph.nodes) || [];
  const edges = (graph && graph.edges) || [];
  const byId = new Map(nodes.map((n) => [n.id, n]));

  // edges → 바닥 살짝 위 라인 세그먼트(방향 무관 단색)
  const pts = [];
  for (const e of edges) {
    const a = byId.get(e.from);
    const b = byId.get(e.to);
    if (!a || !b) continue;
    pts.push(new THREE.Vector3(a.x, 0.012, -a.y), new THREE.Vector3(b.x, 0.012, -b.y));
  }
  if (pts.length) {
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    group.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
      color: new THREE.Color('#8fb4e0'), transparent: true, opacity: 1.0,
    })));
  }

  // nodes → 바닥에 눕힌 작은 디스크 + id 라벨(geometry/material 공유)
  const disc = new THREE.CircleGeometry(0.0175, 20);
  const nodeMat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#86b8ff') });
  for (const n of nodes) {
    const m = new THREE.Mesh(disc, nodeMat);
    m.rotation.x = -Math.PI / 2;
    m.position.set(n.x, 0.014, -n.y);
    group.add(m);
    const lbl = makeNodeLabel(String(n.id));
    lbl.position.set(n.x, 0.11, -n.y);
    group.add(lbl);
  }
  return group;
}

// Default oblique camera framing — centered on the ACTIVE ZONE (the rovers' real
// operating area) so they appear large/centered instead of lost in a big floor.
function defaultCameraPose() {
  const cx = (ACTIVE_ZONE.x_min + ACTIVE_ZONE.x_max) * 0.5; // ~3.68
  const cy = (ACTIVE_ZONE.y_min + ACTIVE_ZONE.y_max) * 0.5; // ~3.37
  return {
    pos: new THREE.Vector3(cx, 2.64, -cy + 2.2),  // 활동영역(4.1×2.54) 프레이밍(아주 약간 줌아웃)
    target: new THREE.Vector3(cx, 0, -cy),
  };
}

export default function DigitalTwin() {
  const mountRef = useRef(null);
  const { poseRef } = useRobotState();
  const { getStatus } = useDiagnostics();
  const { dispatch, queue, systemState } = useMission();

  // Keep latest diagnostics-derived state accessible inside the rAF loop.
  const getStatusRef = useRef(getStatus);
  getStatusRef.current = getStatus;

  // Bridge mission state into the (run-once) three.js effect via refs:
  //   dispatchRef — click handler publishes /mission/dispatch.
  //   queuedRef   — set of part ids currently in the queue → glow their cubes.
  const dispatchRef = useRef(dispatch);
  dispatchRef.current = dispatch;
  const queuedRef = useRef(new Set());
  queuedRef.current = new Set((queue || []).map((q) => q.part));

  const resetViewRef = useRef(() => {});
  // STALE flags surfaced to the DOM overlay (set from rAF, read on render via state).
  const [staleFlags, setStaleFlags] = useState({ leader: false, follower: false });
  const staleFlagsRef = useRef(staleFlags);

  // Lane-graph debug overlay toggle (read inside the rAF loop without re-running the effect).
  const [showGraph, setShowGraph] = useState(true);
  const showGraphRef = useRef(showGraph);
  showGraphRef.current = showGraph;
  // Mirror host mission state into a ref so the rAF loop can gate the route overlay
  // without re-running the scene effect (same bridge pattern as showGraphRef).
  const systemStateRef = useRef(systemState);
  systemStateRef.current = systemState;

  // Live leader route (/leader/route). The leader publishes its planned node path; we
  // render the node_ids verbatim (no client-side routing — pure topic consumer). The
  // follower has no route topic by design (it branches internally), so this is leader-only.
  const routeRef = useRef({ ids: [], rx: 0 });
  useSubscription(
    '/leader/route',
    'host_mission_interfaces/msg/LeaderRoute',
    (m) => {
      routeRef.current = { ids: (m && m.node_ids) || [], rx: Date.now() };
    },
    []
  );

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;

    const width = mount.clientWidth || 800;
    const height = mount.clientHeight || 500;

    // --- Renderer ---
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);

    // --- Scene ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.background);

    // --- Camera ---
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    const def = defaultCameraPose();
    camera.position.copy(def.pos);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.copy(def.target);
    controls.update();

    resetViewRef.current = () => {
      const d = defaultCameraPose();
      camera.position.copy(d.pos);
      controls.target.copy(d.target);
      controls.update();
    };

    // --- Lights ---
    const hemi = new THREE.HemisphereLight(0xffffff, 0x9a9a92, 1.0);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 0.42);
    dir.position.set(ROOM.width * 0.3, 7, -ROOM.depth * 0.2);
    dir.castShadow = true;
    dir.shadow.mapSize.set(2048, 2048);
    dir.shadow.camera.near = 1;
    dir.shadow.camera.far = 30;
    dir.shadow.camera.left = -6;
    dir.shadow.camera.right = 6;
    dir.shadow.camera.top = 6;
    dir.shadow.camera.bottom = -6;
    scene.add(dir);

    // --- Floor (room-sized, centered so x in [0,W], y in [0,D]) ---
    const floorMat = new THREE.MeshStandardMaterial({
      color: COLORS.floor,
      roughness: 1,
      metalness: 0,
    });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(ROOM.width, ROOM.depth), floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(ROOM.width / 2, 0, -ROOM.depth / 2);
    floor.receiveShadow = true;
    scene.add(floor);

    // --- Grid (활동영역에만) ---
    scene.add(buildActiveGrid(ACTIVE_ZONE));

    // --- Map-driven static context: zones + shelves from /map.json (live) ---
    // path editor가 유지하는 단일 진실. 경로(graph)는 미표시, zone·선반만.
    const mapGroup = new THREE.Group();
    scene.add(mapGroup);
    let graphGroup = null; // lane-graph debug overlay (built async from /map.json, toggled in rAF)
    let nodeById = new Map(); // node id → {x,y} for the live route overlay (filled on map load)
    const partMeshes = []; // clickable part cubes on shelves (raycaster targets)
    fetch('/map.json')
      .then((r) => r.json())
      .then((map) => {
        addMapElements(mapGroup, map);
        graphGroup = buildGraph(map && map.graph);
        graphGroup.visible = showGraphRef.current;
        scene.add(graphGroup);
        nodeById = new Map(((map && map.graph && map.graph.nodes) || []).map((n) => [n.id, n]));
        for (const s of (map && map.shelves) || []) {
          if (!s.part) continue;
          const pm = buildPart(s);
          mapGroup.add(pm);
          partMeshes.push(pm);
        }
      })
      .catch((e) => console.warn('[twin] /map.json 로드 실패:', e));

    // --- Live route overlay: /leader/route (LeaderRoute.node_ids) → bright path on
    // the floor. Pure consumer of the published route (no client-side routing). The
    // geometry is rebuilt only when node_ids change; hidden when stale/empty (see rAF).
    const routeMat = new LineMaterial({
      color: 0x36e08a, linewidth: 5, transparent: true, opacity: 0.95,
    });
    routeMat.resolution.set(width, height); // px-space linewidth → needs canvas size
    const routeLine = new Line2(new LineGeometry(), routeMat);
    routeLine.visible = false;
    scene.add(routeLine);
    let routeKey = ''; // last-drawn node_ids signature (skip rebuild when unchanged)

    // --- Click-to-queue: raycast the part cubes. A click (not an OrbitControls
    // drag) on a part publishes /mission/dispatch "fetch:<id>" via useMission. ---
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let downX = 0;
    let downY = 0;
    let downT = 0;
    const setNdc = (ev) => {
      const rect = renderer.domElement.getBoundingClientRect();
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
      if (part && dispatchRef.current) dispatchRef.current(part.userData.partId);
    };
    const onPointerMove = (ev) => {
      setNdc(ev);
      renderer.domElement.style.cursor = pickPart() ? 'pointer' : '';
    };
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerup', onPointerUp);
    renderer.domElement.addEventListener('pointermove', onPointerMove);

    // --- Rovers ---
    const rovers = {};
    const labels = {};
    rovers.leader = buildRover();
    rovers.follower = buildRover();
    loadRover(rovers.leader, '/urdf/turtlebot3_manipulation_description/urdf/turtlebot3_manipulation.urdf');
    loadRover(rovers.follower, '/urdf/turtlebot3_description/urdf/turtlebot3_waffle.urdf');
    for (const r of ROBOTS) {
      scene.add(rovers[r]);
      const label = makeLabel();
      rovers[r].add(label);
      labels[r] = label;
    }

    // Smoothed render state, lerped toward target each frame.
    const render = {
      leader: { x: ROOM.width / 2, y: ROOM.depth / 2, yaw: 0, init: false },
      follower: { x: ROOM.width / 2, y: ROOM.depth / 2, yaw: 0, init: false },
    };

    // --- Animation loop ---
    let raf;
    let lastStaleEmit = 0;
    function animate() {
      raf = requestAnimationFrame(animate);
      const now = Date.now();
      const newStale = {};
      if (graphGroup) graphGroup.visible = showGraphRef.current;

      // Live route overlay (/leader/route): hold the last published node path until a new
      // LeaderRoute changes node_ids (event-driven topic — see ROUTE hold note above), and
      // clear it when no mission is running (INACTIVE_STATES) so a stale latched route does
      // not linger on the floor after the system returns to IDLE/DONE.
      // Rebuild geometry only when node_ids change; hide when empty or < 2 mappable nodes.
      const rt = routeRef.current;
      const active = !INACTIVE_STATES.has(systemStateRef.current);
      const key = active ? rt.ids.join(',') : '';
      if (key !== routeKey) {
        routeKey = key;
        const pts = [];
        if (key) {
          for (const id of rt.ids) {
            const n = nodeById.get(id);
            if (n) pts.push(n.x, 0.05, -n.y); // sit above the dim base graph (0.012)
          }
        }
        if (pts.length >= 6) {
          routeLine.geometry.setPositions(pts);
          routeLine.visible = true;
        } else {
          routeLine.visible = false;
        }
      }

      for (const r of ROBOTS) {
        const target = poseRef.current[r];
        const rs = render[r];
        // present = 실제 pose가 한 번이라도 도착했는가 (rxWall은 첫 메시지 전까지 0).
        // 미수신이면 렌더하지 않는다 — 안 그러면 render 기본값(ROOM 중앙)에 반투명
        // 고스트가 박혀 "인식 안 됐는데 중앙에 떠 있는" 버그가 된다.
        const present = !!target.rxWall;
        const age = present ? now - target.rxWall : Infinity;
        const stale = age > STALE_MS;
        // 미수신이거나 HIDE_MS 초과(소스 끊겨 한참 잔상)면 통째 숨김 — "없으면 안 뜬다".
        const visible = present && age <= HIDE_MS;

        rovers[r].visible = visible;     // 숨김=라벨(자식)도 함께 사라짐
        newStale[r] = visible && stale;  // 숨겨진/부재 로버는 STALE 태그도 안 띄움
        if (!visible) continue;

        // Snap on first valid pose so it doesn't slide in from the origin.
        if (!rs.init) {
          rs.x = target.x;
          rs.y = target.y;
          rs.yaw = target.yaw;
          rs.init = true;
        }

        // STALENESS GUARD: only interpolate toward target while fresh.
        if (!stale) {
          const t = 0.18; // smoothing factor (~time-constant @60fps)
          rs.x += (target.x - rs.x) * t;
          rs.y += (target.y - rs.y) * t;
          rs.yaw = lerpAngle(rs.yaw, target.yaw, t);
        }

        // Apply pose (see COORDINATE MAPPING note above).
        const p = rosToScene(rs.x, rs.y);
        rovers[r].position.set(p.x, 0, p.z);
        rovers[r].rotation.y = yawToSceneRot(rs.yaw);

        // Dim on STALE (링 제거 — 모델 투명도로만 표시).
        const opacity = stale ? 0.32 : 1.0;
        for (const mat of rovers[r].userData.dimmable) {
          mat.transparent = true;
          mat.opacity = opacity;
        }

        // 미니멀 태그: 이름만(STALE은 DOM 태그로 별도 표시).
        drawLabel(labels[r], r.toUpperCase());
        labels[r].material.opacity = stale ? 0.45 : 0.95;
      }

      // Queued feedback — keep the cube's OWN hue (subtle emissive brighten) and recolor its
      // name label (white → green). No jarring hue swap. Reflects the real /mission/queue.
      for (const pm of partMeshes) {
        const queued = queuedRef.current.has(pm.userData.partId);
        if (queued) {
          pm.userData.mat.emissive.copy(pm.userData.mat.color); // same hue, lit up
          pm.userData.mat.emissiveIntensity = 0.5;
        } else {
          pm.userData.mat.emissive.set('#000000');
          pm.userData.mat.emissiveIntensity = 0;
        }
        if (pm.userData.label) {
          pm.userData.label.material.color.set(queued ? '#5fd98a' : '#dfe4ea');
        }
      }

      // Surface STALE flags to React at most ~5Hz (overlay tags).
      if (
        now - lastStaleEmit > 200 &&
        (newStale.leader !== staleFlagsRef.current.leader ||
          newStale.follower !== staleFlagsRef.current.follower)
      ) {
        lastStaleEmit = now;
        staleFlagsRef.current = newStale;
        setStaleFlags(newStale);
      }

      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    // --- Resize ---
    const ro = new ResizeObserver(() => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (w === 0 || h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      routeMat.resolution.set(w, h); // keep px-space route linewidth correct
    });
    ro.observe(mount);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="twin">
      <div className="twin-bar">
        <span className="twin-title">Digital Twin View</span>
        <div className="twin-bar-btns">
          <button
            type="button"
            className="twin-reset"
            aria-pressed={showGraph}
            onClick={() => setShowGraph((v) => !v)}
          >
            Graph: {showGraph ? 'ON' : 'OFF'}
          </button>
          <button type="button" className="twin-reset" onClick={() => resetViewRef.current()}>
            Reset View
          </button>
        </div>
      </div>
      <div ref={mountRef} className="twin-canvas" />
      <div className="twin-overlay">
        {ROBOTS.map((r) =>
          staleFlags[r] ? (
            <span key={r} className="twin-stale-tag">
              ⌧ {r} STALE
            </span>
          ) : null
        )}
      </div>
    </div>
  );
}
