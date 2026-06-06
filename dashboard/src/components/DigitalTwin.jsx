import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { useRobotState, ROBOTS } from '../ros/useRobotState';
import { useDiagnostics } from '../ros/useDiagnostics';
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

function makeMapLabel(text, css, x, y) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(20,24,30,0.85)';
  roundRect(ctx, 4, 4, canvas.width - 8, canvas.height - 8, 10); ctx.fill();
  ctx.fillStyle = css; ctx.fillRect(4, 4, 8, canvas.height - 8);
  ctx.fillStyle = '#e7eaee'; ctx.font = 'bold 30px sans-serif'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 20, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas); tex.minFilter = THREE.LinearFilter;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set(0.72, 0.18, 1);
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

  // Keep latest diagnostics-derived state accessible inside the rAF loop.
  const getStatusRef = useRef(getStatus);
  getStatusRef.current = getStatus;

  const resetViewRef = useRef(() => {});
  // STALE flags surfaced to the DOM overlay (set from rAF, read on render via state).
  const [staleFlags, setStaleFlags] = useState({ leader: false, follower: false });
  const staleFlagsRef = useRef(staleFlags);

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
    fetch('/map.json')
      .then((r) => r.json())
      .then((map) => addMapElements(mapGroup, map))
      .catch((e) => console.warn('[twin] /map.json 로드 실패:', e));

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

      for (const r of ROBOTS) {
        const target = poseRef.current[r];
        const rs = render[r];
        const age = target.rxWall ? now - target.rxWall : Infinity;
        const stale = age > STALE_MS || !target.rxWall;
        newStale[r] = stale;

        // Snap on first valid pose so it doesn't slide in from the origin.
        if (!rs.init && target.rxWall) {
          rs.x = target.x;
          rs.y = target.y;
          rs.yaw = target.yaw;
          rs.init = true;
        }

        // STALENESS GUARD: only interpolate toward target while fresh.
        if (!stale && rs.init) {
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
    });
    ro.observe(mount);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
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
        <button type="button" className="twin-reset" onClick={() => resetViewRef.current()}>
          Reset View
        </button>
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
