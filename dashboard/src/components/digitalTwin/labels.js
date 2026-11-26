import * as THREE from 'three';

const LABEL_FONT = 'bold 30px sans-serif';
const LABEL_H = 64;
export const LABEL_PX2WORLD = 0.5 / 256; // fixed pixel->world scale: same glyph size everywhere

// Outlined text sprite (faces the camera, depthTest off) sized to the text width.
export function makeTextSprite(text, fill, px2world = LABEL_PX2WORLD) {
  const canvas = document.createElement('canvas');
  let ctx = canvas.getContext('2d');
  ctx.font = LABEL_FONT;
  const textW = Math.ceil(ctx.measureText(text).width);
  canvas.width = 16 + textW + 16;
  canvas.height = LABEL_H;
  ctx = canvas.getContext('2d'); // resizing resets the context; re-acquire
  ctx.font = LABEL_FONT;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(8,10,14,0.92)'; // outline instead of a background
  ctx.lineWidth = 6;
  ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
  ctx.fillStyle = fill;
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set(canvas.width * px2world, canvas.height * px2world, 1);
  return sp;
}

// Rover tag: a fixed-size canvas sprite floating above each rover; text redrawn on change.
export function makeRoverLabel() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 80;
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.6, 0.1875, 1); // keeps the 256:80 aspect
  sprite.position.y = 0.46;
  sprite.userData = { canvas, ctx: canvas.getContext('2d'), tex, text: '' };
  return sprite;
}

// Name text only, outlined for contrast.
export function drawRoverLabel(sprite, text) {
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
