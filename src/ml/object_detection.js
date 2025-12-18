import { GESTURE_OVERLAY, PREVIEW_VIDEO, STATUS } from '../domRefs.js';
import { state } from '../state.js';
import { clearOverlay, resizeOverlay } from './overlay.js';
import { renderProbabilities } from '../ui/probabilities.js';

const MP_VERSION = '0.10.8';
const OBJECT_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite';
const MAX_RESULTS = 5;
const SCORE_THRESHOLD = 0.5;
const BOX_LINE_WIDTH = 6;
const BOX_CORNER_RADIUS = 14;
const LABEL_FONT = '600 15px DM Sans, Arial, sans-serif';
const LABEL_PADDING_X = 8;
const LABEL_PADDING_Y = 6;
const LABEL_GAP = 8;
const LABEL_ACCENT_WIDTH = 6;

let objectLoopHandle = null;

export async function ensureObjectDetector() {
  if (state.objectDetector) return state.objectDetector;
  if (state.objectInitPromise) return state.objectInitPromise;

  state.objectInitPromise = (async () => {
    try {
      const vision = await import(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`
      );
      const fileset = await vision.FilesetResolver.forVisionTasks(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
      );
      const detector = await vision.ObjectDetector.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: OBJECT_MODEL_URL,
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        maxResults: MAX_RESULTS,
        scoreThreshold: SCORE_THRESHOLD,
      });

      state.objectDetector = detector;

      if (STATUS) {
        STATUS.innerText = 'Object Detector bereit.';
      }

      return detector;
    } catch (err) {
      console.error(err);
      state.objectDetector = null;
      state.objectInitPromise = null;
      if (STATUS) {
        STATUS.innerText = 'Object Detector konnte nicht geladen werden.';
      }
      return null;
    }
  })();

  return state.objectInitPromise;
}

export function runObjectLoop() {
  if (objectLoopHandle || state.currentMode !== 'object_detection') return;

  const loop = async () => {
    if (state.currentMode !== 'object_detection') {
      stopObjectLoop();
      return;
    }
    objectLoopHandle = window.requestAnimationFrame(loop);
    await runObjectStep();
  };

  objectLoopHandle = window.requestAnimationFrame(loop);
}

export function stopObjectLoop() {
  if (objectLoopHandle) {
    window.cancelAnimationFrame(objectLoopHandle);
    objectLoopHandle = null;
  }
  clearOverlay();
  renderProbabilities([], -1, []);
}

export async function runObjectStep() {
  if (state.currentMode !== 'object_detection') return;
  if (state.objectBusy) return;
  if (!state.previewReady) return;
  if (!GESTURE_OVERLAY) return;
  if (
    !PREVIEW_VIDEO ||
    PREVIEW_VIDEO.readyState < 2 ||
    !PREVIEW_VIDEO.videoWidth ||
    !PREVIEW_VIDEO.videoHeight
  ) {
    return;
  }

  state.objectBusy = true;
  try {
    const detector = await ensureObjectDetector();
    if (!detector) return;
    if (state.currentMode !== 'object_detection') return;

    const nowInMs = performance.now();
    const result = detector.detectForVideo(PREVIEW_VIDEO, nowInMs);
    const detections = result?.detections || [];

    if (state.currentMode !== 'object_detection') return;
    if (!detections.length) {
      clearOverlay();
      renderProbabilities([], -1, []);
      return;
    }

    drawObjectOverlay(detections);

    const topCategories = detections
      .map((d) => d?.categories?.[0] || null)
      .filter(Boolean)
      .slice(0, MAX_RESULTS);
    const names = topCategories.map((c) => c.displayName || c.categoryName || 'Objekt');
    const probs = topCategories.map((c) => c.score ?? 0);
    const bestIndex =
      probs.length > 0
        ? probs.reduce(
            (bestIdx, value, idx, arr) => (value > arr[bestIdx] ? idx : bestIdx),
            0
          )
        : -1;
    renderProbabilities(probs, bestIndex, names);
  } catch (err) {
    console.error(err);
  } finally {
    state.objectBusy = false;
  }
}

function drawObjectOverlay(detections = []) {
  if (!GESTURE_OVERLAY) return;
  resizeOverlay();
  const ctx = GESTURE_OVERLAY.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, GESTURE_OVERLAY.width, GESTURE_OVERLAY.height);

  const width = GESTURE_OVERLAY.width || 0;
  const height = GESTURE_OVERLAY.height || 0;
  if (!width || !height) return;

  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.font = LABEL_FONT;
  ctx.textBaseline = 'top';

  detections.slice(0, MAX_RESULTS).forEach((detection, idx) => {
    const bbox = detection?.boundingBox;
    if (!bbox) return;

    const raw = toCanvasBox(bbox, width, height);
    const { y, w, h } = raw;
    const x = mirrorX(raw.x, w, width);
    if (!w || !h) return;

    const style = getDetectionStyle(idx);
    drawBoundingBox(ctx, { x, y, w, h }, style);

    const category = detection?.categories?.[0];
    const label = category?.displayName || category?.categoryName || 'Objekt';
    const score =
      typeof category?.score === 'number' && Number.isFinite(category.score) ? category.score : null;
    const text = score === null ? label : `${label} ${Math.round(score * 100)}%`;

    drawLabel(ctx, text, x, y, width, height, style.accent);
  });
}

function mirrorX(x, w, canvasW) {
  const mirrored = canvasW - x - w;
  const maxX = Math.max(0, canvasW - w);
  return Math.max(0, Math.min(mirrored, maxX));
}

function getDetectionStyle(idx) {
  const accent = idx === 0 ? '#3f73ff' : '#55c3ff';
  const fill = idx === 0 ? 'rgba(63, 115, 255, 0.18)' : 'rgba(85, 195, 255, 0.12)';
  return { accent, fill };
}

function drawBoundingBox(ctx, { x, y, w, h }, { accent, fill }) {
  const radius = Math.max(6, Math.min(BOX_CORNER_RADIUS, w / 5, h / 5));

  ctx.save();

  if (fill) {
    ctx.fillStyle = fill;
    roundedRect(ctx, x, y, w, h, radius);
    ctx.fill();
  }

  ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
  ctx.shadowBlur = 10;
  ctx.shadowOffsetY = 2;

  ctx.lineWidth = BOX_LINE_WIDTH + 2;
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)';
  roundedRect(ctx, x, y, w, h, radius);
  ctx.stroke();

  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.shadowOffsetY = 0;

  ctx.lineWidth = BOX_LINE_WIDTH;
  ctx.strokeStyle = accent;
  roundedRect(ctx, x, y, w, h, radius);
  ctx.stroke();

  ctx.restore();
}

function roundedRect(ctx, x, y, w, h, r) {
  const radius = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') {
    ctx.roundRect(x, y, w, h, radius);
    return;
  }
  const right = x + w;
  const bottom = y + h;
  ctx.moveTo(x + radius, y);
  ctx.lineTo(right - radius, y);
  ctx.quadraticCurveTo(right, y, right, y + radius);
  ctx.lineTo(right, bottom - radius);
  ctx.quadraticCurveTo(right, bottom, right - radius, bottom);
  ctx.lineTo(x + radius, bottom);
  ctx.quadraticCurveTo(x, bottom, x, bottom - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
}

function toCanvasBox(bbox, canvasW, canvasH) {
  const boxW = Number(bbox.width ?? 0);
  const boxH = Number(bbox.height ?? 0);

  const hasOrigin = bbox.originX !== undefined || bbox.originY !== undefined;
  const hasCenter = bbox.xCenter !== undefined || bbox.yCenter !== undefined;

  let originX = Number(bbox.originX ?? 0);
  let originY = Number(bbox.originY ?? 0);

  if (!hasOrigin && hasCenter) {
    const xCenter = Number(bbox.xCenter ?? 0);
    const yCenter = Number(bbox.yCenter ?? 0);
    originX = xCenter - boxW / 2;
    originY = yCenter - boxH / 2;
  }

  const isNormalized =
    boxW >= 0 &&
    boxW <= 1 &&
    boxH >= 0 &&
    boxH <= 1 &&
    originX >= -1 &&
    originX <= 1 &&
    originY >= -1 &&
    originY <= 1;

  const x = isNormalized ? originX * canvasW : originX;
  const y = isNormalized ? originY * canvasH : originY;
  const w = isNormalized ? boxW * canvasW : boxW;
  const h = isNormalized ? boxH * canvasH : boxH;

  const clampedW = Math.max(0, Math.min(w, canvasW));
  const clampedH = Math.max(0, Math.min(h, canvasH));
  const clampedX = Math.max(0, Math.min(x, canvasW - clampedW));
  const clampedY = Math.max(0, Math.min(y, canvasH - clampedH));

  return { x: clampedX, y: clampedY, w: clampedW, h: clampedH };
}

function drawLabel(ctx, text, x, y, canvasW, canvasH, accentColor) {
  if (!text) return;
  const maxTextWidth = Math.max(0, canvasW - (LABEL_PADDING_X * 2 + LABEL_ACCENT_WIDTH + 16));
  const fittedText = fitTextToWidth(ctx, text, maxTextWidth);
  const metrics = ctx.measureText(fittedText);
  const textW = Math.ceil(metrics.width);
  const textH = 18;

  const labelW = textW + LABEL_PADDING_X * 2 + LABEL_ACCENT_WIDTH;
  const labelH = textH + LABEL_PADDING_Y * 2;

  const maxX = Math.max(0, canvasW - labelW);
  const labelX = Math.max(0, Math.min(x, maxX));

  const aboveY = y - labelH - LABEL_GAP;
  const preferY = aboveY >= 0 ? aboveY : y + LABEL_GAP;
  const labelY = Math.max(0, Math.min(preferY, Math.max(0, canvasH - labelH)));

  const radius = Math.max(8, Math.min(12, labelH / 2));

  ctx.save();
  ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
  ctx.shadowBlur = 10;
  ctx.shadowOffsetY = 2;

  ctx.fillStyle = 'rgba(15, 17, 21, 0.78)';
  roundedRect(ctx, labelX, labelY, labelW, labelH, radius);
  ctx.fill();

  ctx.fillStyle = accentColor;
  roundedRect(ctx, labelX, labelY, LABEL_ACCENT_WIDTH, labelH, radius);
  ctx.fill();

  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.shadowOffsetY = 0;

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
  roundedRect(ctx, labelX, labelY, labelW, labelH, radius);
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.fillText(fittedText, labelX + LABEL_ACCENT_WIDTH + LABEL_PADDING_X, labelY + LABEL_PADDING_Y);
  ctx.restore();
}

function fitTextToWidth(ctx, text, maxWidth) {
  if (!maxWidth || maxWidth <= 0) return '';
  if (ctx.measureText(text).width <= maxWidth) return text;
  const ellipsis = '…';
  let start = 0;
  let end = text.length;
  while (start < end) {
    const mid = Math.floor((start + end) / 2);
    const candidate = text.slice(0, mid).trimEnd() + ellipsis;
    if (ctx.measureText(candidate).width <= maxWidth) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  const best = Math.max(1, end - 1);
  return text.slice(0, best).trimEnd() + ellipsis;
}
