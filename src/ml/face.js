import { GESTURE_OVERLAY, PREVIEW_VIDEO, STATUS } from '../domRefs.js';
import { state } from '../state.js';
import { clearOverlay, resizeOverlay } from './overlay.js';
import { renderProbabilities } from '../ui/probabilities.js';

const BLENDSHAPE_LABELS_DE = {
  neutral: 'Neutral',
  browDownLeft: 'Augenbraue unten links',
  browDownRight: 'Augenbraue unten rechts',
  browInnerUp: 'Innere Braue hoch',
  browOuterUpLeft: 'Äußere Braue hoch links',
  browOuterUpRight: 'Äußere Braue hoch rechts',
  cheekPuff: 'Wangen aufblasen',
  cheekSquintLeft: 'Wangenkneifen links',
  cheekSquintRight: 'Wangenkneifen rechts',
  eyeBlinkLeft: 'Augenblinzeln links',
  eyeBlinkRight: 'Augenblinzeln rechts',
  eyeLookDownLeft: 'Blick nach unten links',
  eyeLookDownRight: 'Blick nach unten rechts',
  eyeLookInLeft: 'Blick nach innen links',
  eyeLookInRight: 'Blick nach innen rechts',
  eyeLookOutLeft: 'Blick nach außen links',
  eyeLookOutRight: 'Blick nach außen rechts',
  eyeLookUpLeft: 'Blick nach oben links',
  eyeLookUpRight: 'Blick nach oben rechts',
  eyeSquintLeft: 'Augen kneifen links',
  eyeSquintRight: 'Augen kneifen rechts',
  eyeWideLeft: 'Augen aufreißen links',
  eyeWideRight: 'Augen aufreißen rechts',
  jawForward: 'Kiefer vor',
  jawLeft: 'Kiefer links',
  jawOpen: 'Kiefer öffnen',
  jawRight: 'Kiefer rechts',
  mouthClose: 'Mund schließen',
  mouthFrownLeft: 'Mundwinkel unten links',
  mouthFrownRight: 'Mundwinkel unten rechts',
  mouthFunnel: 'Mund spitzen',
  mouthLeft: 'Mund nach links',
  mouthLowerDownLeft: 'Unterlippe runter links',
  mouthLowerDownRight: 'Unterlippe runter rechts',
  mouthPressLeft: 'Lippe pressen links',
  mouthPressRight: 'Lippe pressen rechts',
  mouthPucker: 'Mund küssen',
  mouthRight: 'Mund nach rechts',
  mouthRollLower: 'Unterlippe einrollen',
  mouthRollUpper: 'Oberlippe einrollen',
  mouthShrugLower: 'Unterlippe heben',
  mouthShrugUpper: 'Oberlippe heben',
  mouthSmileLeft: 'Lächeln links',
  mouthSmileRight: 'Lächeln rechts',
  mouthStretchLeft: 'Mund ziehen links',
  mouthStretchRight: 'Mund ziehen rechts',
  mouthUpperUpLeft: 'Oberlippe hoch links',
  mouthUpperUpRight: 'Oberlippe hoch rechts',
  noseSneerLeft: 'Nase rümpfen links',
  noseSneerRight: 'Nase rümpfen rechts',
  mouthDimpleLeft: 'Grübchen links',
  mouthDimpleRight: 'Grübchen rechts',
  tongueOut: 'Zunge raus',
};

const BLENDSHAPE_TOP10_WHITELIST = new Set([
  'mouthSmileLeft',
  'mouthSmileRight',
  'eyeBlinkLeft',
  'eyeBlinkRight',
  'browInnerUp',
  'browDownLeft',
  'browDownRight',
  'jawOpen',
  'mouthPucker',
  'cheekPuff',
  'tongueOut',
  'eyeWideLeft',
  'eyeWideRight',
  'mouthFrownLeft',
  'mouthFrownRight',
]);

let faceLoopHandle = null;

function translateBlendshape(name) {
  if (!name) return '';
  return BLENDSHAPE_LABELS_DE[name] || name;
}

export async function ensureFaceLandmarker() {
  if (state.faceLandmarker) return state.faceLandmarker;
  if (state.faceInitPromise) return state.faceInitPromise;

  state.faceInitPromise = (async () => {
    try {
      const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3');
      const fileset = await vision.FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
      );
      const landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        outputFaceBlendshapes: true,
        numFaces: 1,
      });
      state.faceLandmarker = landmarker;
      state.faceVision = vision;
      if (GESTURE_OVERLAY) {
        const ctx = GESTURE_OVERLAY.getContext('2d');
        state.faceDrawingUtils = new vision.DrawingUtils(ctx);
      }
      if (STATUS) {
        STATUS.innerText = 'Face Landmarker bereit.';
      }
      return landmarker;
    } catch (err) {
      console.error(err);
      state.faceLandmarker = null;
      if (STATUS) {
        STATUS.innerText = 'Face Landmarker konnte nicht geladen werden.';
      }
      return null;
    }
  })();

  return state.faceInitPromise;
}

export function runFaceLoop() {
  if (faceLoopHandle || state.currentMode !== 'face') return;

  const loop = async () => {
    if (state.currentMode !== 'face') {
      stopFaceLoop();
      return;
    }
    faceLoopHandle = window.requestAnimationFrame(loop);
    await runFaceStep();
  };

  faceLoopHandle = window.requestAnimationFrame(loop);
}

export function stopFaceLoop() {
  if (faceLoopHandle) {
    window.cancelAnimationFrame(faceLoopHandle);
    faceLoopHandle = null;
  }
  clearOverlay();
  renderProbabilities([], -1, []);
}

export async function runFaceStep() {
  if (state.faceBusy) return;
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
  state.faceBusy = true;
  try {
    const landmarker = await ensureFaceLandmarker();
    if (!landmarker) return;
    const nowInMs = performance.now();
    const result = landmarker.detectForVideo(PREVIEW_VIDEO, nowInMs);
    const landmarks = result?.faceLandmarks?.[0];
    if (!landmarks) {
      clearOverlay();
      renderProbabilities([], -1, []);
      return;
    }
    drawFaceOverlay(landmarks);

    const blendShapes = result?.faceBlendshapes?.[0]?.categories || [];
    if (blendShapes.length) {
      const sorted = [...blendShapes]
        .map((shape) => ({
          name: translateBlendshape(shape.displayName || shape.categoryName),
          rawName: shape.displayName || shape.categoryName,
          score: shape.score,
        }))
        .filter((shape) => BLENDSHAPE_TOP10_WHITELIST.has(shape.rawName))
        .sort((a, b) => b.score - a.score)
        .slice(0, 10);
      const names = sorted.map((s) => s.name);
      const probs = sorted.map((s) => s.score);
      const bestIndex =
        probs.length > 0
          ? probs.reduce((bestIdx, value, idx, arr) => (value > arr[bestIdx] ? idx : bestIdx), 0)
          : -1;
      renderProbabilities(probs, bestIndex, names);
    } else {
      renderProbabilities([], -1, []);
    }
  } catch (err) {
    console.error(err);
  } finally {
    state.faceBusy = false;
  }
}

function drawFaceOverlay(landmarks = []) {
  if (!GESTURE_OVERLAY) return;
  resizeOverlay();
  if (!state.faceDrawingUtils) return;
  const ctx = GESTURE_OVERLAY.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, GESTURE_OVERLAY.width, GESTURE_OVERLAY.height);

  const FaceLandmarker = state.faceVision?.FaceLandmarker;
  if (!FaceLandmarker) return;

  state.faceDrawingUtils.drawConnectors(
    landmarks,
    FaceLandmarker.FACE_LANDMARKS_TESSELATION,
    { color: '#C0C0C070', lineWidth: 1 }
  );
  state.faceDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {
    color: '#FF3030',
  });
  state.faceDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, {
    color: '#FF3030',
  });
  state.faceDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {
    color: '#30FF30',
  });
  state.faceDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, {
    color: '#30FF30',
  });
  state.faceDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
    color: '#E0E0E0',
  });
  state.faceDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
    color: '#E0E0E0',
  });
  state.faceDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, {
    color: '#FF3030',
  });
  state.faceDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, {
    color: '#30FF30',
  });
}
