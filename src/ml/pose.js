import { CAPTURE_VIDEO, GESTURE_OVERLAY, PREVIEW_VIDEO, STATUS } from '../domRefs.js';
import { captureCanvas } from '../camera/webcam.js';
import { STOP_DATA_GATHER } from '../constants.js';
import { state } from '../state.js';
import { resizeOverlay } from './overlay.js';
import { updateExampleCounts } from '../ui/classes.js';

const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const MP_VERSION = '0.10.8';
const FEATURE_SIZE = 99; // 33 Landmark-Punkte * 3 (x, y, z)
const SAMPLE_INTERVAL_MS = 120;
let poseLoopHandle = null;
let captureDrawingUtils = null;

export function resetPoseSamples() {
  state.poseSamples.length = 0;
}

export async function ensurePoseLandmarker() {
  if (state.poseLandmarker) return state.poseLandmarker;
  if (state.poseInitPromise) return state.poseInitPromise;

  state.poseInitPromise = (async () => {
    try {
      const vision = await import(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`
      );
      const fileset = await vision.FilesetResolver.forVisionTasks(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
      );
      const landmarker = await vision.PoseLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: POSE_MODEL_URL,
          delegate: 'CPU',
        },
        numPoses: 1,
        runningMode: 'VIDEO',
      });

      state.poseVision = vision;
      state.poseLandmarker = landmarker;

      if (GESTURE_OVERLAY) {
        const ctx = GESTURE_OVERLAY.getContext('2d');
        state.poseDrawingUtils = ctx ? new vision.DrawingUtils(ctx) : null;
      }
      if (captureCanvas) {
        const captureCtx = captureCanvas.getContext('2d');
        captureDrawingUtils = captureCtx ? new vision.DrawingUtils(captureCtx) : null;
      }

      if (STATUS) {
        STATUS.innerText = 'Pose Landmarker bereit.';
      }

      return landmarker;
    } catch (error) {
      console.error(error);
      state.poseLandmarker = null;
      state.poseInitPromise = null;
      if (STATUS) {
        STATUS.innerText = 'Pose Landmarker konnte nicht geladen werden.';
      }
      return null;
    }
  })();

  return state.poseInitPromise;
}

function drawSkeletonOnCanvas(landmarks) {
  if (!captureCanvas) return;
  const ctx = captureCanvas.getContext('2d');
  if (!ctx) return;

  const width = CAPTURE_VIDEO?.videoWidth || captureCanvas.width;
  const height = CAPTURE_VIDEO?.videoHeight || captureCanvas.height;

  if (width && height) {
    captureCanvas.width = width;
    captureCanvas.height = height;
  }

  ctx.clearRect(0, 0, captureCanvas.width || width || 0, captureCanvas.height || height || 0);

  if (!landmarks || !landmarks.length) return;

  const PoseLandmarker = state.poseVision?.PoseLandmarker;
  if (!PoseLandmarker || !state.poseVision) return;

  if (!captureDrawingUtils) {
    captureDrawingUtils = new state.poseVision.DrawingUtils(ctx);
  }

  captureDrawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: '#55c3ff',
    lineWidth: 3,
  });
  captureDrawingUtils.drawLandmarks(landmarks, {
    color: '#0066ff',
    lineWidth: 2,
  });
}

export function runPoseLoop() {
  if (poseLoopHandle || state.currentMode !== 'pose') return;

  const loop = async () => {
    if (state.currentMode !== 'pose') {
      stopPoseLoop();
      return;
    }

    poseLoopHandle = window.requestAnimationFrame(loop);

    const detection = await detectPoseLandmarks(CAPTURE_VIDEO);
    drawSkeletonOnCanvas(detection?.landmarks ?? null);

    if (!detection || detection.vector.length !== FEATURE_SIZE) return;
    if (state.gatherDataState === STOP_DATA_GATHER) return;

    const now = performance.now();
    if (now - state.poseLastSampleTs < SAMPLE_INTERVAL_MS) return;

    state.poseSamples.push({
      landmarks: detection.vector,
      labelId: state.gatherDataState,
    });
    state.poseLastSampleTs = now;

    if (state.examplesCount[state.gatherDataState] === undefined) {
      state.examplesCount[state.gatherDataState] = 0;
    }
    state.examplesCount[state.gatherDataState]++;
    updateExampleCounts();
  };

  poseLoopHandle = window.requestAnimationFrame(loop);
}

export function stopPoseLoop() {
  if (poseLoopHandle) {
    window.cancelAnimationFrame(poseLoopHandle);
    poseLoopHandle = null;
  }
  state.poseLastSampleTs = 0;
  drawSkeletonOnCanvas(null);
}

export async function predictPose() {
  if (!state.model) return null;
  const detection = await detectPoseLandmarks(PREVIEW_VIDEO);
  if (!detection || detection.vector.length !== FEATURE_SIZE) {
    clearPoseOverlay();
    return null;
  }

  drawPoseOverlay(detection.landmarks);

  const probabilities = tf.tidy(() => {
    const input = tf.tensor2d([detection.vector], [1, FEATURE_SIZE]);
    const prediction = state.model.predict(input).squeeze();
    return prediction.arraySync();
  });

  const bestIndex =
    probabilities.length > 0
      ? probabilities.reduce(
          (bestIdx, value, idx, arr) => (value > arr[bestIdx] ? idx : bestIdx),
          0
        )
      : -1;

  return { probabilities, bestIndex };
}

export async function trainPoseModel({ batchSize, epochs, learningRate, onEpochEnd }) {
  if (!state.poseSamples.length) {
    throw new Error('Keine Posen-Beispiele gesammelt.');
  }

  const outputUnits = Math.max(state.classNames.length, 1);

  if (state.model) {
    state.model.dispose();
  }

  state.model = buildPoseClassifier(outputUnits, learningRate);

  const xs = tf.tensor2d(
    state.poseSamples.map((sample) => sample.landmarks),
    [state.poseSamples.length, FEATURE_SIZE]
  );
  const labelTensor = tf.tensor1d(
    state.poseSamples.map((sample) => sample.labelId),
    'int32'
  );
  const ys = tf.oneHot(labelTensor, outputUnits);

  try {
    await state.model.fit(xs, ys, {
      shuffle: true,
      batchSize,
      epochs,
      callbacks: {
        onEpochEnd,
      },
    });
  } finally {
    xs.dispose();
    ys.dispose();
    labelTensor.dispose();
  }

  return state.model;
}

function buildPoseClassifier(outputUnits, learningRate) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [FEATURE_SIZE], units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: outputUnits, activation: 'softmax' }));

  const lr = sanitizeLearningRate(learningRate);
  model.compile({
    optimizer: tf.train.adam(lr),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

export async function detectPoseLandmarks(videoEl) {
  if (!videoEl || videoEl.readyState < 2 || !videoEl.videoWidth || !videoEl.videoHeight) {
    return null;
  }
  if (state.poseBusy) return null;

  const landmarker = await ensurePoseLandmarker();
  if (!landmarker) return null;

  state.poseBusy = true;
  try {
    const nowInMs = performance.now();
    const result = landmarker.detectForVideo(videoEl, nowInMs);
    const landmarks = result?.landmarks?.[0];
    if (!landmarks || !landmarks.length) return null;

    return { landmarks, vector: flattenLandmarks(landmarks) };
  } catch (error) {
    console.error(error);
    return null;
  } finally {
    state.poseBusy = false;
  }
}

function flattenLandmarks(landmarks = []) {
  const flat = [];
  for (let i = 0; i < landmarks.length; i++) {
    const point = landmarks[i];
    flat.push(point.x ?? 0);
    flat.push(point.y ?? 0);
    flat.push(point.z ?? 0);
  }
  return flat;
}

export function drawPoseOverlay(landmarks = []) {
  if (!GESTURE_OVERLAY) return;
  resizeOverlay();
  const ctx = GESTURE_OVERLAY.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, GESTURE_OVERLAY.width, GESTURE_OVERLAY.height);

  const PoseLandmarker = state.poseVision?.PoseLandmarker;
  if (!PoseLandmarker || !state.poseDrawingUtils) return;

  state.poseDrawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: '#55c3ff',
    lineWidth: 3,
  });
  state.poseDrawingUtils.drawLandmarks(landmarks, {
    color: '#0066ff',
    lineWidth: 2,
  });
}

function clearPoseOverlay() {
  if (!GESTURE_OVERLAY) return;
  const ctx = GESTURE_OVERLAY.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, GESTURE_OVERLAY.width, GESTURE_OVERLAY.height);
}

function sanitizeLearningRate(value) {
  const lr = Number(value);
  if (!Number.isFinite(lr) || lr <= 0) return 0.001;
  return lr;
}
