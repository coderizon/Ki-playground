import {
  STATUS,
  audioSpectrogramCanvas,
  audioTimeline,
  audioTimelineFill,
  audioTimelineLabel,
} from '../domRefs.js';
import { state } from '../state.js';
import { renderProbabilities } from '../ui/probabilities.js';

const BACKGROUND_LABEL = '_background_noise_';
const DEFAULT_GAIN = 2.5;
const AUDIO_CLASS_SECONDS = 10;
const AUDIO_BACKGROUND_SECONDS_DESKTOP = 20;
const AUDIO_BACKGROUND_SECONDS_MOBILE = 10;
const COLLECT_EXAMPLE_TIMEOUT_MS = 8000;
const MOBILE_COLLECT_RETRIES = 1;

let baseRecognizer = null;
let transferRecognizer = null;
let isListening = false;
let activeBurstAbort = null;

let originalGetUserMedia = null;
let isGetUserMediaPatched = false;
const activeAudioCleanups = new Set();
let sharedAudioContext = null;

export function getBackgroundLabel() {
  return BACKGROUND_LABEL;
}

export function getAudioCollectionSeconds(classIndex) {
  if (classIndex === 0) {
    return isLikelyMobileDevice()
      ? AUDIO_BACKGROUND_SECONDS_MOBILE
      : AUDIO_BACKGROUND_SECONDS_DESKTOP;
  }
  return AUDIO_CLASS_SECONDS;
}

export async function ensureAudioInitialized() {
  if (state.isAudioInitialized && state.audioRecognizer) {
    return state.audioRecognizer;
  }

  patchGetUserMediaForRawAudio();

  if (typeof window.speechCommands === 'undefined') {
    throw new Error(
      'speech-commands ist nicht geladen. Bitte @tensorflow-models/speech-commands einbinden.'
    );
  }

  if (!baseRecognizer) {
    baseRecognizer = window.speechCommands.create('BROWSER_FFT');
    if (STATUS) STATUS.textContent = 'Audio-Modell wird geladen...';
    await baseRecognizer.ensureModelLoaded();
  }

  if (!transferRecognizer) {
    transferRecognizer = baseRecognizer.createTransfer('ki-playground-audio');
  }

  state.audioRecognizer = transferRecognizer;
  state.isAudioInitialized = true;

  return transferRecognizer;
}

export async function collectBurstForClassIndex(classIndex, { onTick } = {}) {
  const recognizer = await ensureAudioInitialized();
  stopAudioPredictionLoop();

  const label =
    classIndex === 0 ? BACKGROUND_LABEL : state.classNames[classIndex] || `Class ${classIndex}`;
  const totalSeconds = getAudioCollectionSeconds(classIndex);
  const totalSamples = totalSeconds;

  const abortController = new AbortController();
  activeBurstAbort = abortController;

  setTimelineVisible(true);
  updateTimeline(0, totalSamples, label);

  const wakeLock = await requestWakeLock();

  try {
    for (let i = 0; i < totalSamples; i++) {
      if (abortController.signal.aborted || state.currentMode !== 'audio') {
        throw new Error('Audio-Aufnahme abgebrochen.');
      }

      const tick = i + 1;
      onTick?.({ tick, total: totalSamples, seconds: totalSeconds, label });
      updateTimeline(tick, totalSamples, label);

      // collectExample takes ~1s and grabs one training example per call.
      await collectExampleWithRetries(recognizer, label, {
        signal: abortController.signal,
        timeoutMs: COLLECT_EXAMPLE_TIMEOUT_MS,
        maxRetries: isLikelyMobileDevice() ? MOBILE_COLLECT_RETRIES : 0,
      });

      if (state.examplesCount[classIndex] === undefined) {
        state.examplesCount[classIndex] = 0;
      }
      state.examplesCount[classIndex]++;
      state.audioSamples[classIndex] = state.examplesCount[classIndex];
    }
  } finally {
    if (activeBurstAbort === abortController) activeBurstAbort = null;
    await releaseWakeLock(wakeLock);
    setTimelineVisible(false);
  }
}

export async function trainAudioModel({ epochs, onEpochEnd } = {}) {
  const recognizer = await ensureAudioInitialized();
  stopAudioPredictionLoop();

  const safeEpochs = sanitizeInteger(epochs, 20);

  const options = { epochs: safeEpochs };
  if (onEpochEnd) {
    options.callback = {
      onEpochEnd,
    };
  }

  return recognizer.train(options);
}

export function startAudioPredictionLoop() {
  if (!state.audioRecognizer) return;
  if (isListening) return;

  const labels = state.audioRecognizer.wordLabels();
  if (!Array.isArray(labels) || !labels.length) {
    renderProbabilities([], -1, state.classNames);
  }

  state.audioRecognizer.listen(
    (result) => {
      tf.tidy(() => {
        const scores = Array.from(result?.scores ?? []);
        const recognizerLabels = state.audioRecognizer?.wordLabels?.() ?? [];
        const probabilities = state.classNames.map((name) => {
          const idx = recognizerLabels.indexOf(name);
          return idx >= 0 ? scores[idx] ?? 0 : 0;
        });

        const bestIndex =
          probabilities.length > 0
            ? probabilities.reduce(
                (bestIdx, value, idx, arr) => (value > arr[bestIdx] ? idx : bestIdx),
                0
              )
            : -1;

        renderProbabilities(probabilities, bestIndex, state.classNames);

        if (result?.spectrogram && audioSpectrogramCanvas) {
          drawSpectrogram(audioSpectrogramCanvas, result.spectrogram);
        }
      });
    },
    {
      includeSpectrogram: true,
      probabilityThreshold: 0,
      invokeCallbackOnNoiseAndUnknown: true,
      overlapFactor: 0.5,
    }
  );

  isListening = true;
}

export function stopAudioLoop() {
  if (activeBurstAbort) {
    activeBurstAbort.abort();
    activeBurstAbort = null;
  }
  stopAudioPredictionLoop();
  stopAllPatchedAudioResources();
  setTimelineVisible(false);
  clearSpectrogram();
}

export function clearAudioExamples() {
  if (!state.audioRecognizer?.clearExamples) return;
  stopAudioPredictionLoop();
  state.audioRecognizer.clearExamples();
  state.audioSamples = [];
}

function stopAudioPredictionLoop() {
  if (!state.audioRecognizer || !isListening) return;
  try {
    state.audioRecognizer.stopListening();
  } catch (error) {
    console.warn(error);
  }
  isListening = false;
}

function patchGetUserMediaForRawAudio() {
  if (isGetUserMediaPatched) return;
  if (!navigator.mediaDevices?.getUserMedia) return;

  originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

  navigator.mediaDevices.getUserMedia = async (constraints) => {
    const wantsAudio = Boolean(constraints && typeof constraints === 'object' && constraints.audio);
    if (!wantsAudio) {
      return originalGetUserMedia(constraints);
    }

    const patchedConstraints = patchAudioConstraints(constraints);
    const rawStream = await originalGetUserMedia(patchedConstraints);
    return applyGainToStream(rawStream, DEFAULT_GAIN);
  };

  isGetUserMediaPatched = true;
}

function patchAudioConstraints(constraints) {
  const patched = { ...constraints };
  const audio = constraints.audio;
  const audioConstraints = audio === true ? {} : { ...(audio || {}) };
  audioConstraints.echoCancellation = false;
  audioConstraints.noiseSuppression = false;
  audioConstraints.autoGainControl = false;
  patched.audio = audioConstraints;
  return patched;
}

function applyGainToStream(rawStream, gainValue) {
  const audioContext = getOrCreateSharedAudioContext();
  if (!audioContext) return rawStream;
  const source = audioContext.createMediaStreamSource(rawStream);
  const gain = audioContext.createGain();
  gain.gain.value = gainValue;
  const destination = audioContext.createMediaStreamDestination();
  source.connect(gain);
  gain.connect(destination);

  const processedStream = destination.stream;
  let cleaned = false;
  const cleanup = () => {
    if (cleaned) return;
    cleaned = true;
    try {
      source.disconnect();
    } catch {}
    try {
      gain.disconnect();
    } catch {}
    try {
      rawStream?.getTracks?.().forEach((t) => t.stop());
    } catch {}
    try {
      processedStream?.getTracks?.().forEach((t) => t.stop());
    } catch {}
    activeAudioCleanups.delete(cleanup);
  };

  // If the consumer stops the track, also stop the original mic stream + context.
  processedStream.getTracks().forEach((track) => {
    track.addEventListener('ended', cleanup, { once: true });
  });

  activeAudioCleanups.add(cleanup);

  // Some browsers require a resume() after user gesture.
  audioContext.resume?.().catch(() => {});

  return processedStream;
}

function stopAllPatchedAudioResources() {
  for (const cleanup of Array.from(activeAudioCleanups)) {
    try {
      cleanup();
    } catch {}
  }
  activeAudioCleanups.clear();
  if (sharedAudioContext && sharedAudioContext.state !== 'closed') {
    sharedAudioContext.close?.().catch(() => {});
  }
  sharedAudioContext = null;
}

function setTimelineVisible(visible) {
  if (!audioTimeline) return;
  audioTimeline.classList.toggle('hidden', !visible);
}

function updateTimeline(tick, total, label) {
  if (audioTimelineFill) {
    const percent = total > 0 ? Math.round((tick / total) * 100) : 0;
    audioTimelineFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
  }
  if (audioTimelineLabel) {
    audioTimelineLabel.textContent = total
      ? `${label}: Sekunde ${Math.min(tick, total)}/${total}`
      : label;
  }
}

function clearSpectrogram() {
  if (!audioSpectrogramCanvas) return;
  const ctx = audioSpectrogramCanvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, audioSpectrogramCanvas.width, audioSpectrogramCanvas.height);
}

function drawSpectrogram(canvas, spectrogram) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const { frameSize, data } = spectrogram;
  if (!frameSize || !data || data.length < frameSize) return;

  const numFrames = Math.floor(data.length / frameSize);
  if (!numFrames) return;

  // Auto-resize once to match CSS size for crispness.
  const cssWidth = canvas.clientWidth || canvas.width;
  const cssHeight = canvas.clientHeight || canvas.height;
  if (canvas.width !== cssWidth) canvas.width = cssWidth;
  if (canvas.height !== cssHeight) canvas.height = cssHeight;

  ctx.imageSmoothingEnabled = false;

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;

  const width = canvas.width;
  const height = canvas.height;
  const image = ctx.createImageData(width, height);
  const out = image.data;

  for (let x = 0; x < width; x++) {
    const frameIndex = Math.min(numFrames - 1, Math.floor((x / width) * numFrames));
    for (let y = 0; y < height; y++) {
      const freqIndex = Math.min(
        frameSize - 1,
        Math.floor(((height - 1 - y) / height) * frameSize)
      );
      const raw = data[frameIndex * frameSize + freqIndex];
      const norm = Math.max(0, Math.min(1, (raw - min) / range));
      const [r, g, b] = paletteBluePurpleYellow(norm);

      const idx = (y * width + x) * 4;
      out[idx + 0] = r;
      out[idx + 1] = g;
      out[idx + 2] = b;
      out[idx + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
}

function paletteBluePurpleYellow(t) {
  // 0 -> blue, 0.5 -> purple, 1 -> yellow
  const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
  const lerp = (a, b, p) => a + (b - a) * p;

  const blue = [8, 60, 255];
  const purple = [160, 60, 255];
  const yellow = [255, 222, 64];

  if (t < 0.5) {
    const p = t / 0.5;
    return [clamp(lerp(blue[0], purple[0], p)), clamp(lerp(blue[1], purple[1], p)), clamp(lerp(blue[2], purple[2], p))];
  }

  const p = (t - 0.5) / 0.5;
  return [clamp(lerp(purple[0], yellow[0], p)), clamp(lerp(purple[1], yellow[1], p)), clamp(lerp(purple[2], yellow[2], p))];
}

function sanitizeInteger(value, fallback) {
  const parsed = parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return parsed;
}

function isLikelyMobileDevice() {
  if (navigator.userAgentData && typeof navigator.userAgentData.mobile === 'boolean') {
    return navigator.userAgentData.mobile;
  }
  const ua = navigator.userAgent || '';
  return /Android|iPhone|iPad|iPod|Mobile|IEMobile|BlackBerry|Opera Mini/i.test(ua);
}

function getOrCreateSharedAudioContext() {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) return null;
  if (sharedAudioContext && sharedAudioContext.state !== 'closed') {
    return sharedAudioContext;
  }
  sharedAudioContext = new AudioContextCtor({ latencyHint: 'interactive' });
  return sharedAudioContext;
}

async function requestWakeLock() {
  const wakeLockApi = navigator.wakeLock;
  if (!wakeLockApi?.request) return null;
  try {
    return await wakeLockApi.request('screen');
  } catch (error) {
    console.warn('[Audio] wakeLock request failed', error);
    return null;
  }
}

async function releaseWakeLock(wakeLock) {
  if (!wakeLock?.release) return;
  try {
    await wakeLock.release();
  } catch (error) {
    console.warn('[Audio] wakeLock release failed', error);
  }
}

async function collectExampleWithRetries(
  recognizer,
  label,
  { signal, timeoutMs, maxRetries } = {}
) {
  const retries = sanitizeInteger(maxRetries, 0);
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (signal?.aborted) {
      throw new Error('Audio-Aufnahme abgebrochen.');
    }
    try {
      await withTimeout(
        recognizer.collectExample(label),
        timeoutMs,
        'Audio-Aufnahme dauert zu lange. Bitte erneut versuchen.'
      );
      return;
    } catch (error) {
      if (attempt >= retries) throw error;
      console.warn('[Audio] collectExample failed, retrying...', error);
      try {
        recognizer.stopListening?.();
      } catch {}
      stopAllPatchedAudioResources();
      await sleep(250);
    }
  }
}

function withTimeout(promise, timeoutMs, timeoutMessage) {
  const ms = sanitizeInteger(timeoutMs, 0);
  if (!ms) return promise;

  let timeoutId = null;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = window.setTimeout(() => {
      reject(new Error(timeoutMessage));
    }, ms);
  });

  return Promise.race([promise, timeoutPromise]).finally(() => {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
  });
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
