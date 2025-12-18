import { STOP_DATA_GATHER } from '../constants.js';
import { state } from '../state.js';
import { CAPTURE_VIDEO, PREVIEW_VIDEO, STATUS, previewSwitchCameraButton } from '../domRefs.js';

export const captureCanvas = document.createElement('canvas');
captureCanvas.className = 'overlay-canvas';

export function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

export function stopCurrentStream() {
  if (!state.currentStream) return;
  state.currentStream.getTracks().forEach((track) => track.stop());
  state.currentStream = null;
  state.videoPlaying = false;
  if (CAPTURE_VIDEO) {
    CAPTURE_VIDEO.srcObject = null;
  }
  if (PREVIEW_VIDEO) {
    PREVIEW_VIDEO.srcObject = null;
  }
}

function attachStreamToVideos(stream) {
  CAPTURE_VIDEO.srcObject = stream;
  PREVIEW_VIDEO.srcObject = stream;
  CAPTURE_VIDEO.addEventListener('loadeddata', function onLoad() {
    state.videoPlaying = true;
    CAPTURE_VIDEO.classList.remove('hidden');
    CAPTURE_VIDEO.removeEventListener('loadeddata', onLoad);
  });
}

export function updateSwitchButtonsLabel() {
  const targetLabel = state.preferredFacingMode === 'user' ? 'Außenkamera' : 'Selfie-Kamera';
  state.switchCameraButtons.forEach((btn) => {
    if (btn) btn.textContent = targetLabel;
  });
  if (previewSwitchCameraButton) {
    previewSwitchCameraButton.textContent = targetLabel;
  }
}

export async function enableCam(allowFallback = true) {
  if (!hasGetUserMedia()) {
    console.warn('getUserMedia() is not supported by your browser');
    return;
  }

  const facingConstraint =
    state.preferredFacingMode === 'environment' ? { exact: 'environment' } : 'user';
  const constraints = {
    video: {
      width: 640,
      height: 480,
      aspectRatio: { ideal: 4 / 3 },
      facingMode: facingConstraint,
    },
  };

  if (state.currentStream) {
    attachStreamToVideos(state.currentStream);
    return;
  }

  try {
    state.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    attachStreamToVideos(state.currentStream);
  } catch (err) {
    console.error(err);
    if (allowFallback && state.preferredFacingMode === 'environment') {
      state.preferredFacingMode = 'user';
      updateSwitchButtonsLabel();
      if (STATUS) {
        STATUS.innerText = 'Außenkamera nicht verfügbar, Selfie-Kamera aktiviert.';
      }
      return enableCam(false);
    }
    if (STATUS) {
      STATUS.innerText = 'Webcam konnte nicht gestartet werden.';
    }
  }
}

export function moveCaptureToSlot(idx) {
  const slot = state.captureSlots.find(
    (s) => parseInt(s.getAttribute('data-class-slot'), 10) === idx
  );
  if (!slot) return;

  if (CAPTURE_VIDEO.parentElement !== slot) {
    slot.innerHTML = '';
    slot.appendChild(CAPTURE_VIDEO);
  }

  if (captureCanvas.parentElement !== slot) {
    slot.appendChild(captureCanvas);
  }
}

export function openWebcamForClass(idx) {
  state.activeClassIndex = idx;
  state.webcamPanels.forEach((panel) => {
    panel.classList.toggle(
      'visible',
      parseInt(panel.getAttribute('data-class-panel'), 10) === idx
    );
  });
  moveCaptureToSlot(idx);
  enableCam();
  if (STATUS) {
    STATUS.innerText = `Webcam geöffnet für ${state.classNames[idx]}. Halte zum Aufnehmen.`;
  }
}

export function hideWebcamPanel(idx) {
  if (state.gatherDataState !== STOP_DATA_GATHER) {
    state.gatherDataState = STOP_DATA_GATHER;
  }
  const panel = state.webcamPanels.find(
    (p) => parseInt(p.getAttribute('data-class-panel'), 10) === idx
  );
  if (panel) {
    panel.classList.remove('visible');
  }
}

export function toggleCameraFacing() {
  state.preferredFacingMode = state.preferredFacingMode === 'user' ? 'environment' : 'user';
  stopCurrentStream();
  moveCaptureToSlot(state.activeClassIndex);
  enableCam();
  updateSwitchButtonsLabel();
  return state.preferredFacingMode === 'environment'
    ? 'Außenkamera aktiviert.'
    : 'Selfie-Kamera aktiviert.';
}
