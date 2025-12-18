import {
  connectArduino,
  isArduinoConnected,
  setArduinoConnectionListener,
} from './bluetooth/arduino.js';
import {
  connectMicrobit,
  isMicrobitConnected,
  setMicrobitConnectionListener,
} from './bluetooth/microbit.js';
import {
  connectCalliope,
  isCalliopeConnected,
  setCalliopeConnectionListener,
} from './bluetooth/calliope.js';
import { MODE_NAMES, STOP_DATA_GATHER } from './constants.js';
import {
  GESTURE_OVERLAY,
  PREVIEW_VIDEO,
  RESET_BUTTON,
  STATUS,
  TRAIN_BUTTON,
  bleArduinoCard,
  bleCalliopeCard,
  bleConnectButton,
  bleMicrobitCard,
  bleModal,
  bleModalBackdrop,
  bleModalClose,
  addClassButton,
  epochsInput,
  batchSizeInput,
  learningRateInput,
  modeLabel,
  modeMenu,
  mobileStepButtons,
} from './domRefs.js';
import { clearOverlay } from './ml/overlay.js';
import { ensureHandLandmarker, runGestureLoop, stopGestureLoop } from './ml/gesture.js';
import { ensureFaceLandmarker, runFaceLoop, stopFaceLoop } from './ml/face.js';
import { loadMobileNetFeatureModel, rebuildModel } from './ml/model.js';
import {
  handleCollectEnd,
  handleCollectStart,
  predictLoop,
  showPreview,
  resetTrainingProgress,
  trainAndPredict,
} from './ml/training.js';
import { state, disposeTrainingData } from './state.js';
import {
  enableCam,
  hideWebcamPanel,
  openWebcamForClass,
  toggleCameraFacing,
  updateSwitchButtonsLabel,
} from './camera/webcam.js';
import { renderProbabilities } from './ui/probabilities.js';
import {
  initializeExistingClasses,
  addNewClassCard,
  updateExampleCounts,
  unlockCapturePanels,
  resetClassCards,
} from './ui/classes.js';
import { toggleModeMenu, closeModeMenu, updateModeMenuActive } from './ui/menu.js';
import { setMobileStep } from './ui/steps.js';

const classCardHandlers = {
  onNameChange: () => renderProbabilities(state.lastPrediction),
  onOpenWebcam: (idx) => openWebcamForClass(idx),
  onCollectStart: handleCollectStart,
  onCollectEnd: handleCollectEnd,
  onSwitchCamera: () => {
    const message = toggleCameraFacing();
    if (STATUS) {
      STATUS.innerText = message;
    }
  },
  onClosePanel: (idx) => hideWebcamPanel(idx),
};

TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', resetApp);
const defaultTrainLabel = TRAIN_BUTTON ? TRAIN_BUTTON.textContent : 'Modell trainieren';

mobileStepButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    setMobileStep(btn.getAttribute('data-step-target'));
  });
});

initializeExistingClasses(classCardHandlers);
rebuildModel();
updateExampleCounts(true);
renderProbabilities([], -1, state.classNames);
setMode('image');
updateSwitchButtonsLabel();
loadMobileNetFeatureModel();
initHyperparamInputs();

if (addClassButton) {
  addClassButton.addEventListener('click', () => {
    addClassAndReset();
  });
  addClassButton.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      addClassAndReset();
    }
  });
}

if (modeLabel) {
  modeLabel.addEventListener('click', toggleModeMenu);
}

if (modeMenu) {
  modeMenu.addEventListener('click', (event) => {
    const target = event.target.closest('[data-mode]');
    const mode = target?.getAttribute('data-mode');
    if (mode) {
      setMode(mode);
      closeModeMenu();
    }
  });
}

document.addEventListener('click', (event) => {
  if (!modeMenu || !modeLabel) return;
  if (!modeMenu.contains(event.target) && !modeLabel.contains(event.target)) {
    closeModeMenu();
  }
});

initBleModal();

function initHyperparamInputs() {
  bindHyperparamInput(epochsInput, 'trainingEpochs', {
    fallback: 10,
    parse: (value) => parseInt(value, 10),
    min: 1,
  });
  bindHyperparamInput(batchSizeInput, 'trainingBatchSize', {
    fallback: 5,
    parse: (value) => parseInt(value, 10),
    min: 1,
  });
  bindHyperparamInput(learningRateInput, 'trainingLearningRate', {
    fallback: 0.001,
    parse: (value) => parseFloat(value),
    min: 0.000001,
  });
}

function bindHyperparamInput(element, stateKey, { fallback, parse, min }) {
  if (!element) return;

  const applyValue = () => {
    const parsed = parse(element.value);
    const valid = Number.isFinite(parsed) && parsed >= min ? parsed : fallback;
    state[stateKey] = valid;
    element.value = valid;
  };

  element.value = state[stateKey] ?? fallback;
  applyValue();
  element.addEventListener('change', applyValue);
  element.addEventListener('blur', applyValue);
}

function initBleModal() {
  if (!bleConnectButton || !bleModal) return;

  const closeModal = () => {
    bleModal.classList.add('hidden');
    bleModalBackdrop?.classList.add('hidden');
    document.body.classList.remove('modal-open');
  };

  const openModal = () => {
    bleModal.classList.remove('hidden');
    bleModalBackdrop?.classList.remove('hidden');
    document.body.classList.add('modal-open');
  };

  bleConnectButton.addEventListener('click', openModal);
  bleModalBackdrop?.addEventListener('click', closeModal);
  bleModalClose?.addEventListener('click', closeModal);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !bleModal.classList.contains('hidden')) {
      closeModal();
    }
  });

  const devices = [
    {
      button: bleArduinoCard,
      stateKey: 'arduinoConnected',
      setListener: setArduinoConnectionListener,
      isConnected: isArduinoConnected,
      connect: connectArduino,
      label: 'Arduino Uno R4',
    },
    {
      button: bleCalliopeCard,
      stateKey: 'calliopeConnected',
      setListener: setCalliopeConnectionListener,
      isConnected: isCalliopeConnected,
      connect: connectCalliope,
      label: 'Calliope Mini',
    },
    {
      button: bleMicrobitCard,
      stateKey: 'microbitConnected',
      setListener: setMicrobitConnectionListener,
      isConnected: isMicrobitConnected,
      connect: connectMicrobit,
      label: 'Micro:bit',
    },
  ];

  devices.forEach((device) => setupBleDeviceCard(device, closeModal));
}

function setupBleDeviceCard(
  { button, stateKey, setListener, isConnected, connect, label },
  closeModal
) {
  if (!button) return;
  const statusEl = button.querySelector('.ble-device-status');

  const setStatus = (text) => {
    if (statusEl) statusEl.textContent = text;
  };

  const updateState = (connected) => {
    state[stateKey] = connected;
    button.classList.toggle('is-connected', connected);
    button.disabled = connected;
    setStatus(connected ? `${label} verbunden` : 'Bereit');
    if (connected) closeModal();
  };

  setListener((connected) => {
    updateState(connected);
  });

  button.addEventListener('click', async () => {
    if (state[stateKey] || isConnected()) {
      updateState(true);
      return;
    }
    setStatus('Verbinde...');
    button.disabled = true;
    try {
      await connect();
    } catch (error) {
      console.error(error);
      setStatus('Verbindung fehlgeschlagen');
      button.disabled = false;
    }
  });
}

function addClassAndReset() {
  addNewClassCard(classCardHandlers);
  state.trainingCompleted = false;
  state.predict = false;
  state.previewReady = false;
  resetTrainingProgress();
  PREVIEW_VIDEO.classList.add('hidden');
  if (state.currentMode === 'image') {
    rebuildModel();
  } else if (state.currentMode === 'gesture' && state.model) {
    state.model.dispose();
    state.model = null;
  }
  renderProbabilities([], -1, state.classNames);
  if (STATUS) {
    STATUS.innerText = `Neue Klasse ${state.classNames[state.classNames.length - 1]} hinzugefügt.`;
  }
  updateSwitchButtonsLabel();
}

async function setMode(newMode) {
  const supportedModes = ['image', 'gesture', 'face'];
  if (!supportedModes.includes(newMode)) return;
  if (newMode === state.currentMode) {
    updateModeMenuActive();
    return;
  }
  stopGestureLoop();
  stopFaceLoop();
  state.currentMode = newMode;
  if (modeLabel) {
    modeLabel.textContent = MODE_NAMES[newMode] || newMode;
  }
  document.body.setAttribute('data-mode', newMode);

  state.predict = false;
  state.previewReady = false;
  state.trainingCompleted = false;
  state.gatherDataState = STOP_DATA_GATHER;
  state.lastPrediction = [];
  disposeTrainingData();
  state.examplesCount.length = 0;
  resetTrainingProgress();
  updateExampleCounts(true);
  resetClassCards(classCardHandlers);
  unlockCapturePanels();
  const clearedNames = newMode === 'face' ? [] : state.classNames;
  renderProbabilities([], -1, clearedNames);
  PREVIEW_VIDEO.classList.add('hidden');
  clearOverlay();
  toggleCaptureControls(true);
  setTrainButtonState(true);

  if (newMode === 'gesture') {
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
      clearOverlay();
    }
    setMobileStep('collect');
    enableCam();
    const landmarker = await ensureHandLandmarker();
    if (state.currentMode === 'gesture' && landmarker) {
      runGestureLoop();
    }
    if (state.currentMode === 'gesture') {
      if (STATUS) {
        STATUS.innerText = 'Gestenerkennung aktiv. Sammle Daten und trainiere.';
      }
      if (state.model) {
        state.model.dispose();
        state.model = null;
      }
    }
  } else if (newMode === 'face') {
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
      clearOverlay();
    }
    setMobileStep('preview');
    enableCam();
    preparePreviewForFace();
    const landmarker = await ensureFaceLandmarker();
    if (state.currentMode === 'face' && landmarker) {
      state.predict = true;
      renderProbabilities([], -1, []);
      runFaceLoop();
      if (STATUS) {
        STATUS.innerText = 'Gesichtserkennung aktiv. Vorschau läuft.';
      }
    }
  } else {
    if (STATUS) {
      STATUS.innerText = 'Bildklassifikation aktiv. Sammle Daten und trainiere.';
    }
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.add('hidden');
    }
    setMobileStep('collect');
    rebuildModel();
  }

  updateModeMenuActive();
}

function resetApp() {
  state.predict = false;
  state.previewReady = false;
  state.trainingCompleted = false;
  state.gatherDataState = STOP_DATA_GATHER;
  state.examplesCount.length = 0;
  resetTrainingProgress();
  disposeTrainingData();
  if (STATUS) {
    STATUS.innerText = 'No data collected';
  }
  PREVIEW_VIDEO.classList.add('hidden');
  unlockCapturePanels();
  if (state.currentMode === 'image') {
    rebuildModel();
    setTrainButtonState(true);
    toggleCaptureControls(true);
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.add('hidden');
    }
  } else if (state.currentMode === 'gesture') {
    if (state.model) {
      state.model.dispose();
      state.model = null;
    }
    setTrainButtonState(true);
    toggleCaptureControls(true);
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
      clearOverlay();
    }
  } else if (state.currentMode === 'face') {
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
      clearOverlay();
    }
    PREVIEW_VIDEO.classList.remove('hidden');
    state.predict = true;
    preparePreviewForFace();
    runFaceLoop();
    if (STATUS) {
      STATUS.innerText = 'Gesichtserkennung aktiv.';
    }
  }
  updateExampleCounts(true);
  state.lastPrediction = [];
  if (state.currentMode === 'face') {
    renderProbabilities([], -1, []);
    setMobileStep('preview');
  } else {
    renderProbabilities([], -1, state.classNames);
    setMobileStep('collect');
  }

  console.log('Tensors in memory: ' + tf.memory().numTensors);
}

window.setAppMode = setMode;

function toggleCaptureControls(enabled) {
  state.openWebcamButtons.forEach((btn) => {
    if (btn) btn.disabled = !enabled;
  });
  state.dataCollectorButtons.forEach((btn) => {
    if (btn) btn.disabled = !enabled;
  });
}

function setTrainButtonState(enabled, label = defaultTrainLabel) {
  if (!TRAIN_BUTTON) return;
  TRAIN_BUTTON.disabled = !enabled;
  TRAIN_BUTTON.textContent = label;
}

function preparePreviewForFace() {
  if (!PREVIEW_VIDEO) return;
  PREVIEW_VIDEO.classList.remove('hidden');
  const hasFrame = () =>
    PREVIEW_VIDEO.readyState >= 2 &&
    PREVIEW_VIDEO.videoWidth > 0 &&
    PREVIEW_VIDEO.videoHeight > 0;

  const markReady = () => {
    if (!hasFrame()) return;
    state.previewReady = true;
    PREVIEW_VIDEO.removeEventListener('loadeddata', markReady);
    PREVIEW_VIDEO.removeEventListener('canplay', markReady);
    PREVIEW_VIDEO.removeEventListener('playing', markReady);
  };

  if (hasFrame()) {
    markReady();
  } else {
    PREVIEW_VIDEO.addEventListener('loadeddata', markReady);
    PREVIEW_VIDEO.addEventListener('canplay', markReady);
    PREVIEW_VIDEO.addEventListener('playing', markReady);
  }
}
