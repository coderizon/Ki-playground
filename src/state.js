import { STOP_DATA_GATHER } from './constants.js';

export const state = {
  currentStream: null,
  activeClassIndex: 0,
  previewReady: false,
  trainingCompleted: false,
  trainingInProgress: false,
  mobilenet: undefined,
  gatherDataState: STOP_DATA_GATHER,
  videoPlaying: false,
  trainingDataInputs: [],
  trainingDataOutputs: [],
  examplesCount: [],
  predict: false,
  model: null,
  preferredFacingMode: 'user',
  lastPrediction: [],
  currentMode: 'image',

  handLandmarker: null,
  handInitPromise: null,
  handBusy: false,
  handDrawingUtils: null,
  handVision: null,
  gestureSamples: [],
  gestureLastSampleTs: 0,
  faceLandmarker: null,
  faceInitPromise: null,
  faceBusy: false,
  faceDrawingUtils: null,
  faceVision: null,
  arduinoConnected: false,
  microbitConnected: false,
  calliopeConnected: false,
  trainingEpochs: 10,
  trainingBatchSize: 5,
  trainingLearningRate: 0.001,
  lastSentLabel: null,
  lastSentAt: 0,
  classNames: [],
  openWebcamButtons: [],
  webcamPanels: [],
  classNameInputs: [],
  captureSlots: [],
  dataCollectorButtons: [],
  countChips: [],
  switchCameraButtons: [],
};

export function disposeTrainingData() {
  for (let i = 0; i < state.trainingDataInputs.length; i++) {
    const input = state.trainingDataInputs[i];
    if (input && typeof input.dispose === 'function') {
      input.dispose();
    }
  }
  state.trainingDataInputs.length = 0;
  state.trainingDataOutputs.length = 0;
  state.gestureSamples.length = 0;
  state.gestureLastSampleTs = 0;
}
