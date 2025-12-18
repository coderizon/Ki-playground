import {
  ARDUINO_SEND_COOLDOWN_MS,
  ARDUINO_SEND_THRESHOLD,
  BAR_COLORS,
} from '../constants.js';
import { probabilityList } from '../domRefs.js';
import { state } from '../state.js';
import { sendToArduino, isArduinoConnected } from '../bluetooth/arduino.js';
import { sendToMicrobit, isMicrobitConnected } from '../bluetooth/microbit.js';
import { sendToCalliope, isCalliopeConnected } from '../bluetooth/calliope.js';

export function renderProbabilities(
  probArray = state.lastPrediction,
  bestIndex = -1,
  names = state.classNames
) {
  const safeValues = names.map((_, idx) => {
    if (probArray && probArray[idx] !== undefined) return probArray[idx];
    return 0;
  });
  state.lastPrediction = safeValues;

  if (state.predict && bestIndex >= 0) {
    maybeSendBlePrediction(safeValues, bestIndex, names);
  }

  if (!probabilityList) return;

  probabilityList.innerHTML = '';

  names.forEach((name, idx) => {
    const value = safeValues[idx] || 0;
    const percent = Math.round(Math.max(0, Math.min(1, value)) * 100);
    const row = document.createElement('div');
    row.className = 'probability-row';
    if (idx === bestIndex && state.predict) {
      row.classList.add('is-top');
    }

    const label = document.createElement('div');
    label.className = 'probability-label';
    label.textContent = name || `Class ${idx + 1}`;

    const bar = document.createElement('div');
    bar.className = 'probability-bar';

    const fill = document.createElement('div');
    fill.className = 'probability-bar-fill';
    const { start, end } = getBarColors(idx);
    fill.style.setProperty('--bar-start', start);
    fill.style.setProperty('--bar-end', end);
    fill.style.width = `${percent}%`;

    const valueEl = document.createElement('span');
    valueEl.className = 'probability-value';
    valueEl.textContent = `${percent}%`;

    bar.appendChild(fill);
    bar.appendChild(valueEl);

    row.appendChild(label);
    row.appendChild(bar);

    probabilityList.appendChild(row);
  });
}

function maybeSendBlePrediction(probArray, bestIndex, names) {
  const hasAnyBleConnection =
    (state.arduinoConnected && isArduinoConnected()) ||
    (state.microbitConnected && isMicrobitConnected()) ||
    (state.calliopeConnected && isCalliopeConnected());
  if (!hasAnyBleConnection) return;
  if (!Array.isArray(probArray) || bestIndex < 0) return;

  const probability = probArray[bestIndex] || 0;
  if (probability < ARDUINO_SEND_THRESHOLD) return;

  const label = names[bestIndex] || `Class ${bestIndex + 1}`;
  const now = Date.now();
  if (label === state.lastSentLabel && now - state.lastSentAt < ARDUINO_SEND_COOLDOWN_MS) return;

  state.lastSentLabel = label;
  state.lastSentAt = now;
  const payload = `${label}:${Math.round(probability * 100)}`;
  if (state.arduinoConnected && isArduinoConnected()) {
    sendToArduino(payload);
  }
  if (state.microbitConnected && isMicrobitConnected()) {
    sendToMicrobit(payload);
  }
  if (state.calliopeConnected && isCalliopeConnected()) {
    sendToCalliope(payload);
  }
}

function getBarColors(idx) {
  const palette = BAR_COLORS[idx % BAR_COLORS.length];
  return { start: palette[0], end: palette[1] };
}
