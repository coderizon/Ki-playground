import { CAPTURE_VIDEO, STATUS, addClassButton, classesColumn } from '../domRefs.js';
import { state } from '../state.js';

const AUDIO_BACKGROUND_LABEL = '_background_noise_';
const AUDIO_BACKGROUND_SECONDS = 20;
const AUDIO_CLASS_SECONDS = 10;

export function initializeExistingClasses(handlers = {}) {
  const existingCards = document.querySelectorAll('.class-card');
  existingCards.forEach((card, idx) => {
    setupClassCard(card, idx, handlers);
  });
}

export function addNewClassCard(handlers = {}) {
  const newIndex = state.classNames.length;
  const newCard = buildClassCardElement(newIndex);
  if (classesColumn && addClassButton) {
    classesColumn.insertBefore(newCard, addClassButton);
  } else if (classesColumn) {
    classesColumn.appendChild(newCard);
  }
  setupClassCard(newCard, newIndex, handlers);
  state.examplesCount[newIndex] = 0;
  updateExampleCounts();
  return newIndex;
}

export function setupClassCard(card, idx, handlers = {}) {
  if (state.currentMode === 'audio') {
    setupAudioClassCard(card, idx, handlers);
    return;
  }

  const {
    onNameChange = () => {},
    onOpenWebcam = () => {},
    onCollectStart = () => {},
    onCollectEnd = () => {},
    onSwitchCamera = () => {},
    onClosePanel = () => {},
  } = handlers;

  const nameInput = card.querySelector('.class-name-input');
  const openBtn = card.querySelector('.open-webcam');
  const panel = card.querySelector('.webcam-panel');
  const closeBtn = panel ? panel.querySelector('.icon-close') : null;
  const switchBtn = panel ? panel.querySelector('.switch-camera') : null;
  const slot = card.querySelector('.capture-slot');
  const collectorBtn = card.querySelector('.dataCollector');
  const countChip = card.querySelector('.count-chip');

  if (!nameInput || !openBtn || !panel || !slot || !collectorBtn || !countChip) return;

  nameInput.setAttribute('data-class-index', idx);
  openBtn.setAttribute('data-class-index', idx);
  panel.setAttribute('data-class-panel', idx);
  slot.setAttribute('data-class-slot', idx);
  collectorBtn.setAttribute('data-1hot', idx);
  countChip.setAttribute('data-count-for', idx);

  const classLabel = nameInput.value || `Class ${idx + 1}`;
  state.classNames[idx] = classLabel;
  collectorBtn.setAttribute('data-name', classLabel);

  state.classNameInputs[idx] = nameInput;
  state.openWebcamButtons[idx] = openBtn;
  state.webcamPanels[idx] = panel;
  state.captureSlots[idx] = slot;
  state.dataCollectorButtons[idx] = collectorBtn;
  state.countChips[idx] = countChip;
  if (switchBtn) {
    state.switchCameraButtons[idx] = switchBtn;
  }

  const clearNameOnce = () => {
    if (nameInput.dataset.cleared === 'true') return;
    nameInput.dataset.cleared = 'true';
    nameInput.value = '';
    state.classNames[idx] = '';
    collectorBtn.setAttribute('data-name', '');
    onNameChange(idx, state.classNames[idx]);
  };

  nameInput.addEventListener('focus', clearNameOnce);
  nameInput.addEventListener('pointerdown', clearNameOnce);

  nameInput.addEventListener('input', () => {
    state.classNames[idx] = nameInput.value || `Class ${idx + 1}`;
    collectorBtn.setAttribute('data-name', state.classNames[idx]);
    if (STATUS) {
      STATUS.innerText = `Klasse ${idx + 1} benannt als ${state.classNames[idx]}.`;
    }
    onNameChange(idx, state.classNames[idx]);
  });

  openBtn.addEventListener('click', () => onOpenWebcam(idx));
  attachCollectorButtonListeners(collectorBtn, onCollectStart, onCollectEnd);

  if (switchBtn) {
    switchBtn.addEventListener('click', () => onSwitchCamera(idx));
  }

  if (closeBtn) {
    closeBtn.setAttribute('data-close-panel', idx);
    closeBtn.addEventListener('click', () => onClosePanel(idx));
  }
}

function setupAudioClassCard(card, idx, handlers = {}) {
  const { onNameChange = () => {}, onAudioCollect = () => Promise.resolve() } = handlers;

  const nameInput = card.querySelector('.class-name-input');
  const collectorBtn = card.querySelector('.dataCollector');
  const countChip = card.querySelector('.count-chip');

  if (!nameInput || !collectorBtn || !countChip) return;

  nameInput.setAttribute('data-class-index', idx);
  collectorBtn.setAttribute('data-1hot', idx);
  countChip.setAttribute('data-count-for', idx);

  const isBackground = idx === 0;
  const classLabel = isBackground ? AUDIO_BACKGROUND_LABEL : nameInput.value || `Class ${idx}`;
  nameInput.value = classLabel;
  state.classNames[idx] = classLabel;
  collectorBtn.setAttribute('data-name', classLabel);

  state.classNameInputs[idx] = nameInput;
  state.dataCollectorButtons[idx] = collectorBtn;
  state.countChips[idx] = countChip;

  // Not used in audio mode, but keep arrays aligned to avoid null crashes.
  state.openWebcamButtons[idx] = null;
  state.webcamPanels[idx] = null;
  state.captureSlots[idx] = null;
  state.switchCameraButtons[idx] = null;

  if (isBackground) {
    nameInput.disabled = true;
    nameInput.setAttribute('aria-disabled', 'true');
  }

  nameInput.addEventListener('input', () => {
    if (isBackground) return;
    if ((state.examplesCount[idx] || 0) > 0) {
      nameInput.value = state.classNames[idx] || nameInput.value;
      if (STATUS) {
        STATUS.innerText = 'Klassenname kann nach dem Sammeln nicht mehr geändert werden.';
      }
      return;
    }
    state.classNames[idx] = nameInput.value || `Class ${idx}`;
    collectorBtn.setAttribute('data-name', state.classNames[idx]);
    onNameChange(idx, state.classNames[idx]);
  });

  collectorBtn.textContent = 'Aufnehmen';
  collectorBtn.addEventListener('click', async () => {
    const secondsTotal = isBackground ? AUDIO_BACKGROUND_SECONDS : AUDIO_CLASS_SECONDS;

    state.dataCollectorButtons.forEach((btn) => {
      if (btn) btn.disabled = true;
    });
    if (addClassButton) addClassButton.classList.add('removed');

    try {
      await onAudioCollect(idx, {
        onTick: ({ tick, total }) => {
          collectorBtn.textContent = `Sekunde ${tick}/${total}`;
        },
      });
      updateExampleCounts();
      if (!isBackground && (state.examplesCount[idx] || 0) > 0) {
        nameInput.disabled = true;
        nameInput.setAttribute('aria-disabled', 'true');
      }
      if (STATUS) {
        STATUS.innerText = `Aufnahme abgeschlossen (${secondsTotal}s).`;
      }
    } catch (error) {
      console.error(error);
      if (STATUS) {
        STATUS.innerText = 'Aufnahme fehlgeschlagen oder abgebrochen.';
      }
    } finally {
      collectorBtn.textContent = 'Aufnehmen';
      state.dataCollectorButtons.forEach((btn) => {
        if (btn) btn.disabled = false;
      });
      if (addClassButton) addClassButton.classList.remove('removed');
    }
  });
}

function attachCollectorButtonListeners(btn, onCollectStart, onCollectEnd) {
  const supportsPointer = 'onpointerdown' in window;
  if (supportsPointer) {
    btn.addEventListener('pointerdown', onCollectStart, { passive: false });
    btn.addEventListener('pointerup', onCollectEnd);
    btn.addEventListener('pointerleave', onCollectEnd);
  } else {
    btn.addEventListener('mousedown', onCollectStart);
    btn.addEventListener('mouseup', onCollectEnd);
    btn.addEventListener('touchstart', onCollectStart, { passive: false });
    btn.addEventListener('touchend', onCollectEnd);
  }
}

function buildClassCardElement(idx) {
  if (state.currentMode === 'audio') {
    return buildAudioClassCardElement(idx);
  }

  const card = document.createElement('div');
  card.className = 'card class-card';
  card.innerHTML = `
    <div class="card-header">
      <div class="title-group editable">
        <input class="class-name-input" data-class-index="${idx}" value="Class ${idx + 1}" aria-label="Klassenname eingeben">
      </div>
      <span class="dots">⋮</span>
    </div>
    <p class="section-label">Beispiele hinzufügen:</p>
    <div class="action-row">
      <button class="open-webcam ghost" data-class-index="${idx}">Webcam</button>
    </div>
    <div class="webcam-panel" data-class-panel="${idx}">
      <div class="panel-top">
        <span>Webcam</span>
        <div class="panel-actions">
          <button class="ghost switch-camera" data-switch-camera aria-label="Kamera wechseln">Außenkamera</button>
          <button class="icon-close" data-close-panel="${idx}" aria-label="Panel schließen">×</button>
        </div>
      </div>
      <div class="count-row">
        <span class="count-chip" data-count-for="${idx}">0 Beispiele</span>
      </div>
      <div class="capture-slot" data-class-slot="${idx}"></div>
      <button class="dataCollector primary block" data-1hot="${idx}" data-name="Class ${idx + 1}">Zum Aufnehmen halten</button>
    </div>
  `;
  return card;
}

function buildAudioClassCardElement(idx) {
  const card = document.createElement('div');
  card.className = 'card class-card';
  const label = idx === 0 ? AUDIO_BACKGROUND_LABEL : `Class ${idx}`;
  card.innerHTML = `
    <div class="card-header">
      <div class="title-group editable">
        <input class="class-name-input" data-class-index="${idx}" value="${label}" aria-label="Klassenname eingeben">
      </div>
      <span class="dots">⋮</span>
    </div>
    <p class="section-label">Beispiele hinzufügen:</p>
    <div class="count-row">
      <span class="count-chip" data-count-for="${idx}">0 Beispiele</span>
    </div>
    <button class="dataCollector primary block" data-1hot="${idx}" data-name="${label}">Aufnehmen</button>
  `;
  return card;
}

export function updateExampleCounts(reset = false) {
  state.countChips.forEach((chip) => {
    if (!chip) return;
    const idx = parseInt(chip.getAttribute('data-count-for'), 10);
    const count = reset ? 0 : state.examplesCount[idx] || 0;
    chip.textContent = `${count} Beispiele`;
  });
}

export function resetClassCards(handlers = {}) {
  if (classesColumn) {
    classesColumn.querySelectorAll('.class-card').forEach((card) => card.remove());
  }

  state.classNames.length = 0;
  state.examplesCount.length = 0;
  state.classNameInputs.length = 0;
  state.openWebcamButtons.length = 0;
  state.webcamPanels.length = 0;
  state.captureSlots.length = 0;
  state.dataCollectorButtons.length = 0;
  state.countChips.length = 0;
  state.switchCameraButtons.length = 0;

  addNewClassCard(handlers);
  updateExampleCounts(true);
}

export function lockCapturePanels() {
  state.webcamPanels.forEach((panel) => panel?.classList?.remove('visible'));
  CAPTURE_VIDEO?.classList?.add('hidden');
  state.openWebcamButtons.forEach((btn) => {
    if (btn) btn.disabled = true;
  });
  state.dataCollectorButtons.forEach((btn) => {
    if (btn) btn.disabled = true;
  });
}

export function unlockCapturePanels() {
  state.openWebcamButtons.forEach((btn) => {
    if (btn) btn.disabled = false;
  });
  state.dataCollectorButtons.forEach((btn) => {
    if (btn) btn.disabled = false;
  });
  if (state.currentMode !== 'audio') {
    CAPTURE_VIDEO?.classList?.remove('hidden');
  }
}
