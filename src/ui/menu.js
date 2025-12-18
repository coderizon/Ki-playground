import { modeMenu } from '../domRefs.js';
import { state } from '../state.js';

export function toggleModeMenu() {
  if (!modeMenu) return;
  modeMenu.classList.toggle('hidden');
  updateModeMenuActive();
}

export function closeModeMenu() {
  if (!modeMenu) return;
  modeMenu.classList.add('hidden');
}

export function updateModeMenuActive() {
  if (!modeMenu) return;
  Array.from(modeMenu.querySelectorAll('[data-mode]')).forEach((btn) => {
    btn.classList.toggle('active', btn.getAttribute('data-mode') === state.currentMode);
  });
}
