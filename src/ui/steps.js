import { bodyEl, mobileStepButtons } from '../domRefs.js';

export function setMobileStep(step) {
  bodyEl.setAttribute('data-step', step);
  mobileStepButtons.forEach((btn) => {
    btn.classList.toggle('active', btn.getAttribute('data-step-target') === step);
  });
}
