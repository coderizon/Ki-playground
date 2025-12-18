import { GESTURE_OVERLAY, PREVIEW_VIDEO } from '../domRefs.js';

export function clearOverlay() {
  if (!GESTURE_OVERLAY) return;
  const ctx = GESTURE_OVERLAY.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, GESTURE_OVERLAY.width, GESTURE_OVERLAY.height);
}

export function resizeOverlay() {
  if (!GESTURE_OVERLAY || !PREVIEW_VIDEO) return;
  const w = PREVIEW_VIDEO.videoWidth;
  const h = PREVIEW_VIDEO.videoHeight;
  if (!w || !h) return;
  if (GESTURE_OVERLAY.width !== w || GESTURE_OVERLAY.height !== h) {
    GESTURE_OVERLAY.width = w;
    GESTURE_OVERLAY.height = h;
  }
}
