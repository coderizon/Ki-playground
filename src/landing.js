const landingPage = document.getElementById('landing-page');
const appShell = document.getElementById('app-shell');
const sideMenu = document.getElementById('sideMenu');
const sideMenuBackdrop = document.getElementById('sideMenuBackdrop');
const sideMenuClose = document.querySelector('.side-menu-close');
const navToggles = document.querySelectorAll('.nav-toggle');
const navHome = document.querySelector('.side-menu-item[data-nav="home"]');

function startApp(targetMode) {
  if (!appShell || !landingPage) return;

  if (typeof window.setAppMode === 'function' && targetMode) {
    window.setAppMode(targetMode);
  }

  landingPage.classList.add('is-hidden');
  landingPage.setAttribute('aria-hidden', 'true');
  appShell.classList.remove('hidden');
  document.body.classList.remove('landing-active');
}

function showLandingPage() {
  if (!appShell || !landingPage) return;
  landingPage.classList.remove('is-hidden');
  landingPage.removeAttribute('aria-hidden');
  appShell.classList.add('hidden');
  document.body.classList.add('landing-active');
}

if (landingPage) {
  const startButtons = landingPage.querySelectorAll('[data-mode]');

  startButtons.forEach((button) => {
    button.addEventListener('click', (event) => {
      event.stopPropagation();
      const mode = button.getAttribute('data-mode');
      const availabilityNode = button.closest('[data-available]');
      const isAvailable = !availabilityNode || availabilityNode.getAttribute('data-available') !== 'false';

      if (button.getAttribute('aria-disabled') === 'true') {
        event.preventDefault();
        return;
      }

      if (!isAvailable) {
        alert('Dieser Modus ist noch in Arbeit.');
        return;
      }

      startApp(mode);
    });
  });

  landingPage.addEventListener('keydown', (event) => {
    const isActivationKey = event.key === 'Enter' || event.key === ' ';
    if (!isActivationKey || !(event.target instanceof Element)) return;
    const modeTarget = event.target.closest('[data-mode]');
    const isNativeButton = modeTarget instanceof HTMLButtonElement;
    if (modeTarget && !isNativeButton) {
      event.preventDefault();
      modeTarget.dispatchEvent(new Event('click', { bubbles: true }));
    }
  });
}

initMarqueeScroller();
initSideMenu();

function initMarqueeScroller() {
  const marquee = document.querySelector('.model-marquee');
  const track = marquee?.querySelector('.model-track');
  if (!marquee || !track) return;

  const isMobileMarquee = window.matchMedia('(max-width: 640px)').matches;
  const AUTO_SPEED = isMobileMarquee ? 0.01 : 0.02; // px per ms, auf Mobile extra langsam
  const SCROLL_DIRECTION = 1; // positiver Offset scrollt die Spur nach links (keine Lücke)
  let isDragging = false;
  let startX = 0;
  let startOffset = 0;
  let offset = 0;
  let lastTime = null;
  let autoScroll = true;
  let activePointerId = null;

  const loopWidth = () => track.scrollWidth / 2;

  const wrapOffset = () => {
    const half = loopWidth();
    if (!half) return;
    while (offset >= half) offset -= half;
    while (offset < 0) offset += half;
  };

  const applyTransform = () => {
    track.style.transform = `translate3d(${-offset}px, 0, 0)`;
  };

  wrapOffset();
  applyTransform();

  const step = (timestamp) => {
    if (lastTime === null) lastTime = timestamp;
    const delta = timestamp - lastTime;
    if (autoScroll && !isDragging) {
      offset += delta * AUTO_SPEED * SCROLL_DIRECTION;
      wrapOffset();
      applyTransform();
    }
    lastTime = timestamp;
    requestAnimationFrame(step);
  };

  requestAnimationFrame(step);

  marquee.addEventListener('pointerdown', (event) => {
    activePointerId = event.pointerId;
    isDragging = true;
    autoScroll = false;
    startX = event.clientX;
    startOffset = offset;
    marquee.classList.add('is-dragging');
  });

  marquee.addEventListener('pointermove', (event) => {
    if (!isDragging || event.pointerId !== activePointerId) return;
    event.preventDefault();
    const delta = event.clientX - startX;
    offset = startOffset - delta;
    wrapOffset();
    applyTransform();
  });

  const endDrag = (event) => {
    if (!isDragging || (event.pointerId && event.pointerId !== activePointerId)) return;
    isDragging = false;
    activePointerId = null;
    marquee.classList.remove('is-dragging');
    autoScroll = true;
    lastTime = null;
  };

  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);
  window.addEventListener('pointerleave', endDrag);

  const handleResize = () => {
    wrapOffset();
    applyTransform();
  };

  window.addEventListener('resize', handleResize);
}

function initSideMenu() {
  if (!sideMenu || !sideMenuBackdrop) return;

  const openMenu = () => {
    sideMenu.classList.remove('hidden');
    sideMenuBackdrop.classList.remove('hidden');
    requestAnimationFrame(() => {
      sideMenu.classList.add('is-open');
    });
  };

  const closeMenu = () => {
    sideMenu.classList.remove('is-open');
    sideMenuBackdrop.classList.add('hidden');
    setTimeout(() => {
      sideMenu.classList.add('hidden');
    }, 200);
  };

  navToggles.forEach((toggle) => {
    toggle.addEventListener('click', () => {
      openMenu();
    });
  });

  if (sideMenuClose) {
    sideMenuClose.addEventListener('click', closeMenu);
  }

  sideMenuBackdrop.addEventListener('click', closeMenu);

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeMenu();
    }
  });

  if (navHome) {
    navHome.addEventListener('click', () => {
      closeMenu();
      showLandingPage();
    });
  }

  window.showLandingPage = showLandingPage;
}
