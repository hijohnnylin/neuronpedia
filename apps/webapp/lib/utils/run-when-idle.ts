// Run `cb` once the browser is idle, falling back to a post-paint callback
// (two rAFs) where `requestIdleCallback` is unavailable (e.g. older mobile
// Chrome / Safari). On the server it runs `cb` synchronously.
//
// Primarily used by the JLens / NLA guided tours to defer driver.js's `drive()`
// — the point at which driver.js mounts its overlay/popover onto
// `document.body` — until React has finished committing. The App Router root
// layout renders `<body>`, so React owns and reconciles its direct children
// (which also include body-level `createPortal(..., document.body)` popups and
// Radix dialog portals). If driver.js mutates `<body>` while React is mid-commit
// (a `useTransition` navigation or a soft `history.replaceState` cache-hydrate),
// React's commit can try to remove/insert a body child that is no longer where
// its fiber tree expects, throwing
// `NotFoundError: Failed to execute 'removeChild'/'insertBefore' on 'Node'`,
// which Next surfaces as "Application error: a client-side exception". The
// overlap window only reliably opens on slow (mobile) devices, so this is what
// keeps the tours from crashing on first load there.
export function runWhenIdle(cb: () => void): void {
  if (typeof window === 'undefined') {
    cb();
    return;
  }
  const ric = (
    window as Window & {
      requestIdleCallback?: (callback: () => void, options?: { timeout?: number }) => number;
    }
  ).requestIdleCallback;
  if (typeof ric === 'function') {
    ric(() => cb(), { timeout: 500 });
  } else {
    // Two rAFs ≈ after the next paint, by which point any pending React commit
    // has flushed.
    window.requestAnimationFrame(() => window.requestAnimationFrame(cb));
  }
}
