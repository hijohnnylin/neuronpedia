'use client';

import { driver, type DriveStep } from 'driver.js';
import 'driver.js/dist/driver.css';
import { useCallback, useEffect, useRef, useState } from 'react';
// Loaded after driver.css so our theme overrides win on equal specificity.
import {
  JLENS_CHAT_ID,
  JLENS_DEMOS_BAR_ID,
  JLENS_JACOBIAN_SPACE_ID,
  JLENS_RIGHT_BUTTONS_ELEMENT_ID,
  JLENS_STEER_COLUMNS_ID,
  JLENS_STEER_OUTPUT_ID,
  JLENS_STEER_PANEL_ID,
  JLENS_STEER_SPIDER_ID,
  JLENS_TOUR_GUIDE_BUTTON_ELEMENT_ID,
  JLENS_TOUR_MODEL_ID,
  JLENS_TOUR_SHARE_ID,
  JLENS_VERBAL_REPORT_SHARE_ID,
} from './jlens-tour-constants';
import './jlens-tour.css';
import { runWhenIdle } from '@/lib/utils/run-when-idle';
import { JLENS_BLOG_URL, JLENS_GITHUB_URL, JLENS_HF_URL, JLENS_PAPER_URL } from './jlens-urls';

// Module-scoped cleanup for a per-step DOM listener (the spider step's
// click-to-advance handler). Only one driver instance is ever live at a time
// (we destroy the previous before creating a new one), so a single slot is
// enough. Cleared when the step is deselected or the tour is destroyed.
let activeStepCleanup: (() => void) | null = null;

function clearStepCleanup() {
  activeStepCleanup?.();
  activeStepCleanup = null;
}

// Reason a tour run is being torn down. Set by the various exit paths
// (skip / close confirm / Done) and read in the main tour's `onDestroyed`
// hook to decide what to do next. All exit paths navigate the page to the
// Verbal Report demo so the user lands on the first demo rather than a blank
// free chat:
//   • 'skip'   → user clicked the footer "Close" early. Go to the Verbal Report
//                demo, then spawn a one-step hint that points at the Tutorial
//                button so they know how to relaunch.
//   • 'close'  → user confirmed the X close early. Go to the Verbal Report demo.
//   • 'finish' → user reached the end (Done). Go to the Verbal Report demo.
type ExitMode = 'skip' | 'close' | 'finish' | null;
let exitMode: ExitMode = null;

// Poll for an element to appear in the DOM, resolving with it once found or
// with `null` if it never shows up within `timeoutMs`. Used to hold the tour
// off until the first step's target has actually rendered — otherwise (e.g.
// right after the start-of-tour navigation, while the shared run is still
// loading) `driver.drive()` would spotlight nothing.
//
// `excludeEl` lets callers ignore a specific (stale) node: when the tour starts
// from an already-loaded demo, that demo's `#jlens-chat` is still mounted at the
// moment we begin polling. Since the start navigation is a non-blocking
// transition, resolving with that stale node would drive the tour before the
// tour share loads (and the node is then swapped out, leaving the spotlight with
// nothing). Passing the previously-mounted node here holds off until a *fresh*
// one (the tour share's, a distinct DOM node) renders.
function waitForElement(
  selector: string,
  { excludeEl = null, timeoutMs = 10000 }: { excludeEl?: Element | null; timeoutMs?: number } = {},
): Promise<Element | null> {
  return new Promise((resolve) => {
    const find = () => {
      const el = document.querySelector(selector);
      return el && el !== excludeEl ? el : null;
    };
    const existing = find();
    if (existing) {
      resolve(existing);
      return;
    }
    const start = Date.now();
    const interval = window.setInterval(() => {
      const el = find();
      if (el) {
        window.clearInterval(interval);
        resolve(el);
      } else if (Date.now() - start >= timeoutMs) {
        window.clearInterval(interval);
        resolve(null);
      }
    }, 50);
  });
}

// Class on the injected "Close" (skip) button. Lives in the popover footer;
// styled in `jlens-tour.css` so it doesn't pick up the primary next/prev
// button theme.
const SKIP_BTN_CLASS = 'jlens-tour-skip-btn';

// Steps (0-indexed) on which the injected "Close" button appears. Update as
// steps are added/removed.
const STEPS_TO_SHOW_SKIP_BTN = [0];

// Placeholder steps. Replace the copy / targets / interactions when building
// the real tour. Steps 2 & 3 spotlight elements that already exist in the
// JLens header; step 1 is a centered welcome popover.
function buildTourSteps(): DriveStep[] {
  return [
    {
      element: `#${JLENS_CHAT_ID}`,
      popover: {
        title: '👋 Welcome to the Jacobian Lens Tutorial',
        showButtons: ['next'],
        description: `
        <p>In this highlighted conversation, we asked an AI (<strong>Qwen</strong>) a question. Please read it.</p>
        <p>This question requires a thinking step: Qwen must first think of the animal that spins webs, <strong>spiders</strong>.</p><p>However, Qwen answers correctly without using any thinking tokens or mentioning spiders.</p><p><strong>Where is Qwen doing its hidden thinking?</strong></p>`,
      },
    },
    {
      element: `#${JLENS_JACOBIAN_SPACE_ID}`,
      popover: {
        title: '🕷️ Spiders Detected!',
        description: `
          <p>
            This is Qwen's <strong>Jacobian space</strong> (J-space): its hidden mental workspace that updates as we chat with it.</p><p>We sort it by count, which shows us that Qwen's most frequently occurring thought in this chat is "spiders".</p>
            <p>We use <strong>Jacobian lens</strong> (J-lens) to reveal an AI's J-space. J-lens are specifically "fitted" for each AI model.</p>
        `,
        side: 'bottom',
        align: 'start',
        showButtons: ['next', 'close'],
      },
    },
    {
      element: `#${JLENS_STEER_SPIDER_ID}`,
      // Re-enable interaction so the user can click the readout's "Swap" action
      // (or the row itself, which is rerouted to swap on this step).
      disableActiveInteraction: false,
      popover: {
        title: '🧑‍🔬 Testing the J-Space',
        description: `
          <p>
            How do we know if Qwen is actually using its J-space as a mental workspace to answer our question?</p>
            <p>To verify this, we'll intercept its thinking by swapping "spiders" with "ants" in its J-space, and observe if its response changes.
          </p>
          <p>Click "Swap" in the "spiders" row.</p>
        `,
        side: 'bottom',
        align: 'start',
        showButtons: ['close'],
      },
      // Advance once the user clicks the readout's swap action. Clicking either
      // the swap pill or the row itself fires the swap flow (see
      // `SidebarTokenRow`); both bubble to this native listener before React's
      // synthetic `stopPropagation` runs at the delegated root. Once clicked,
      // wait for the steer panel to mount so the next step has a target.
      onHighlighted: (element, _step, opts) => {
        clearStepCleanup();
        if (!element) return;
        const onClick = () => {
          clearStepCleanup();
          waitForElement(`#${JLENS_STEER_PANEL_ID}`).then(() => {
            if (opts.driver.isActive()) opts.driver.moveNext();
          });
        };
        element.addEventListener('click', onClick, { once: true });
        activeStepCleanup = () => element.removeEventListener('click', onClick);
      },
      onDeselected: () => {
        clearStepCleanup();
      },
    },
    {
      element: `#${JLENS_STEER_PANEL_ID}`,
      disableActiveInteraction: false,
      popover: {
        title: '🔀 Replace Spiders with Ants',
        description: `<p>We've entered "ants" as the thought to swap in for "spiders" in the J-space.</p>
        <p>Click <strong>Swap</strong> to proceed.</p>`,
        side: 'bottom',
        align: 'start',
        // The user advances by clicking "Swap"; keep only the close button.
        showButtons: ['close'],
      },
      // Advance once the user clicks "Swap" and the swapped output has rendered.
      // The click bubbles from the Swap button up to this native listener; we
      // then wait for the steered column's token chips to appear (the precached
      // swap result) so the next step spotlights both fully-rendered columns.
      onHighlighted: (element, _step, opts) => {
        clearStepCleanup();
        if (!element) return;
        const onClick = () => {
          clearStepCleanup();
          waitForElement(`#${JLENS_STEER_OUTPUT_ID} [data-token-position]`).then(() => {
            if (opts.driver.isActive()) opts.driver.moveNext();
          });
        };
        element.addEventListener('click', onClick, { once: true });
        activeStepCleanup = () => element.removeEventListener('click', onClick);
      },
      onDeselected: () => {
        clearStepCleanup();
      },
    },
    {
      element: `#${JLENS_STEER_COLUMNS_ID}`,
      popover: {
        title: '✅ Eight Legs ➡️ Six Legs',
        description: `<p>Replacing "spiders" with "ants" in Qwen's J-space changes its response from eight legs to six legs.</p><p>Because Qwen's response changed to match the number of legs that ants have, we can infer that Qwen is indeed using its J-space as a mental workspace to answer our question. Neat!</p>`,
        side: 'top',
        align: 'start',
        showButtons: ['next', 'close'],
      },
    },
    {
      element: `#${JLENS_DEMOS_BAR_ID}`,
      popover: {
        title: '📚 Demos + DIY',
        description: `<p>In this tutorial, we've defined a mental "workspace" very casually (or "causally"? 😉). The <a href="${JLENS_PAPER_URL}">Jacobian lens paper</a> has a rigorous definition of workspace with five specific characteristics.</p>
        <p>Click through these demos to learn how J-space fulfills each characteristic of a workspace.</p><p>Then, use Free Chat to run your own experiments.</p>`,
        side: 'bottom',
        align: 'center',
        showButtons: ['next', 'close'],
      },
    },
    {
      element: `#${JLENS_RIGHT_BUTTONS_ELEMENT_ID}`,
      popover: {
        title: '🧑‍🏫 Learn More',
        description: `
          <p>
            Finally, explore the technical details of J-space and J-lens.
          </p>
          <p>
          <ul>
          <li><a href="${JLENS_PAPER_URL}"><strong>Paper</strong></a>: Full technical report by Anthropic</li>
          <li><a href="${JLENS_BLOG_URL}"><strong>Blog</strong></a>: Overview of Jacobian lens</li>
          <li><a href="${JLENS_GITHUB_URL}"><strong>Code</strong></a>: Train and load Jacobian lens</li>
          <li><a href="${JLENS_HF_URL}"><strong>Pre-fitted Lens</strong></a>: 30+ lenses for base and chat models</li>
          </ul>
          </p>
        `,
        side: 'bottom',
        align: 'center',
        showButtons: ['next'],
        // Override the default Done behavior so we can run finish logic in
        // `onDestroyed`. Setting `exitMode = 'finish'` is the signal the
        // destroy hook reads.
        onNextClick: (_element, _step, opts) => {
          exitMode = 'finish';
          opts.driver.destroy();
        },
      },
    },
  ];
}

type JlensTourOptions = {
  // Soft-navigates the page to a new href (same `/[modelId]/jlens` route, so
  // the React tree — and the live driver overlay — stays mounted). Used at
  // tour start to open the featured share the steps are built around.
  navigate: (href: string) => void;
};

export function useJlensTour(options: JlensTourOptions) {
  const driverRef = useRef<ReturnType<typeof driver> | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // The step driver.js is currently spotlighting, exposed so the rest of the
  // JLens tree can react to specific steps. Kept in sync via the driver's
  // highlight/destroy hooks below.
  const [activeStep, setActiveStep] = useState<DriveStep | null>(null);

  // Destroy any live driver instance on unmount so a route change mid-tour
  // doesn't leave a dangling overlay.
  useEffect(
    () => () => {
      driverRef.current?.destroy();
      driverRef.current = null;
    },
    [],
  );

  // Drop the featured tour share and send the user to the Verbal Report demo.
  // Used by every exit path (early or after the last step) so leaving the
  // tutorial lands them on the first demo rather than the locked tour share.
  const goToVerbalReport = useCallback(() => {
    optionsRef.current.navigate(`/${JLENS_TOUR_MODEL_ID}/jlens?shareId=${JLENS_VERBAL_REPORT_SHARE_ID}`);
  }, []);

  // Spawn a one-step driver that points at the Tutorial button so users who
  // close the tour early still know how to bring it back later.
  const showGuideHighlight = useCallback(() => {
    driverRef.current?.destroy();
    const hint = driver({
      showProgress: false,
      allowClose: true,
      smoothScroll: true,
      popoverClass: 'jlens-driver-popover',
      steps: [
        {
          element: `#${JLENS_TOUR_GUIDE_BUTTON_ELEMENT_ID}`,
          popover: {
            title: '',
            description: `<p>Restart the tutorial by clicking <strong>Tutorial</strong>.</p>`,
            showButtons: ['next'],
            doneBtnText: 'Got it',
            side: 'bottom',
            align: 'center',
          },
        },
      ],
      onDestroyed: () => {
        if (driverRef.current === hint) driverRef.current = null;
      },
    });
    driverRef.current = hint;
    // Same rationale as the main tour: this hint is spawned right after a
    // navigation to the Verbal Report demo, so defer driver.js's body mutation
    // until that transition has committed.
    runWhenIdle(() => {
      if (driverRef.current !== hint) return;
      hint.drive();
    });
  }, []);

  const startTour = useCallback(() => {
    driverRef.current?.destroy();
    // Open the featured share every step is built around. Skip the navigation
    // if the page is already on it so we don't kick off a redundant round trip.
    const target = `/${JLENS_TOUR_MODEL_ID}/jlens?shareId=${JLENS_TOUR_SHARE_ID}`;
    // If we're navigating away from an already-loaded run, remember its
    // `#jlens-chat` node so we can wait for the tour share's (fresh) one below
    // rather than driving on the stale, about-to-be-unmounted element.
    let staleChat: Element | null = null;
    if (typeof window !== 'undefined' && `${window.location.pathname}${window.location.search}` !== target) {
      staleChat = document.querySelector(`#${JLENS_CHAT_ID}`);
      optionsRef.current.navigate(target);
    }
    // Reset any stale exit signal from a prior, abandoned tour run so the
    // new instance always starts in a known state.
    exitMode = null;
    const instance = driver({
      showProgress: false,
      allowClose: true,
      disableActiveInteraction: true,
      // Slightly lighter than driver.js's default 0.7 backdrop so the page
      // stays a bit more visible behind the spotlight.
      overlayOpacity: 0.5,
      // Block the two implicit "cancel" gestures so the only way out is via
      // the close button (which we intercept below). Outside-click and Esc
      // would otherwise destroy the tour silently.
      overlayClickBehavior: () => {},
      allowKeyboardControl: false,
      steps: buildTourSteps(),
      popoverClass: 'jlens-driver-popover',
      // Publish the active step as each spotlight begins so consumers (via
      // `JlensTourStepContext`) can adjust their UI for specific steps.
      onHighlightStarted: (_element, step) => {
        setActiveStep(step);
      },
      // Inject a "Close" button into the bottom-left of the early steps'
      // footers. driver.js fires this hook on every popover render (including
      // resize re-renders), so we guard against duplicates.
      onPopoverRender: (popover, opts) => {
        if (!STEPS_TO_SHOW_SKIP_BTN.includes(opts.state.activeIndex ?? 0)) return;
        const footer = popover.footer;
        if (!footer) return;
        if (footer.querySelector(`.${SKIP_BTN_CLASS}`)) return;
        const skipBtn = document.createElement('button');
        skipBtn.type = 'button';
        skipBtn.className = SKIP_BTN_CLASS;
        skipBtn.textContent = 'Close';
        skipBtn.addEventListener('click', () => {
          // eslint-disable-next-line no-alert
          if (confirm('Are you sure you want to exit the tutorial early?')) {
            exitMode = 'skip';
            opts.driver.destroy();
          }
        });
        footer.prepend(skipBtn);
      },
      // Custom close behavior on every step: confirm with the user before
      // tearing the tour down.
      onCloseClick: (_element, _step, opts) => {
        // eslint-disable-next-line no-alert
        const confirmed = window.confirm('Exit the tutorial?');
        if (!confirmed) return;
        exitMode = 'close';
        opts.driver.destroy();
      },
      onDestroyed: () => {
        const mode = exitMode;
        exitMode = null;
        clearStepCleanup();
        setActiveStep(null);
        if (driverRef.current === instance) driverRef.current = null;
        // Defer follow-up actions one frame so driver.js can finish tearing
        // down its overlay before we navigate the page / mount the next overlay.
        // Every exit path drops the featured tour share and sends the user to
        // the Verbal Report demo; skip additionally re-points at the Tutorial
        // button so users can relaunch.
        if (mode === 'skip') {
          requestAnimationFrame(() => {
            goToVerbalReport();
            showGuideHighlight();
          });
          return;
        }
        if (mode === 'close' || mode === 'finish') {
          requestAnimationFrame(() => goToVerbalReport());
        }
      },
    });
    driverRef.current = instance;
    // Wait for the first step's target to render before driving. Right after
    // the start-of-tour navigation the shared run is still loading, so
    // `#jlens-chat` (the first spotlight) may not exist yet — driving now would
    // spotlight nothing. When starting from another loaded run, its stale
    // `#jlens-chat` is still mounted, so ignore it and hold out for the tour
    // share's fresh node. Poll until it appears (or we time out), then start.
    waitForElement(`#${JLENS_CHAT_ID}`, { excludeEl: staleChat }).then(() => {
      // Bail if this instance was torn down / replaced while we were waiting.
      if (driverRef.current !== instance) return;
      // Defer the actual drive until the browser is idle. `drive()` mounts
      // driver.js's overlay/popover onto `document.body`; running it while the
      // start-of-tour navigation transition is still committing races React's
      // own body-child reconciliation and crashes on mobile (see `runWhenIdle`).
      runWhenIdle(() => {
        if (driverRef.current !== instance) return;
        instance.drive();
      });
    });
  }, [showGuideHighlight, goToVerbalReport]);

  // `startTour` kicks off the tour; `activeStep` is the currently-spotlighted
  // step (or `null`), for consumers that special-case their UI per step.
  return { startTour, activeStep };
}
