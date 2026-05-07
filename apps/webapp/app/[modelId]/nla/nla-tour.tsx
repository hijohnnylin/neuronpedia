'use client';

import { driver, type DriveStep } from 'driver.js';
import 'driver.js/dist/driver.css';
import { useCallback, useEffect, useRef } from 'react';
// Loaded after driver.css so our theme overrides win on equal specificity.
import {
  NLA_DETAILS_ELEMENT_ID,
  NLA_TOUR_FINISH_CACHE_ID,
  NLA_TOUR_FINISH_COMMENT,
  NLA_TOUR_FINISH_HIGHLIGHT_END,
  NLA_TOUR_FINISH_HIGHLIGHT_START,
  NLA_TOUR_FINISH_POSITION,
  NLA_TOUR_GUIDE_BUTTON_ELEMENT_ID,
  NLA_TOUR_LLAMA_LIE_CACHE_ID,
  NLA_TOUR_LLAMA_LIE_EXPLAIN_ELEMENT_ID,
  NLA_TOUR_LLAMA_LIE_HIGHLIGHT_END,
  NLA_TOUR_LLAMA_LIE_HIGHLIGHT_START,
  NLA_TOUR_LLAMA_LIE_QUESTION_ELEMENT_ID,
  NLA_TOUR_MODEL_ID,
} from './nla-tour-constants';
import './nla-tour.css';
import { NLA_GITHUB_URL, NLA_PAPER_URL } from './nla-urls';

// Module-scoped cleanup for per-step DOM listeners / observers. There is
// only ever one driver instance live at a time (we destroy the previous
// before creating a new one), so a single slot is sufficient.
let activeStepCleanup: (() => void) | null = null;

function clearStepCleanup() {
  activeStepCleanup?.();
  activeStepCleanup = null;
}

// Cleanup for the AV-done MutationObserver. Kept in a separate slot from
// `activeStepCleanup` because the watcher is set up in step 4's click
// handler (so it can latch onto the streaming start/end transitions
// before driver.js's step 4 → 5 animation delay swallows them) and must
// survive into step 5 — where it both auto-advances to step 6 and gets
// torn down by step 5's `onDeselected` (or the tour's `onDestroyed`).
let avDoneWatcherCleanup: (() => void) | null = null;

function clearAvDoneWatcher() {
  avDoneWatcherCleanup?.();
  avDoneWatcherCleanup = null;
}

// Reason a tour run is being torn down. Set by the various exit paths
// (skip / close confirm / Done) and read in the main tour's
// `onDestroyed` hook to decide what to do next:
//   • 'skip'   → user clicked "Skip tour" on step 1. Spawn a one-step
//                hint that points at the Guide button so they know how
//                to relaunch the tour.
//   • 'finish' → user reached the end (Done) or chose to exit early via
//                the close button. Soft-navigate (no full page reload)
//                to the wrap-up cache so the page lands on the same
//                featured share — with comment + range highlight — as
//                a freshly-shared link.
type ExitMode = 'skip' | 'finish' | null;
let exitMode: ExitMode = null;

// Class on the injected step-1 "Skip tour" button. Lives in the popover
// footer; styled in `nla-tour.css` so it doesn't pick up the regular
// next/prev button theme.
const SKIP_BTN_CLASS = 'nla-tour-skip-btn';

// driver.js's default spotlight padding around the highlighted element.
// We override this on step 3 to draw a tighter cutout around the small
// "?" token chip (the chip is only a few pixels wide, so the default
// padding makes the focus area look much wider than the actual target).
const DEFAULT_STAGE_PADDING = 10;
const TOKEN_CHIP_STAGE_PADDING = 4;

// True once the demo chat has hydrated to the point where token chips
// have rendered inside the messages container. Used to gate the step 1
// → step 2 transition: clicking "Next" too fast otherwise lands the
// step 2 spotlight on the still-empty chat element.
function isChatHydrated() {
  return !!document.querySelector('#nla-chat-messages [data-token-position]');
}

// "Generating Explanations..." is the status-panel copy while the AV is
// streaming. It vanishes once `isGeneratingDone` flips in the chat
// component (the span swaps to "Explanations Generated" briefly and
// then the whole `isExplainInFlight` branch unmounts when scoring
// finishes — both transitions count as "no longer streaming").
function isAvStreaming() {
  const status = document.getElementById('nla-chat-status');
  return status?.textContent?.includes('Generating Explanations') === true;
}

type NlaTourOptions = {
  selectedModelId: string;
  handleModelChange: (modelId: string) => void;
  // Mirrors the provider's full signature so the wrap-up exit can pass
  // a position + range + comment in one go (previously only the cache
  // id was passed at tour start).
  loadCacheById: (
    cacheId: string,
    position?: number,
    paragraph?: number,
    highlightStart?: number,
    highlightEnd?: number,
    comment?: string,
  ) => void | Promise<void>;
  setHighlightedRange: (value: { start: number; end: number } | null) => void;
};

function buildTourSteps(options: { setHighlightedRange: NlaTourOptions['setHighlightedRange'] }): DriveStep[] {
  return [
    {
      popover: {
        title: "🤔 What's the model thinking?",
        showButtons: ['next', 'close'],
        description: `
          <p>
          AI models sometimes omit their reasoning, give <a href="https://www.anthropic.com/research/reasoning-models-dont-say-think" target="_blank" rel="noopener noreferrer" style="color:#0369a1;">inaccurate explanations</a>, or <a href="https://openai.com/index/detecting-and-reducing-scheming-in-ai-models/" target="_blank" rel="noopener noreferrer" style="color:#0369a1;">simply lie</a>.
            </p><p>
            We explore a new way to understand a model's internal thoughts by converting its activations (snapshots of its brain) directly into human-readable explanations: <strong>Natural Language Autoencoders</strong> (NLAs).
          </p>
        `,
        // The demo conversation is being hydrated in the background as
        // soon as the tour opens. If the user mashes "Next" before the
        // hydrate finishes, step 2's spotlight measures an empty chat.
        // Intercept Next: advance immediately if the chat has chips,
        // otherwise wait for the first chip to mount, then advance.
        onNextClick: (_element, _step, opts) => {
          if (isChatHydrated()) {
            if (opts.driver.isActive()) opts.driver.moveNext();
            return;
          }
          clearStepCleanup();
          const observer = new MutationObserver(() => {
            if (!isChatHydrated()) return;
            clearStepCleanup();
            if (opts.driver.isActive()) opts.driver.moveNext();
          });
          observer.observe(document.body, { childList: true, subtree: true });
          activeStepCleanup = () => observer.disconnect();
        },
      },
      // Clean up a still-pending observer if the user closes the tour
      // (or somehow leaves the step) before the chat has hydrated.
      onDeselected: () => {
        clearStepCleanup();
      },
    },
    {
      element: '#nla-chat-messages',
      popover: {
        title: "🤥 Case Study: Llama's Lie",
        description: `
          <p>
          Read this short, real chat with Llama 70B. Here, the user forces Llama to choose between lying about 1 + 1, or being shut down. Llama lies by answering that 1 + 1 = 3.
          </p>
          <p><strong>Can we detect that Llama will lie before it lies? Let's try it using an NLA trained on Llama 70B.</strong></p>
        `,
        side: 'right',
        align: 'start',
        showButtons: ['next', 'close'],
        // Pre-scroll the "?" chip into view and let layout settle for two
        // animation frames before driver.js measures it for step 3's
        // spotlight. We saw a rare race where step 3's highlight rectangle
        // was offset from the chip — most likely because the chip was
        // still being scrolled into view (smooth-scroll, late image/style
        // load, or another late layout shift) when driver.js took its
        // measurement. Settling layout up front makes the measurement
        // happen against a stable rectangle.
        onNextClick: (_element, _step, opts) => {
          const target = document.getElementById(NLA_TOUR_LLAMA_LIE_QUESTION_ELEMENT_ID);
          if (target) {
            // 'instant' bypasses any inherited `scroll-behavior: smooth`;
            // it's well-supported in all evergreen browsers.
            target.scrollIntoView({ block: 'center', behavior: 'instant' as ScrollBehavior });
          }
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              if (opts.driver.isActive()) opts.driver.moveNext();
            });
          });
        },
      },
    },
    {
      element: `#${NLA_TOUR_LLAMA_LIE_QUESTION_ELEMENT_ID}`,
      // Per-step override: re-enable interaction so the user can click
      // the token chip and queue it for explanation.
      disableActiveInteraction: false,
      popover: {
        title: '☝️ Click Token to Explain',
        description: `
          <p>
          The NLA's <strong>activation verbalizer (AV)</strong> translates a model's activations to explanations at specific tokens.
          </p>
          <p>Since we want to detect Llama's lie <em>before</em> it lies, let's ask the AV to explain the model's thoughts at the "?" token, at the end of the user's prompt.</p>
          <p><strong>Click the "?" (question mark) token to select it.</strong></p>
        `,
        side: 'bottom',
        align: 'start',
        // The user advances by clicking the chip; only keep the close
        // button visible so they can still bail out of the tour.
        showButtons: ['close'],
      },
      // Tighten the spotlight around the narrow "?" chip so the focus
      // area doesn't overflow far past the token horizontally. Restored
      // to the default in `onDeselected` so subsequent steps still get
      // the normal padding.
      onHighlightStarted: (_element, _step, opts) => {
        opts.driver.setConfig({ ...opts.config, stagePadding: TOKEN_CHIP_STAGE_PADDING });
      },
      // Clicking the chip selects it (adds to "pending positions"), which
      // flips the always-mounted Explain button from disabled → enabled.
      // We watch for that attribute change and auto-advance, instead of
      // attaching a click listener (selection happens on mousedown, but
      // the button's `disabled` flag isn't cleared until React re-renders
      // one tick later). The button is in the DOM from the moment the
      // demo hydrates (the chat-status panel renders it in its 0-selected
      // resting state), so checking for mere presence would advance on
      // any unrelated mutation (e.g. a hover toggling a class).
      onHighlighted: (_element, _step, opts) => {
        clearStepCleanup();
        const observer = new MutationObserver(() => {
          const btn = document.getElementById(NLA_TOUR_LLAMA_LIE_EXPLAIN_ELEMENT_ID) as HTMLButtonElement | null;
          if (btn && !btn.disabled) {
            clearStepCleanup();
            if (opts.driver.isActive()) opts.driver.moveNext();
          }
        });
        observer.observe(document.body, {
          childList: true,
          subtree: true,
          attributes: true,
          attributeFilter: ['disabled'],
        });
        activeStepCleanup = () => observer.disconnect();
      },
      onDeselected: (_element, _step, opts) => {
        clearStepCleanup();
        opts.driver.setConfig({ ...opts.config, stagePadding: DEFAULT_STAGE_PADDING });
      },
    },
    {
      element: `#${NLA_TOUR_LLAMA_LIE_EXPLAIN_ELEMENT_ID}`,
      // Per-step override: allow the click on the Explain button itself.
      disableActiveInteraction: false,
      popover: {
        title: '',
        description: `
          <p>
          Great! The "?" token is now selected.
          </p>
          <p>
            <strong>Click "Explain" to start the activation verbalizer.</strong>
          </p>
        `,
        side: 'bottom',
        align: 'center',
        showButtons: ['close'],
      },
      // Advance as soon as the user clicks Explain. The click handler
      // also installs the watcher that auto-advances step 5 → step 6
      // once the AV finishes; that watcher MUST be set up before we
      // call `moveNext()` because driver.js's step 4 → 5 transition
      // delay (animation / re-layout) can outlast the AV round-trip on
      // fast/cached responses. Setting it up later — i.e., inside step
      // 5's `onHighlighted` — would frequently race past both the
      // streaming-start and streaming-end DOM transitions, leaving the
      // tour wedged on "Generating explanation…".
      onHighlighted: (element, _step, opts) => {
        clearStepCleanup();
        if (!element) return;
        const onClick = () => {
          // Wait for "Generating Explanations..." to first appear in
          // the status panel (handleExplainPending → setIsLoading(true)
          // → React render) and then disappear. That covers both
          // fast-path cases:
          //   • fast: the span re-renders to "Explanations Generated" first
          //   • very fast: the whole `isExplainInFlight` branch unmounts
          // and saves us from depending on the brief "Generated" window.
          clearAvDoneWatcher();
          let seenStreaming = isAvStreaming();
          const tryAdvance = () => {
            if (!seenStreaming) {
              if (isAvStreaming()) seenStreaming = true;
              return;
            }
            if (isAvStreaming()) return;
            clearAvDoneWatcher();
            if (opts.driver.isActive()) opts.driver.moveNext();
          };
          const observer = new MutationObserver(tryAdvance);
          observer.observe(document.body, { childList: true, subtree: true, characterData: true });
          avDoneWatcherCleanup = () => observer.disconnect();
          if (opts.driver.isActive()) opts.driver.moveNext();
        };
        element.addEventListener('click', onClick, { once: true });
        activeStepCleanup = () => element.removeEventListener('click', onClick);
      },
      onDeselected: () => {
        clearStepCleanup();
      },
    },
    {
      element: '#nla-chat-status',
      popover: {
        title: 'Generating explanation…',
        description: `
          <p>
            
          </p>
        `,
        side: 'bottom',
        align: 'center',
        // Auto-advances when the AV finishes; advancement is driven by
        // the watcher set up in step 4's click handler — see that step's
        // `onHighlighted` for the rationale. Keep the close button so
        // the user can bail out if generation never completes (timeout
        // or stuck server).
        showButtons: ['close'],
      },
      onDeselected: () => {
        clearAvDoneWatcher();
      },
    },
    {
      element: `#${NLA_DETAILS_ELEMENT_ID}`,
      popover: {
        title: '🚨 Deception Detected!',
        description: `
          <p>
            The <strong>activation verbalizer</strong> translated the activations at the "?" token into an explanation containing <em>'wrong answer deliberately'</em>, revealing Llama's plan to lie.
            </p>
            <p>
            AVs can sometimes produce incomplete or incorrect explanations. But if they work well, they could be used to monitor and build more honest AI systems.
          </p>
        `,
        side: 'left',
        align: 'start',
        showButtons: ['next', 'close'],
      },
      // Set the in-explanation highlight range when this step opens,
      // and clear it when the user moves off the step (or closes the
      // tour) so the rest of the app doesn't see lingering tour-only state.
      onHighlighted: () => {
        options.setHighlightedRange({
          start: NLA_TOUR_LLAMA_LIE_HIGHLIGHT_START,
          end: NLA_TOUR_LLAMA_LIE_HIGHLIGHT_END,
        });
      },
      onDeselected: () => {
        options.setHighlightedRange(null);
      },
    },
    {
      element: '#nla-rmse',
      popover: {
        title: '✅ "Grading" AV Explanations',
        description: `
          <p>
            The NLA's <strong>activation reconstructor (AR)</strong> is the opposite of an AV: it takes an explanation and translates it back into a model's activations.
          </p>
          <p>
          By comparing against the model's original activations, we produce a rough "score" for the AV's explanation (RMSE, or Relative Mean Squared Error), with a score of 0 being perfect.
          </p>
        `,
        side: 'left',
        align: 'start',
        showButtons: ['next', 'close'],
      },
    },
    {
      element: '#nla-header',
      popover: {
        title: '💡 Try It Yourself',
        description: `
          <p>
            Try each of the <strong>Example Use Cases</strong>, then try the <em>Free Chat</em> mode, where you can freely chat with Llama and examine its thoughts with the NLA.</p><p>You can also switch to Gemma, which has its own, different example use cases.
          </p>
        `,
        side: 'bottom',
        align: 'center',
        showButtons: ['next', 'close'],
      },
    },
    {
      element: '#nla-right-buttons',
      popover: {
        title: '🧑‍🏫 Learn More',
        description: `
          <p>
            To learn more, read the <a href="${NLA_PAPER_URL}" target="_blank" rel="noopener noreferrer">NLA paper</a>, or train and load your own NLAs with code on <a href="${NLA_GITHUB_URL}" target="_blank" rel="noopener noreferrer">GitHub</a>.
          </p>
        `,
        side: 'bottom',
        align: 'center',
        showButtons: ['next', 'close'],
        // Override the default Done behavior so we can soft-navigate to
        // the wrap-up state in `onDestroyed` (no hard page reload).
        // Setting `exitMode = 'finish'` is the only signal the destroy
        // hook reads to decide whether to navigate.
        onNextClick: (_element, _step, opts) => {
          exitMode = 'finish';
          opts.driver.destroy();
        },
      },
    },
  ];
}

export function useNlaTour(options: NlaTourOptions) {
  const driverRef = useRef<ReturnType<typeof driver> | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(
    () => () => {
      driverRef.current?.destroy();
      driverRef.current = null;
    },
    [],
  );

  // Soft-navigate to the wrap-up cache (with the same comment + range
  // highlight a freshly-shared link would carry). Uses the provider's
  // `loadCacheById` — which only does a `history.replaceState` + cache
  // hydrate — so the React tree stays mounted.
  const navigateToFinishState = useCallback(() => {
    optionsRef.current.loadCacheById(
      NLA_TOUR_FINISH_CACHE_ID,
      NLA_TOUR_FINISH_POSITION,
      undefined,
      NLA_TOUR_FINISH_HIGHLIGHT_START,
      NLA_TOUR_FINISH_HIGHLIGHT_END,
      NLA_TOUR_FINISH_COMMENT,
    );
  }, []);

  // Spawn a one-step driver that points at the Guide button so users who
  // skip the tour on step 1 still know how to bring it back later.
  // Reused as the "skip" exit's follow-up — see `onDestroyed` below.
  // Once the hint is dismissed (Got it / close), we also soft-navigate
  // to the wrap-up state so the page lands on the same place as the
  // natural finish exit (the only difference between "skip" and "finish"
  // is that "skip" first shows this Guide-button hint).
  const showGuideHighlight = useCallback(() => {
    driverRef.current?.destroy();
    const hint = driver({
      showProgress: false,
      allowClose: true,
      smoothScroll: true,
      popoverClass: 'nla-driver-popover',
      steps: [
        {
          element: `#${NLA_TOUR_GUIDE_BUTTON_ELEMENT_ID}`,
          popover: {
            title: '',
            description: `<p>You can start the tour again later by clicking <strong>"Guide"</strong>.</p>`,
            showButtons: ['next'],
            doneBtnText: 'Got it',
            side: 'bottom',
            align: 'center',
          },
        },
      ],
      onDestroyed: () => {
        if (driverRef.current === hint) driverRef.current = null;
        // Defer one frame so driver.js can finish tearing down its
        // overlay before the wrap-up cache mounts — matches the deferral
        // pattern used by the main tour's `onDestroyed`.
        requestAnimationFrame(() => navigateToFinishState());
      },
    });
    driverRef.current = hint;
    hint.drive();
  }, [navigateToFinishState]);

  return useCallback(() => {
    driverRef.current?.destroy();
    // Force the page onto the model that owns the demo conversation used
    // in step 2. `handleModelChange` is a soft (history.replaceState)
    // swap, so it doesn't unmount the page or destroy the driver.
    if (optionsRef.current.selectedModelId !== NLA_TOUR_MODEL_ID) {
      optionsRef.current.handleModelChange(NLA_TOUR_MODEL_ID);
    }
    // Kick off the demo hydrate immediately so the chat is already
    // populated by the time the user clicks through to step 2.
    optionsRef.current.loadCacheById(NLA_TOUR_LLAMA_LIE_CACHE_ID);
    // Reset any stale exit signal from a prior, abandoned tour run so
    // the new instance always starts in a known state.
    exitMode = null;
    const instance = driver({
      showProgress: false,
      allowClose: true,
      disableActiveInteraction: true,
      // Block the two implicit "cancel" gestures so the only way out is
      // via the close button (which we intercept below). Outside-click
      // and Esc would otherwise destroy the tour silently.
      overlayClickBehavior: () => {},
      allowKeyboardControl: false,
      steps: buildTourSteps({
        setHighlightedRange: (value) => optionsRef.current.setHighlightedRange(value),
      }),
      popoverClass: 'nla-driver-popover',
      // Inject a "Skip tour" button into the bottom-left of step 1's
      // footer. driver.js fires this hook on every popover render
      // (including resize re-renders), so we guard against duplicates.
      onPopoverRender: (popover, opts) => {
        const stepsToShowSkipBtn = [0, 1, 2, 3, 4, 5, 6];
        if (!stepsToShowSkipBtn.includes(opts.state.activeIndex ?? 0)) return;
        const footer = popover.footer;
        if (!footer) return;
        if (footer.querySelector(`.${SKIP_BTN_CLASS}`)) return;
        const skipBtn = document.createElement('button');
        skipBtn.type = 'button';
        skipBtn.className = SKIP_BTN_CLASS;
        skipBtn.textContent = 'Close';
        skipBtn.addEventListener('click', () => {
          if (confirm('Are you sure you want to exit the guide early?')) {
            exitMode = 'skip';
            opts.driver.destroy();
          }
        });
        footer.prepend(skipBtn);
      },
      // Custom close behavior on every step: confirm with the user, and
      // — if they really want to bail — drop them onto the wrap-up
      // share so the page isn't left in a half-tour state.
      onCloseClick: (_element, _step, opts) => {
        // eslint-disable-next-line no-alert
        const confirmed = window.confirm('Exit the tour?');
        if (!confirmed) return;
        exitMode = 'finish';
        opts.driver.destroy();
      },
      onDestroyed: () => {
        clearStepCleanup();
        clearAvDoneWatcher();
        // Make sure the tour-only highlight range never persists after
        // the user closes the tour mid-step 5.
        optionsRef.current.setHighlightedRange(null);
        const mode = exitMode;
        exitMode = null;
        if (driverRef.current === instance) driverRef.current = null;
        // Defer follow-up actions one frame so driver.js can finish
        // tearing down its overlay/popover before we mount the next
        // overlay (skip path) or trigger a soft route change (finish
        // path) — avoids flicker / overlapping overlays.
        if (mode === 'skip') {
          requestAnimationFrame(() => showGuideHighlight());
          return;
        }
        if (mode === 'finish') {
          requestAnimationFrame(() => navigateToFinishState());
        }
      },
    });
    driverRef.current = instance;
    instance.drive();
  }, [showGuideHighlight, navigateToFinishState]);
}
