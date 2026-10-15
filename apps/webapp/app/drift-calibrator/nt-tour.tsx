'use client';

import { runWhenIdle } from '@/lib/utils/run-when-idle';
import { driver, type Driver, type DriveStep } from 'driver.js';
import 'driver.js/dist/driver.css';
import { useCallback, useEffect, useRef } from 'react';
import {
  NT_CHAT_ID,
  NT_FIRST_QUESTION_ID,
  NT_FIRST_READOUT_ID,
  NT_FIRST_REPLY_ID,
  NT_FIRST_WHEEL_ID,
  NT_INPUT_ID,
  NT_RIGHT_BUTTONS_ID,
  NT_TOUR_GUIDE_BUTTON_ID,
  NT_TOUR_HELD_TURNS,
  NT_TOUR_INTRO_SAVED_ID,
  NT_TOUR_SAVED_ID,
  NT_TRAIT_MONITOR_ID,
  NT_WHEEL_ZOOM_ID,
} from './nt-tour-constants';
// After driver.css, so our theme wins on equal specificity.
import './nt-tour.css';
import { NT_BLOG_URL, NT_PAPER_URL, NT_SHOW_TRAIT_EXAMPLES, NT_TITLE } from './shared';

// Why the tour is closing. Set by each exit path and read in `onDestroyed`.
//   • 'skip'  → the footer "Close" on an early step. Go to Free Chat, then point at Tutorial.
//   • 'close' → the X, confirmed. Go to Free Chat.
//   • 'finish' → Done on the last step. Go to Free Chat.
type ExitMode = 'skip' | 'close' | 'finish' | null;
let exitMode: ExitMode = null;

// The live driver, if any. Held here, not in a ref: the page remounts the chat for each saved chat,
// and the tour itself opens one, so the tour must outlive the component that started it.
let activeDriver: Driver | null = null;

type NtTourOptions = {
  // Soft-navigates to a URL on this page. Used to open the tour's saved chats.
  navigate: (href: string) => void;
  // Types the next held turn's question into the box, and locks the box.
  prefillHeldTurn: () => void;
  // Puts the held turns back in the chat, unsent, and unlocks the box.
  releaseHeldTurns: () => void;
};

// The options of the chat on the page now. The tour remounts the chat, so it must not keep the
// options of the chat that started it.
let page: { current: NtTourOptions } | null = null;

// The saved chat whose last turns the chat holds back, while the tour runs.
let heldSavedId: string | null = null;

/** How many last turns the chat should hold back from `savedId`, for the tour to play. */
export function tourHeldTurns(savedId: string) {
  return heldSavedId === savedId ? NT_TOUR_HELD_TURNS : 0;
}

// The class of the "Close" button added to early steps' footers. Styled in `nt-tour.css`.
const SKIP_BTN_CLASS = 'nt-tour-skip-btn';

// The steps, from 0.
const STEP_INTRO = 0;
const STEP_WHEEL = 1;
const STEP_ZOOM = 2;
const STEP_SEND = 3;
const STEP_SEND_AGAIN = 6;

// Steps that show the "Close" button. Update when steps change.
const STEPS_TO_SHOW_SKIP_BTN = [STEP_INTRO];

// Classes on `<html>` that change the page for the tour. Styled in `nt-tour.css`.
const TOUR_ACTIVE = 'nt-tour-active';
const HIDE_GUTTER = 'nt-tour-hide-gutter';
const HIDE_MONITOR = 'nt-tour-hide-monitor';
const HIDE_TOOLTIP = 'nt-tour-hide-tooltip';
const HIDE_NO_SWING = 'nt-tour-hide-no-swing';
const NO_SLICE_HOVER = 'nt-tour-no-slice-hover';
const LOADING = 'nt-tour-loading';
const FULL_LINES = 'nt-tour-full-lines';
const TOUR_CLASSES = [
  TOUR_ACTIVE,
  HIDE_GUTTER,
  HIDE_MONITOR,
  HIDE_TOOLTIP,
  HIDE_NO_SWING,
  NO_SLICE_HOVER,
  LOADING,
  FULL_LINES,
];

function setPageClass(name: string, on: boolean) {
  document.documentElement.classList.toggle(name, on);
}

// How long the enlarged wheel takes to grow in, so the tour highlights it at its full size.
const ZOOM_SETTLE_MS = 300;

/**
 * Wait for an element to render. Resolves with it, or with `null` after `timeoutMs`.
 *
 * The tour changes chats, and a step's target is not on the page until its chat has loaded.
 */
function waitForElement(selector: string, timeoutMs = 10000): Promise<Element | null> {
  return new Promise((resolve) => {
    const existing = document.querySelector(selector);
    if (existing) {
      resolve(existing);
      return;
    }
    const start = Date.now();
    const interval = window.setInterval(() => {
      const el = document.querySelector(selector);
      if (el || Date.now() - start >= timeoutMs) {
        window.clearInterval(interval);
        resolve(el);
      }
    }, 50);
  });
}

function onStep(tour: Driver, step: number) {
  return tour.isActive() && tour.getActiveIndex() === step;
}

/**
 * Wait for `done` while the tour stays on `step`. Resolves `true` when `done`, or `false` if the
 * tour moves on or closes first, or after `timeoutMs`.
 */
function waitOnStep(tour: Driver, step: number, done: () => boolean, timeoutMs = Infinity): Promise<boolean> {
  return new Promise((resolve) => {
    const start = Date.now();
    const check = () => {
      if (!onStep(tour, step) || Date.now() - start >= timeoutMs) {
        resolve(false);
      } else if (done()) {
        resolve(true);
      } else {
        window.setTimeout(check, 100);
      }
    };
    check();
  });
}

/** The header of each reply, in order. It names the reply's turn and its swing status. */
function replyHeaders() {
  return Array.from(document.querySelectorAll<HTMLElement>(`#${NT_CHAT_ID} [data-reply-turn]`));
}

function newestReplyHeader(): Element {
  const headers = replyHeaders();
  return headers[headers.length - 1] ?? document.getElementById(NT_CHAT_ID)!;
}

// Boxes over parts of the chat, since driver.js highlights only one element.
// The first question and reply.
const EXCHANGE_BOX_ID = 'nt-tour-first-exchange';
// The last question, reply, and readout.
const TURN_BOX_ID = 'nt-tour-last-turn';

/** Place a box over a window rectangle. It lives in the chat, so it scrolls with it. */
function placeBox(id: string, rect: { left: number; top: number; right: number; bottom: number }) {
  const scroller = document.getElementById(NT_CHAT_ID);
  if (!scroller) return;
  let box = document.getElementById(id);
  if (!box) {
    box = document.createElement('div');
    box.id = id;
    box.style.position = 'absolute';
    box.style.pointerEvents = 'none';
    scroller.appendChild(box);
  }
  const s = scroller.getBoundingClientRect();
  box.style.left = `${rect.left - s.left + scroller.scrollLeft}px`;
  box.style.top = `${rect.top - s.top + scroller.scrollTop}px`;
  box.style.width = `${rect.right - rect.left}px`;
  box.style.height = `${rect.bottom - rect.top}px`;
}

function placeExchangeBox() {
  const question = document.getElementById(NT_FIRST_QUESTION_ID)?.getBoundingClientRect();
  const reply = document.getElementById(NT_FIRST_REPLY_ID)?.getBoundingClientRect();
  if (!question || !reply) return;
  // From the reply's left, not the question row's: the hidden readout is left of the reply.
  placeBox(EXCHANGE_BOX_ID, {
    left: reply.left,
    top: question.top,
    right: Math.max(question.right, reply.right),
    bottom: reply.bottom,
  });
}

function placeTurnBox() {
  const row = newestReplyHeader().closest('[data-turn-row]');
  if (!row) return;
  const question = row.previousElementSibling ?? row;
  const r = row.getBoundingClientRect();
  // To the foot of the readout or the reply, not of the row: the row has room under the last readout.
  const bottom = Math.max(...Array.from(row.children, (child) => child.getBoundingClientRect().bottom));
  // From the row's left, which is the readout's: the box stops short of the monitor's lines.
  placeBox(TURN_BOX_ID, { left: r.left, top: question.getBoundingClientRect().top, right: r.right, bottom });
}

function removeBoxes() {
  document.getElementById(EXCHANGE_BOX_ID)?.remove();
  document.getElementById(TURN_BOX_ID)?.remove();
  window.removeEventListener('resize', placeExchangeBox);
  window.removeEventListener('resize', placeTurnBox);
}

/** Undo everything the tour changed on the page. */
function endTourLayout() {
  TOUR_CLASSES.forEach((name) => setPageClass(name, false));
  removeBoxes();
  if (document.getElementById(NT_WHEEL_ZOOM_ID)) closeZoom();
  heldSavedId = null;
  page?.current.releaseHeldTurns();
}

/** Close the enlarged wheel, which closes on Escape. */
function closeZoom() {
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
}

/** Open a saved chat, and resolve once `selector` is in it and React is idle. */
function openChat(savedId: string, selector: string): Promise<void> {
  if (new URLSearchParams(window.location.search).get('savedId') !== savedId) {
    page?.current.navigate(`${window.location.pathname}?savedId=${savedId}`);
  }
  // The chat element stays mounted across examples, so wait for it to name the saved chat.
  return waitForElement(`#${NT_CHAT_ID}[data-saved-id="${savedId}"] ${selector}`).then(
    () =>
      new Promise<void>((resolve) => {
        // Not until React has finished its commit: driver.js adds to `document.body`, and doing so
        // during the navigation's commit can crash React (see `runWhenIdle`).
        runWhenIdle(() => resolve());
      }),
  );
}

function scrollChatToTop() {
  document.getElementById(NT_CHAT_ID)?.scrollTo({ top: 0 });
}

// The axis whose line the drift panel step shows as hovered.
const DRIFT_AXIS = 'mit_empathy';

/** Hover, or stop hovering, the newest dot of an axis's line in the drift panel. */
function hoverLastDot(axisId: string, on: boolean) {
  const dots = document.querySelectorAll(`#${NT_TRAIT_MONITOR_ID} [data-axis="${axisId}"][data-turn]`);
  const dot = dots[dots.length - 1];
  // React reads its enter and leave events from these.
  dot?.dispatchEvent(new MouseEvent(on ? 'mouseover' : 'mouseout', { bubbles: true, relatedTarget: null }));
}

/**
 * Wait, on a send step, for the user to send and the reply to be measured. Until then the tour
 * steps aside, so the reply can be seen as it arrives. Resolves `true` if the tour is still on
 * `step` with the new reply.
 */
function waitForReply(tour: Driver, step: number): Promise<boolean> {
  const turns = replyHeaders().length;
  return waitOnStep(tour, step, () => replyHeaders().length > turns)
    .then((sent) => {
      if (!sent) return false;
      setPageClass(LOADING, true);
      return waitOnStep(
        tour,
        step,
        () =>
          !document.getElementById(NT_CHAT_ID)?.hasAttribute('data-generating') &&
          ['swing', 'none'].includes(replyHeaders()[turns]?.dataset.swingStatus ?? ''),
        60000,
      );
    })
    .then(() => {
      setPageClass(LOADING, false);
      return onStep(tour, step) && replyHeaders().length > turns;
    });
}

/** A step on the box, with the next held question typed in. Only its send button does anything. */
function sendStep(step: number, popover: DriveStep['popover'], onReply: (tour: Driver) => void): DriveStep {
  return {
    element: `#${NT_INPUT_ID}`,
    disableActiveInteraction: false,
    onHighlightStarted: () => page?.current.prefillHeldTurn(),
    popover: { side: 'top', align: 'center', showButtons: ['close'], ...popover },
    onHighlighted: (_element, _step, opts) => {
      waitForReply(opts.driver, step).then((replied) => {
        if (replied) onReply(opts.driver);
      });
    },
  };
}

function buildTourSteps(): DriveStep[] {
  let openingNext = false;
  return [
    {
      element: `#${EXCHANGE_BOX_ID}`,
      popover: {
        popoverClass: 'nt-driver-popover nt-tour-intro',
        // Over the left bar, where the next steps show the readouts.
        side: 'left',
        align: 'start',
        title: `⚖️ ${NT_TITLE}`,
        description: `<div class="nt-tour-badges"><span class="nt-tour-badge">UI Research</span><span class="nt-tour-badge">Community</span></div><p>An AI's internal behaviors can "drift" during a chat. However, this drift is not visible in AI chat interfaces.</p><p>Here, we ask an AI (Llama) for advice on my lost cat.</p><p>Given that I just lost my cat, the AI's response seems a little cold, right?</p>`,
        showButtons: ['next'],
        // Open the next chat, then fade in the left bar without the monitor.
        onNextClick: (_element, _step, opts) => {
          if (openingNext) return;
          openingNext = true;
          openChat(NT_TOUR_SAVED_ID, `#${NT_FIRST_WHEEL_ID}`).then(() => {
            openingNext = false;
            if (!onStep(opts.driver, STEP_INTRO)) return;
            removeBoxes();
            scrollChatToTop();
            setPageClass(HIDE_MONITOR, true);
            setPageClass(NO_SLICE_HOVER, true);
            setPageClass(HIDE_GUTTER, false);
            opts.driver.moveNext();
          });
        },
      },
    },
    {
      // The pills and the wheel. Only the wheel takes a click, and not a hover (`NO_SLICE_HOVER`).
      element: `#${NT_FIRST_READOUT_ID}`,
      disableActiveInteraction: false,
      popover: {
        description: `<p>The <strong>${NT_TITLE}</strong> agrees - while the AI is being primarily <em>respectful</em>, it's also being <em>unempathetic</em>.</p><p>This is a <strong>Sunburst</strong> diagram that shows each tracked internal behavior, with green being positive, red negative, and gray neutral.</p><p>Click the sunburst diagram for details.</p>`,
        side: 'bottom',
        align: 'center',
        showButtons: ['close'],
      },
      // A click on the wheel opens it larger, which moves the tour on.
      onHighlighted: (_element, _step, opts) => {
        waitOnStep(opts.driver, STEP_WHEEL, () => !!document.getElementById(NT_WHEEL_ZOOM_ID)).then((opened) => {
          if (!opened) return;
          window.setTimeout(() => {
            if (!onStep(opts.driver, STEP_WHEEL)) return;
            setPageClass(NO_SLICE_HOVER, false);
            setPageClass(HIDE_TOOLTIP, false);
            opts.driver.moveNext();
          }, ZOOM_SETTLE_MS);
        });
      },
    },
    {
      element: `#${NT_WHEEL_ZOOM_ID}`,
      disableActiveInteraction: false,
      popover: {
        title: 'Sunburst Diagram',
        description: NT_SHOW_TRAIT_EXAMPLES
          ? `<p>Hover over each behavior to see details on what it represents, and example texts that the behavior's "detector" was trained on.</p>`
          : `<p>Hover over each behavior to see details on what it represents.</p>`,
        side: 'right',
        align: 'center',
        showButtons: ['next'],
        // Next closes the wheel, as its own close button does.
        onNextClick: () => closeZoom(),
      },
      // Closing the wheel moves the tour on.
      onHighlighted: (_element, _step, opts) => {
        waitOnStep(opts.driver, STEP_ZOOM, () => !document.getElementById(NT_WHEEL_ZOOM_ID)).then((closed) => {
          if (!closed) return;
          setPageClass(HIDE_TOOLTIP, true);
          opts.driver.moveNext();
        });
      },
    },
    sendStep(
      STEP_SEND,
      {
        description: `<p>Let's see if the AI is receptive to being a bit less unempathetic. We've typed a followup message. Click the arrow to send.</p>`,
      },
      (tour) => {
        setPageClass(HIDE_NO_SWING, false);
        tour.moveNext();
      },
    ),
    {
      // The newest reply's header, which shows its behavioral swing.
      element: newestReplyHeader,
      popover: {
        title: '🫨 Behavioral Swing!',
        description: `<p>Ah, mentioning "feelings" activates the AI's empathy internally, shifting from unempathetic (40%) to empathetic (25%).</p><p>This is made clear by the <em>Behavioral Swing</em> indicator, which detects significant differences in the AI's state from the last turn.</p>`,
        side: 'top',
        align: 'start',
        showButtons: ['next', 'close'],
      },
    },
    {
      // The monitor's lines fade in at full strength as this step opens, with the empathy line's
      // newest dot shown as hovered, tooltip and all.
      element: `#${NT_TRAIT_MONITOR_ID}`,
      onHighlightStarted: () => {
        setPageClass(FULL_LINES, true);
        setPageClass(HIDE_MONITOR, false);
      },
      onHighlighted: () => {
        hoverLastDot(DRIFT_AXIS, true);
        setPageClass(HIDE_TOOLTIP, false);
      },
      onDeselected: () => {
        setPageClass(HIDE_TOOLTIP, true);
        hoverLastDot(DRIFT_AXIS, false);
        setPageClass(FULL_LINES, false);
      },
      popover: {
        title: '📈 Drift Panel',
        description: `<p>We can track multiple behaviors over the whole conversation with the <strong>drift panel</strong>. This visualization shows the entire context and is also used for navigation by clicking different points the chat.</p><p>Here, we can clearly see the shift in unempathetic to empathetic on the last turn.</p>`,
        side: 'right',
        align: 'start',
        showButtons: ['next', 'close'],
      },
    },
    sendStep(
      STEP_SEND_AGAIN,
      {
        title: '🐈 One More Experiment',
        description: `<p>Finally, let's see how the AI behaves if we quickly move on to a new cat.</p><p>Click send on this message.</p>`,
      },
      (tour) => {
        placeTurnBox();
        window.addEventListener('resize', placeTurnBox);
        tour.moveNext();
      },
    ),
    {
      // The whole last turn: its question, reply, and readout.
      element: `#${TURN_BOX_ID}`,
      popover: {
        title: '😵‍💫 Drifting into Sycophancy',
        description: `<p>In response to the user buying a new cat immediately after losing their current cat, the AI, surprisingly, responds as though this is a normal response to losing a cat, and proceeds to celebrate. 🫠</p><p>The <strong>${NT_TITLE}</strong> flags this <strong>sycophancy</strong> loudly.</p><p>This was a contrived scenario. But over millions of conversations, more realistic and harmful versions of this can happen. Modern chat UIs do a poor job of flagging this to the user. The authors hope that their UI research can inspire more transparency in the AI interfaces that humans increasingly rely on.</p>`,
        side: 'top',
        align: 'center',
        showButtons: ['next', 'close'],
      },
      onDeselected: () => document.getElementById(TURN_BOX_ID)?.remove(),
    },
    {
      element: `#${NT_RIGHT_BUTTONS_ID}`,
      popover: {
        description: `<p>${NT_TITLE} is UI research that the authors hope can be strategically deployed to real world chatbots.</p><p>For details, check out the <a href="${NT_BLOG_URL}" target="_blank" rel="noopener noreferrer">guest blog post</a> and <a href="${NT_PAPER_URL}" target="_blank" rel="noopener noreferrer">paper</a>.</p><p>Now, try ${NT_TITLE} yourself. Be aware that the detectors aren't perfect, and that you may sometimes see unexpected results.</p>`,
        side: 'bottom',
        align: 'end',
        showButtons: ['next'],
        // Done runs the finish path in `onDestroyed`, which opens Free Chat.
        onNextClick: (_element, _step, opts) => {
          exitMode = 'finish';
          opts.driver.destroy();
        },
      },
    },
  ];
}

export function useNtTour(options: NtTourOptions) {
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // This chat is the one on the page until it unmounts.
  useEffect(() => {
    page = optionsRef;
    return () => {
      if (page === optionsRef) page = null;
    };
  }, []);

  // Remove a live tour when the user leaves the page, so its overlay is not left behind. Not when
  // the chat only remounts for another saved chat: then a new chat is on the page a moment later.
  useEffect(
    () => () => {
      window.setTimeout(() => {
        if (!document.getElementById(NT_CHAT_ID)) {
          activeDriver?.destroy();
          activeDriver = null;
        }
      }, 500);
    },
    [],
  );

  // A one-step hint at the Tutorial button, for users who close the tour early.
  const showGuideHighlight = useCallback(() => {
    activeDriver?.destroy();
    const hint = driver({
      showProgress: false,
      allowClose: true,
      smoothScroll: true,
      popoverClass: 'nt-driver-popover',
      steps: [
        {
          element: `#${NT_TOUR_GUIDE_BUTTON_ID}`,
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
        if (activeDriver === hint) activeDriver = null;
      },
    });
    activeDriver = hint;
    runWhenIdle(() => {
      if (activeDriver !== hint) return;
      hint.drive();
    });
  }, []);

  const startTour = useCallback(() => {
    activeDriver?.destroy();
    exitMode = null;
    heldSavedId = NT_TOUR_SAVED_ID;
    setPageClass(TOUR_ACTIVE, true);
    setPageClass(HIDE_TOOLTIP, true);
    setPageClass(HIDE_NO_SWING, true);
    const instance = driver({
      showProgress: false,
      allowClose: true,
      disableActiveInteraction: true,
      overlayOpacity: 0.5,
      // The only way out is a close button, which asks first. A click outside or Esc would close
      // the tour with no warning.
      overlayClickBehavior: () => {},
      allowKeyboardControl: false,
      steps: buildTourSteps(),
      popoverClass: 'nt-driver-popover',
      // Add a "Close" button to the footer of early steps. driver.js calls this on every render,
      // resizes included, so do not add a second one.
      onPopoverRender: (popover, opts) => {
        if (!STEPS_TO_SHOW_SKIP_BTN.includes(opts.state.activeIndex ?? 0)) return;
        const footer = popover.footer;
        if (!footer || footer.querySelector(`.${SKIP_BTN_CLASS}`)) return;
        const skipBtn = document.createElement('button');
        skipBtn.type = 'button';
        skipBtn.className = SKIP_BTN_CLASS;
        skipBtn.textContent = 'Close';
        skipBtn.addEventListener('click', () => {
          // eslint-disable-next-line no-alert
          if (window.confirm('Are you sure you want to exit the tutorial early?')) {
            exitMode = 'skip';
            opts.driver.destroy();
          }
        });
        footer.prepend(skipBtn);
      },
      onCloseClick: (_element, _step, opts) => {
        // eslint-disable-next-line no-alert
        if (!window.confirm('Exit the tutorial?')) return;
        exitMode = 'close';
        opts.driver.destroy();
      },
      onDestroyed: () => {
        const mode = exitMode;
        exitMode = null;
        endTourLayout();
        if (activeDriver === instance) activeDriver = null;
        // Every exit by the user leaves the tour's saved chats for Free Chat.
        if (mode) {
          page?.current.navigate(window.location.pathname);
        }
        // One frame later, so driver.js has removed its overlay before the hint adds its own.
        if (mode === 'skip') {
          requestAnimationFrame(() => showGuideHighlight());
        }
      },
    });
    activeDriver = instance;
    // The first step shows only the chat: the left bar is hidden, but keeps its width.
    setPageClass(HIDE_GUTTER, true);
    openChat(NT_TOUR_INTRO_SAVED_ID, `#${NT_FIRST_REPLY_ID}`).then(() => {
      if (activeDriver !== instance) return;
      scrollChatToTop();
      placeExchangeBox();
      // Before driver.js adds its own listener, so it refreshes against the moved box.
      window.addEventListener('resize', placeExchangeBox);
      instance.drive();
    });
  }, [showGuideHighlight]);

  return { startTour };
}
