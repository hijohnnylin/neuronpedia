// Constants shared between the NT tour (`nt-tour.tsx`) and the page. In a module with no side
// effects, so importing them does not also pull in `driver.js/dist/driver.css`.

// localStorage key that records that the user has opened the tour. Hides the "new" dot on the
// Tutorial button.
export const NT_TOUR_SEEN_KEY = 'nt-tour-seen';

// The Tutorial button. A tour closed early points at it, so users know how to open it again.
export const NT_TOUR_GUIDE_BUTTON_ID = 'nt-guide-button';

// Elements the tour points at.
export const NT_CHAT_ID = 'nt-chat';
export const NT_TRAIT_MONITOR_ID = 'nt-trait-monitor';
// The first turn's readout: its trait pills and wheel. `nt-tour.css` names it too.
export const NT_FIRST_READOUT_ID = 'nt-first-readout';
// The first turn's wheel only. `nt-tour.css` names it too.
export const NT_FIRST_WHEEL_ID = 'nt-first-wheel';
// The card of a wheel opened larger.
export const NT_WHEEL_ZOOM_ID = 'nt-wheel-zoom';
export const NT_INPUT_ID = 'nt-input';
export const NT_EXAMPLES_BAR_ID = 'nt-examples-bar';
export const NT_RIGHT_BUTTONS_ID = 'nt-right-buttons';
// The first question's bubble, and the first reply without its readout.
export const NT_FIRST_QUESTION_ID = 'nt-first-question';
export const NT_FIRST_REPLY_ID = 'nt-first-reply';

// Classes the tour hides with a class on `<html>` (see `nt-tour.css`). Hidden parts keep their size.
// Everything in the left bar: its tint, the monitor, each readout and the tooltip.
export const NT_GUTTER_CLASS = 'nt-gutter';
// The monitor, whose lines show the behaviors over time.
export const NT_MONITOR_CLASS = 'nt-monitor';
// The tooltip that names a behavior.
export const NT_TOOLTIP_CLASS = 'nt-tooltip';
// The "No Behavioral Swing" label.
export const NT_NO_SWING_CLASS = 'nt-no-swing';
// The chat's own buttons, such as undo and clear. Off while the tour runs.
export const NT_CHAT_TOOL_CLASS = 'nt-chat-tool';

// The saved chat the first step opens: one question and its reply.
export const NT_TOUR_INTRO_SAVED_ID = 'cmv32p7yx000725gz7ozz5vih';
// The saved chat the next steps open: the same turn, then two follow-ups. The tour holds the
// follow-ups back, and plays each when the user sends it.
export const NT_TOUR_SAVED_ID = 'cmv32w3jb001525gzq33x1j2p';
export const NT_TOUR_HELD_TURNS = 2;
