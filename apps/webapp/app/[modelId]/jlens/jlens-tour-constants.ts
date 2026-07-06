// Constants shared between the JLens tour (`jlens-tour.tsx`) and the page
// client (`jlens-page-client.tsx`). Kept in a side-effect-free module so
// importing these from anywhere does not also pull in
// `driver.js/dist/driver.css` (which the tour module imports).

// localStorage key that records whether the user has seen the JLens tour.
// Used both to auto-start the tour for first-time visitors and to hide the
// "new" indicator dot on the Tutorial button once it's been seen.
export const JLENS_TOUR_SEEN_KEY = 'jlens-tour-seen';

// DOM id assigned to the "Tutorial" button in the JLens header. The "skip
// tour" exit path spotlights this button with a one-step hint so users know
// how to relaunch the tour later. Also used to hide the "new" dot.
export const JLENS_TOUR_GUIDE_BUTTON_ELEMENT_ID = 'jlens-guide-button';

// Stable DOM ids the tour spotlights. These target existing elements in the
// JLens header. Add more here (and assign matching `id={...}` in the relevant
// components) as the real tour steps are built out.
export const JLENS_CHAT_ID = 'jlens-chat';
export const JLENS_JACOBIAN_SPACE_ID = 'jlens-jacobian-space';
export const JLENS_STEER_SPIDER_ID = 'jlens-steer-spider';
export const JLENS_STEER_PANEL_ID = 'jlens-steer-panel';
// Wraps the side-by-side default + steered output columns (only present while
// steering). Spotlighted after the swap so the user sees both columns at once.
export const JLENS_STEER_COLUMNS_ID = 'jlens-steer-columns';
// The steered-output column specifically; used to detect when the swap result
// has rendered (its token chips appear) so the tour can advance.
export const JLENS_STEER_OUTPUT_ID = 'jlens-steer-output';
export const JLENS_HEADER_ELEMENT_ID = 'jlens-header';
export const JLENS_RIGHT_BUTTONS_ELEMENT_ID = 'jlens-right-buttons';
// The top demos/example-use-cases bar (the row of demo tabs, e.g. Verbal Report
// through Selective Mediation) in the model selector.
export const JLENS_DEMOS_BAR_ID = 'jlens-demos-bar';

// Model + shared run the tour opens on when it first starts. Starting the tour
// soft-navigates the page to this share so every step spotlights the same
// featured run.
export const JLENS_TOUR_MODEL_ID = 'qwen3.6-27b';
export const JLENS_TOUR_SHARE_ID = 'cmr2kx72r000mpt2x71l23e0z';

// The "Verbal Report" demo share. Exiting the tour (early or after the last
// step) drops the user here so they land on the first demo rather than a blank
// free chat. Keep in sync with the demo list in `jlens-model-selector.tsx`.
export const JLENS_VERBAL_REPORT_SHARE_ID = 'cmr1qav5a0008pt2xhsvp0scq';
