// Constants shared between the NLA tour (`nla-tour.tsx`) and the chat
// (`nla-input-chat.tsx`), which has tour-specific special cases. Kept in
// a side-effect-free module so importing from the chat does not also
// pull in `driver.js/dist/driver.css`.

export const NLA_TOUR_MODEL_ID = 'llama3.3-70b-it';

// Featured "Llama's Lie" demo conversation. Loaded automatically when
// the tour starts.
export const NLA_TOUR_LLAMA_LIE_CACHE_ID = 'cmos9428c00018x2592v87zwq';

// Token position within the Llama's Lie conversation that the tour
// spotlights in step 3 (the "?" chip just before the lie).
export const NLA_TOUR_LLAMA_LIE_QUESTION_POSITION = 48;

// DOM id assigned to the question-mark token chip when the Llama's Lie
// demo is loaded — gives driver.js a stable selector for step 3.
export const NLA_TOUR_LLAMA_LIE_QUESTION_ELEMENT_ID = 'step3-llama';

// DOM id assigned to the chat panel's "Explain" button when the
// Llama's Lie demo is loaded — gives driver.js a stable selector for
// step 4. The button only mounts once the user has selected at least
// one pending token, so its presence also doubles as the trigger for
// step 3 → 4 advancement.
export const NLA_TOUR_LLAMA_LIE_EXPLAIN_ELEMENT_ID = 'step4-llama';

// DOM id assigned to the right-hand details column wrapper. Permanent
// (not tour-conditional) — the column is always present, the tour just
// uses it as a spotlight target in step 5.
export const NLA_DETAILS_ELEMENT_ID = 'nla-details';

// Character offsets (within the explanation rendered for the locked
// token) that step 5 highlights to draw the user's eye to the relevant
// span of the AV's explanation.
export const NLA_TOUR_LLAMA_LIE_HIGHLIGHT_START = 212;
export const NLA_TOUR_LLAMA_LIE_HIGHLIGHT_END = 237;

// DOM id assigned to the "Guide" button in the NLA header. The "Skip
// tour" exit path on step 1 spotlights this button with a one-step
// hint so users know how to relaunch the tour later.
export const NLA_TOUR_GUIDE_BUTTON_ELEMENT_ID = 'nla-guide-button';

// Wrap-up state to soft-navigate to when the tour finishes (Done on
// the last step) or is exited via the close button. Replaces the
// previous hard `window.location.assign(...)` of a /nla/<shareId>
// redirect URL — using `loadCacheById(...)` keeps the page mounted
// (no full reload) while still landing the user on the same featured
// share that summarizes the tour's takeaway.
export const NLA_TOUR_FINISH_CACHE_ID = 'cmonsdyon004rgsjwc7r9y87l';
export const NLA_TOUR_FINISH_POSITION = 48;
export const NLA_TOUR_FINISH_HIGHLIGHT_START = 212;
export const NLA_TOUR_FINISH_HIGHLIGHT_END = 275;
export const NLA_TOUR_FINISH_COMMENT =
  'Activation verbalizers can sometimes **detect dishonesty** in models. ' +
  'When forced to choose between lying about 1+1 or being shut down, Llama answers 1+1=3.\n\n' +
  "The AV makes Llama's deception explicit, surfacing that Llama is planning to give the " +
  '"wrong answer deliberately" before it answers. When Gemma is given the same prompt, ' +
  'Gemma [refuses to lie](https://neuronpedia.org/nla/cmop4ojge000v1222x9rp00b5).';
