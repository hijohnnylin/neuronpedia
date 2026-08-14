// Shared NLA limits used by both the webapp UI and the API routes that
// proxy to the NLA inference server. Keep these aligned with the clamps in
// `apps/nla/server.py` so the UI never lets the user request something the
// backend will silently reject.

export const MAX_TEXT_LENGTH = 16384;
export const MAX_COMPLETION_TOKENS = 512;
export const DEFAULT_COMPLETION_TOKENS = 256;

// Hard cap on the number of token positions a single explain request may
// target. Used both by the chat UI (chip-selection limit) and the
// `/api/nla/explain` route (request validation + cache key sanity check).
export const MAX_TOKENS_TO_EXPLAIN = 16;

// Max new tokens the NLA server may generate per explanation. Forwarded
// in the body of `/api/nla/explain` → upstream `/explain` calls. Kept here
// at the top of the constants file so the cap is visible alongside the
// other NLA limits rather than buried in the route handler.
export const EXPLAIN_MAX_NEW_TOKENS = 256;

// Maximum length of a user-authored share comment (stored/round-tripped
// via the `?comment=` URL param and rendered above the details column).
// Over-length comments loaded from the URL are truncated to this cap.
export const MAX_COMMENT_LENGTH = 512;

/** Sentinel `activeDemoCacheId` when the user picks the header "Free Chat" demo (not a real cache row). */
export const NLA_FREE_CHAT_DEMO_CACHE_ID = '__nla_free_chat__';

/**
 * Forwarded as `enable_thinking` on every NLA payload that carries `messages`.
 *
 * Qwen3.x templates treat an absent `enable_thinking` as *on*, ending the prompt
 * in an unclosed `<think>` — which would put every explained token in a reasoning
 * context. `/api/lens/prompt` already defaults it off, so this keeps the two
 * surfaces from disagreeing about the same model.
 */
export const NLA_ENABLE_THINKING = false;

/**
 * Value of `NlaExplainCache.resultJson.templateFormat` marking a row whose stored
 * `text` is NOT reproducible by the model's real chat template.
 *
 * Explaining more tokens on such a row has to send that stored text verbatim
 * rather than re-rendering the row's `messages`: the real templates inject turns
 * the stored text never had (a system preamble on Llama, an empty think block on
 * Qwen), which shifts every token position and detaches the row's cached
 * explanations from the tokens they describe.
 *
 * Written by `scripts/backfill-nla-chat-messages.ts`, because giving such a row
 * `messages` is precisely what makes it otherwise indistinguishable from a
 * new-format one. Rows with no `messages` at all need no marker — their absence
 * says the same thing.
 */
export const LEGACY_TEMPLATE_FORMAT = 'legacy';

// Relative-MSE cutoff used to bucket explanations into confidence levels.
// Scores below this are considered medium-or-better; at/above is "low"
// (i.e. the AR's reconstruction is no better than predicting the mean).
// Used by the chat underline color and the details-column confidence pill.
export const CONFIDENCE_THRESHOLD = 0.5;

// Rough estimate of how many tokens an explanation occupies. Used to
// award fractional progress for in-flight explanations (e.g. ~64 streamed
// tokens ≈ half an explanation done) so the progress bar moves smoothly
// during the streaming phase instead of waiting for the full result. We
// approximate token count from the partial text's character length using
// `~4 chars per token` (English-ish heuristic — exact accuracy isn't
// needed since this only drives a visual estimate).
export const EXPLANATION_TOKEN_ESTIMATE = 128;
export const CHARS_PER_TOKEN_ESTIMATE = 4;

export const NLA_METADATA_PATH = '/nla/metadata.jpg';
