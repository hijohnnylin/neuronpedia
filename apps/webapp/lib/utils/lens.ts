// Shared types + constants for the streaming Jacobian/Logit lens endpoint
// (`POST /v1/lens/prompt` on the inference server). Pure types only — safe to
// import from both server and client code.
//
// The endpoint streams NDJSON (one JSON message per line): a single `meta`
// message, then one `token` message per sequence position, then a final
// `done` message (or an `error` message). When `stream` is false the same
// messages are buffered into a single `{ meta, tokens, done }` object.

// Canonical string values for the two lens types. Reference these (e.g.
// `LensType.JACOBIAN_LENS`) instead of bare string literals to avoid typos.
export const LensType = {
  LOGIT_LENS: 'LOGIT_LENS',
  JACOBIAN_LENS: 'JACOBIAN_LENS',
} as const;
export type LensType = (typeof LensType)[keyof typeof LensType];

export const LENS_TYPES = [LensType.LOGIT_LENS, LensType.JACOBIAN_LENS] as const;

// Lens display mode used by the UI toggle: a single lens type, or DIFF (two
// columns showing each lens's advantage over the other). The single-mode values
// intentionally match `LensType`.
export const LensMode = {
  JACOBIAN_LENS: LensType.JACOBIAN_LENS,
  LOGIT_LENS: LensType.LOGIT_LENS,
  DIFF: 'DIFF',
} as const;
export type LensMode = (typeof LensMode)[keyof typeof LensMode];

export const LENS_MODES = [LensMode.JACOBIAN_LENS, LensMode.LOGIT_LENS, LensMode.DIFF] as const;

// Neuronpedia model id (route segment) for the default jlens model. The
// underlying HF/TransformerLens id (e.g. google/gemma-3-4b-pt) is resolved
// server-side from the model's tlensId, so this must be the slash-free NP id.
export const DEFAULT_JLENS_MODEL_ID = 'qwen3.6-27b';

// Social/OpenGraph preview image for the jlens landing page. Resolved against
// `ASSET_BASE_URL` (site-assets bucket), so this is the path within that bucket.
export const JLENS_METADATA_PATH = '/jlens/jlens.jpg';
export const DEFAULT_LENS_TOP_N = 8;
export const DEFAULT_LENS_TEMPERATURE = 0;
export const DEFAULT_LENS_COMPLETION_TOKENS = 128;
// The single-shot completion interface caps generation lower than chat.
export const MAX_LENS_COMPLETION_TOKENS_COMPLETION = 128;
// Default number of generated tokens for the completion interface (lower than
// the cap above; the chat interface uses DEFAULT_LENS_COMPLETION_TOKENS).
export const DEFAULT_LENS_COMPLETION_TOKENS_COMPLETION = 32;

// Character caps on user-supplied input. Enforced on the frontend (so the UI
// won't let you exceed them) and re-validated on the API (so direct callers are
// rejected too).
//   - Chat: each user message.
//   - Chat: the optional assistant prefill the user types. Enforced on the
//     frontend only — the API can't distinguish a prefill from generated /
//     edited assistant content (which is legitimately longer) since both arrive
//     as assistant messages in the same `chat` payload.
//   - Completion: the single prompt.
export const MAX_LENS_CHAT_USER_CHARS = 1024;
export const MAX_LENS_CHAT_PREFILL_CHARS = 512;
export const MAX_LENS_COMPLETION_PROMPT_CHARS = 1024;

// Steering: default/extent of the strength control (a signed fraction of each
// position's residual norm; negative suppresses the selected readout).
export const DEFAULT_LENS_STEER_STRENGTH = -0.1;
export const MAX_LENS_STEER_STRENGTH = 2;
export const LENS_STEER_STRENGTH_STEP = 0.1;

export interface LensChatMessage {
  role: string;
  content: string;
}

// A single readout to steer on. `token` is the EXACT decoded token string
// (whitespace preserved, e.g. " cat") as it appeared in a read-out slice; the
// server resolves it back to a vocab id. `type` selects which lens's readout
// direction to use (Jacobian: J_bar^T·w_t; Logit: plain unembedding w_t).
export interface LensSteerToken {
  token: string;
  type: LensType;
}

export interface LensPromptRequest {
  model: string;
  // One or more lens types to compute. Requesting both is essentially free
  // (the model is run once and residuals are shared).
  type: LensType[];
  // Provide exactly one of `prompt` (raw text) or `chat` (chat-formatted).
  prompt?: string | null;
  chat?: LensChatMessage[] | null;
  top_n?: number;
  // Layers to read out. Empty/omitted = all available layers for the lens
  // type. The model's final layer is always included.
  layers?: number[];
  max_seq_len?: number | null;
  prepend_bos?: boolean;
  // Enable "thinking" mode when applying a chat template (chat requests only).
  enable_thinking?: boolean;
  // Stream results as NDJSON (one message per line).
  stream?: boolean;
  // Sampling temperature for generated tokens. 0 = greedy (argmax).
  temperature?: number;
  // Number of tokens to generate after the prompt. 0 = lens over the prompt
  // only (no generation).
  num_completion_tokens?: number;
  // Token ids the client already has read-outs for (the previous response's
  // prompt + generated tokens, in order). The server reuses the longest common
  // token-id prefix and only recomputes/streams the new positions.
  cached_token_ids?: number[];
  // Exact input token ids to read out over, bypassing tokenization. When
  // provided, `prompt`/`chat` are ignored and generation is disabled
  // (num_completion_tokens forced to 0) — used to faithfully reproduce a
  // previously-computed run (e.g. a shared jlens link).
  input_token_ids?: number[];
  // Steering: readouts to additively inject (negatively, to suppress) into the
  // residual stream at every position, during prefill + generation. When set
  // with a non-zero `steer_strength`, prefix-reuse is disabled server-side.
  steer_tokens?: LensSteerToken[];
  // Layers to inject the steering direction at (the selected layer range).
  // Empty/omitted = the read-out layers.
  steer_layers?: number[];
  // Signed steering strength as a fraction of each position's residual norm
  // (negative suppresses the readout). 0/omitted = no steering.
  steer_strength?: number;
  // When true, ablate (project out) the readout direction from the residual
  // instead of additively steering. Mutually exclusive with steer_strength.
  steer_ablate?: boolean;
  // SWAP: when set, replace the source readout (steer_tokens[0]) with this
  // target readout at every steered layer/position (subtract the source
  // projection, add it back along the target). Takes precedence over
  // steer_strength / steer_ablate. `type` should match the source readout.
  swap_token?: LensSteerToken;
  // Whether to apply the steer/swap intervention to generated tokens too. When
  // false (default), only the prompt positions are intervened on; generation
  // continues from the steered prompt but new positions are left unmodified.
  steer_generated_tokens?: boolean;
  // Whether to drop non-word tokens (punctuation/whitespace/symbol/special)
  // from each position's per-layer read-out BEFORE selecting the top-n, so the
  // returned tokens are predominantly interesting word tokens. The model's true
  // top-1 (output) token per layer is always preserved. Probabilities stay the
  // model's real (full-vocab) probabilities. Defaults to true server-side.
  filter_non_word_tokens?: boolean;
  // When true, if the chosen server is already busy with another request, it
  // returns 429 immediately (instead of queueing) so the caller can fail over
  // to another server. Set internally by the failover logic in
  // `lensPromptStream`; not part of the public request surface. Defaults to
  // false server-side (queue/wait for the lock as before).
  fail_if_busy?: boolean;
}

// Lens read-out for one (position, lens_type). All token references are
// decoded STRINGS, never ids.
export interface LensTypeSlice {
  type: LensType;
  // [n_layers][top_n]
  top_tokens: string[][];
  top_probs: number[][];
}

// A single chat-formatted prompt token, sent up-front (before inference) so the
// client can render the conversation structure immediately.
export interface LensPromptToken {
  position: number;
  token: string;
  // Token id, echoed so the client can send it back as `cached_token_ids` on
  // the next turn for prefix-reuse matching.
  id: number;
  is_generated: boolean;
}

// Emitted right after `meta` and before inference begins: the chat-formatted
// prompt tokens (no lens read-outs yet) so the UI can render the full
// conversation shape (user turn + assistant scaffold) right away.
export interface LensPromptTokensMessage {
  kind: 'prompt';
  tokens: LensPromptToken[];
}

// First streamed message: the shared request context.
export interface LensMetaMessage {
  kind: 'meta';
  model: string;
  types: LensType[];
  // Selected layers per lens type (identical for every position).
  layers_by_type: Record<string, number[]>;
  top_n: number;
  prompt_len: number;
  num_completion_tokens: number;
  temperature: number;
  prepend_bos: boolean;
  // Number of leading prompt positions whose read-outs were reused from the
  // client's cache (skipped this run). Token messages are only emitted for
  // positions >= reuse_len; the client keeps its prior results for the rest.
  reuse_len: number;
}

// One per token position: the token plus its per-type lens slices.
export interface LensTokenMessage {
  kind: 'token';
  position: number;
  token: string;
  // Token id, echoed so the client can send it back as `cached_token_ids` on
  // the next turn for prefix-reuse matching.
  id: number;
  is_generated: boolean;
  results: LensTypeSlice[];
}

// Final streamed message.
export interface LensDoneMessage {
  kind: 'done';
  seq_len: number;
  prompt_len: number;
  vocab_size: number;
  completion: string;
}

export interface LensErrorMessage {
  kind: 'error';
  error: string;
}

export type LensStreamMessage =
  | LensMetaMessage
  | LensPromptTokensMessage
  | LensTokenMessage
  | LensDoneMessage
  | LensErrorMessage;

// Non-streaming response: the same messages buffered into one object.
export interface LensPromptResponse {
  meta: LensMetaMessage;
  tokens: LensTokenMessage[];
  done: LensDoneMessage;
}

export const LENS_TYPE_LABELS: Record<LensType, string> = {
  [LensType.JACOBIAN_LENS]: 'Jacobian Lens',
  [LensType.LOGIT_LENS]: 'Logit Lens',
};

// Preferred order for display columns / default selection.
export const LENS_TYPE_ORDER: LensType[] = [LensType.JACOBIAN_LENS, LensType.LOGIT_LENS];
