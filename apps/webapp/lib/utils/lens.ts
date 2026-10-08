import type { components } from '@/lib/api/inference';
import { JPP_LENS_MODEL_IDS } from '@/lib/env';
// Shared types + constants for the streaming Jacobian/Logit lens endpoint
// (`POST /v1/lens/prompt` on the inference server). Pure types only — safe to
// import from both server and client code.
//
// The endpoint streams NDJSON (one JSON message per line): a single `meta`
// message, then one `token` message per sequence position, then a final
// `done` message (or an `error` message). When `stream` is false the same
// messages are buffered into a single `{ meta, tokens, done }` object.

// Canonical string values for the lens types. Reference these (e.g.
// `LensType.JACOBIAN_LENS`) instead of bare string literals to avoid typos.
// JPP_LENS (the J++ Lens) has the same form as the Jacobian lens, from another fit.
export const LensType = {
  LOGIT_LENS: 'LOGIT_LENS',
  JACOBIAN_LENS: 'JACOBIAN_LENS',
  JPP_LENS: 'JPP_LENS',
} as const;
export type LensType = (typeof LensType)[keyof typeof LensType];

export const LENS_TYPES = [LensType.LOGIT_LENS, LensType.JACOBIAN_LENS, LensType.JPP_LENS] as const;

// The types that carry a layer through a fitted J_bar (the logit lens does not).
export function isJBarLensType(t: LensType): boolean {
  return t === LensType.JACOBIAN_LENS || t === LensType.JPP_LENS;
}

// Whether the inference servers for `modelId` serve the J++ Lens.
export function jppLensAvailable(modelId: string | null | undefined): boolean {
  return !!modelId && JPP_LENS_MODEL_IDS.includes(modelId);
}

// The share's stored lens tab. DIFF is the old two-column view; it now loads as
// the Jacobian and Logit columns.
export const LensMode = {
  JACOBIAN_LENS: LensType.JACOBIAN_LENS,
  LOGIT_LENS: LensType.LOGIT_LENS,
  DIFF: 'DIFF',
} as const;
export type LensMode = (typeof LensMode)[keyof typeof LensMode];

export const LENS_MODES = [LensMode.JACOBIAN_LENS, LensMode.LOGIT_LENS, LensMode.DIFF] as const;

// A column the UI can show. The oracle is not a `LensType`: its read comes from
// `/v1/lens/oracle`, one position at a time, as text.
export const LensColumn = {
  JACOBIAN_LENS: LensType.JACOBIAN_LENS,
  JPP_LENS: LensType.JPP_LENS,
  ORACLE_LENS: 'ORACLE_LENS',
  LOGIT_LENS: LensType.LOGIT_LENS,
} as const;
export type LensColumn = (typeof LensColumn)[keyof typeof LensColumn];

// Display order of the columns (and of the toggle buttons).
export const LENS_COLUMN_ORDER: LensColumn[] = [
  LensColumn.JACOBIAN_LENS,
  LensColumn.JPP_LENS,
  LensColumn.ORACLE_LENS,
  LensColumn.LOGIT_LENS,
];

export function isLensTypeColumn(c: LensColumn): c is LensType {
  return c === LensColumn.JACOBIAN_LENS || c === LensColumn.JPP_LENS || c === LensColumn.LOGIT_LENS;
}

// The columns a legacy share tab opens with.
export function lensColumnsFromTab(tab: string | null | undefined): LensColumn[] {
  if (tab === LensMode.DIFF) {
    return [LensColumn.JACOBIAN_LENS, LensColumn.LOGIT_LENS];
  }
  if (tab === LensMode.LOGIT_LENS) {
    return [LensColumn.LOGIT_LENS];
  }
  return [LensColumn.JACOBIAN_LENS];
}

// A share's columns: `lensColumns` in display order, or the legacy tab's
// columns when `lensColumns` has none.
export function shareLensColumns(lensColumns: string[] | null | undefined, tab: string | null | undefined) {
  const columns = LENS_COLUMN_ORDER.filter((c) => lensColumns?.includes(c));
  return columns.length > 0 ? columns : lensColumnsFromTab(tab);
}

// The legacy tab stored next to `lensColumns`: the token lenses in `columns`.
export function lensTabFromColumns(columns: LensColumn[]): LensMode {
  const j = columns.includes(LensColumn.JACOBIAN_LENS);
  const l = columns.includes(LensColumn.LOGIT_LENS);
  if (j && l) {
    return LensMode.DIFF;
  }
  return l ? LensMode.LOGIT_LENS : LensMode.JACOBIAN_LENS;
}

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
// Ceiling on generated tokens, per model: the chat interface's slider maximum,
// and the bound both `/api/lens/prompt` and `/api/lens/share` validate against.
// All three have to agree — a run the UI lets you generate must be one the
// prompt endpoint accepts and the share endpoint can then store. Only the models
// listed below get the highest ceiling; the rest stop at the lower default.
export const MAX_LENS_COMPLETION_TOKENS = 2048;
export const DEFAULT_MAX_LENS_COMPLETION_TOKENS = 1024;
const MAX_LENS_COMPLETION_TOKENS_BY_MODEL: Record<string, number> = {
  'deepseek-v4-flash': MAX_LENS_COMPLETION_TOKENS,
};
export function maxLensCompletionTokens(modelId: string): number {
  return MAX_LENS_COMPLETION_TOKENS_BY_MODEL[modelId] ?? DEFAULT_MAX_LENS_COMPLETION_TOKENS;
}
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

// A tool call in an assistant message. `arguments` is the call's JSON object.
export interface LensChatToolCall {
  name: string;
  arguments?: Record<string, unknown>;
  id?: string;
}

// `role` is `system`, `user`, `assistant` or `tool`. The chat template renders
// `toolCalls` (assistant only) and `toolCallId` (tool only, the call it answers).
export interface LensChatMessage {
  role: string;
  content: string;
  toolCalls?: LensChatToolCall[];
  toolCallId?: string;
}

// A tool definition, in the OpenAI function-schema shape that chat templates
// read: `{ type: 'function', function: { name, description, parameters } }`.
export type LensChatTool = Record<string, unknown>;

// Chat payload limits shared by `/api/lens/prompt` and `/api/lens/share`.
export const MAX_LENS_CHAT_TOOL_CALLS = 64;
export const MAX_LENS_CHAT_TOOLS = 128;
export const MAX_LENS_CHAT_TOOL_NAME_CHARS = 256;
export const MAX_LENS_CHAT_TOOL_ARGUMENT_CHARS = 10000;
export const MAX_LENS_CHAT_TOOLS_CHARS = 50000;

// A single readout to steer on. `token` is the EXACT decoded token string
// (whitespace preserved, e.g. " cat") as it appeared in a read-out slice; the
// server resolves it back to a vocab id. `type` selects which lens's readout
// direction to use (Jacobian or J++: J_bar^T·w_t; Logit: plain unembedding w_t).
export interface LensSteerToken {
  token: string;
  type: LensType;
}

/**
 * The lens request as inference declares it, in camelCase.
 *
 * Derived from the spec rather than hand-written: this used to be a snake_case duplicate of
 * the python model, which meant every field existed in three places and the route had to
 * translate between two of them.
 *
 * Every field is optional because the server supplies defaults; `model` and `type` are the
 * only ones it truly needs, and omitting them surfaces as a 422.
 */
export type LensPromptRequest = Partial<components['schemas']['LensPromptRequest']>;

// Error body for a lens request refused before the stream starts. `token` and
// `suggestedToken` are set when a steer/swap token is not in the vocabulary.
export type LensErrorResponse = components['schemas']['LensErrorResponse'];

// The streamed NDJSON frames, mirroring the models in apps/inference's lens/prompt.py.
//
// Hand-written because NDJSON frames never reach openapi.json, so there is nothing to
// generate from. They are not renamed on the way through: inference declares these frames on
// `PublicFrameSchema`, which leaves the field names un-aliased precisely because they are what
// `/api/lens/prompt` publishes and what the stored share blobs contain. Inference's
// test_lens_frame_contract.py pins the names, so a change there fails before it reaches here.

// Lens read-out for one (position, lens_type). All token references are
// decoded STRINGS, never ids.
export interface LensTypeSlice {
  type: LensType;
  // [n_layers][top_n]
  top_tokens: string[][];
  top_probs: number[][];
}

// Per-token chat-span metadata (role / channel / section / message index),
// computed server-side by the engine's tokenize layer (the single source of
// truth for message boundaries). All fields are null on raw-text / reproduction
// requests that carry no chat messages, in which case the client renders the
// tokens plainly. `section` is one of "header" | "content" | "footer" |
// "scaffold"; `channel` is a harmony channel (analysis/final/commentary) or null.
export interface LensTokenSpan {
  message_index?: number | null;
  role?: string | null;
  channel?: string | null;
  section?: string | null;
}

// True on the 2nd..nth position of a character that is split across tokens (an
// emoji, typically). The server repeats the whole glyph in `token` at every
// contributing position so each chip renders it; this flag is what makes that
// repetition reversible, so anything rebuilding TEXT from tokens must go through
// `tokensToText` (jlens-chat-format.ts) rather than joining `token`. Optional
// because runs predating the flag (stored fixtures) simply don't carry it.
export interface LensCharContinuation {
  is_char_continuation?: boolean;
}

// A single chat-formatted prompt token, sent up-front (before inference) so the
// client can render the conversation structure immediately.
export interface LensPromptToken extends LensTokenSpan, LensCharContinuation {
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
  // Layers the server's oracle lens reads. Empty (or absent, on older servers
  // and stored runs) when the server has no oracle.
  oracle_layers?: number[];
}

// One per token position: the token plus its per-type lens slices.
export interface LensTokenMessage extends LensTokenSpan, LensCharContinuation {
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
  [LensType.JPP_LENS]: 'J++ Lens',
  [LensType.LOGIT_LENS]: 'Logit Lens',
};

// Short labels for a lens type's read-out, its sidebar and its space.
export const LENS_TYPE_SHORT_LABELS: Record<LensType, string> = {
  [LensType.JACOBIAN_LENS]: 'J-Lens',
  [LensType.JPP_LENS]: 'J++ Lens',
  [LensType.LOGIT_LENS]: 'Logit Lens',
};
export const LENS_TYPE_READOUT_LABELS: Record<LensType, string> = {
  [LensType.JACOBIAN_LENS]: 'J-Lens Readout',
  [LensType.JPP_LENS]: 'J++ Readout',
  [LensType.LOGIT_LENS]: 'Logit Lens',
};
export const LENS_TYPE_SPACE_LABELS: Record<LensType, string> = {
  [LensType.JACOBIAN_LENS]: 'J-Space',
  [LensType.JPP_LENS]: 'J++ Space',
  [LensType.LOGIT_LENS]: 'Logit Lens',
};

export const LENS_COLUMN_LABELS: Record<LensColumn, string> = {
  [LensColumn.JACOBIAN_LENS]: 'Jacobian',
  [LensColumn.JPP_LENS]: 'J++',
  [LensColumn.ORACLE_LENS]: 'Oracle',
  [LensColumn.LOGIT_LENS]: 'Logit',
};

// ---- Oracle lens (`POST /v1/lens/oracle`) ---------------------------------
// One read describes the activation at one position, in words, at each oracle
// layer. Frames mirror the models in apps/inference's lens/oracle.py.

// Every read is one bullet of at most ORACLE_MAX_TOKENS tokens.
export const ORACLE_BULLETS = 1;
export const ORACLE_MAX_TOKENS = 32;
// The largest bullet cap a stored share can have.
export const MAX_ORACLE_BULLETS = 5;
export const MAX_ORACLE_POSITIONS = 64;

export type LensOracleRequest = Partial<components['schemas']['LensOracleRequest']>;

// `position` and `token` are the first of `positions` and `tokens`. Servers
// that read one position per request send neither list.
export interface LensOracleMetaMessage {
  kind: 'meta';
  model: string;
  adapter: string;
  position: number;
  token: string;
  positions?: number[];
  tokens?: string[];
  layers: number[];
  max_bullets: number;
}

// One layer's read at one position. Reads arrive in the order they end.
// Servers that read one position per request send no `position`.
export interface LensOracleReadMessage {
  kind: 'read';
  position?: number;
  layer: number;
  bullets: string[];
  text: string;
  // "bullets" (the cap), "eos", or "length".
  finish: string;
  cached: boolean;
  // Set by the webapp: lets `/api/lens/share` store this read as is.
  sig?: string;
}

export interface LensOracleDoneMessage {
  kind: 'done';
  elapsed_ms: number;
  cached_layers: number;
}

// One layer's text so far, while its read runs. Its `read` comes after.
export interface LensOraclePartialMessage {
  kind: 'partial';
  position?: number;
  layer: number;
  text: string;
}

export type LensOracleStreamMessage =
  | LensOracleMetaMessage
  | LensOraclePartialMessage
  | LensOracleReadMessage
  | LensOracleDoneMessage
  | LensErrorMessage;

// Preferred order for display columns / default selection.
export const LENS_TYPE_ORDER: LensType[] = [LensType.JACOBIAN_LENS, LensType.JPP_LENS, LensType.LOGIT_LENS];

// The types to request for `modelId`: the J++ Lens only where its servers have it.
export function requestLensTypes(modelId: string | null | undefined): LensType[] {
  return LENS_TYPE_ORDER.filter((t) => t !== LensType.JPP_LENS || jppLensAvailable(modelId));
}
