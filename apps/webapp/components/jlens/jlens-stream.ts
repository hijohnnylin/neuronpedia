// Client helpers to POST to `/api/lens/prompt` and `/api/lens/oracle` and
// consume their NDJSON streams (one JSON message per line). They invoke the
// callbacks as messages arrive, and throw on `error` messages or a non-ok
// response.

import {
  LensChatMessage,
  LensChatTool,
  LensDoneMessage,
  LensMetaMessage,
  LensOracleMetaMessage,
  LensOraclePartialMessage,
  LensOracleReadMessage,
  LensOracleStreamMessage,
  LensPromptTokensMessage,
  LensSteerToken,
  LensStreamMessage,
  LensTokenMessage,
  LensType,
} from '@/lib/utils/lens';

// A steer/swap token that is not a single vocab token. `suggestedToken` is the
// closest token that is, or null if the server found none.
export class LensUnknownTokenError extends Error {
  token: string;

  suggestedToken: string | null;

  constructor(message: string, token: string, suggestedToken: string | null) {
    super(message);
    this.name = 'LensUnknownTokenError';
    this.token = token;
    this.suggestedToken = suggestedToken;
  }
}

export interface RunLensStreamParams {
  modelId: string;
  prompt?: string;
  chat?: LensChatMessage[];
  // Tool definitions for `chat`, rendered into the prompt by the chat template.
  tools?: LensChatTool[];
  type: LensType[];
  topN: number;
  temperature: number;
  numCompletionTokens: number;
  prependBos?: boolean;
  enableThinking?: boolean;
  // Token ids the client already has read-outs for (prefix-reuse). The server
  // reuses the longest common token-id prefix and streams only new positions.
  cachedTokenIds?: number[];
  // Exact input token ids to read out over, bypassing tokenization/generation.
  // Used to re-run a prior run's read-outs (e.g. toggling the non-word filter)
  // without re-tokenizing or re-generating. When set, `prompt`/`chat` are
  // ignored server-side.
  inputTokenIds?: number[];
  // Whether to drop non-word tokens from each position's top-n read-out
  // server-side (the true top-1 per layer is always kept). Defaults to true.
  filterNonWordTokens?: boolean;
  // Steering: readouts to additively suppress, the layers to inject at, and the
  // signed strength. When set, the server disables prefix-reuse.
  steerTokens?: LensSteerToken[];
  steerLayers?: number[];
  steerStrength?: number;
  // When true, ablate (project out) the readout direction instead of additively
  // steering (mutually exclusive with steerStrength).
  steerAblate?: boolean;
  // SWAP: when set, replace the source readout (steerTokens[0]) with this target
  // readout (subtract source projection, add it back along the target). Takes
  // precedence over steerStrength / steerAblate.
  swapToken?: LensSteerToken;
  // Apply the steer/swap intervention to generated tokens too (default false =
  // prompt positions only).
  steerGeneratedTokens?: boolean;
  signal?: AbortSignal;
  onMeta?: (meta: LensMetaMessage) => void;
  onPromptTokens?: (prompt: LensPromptTokensMessage) => void;
  onToken?: (token: LensTokenMessage) => void;
  onDone?: (done: LensDoneMessage) => void;
  // Remaining requests in the current hourly window for `/api/lens/prompt`,
  // surfaced via the `x-limit-remaining` response header (set by the top-level
  // rate-limit middleware). Called with the parsed number after every request,
  // or with `0` when the request is rejected as rate-limited (the 429 response
  // doesn't carry the header).
  onRateLimit?: (remaining: number) => void;
}

export async function runLensStream(params: RunLensStreamParams): Promise<void> {
  const {
    modelId,
    prompt,
    chat,
    tools,
    type,
    topN,
    temperature,
    numCompletionTokens,
    prependBos,
    enableThinking,
    cachedTokenIds,
    inputTokenIds,
    filterNonWordTokens,
    steerTokens,
    steerLayers,
    steerStrength,
    steerAblate,
    swapToken,
    steerGeneratedTokens,
    signal,
    onMeta,
    onPromptTokens,
    onToken,
    onDone,
    onRateLimit,
  } = params;

  const res = await fetch('/api/lens/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      modelId,
      prompt,
      chat,
      tools: tools?.length ? tools : undefined,
      type,
      topN,
      temperature,
      numCompletionTokens,
      prependBos,
      enableThinking,
      cachedTokenIds,
      inputTokenIds,
      filterNonWordTokens,
      steerTokens,
      steerLayers,
      steerStrength,
      steerAblate,
      swapToken,
      steerGeneratedTokens,
    }),
    signal,
  });

  const remainingHeader = res.headers.get('x-limit-remaining');
  if (remainingHeader !== null) {
    onRateLimit?.(Number(remainingHeader));
  }

  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => ({}));
    // The middleware rate-limit 429 body carries `limitPerWindow` (and doesn't
    // set the `x-limit-remaining` header on the rejection), so surface a
    // friendly message and zero out the counter. Other 429s (e.g. inference
    // server busy) fall through to the generic error.
    if (res.status === 429 && typeof data?.limitPerWindow === 'number') {
      onRateLimit?.(0);
      throw new Error('Hourly limit reached. Please wait a bit and try again later.');
    }
    const message = data.error ?? `Request failed (${res.status})`;
    if (typeof data?.token === 'string') {
      throw new LensUnknownTokenError(message, data.token, data.suggestedToken ?? null);
    }
    throw new Error(message);
  }

  await readNdjson<LensStreamMessage>(res.body, (msg) => {
    switch (msg.kind) {
      case 'meta':
        onMeta?.(msg);
        break;
      case 'prompt':
        onPromptTokens?.(msg);
        break;
      case 'token':
        onToken?.(msg);
        break;
      case 'done':
        onDone?.(msg);
        break;
      case 'error':
        throw new Error(msg.error || 'Lens stream error');
      default:
        break;
    }
  });
}

// Calls `onMessage` with each JSON line of an NDJSON body. Lines that do not
// parse are skipped.
async function readNdjson<T>(body: NonNullable<Response['body']>, onMessage: (msg: T) => void): Promise<void> {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  const handleLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    let msg: T;
    try {
      msg = JSON.parse(trimmed) as T;
    } catch {
      return;
    }
    onMessage(msg);
  };
  while (true) {
    // eslint-disable-next-line no-await-in-loop
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += value;
    let newlineIdx = buffer.indexOf('\n');
    while (newlineIdx !== -1) {
      const line = buffer.slice(0, newlineIdx);
      buffer = buffer.slice(newlineIdx + 1);
      handleLine(line);
      newlineIdx = buffer.indexOf('\n');
    }
  }
  // Flush any trailing line (no final newline).
  if (buffer.trim()) {
    handleLine(buffer);
  }
}

export interface RunOracleStreamParams {
  modelId: string;
  // The token ids of the run; only `tokenIds[0..position]` affect the read.
  tokenIds: number[];
  position: number;
  // More positions to read in the same request.
  positions?: number[];
  layers?: number[];
  // False: no `onPartial` calls.
  partial?: boolean;
  signal?: AbortSignal;
  onMeta?: (msg: LensOracleMetaMessage) => void;
  // A layer's text so far, before its read.
  onPartial?: (msg: LensOraclePartialMessage) => void;
  onRead?: (msg: LensOracleReadMessage) => void;
}

// An error whose status says the servers have no oracle for this model.
export class OracleUnavailableError extends Error {}

export async function runOracleStream(params: RunOracleStreamParams): Promise<void> {
  const { modelId, tokenIds, position, positions, layers, partial, signal } = params;
  const { onMeta, onPartial, onRead } = params;
  const res = await fetch('/api/lens/oracle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modelId, tokenIds, position, positions, layers, partial }),
    signal,
  });
  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => ({}));
    if (res.status === 404) {
      throw new OracleUnavailableError(data.error ?? 'The oracle lens is not available for this model.');
    }
    if (res.status === 429 && typeof data?.limitPerWindow === 'number') {
      throw new Error('Hourly limit reached. Please wait a bit and try again later.');
    }
    throw new Error(data.error ?? `Request failed (${res.status})`);
  }
  await readNdjson<LensOracleStreamMessage>(res.body, (msg) => {
    if (msg.kind === 'meta') {
      onMeta?.(msg);
    } else if (msg.kind === 'partial') {
      onPartial?.(msg);
    } else if (msg.kind === 'read') {
      onRead?.(msg);
    } else if (msg.kind === 'error') {
      throw new Error(msg.error || 'Oracle stream error');
    }
  });
}
