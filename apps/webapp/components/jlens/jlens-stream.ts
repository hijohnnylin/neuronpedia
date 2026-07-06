// Client helper to POST to `/api/lens/prompt` and consume the NDJSON stream
// (one JSON message per line). Invokes the provided callbacks as `meta`,
// `token`, and `done` messages arrive, and throws on `error` messages or a
// non-ok response.

import {
  LensChatMessage,
  LensDoneMessage,
  LensMetaMessage,
  LensPromptTokensMessage,
  LensSteerToken,
  LensStreamMessage,
  LensTokenMessage,
  LensType,
} from '@/lib/utils/lens';

export interface RunLensStreamParams {
  modelId: string;
  prompt?: string;
  chat?: LensChatMessage[];
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
    throw new Error(data.error ?? `Request failed (${res.status})`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';

  const handleLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    let msg: LensStreamMessage;
    try {
      msg = JSON.parse(trimmed) as LensStreamMessage;
    } catch {
      return;
    }
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
  };

  // eslint-disable-next-line no-constant-condition
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
