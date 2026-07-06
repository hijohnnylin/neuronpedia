import { prisma } from '@/lib/db';
import { nlaFetch } from '@/lib/db/nla-source';
import { formatChatForModel, isChatMessageArray } from '@/lib/nla-chat-template';
import { EXPLAIN_MAX_NEW_TOKENS, MAX_TEXT_LENGTH, MAX_TOKENS_TO_EXPLAIN } from '@/lib/nla-constants';
import { nlaExplainTextHash } from '@/lib/nla-explain-cache-hash';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

type SseMeta = { layer_index?: number; total?: number; prompt_length?: number };

type ExplainResultRecord = Record<string, unknown> & { position?: number };

type ParsedExplainPayload = {
  results: ExplainResultRecord[];
  layer_index: number;
  prompt_length: number;
};

function sortPositions(xs: number[]): number[] {
  return [...xs].sort((a, b) => a - b);
}

function parseSseLines(lines: string[]): { results: ExplainResultRecord[]; meta: SseMeta | null } {
  const results: ExplainResultRecord[] = [];
  let meta: SseMeta | null = null;
  lines
    .filter((line) => line.startsWith('data: '))
    .forEach((line) => {
      const data = line.slice(6).trim();
      if (data === '[DONE]') return;
      try {
        const parsed = JSON.parse(data);
        if ('layer_index' in parsed && 'total' in parsed) {
          meta = parsed;
        } else if ('description' in parsed) {
          results.push(parsed);
        }
      } catch {
        // skip malformed events
      }
    });
  return { results, meta };
}

function positionForSort(r: ExplainResultRecord): number {
  const p = r.position;
  return typeof p === 'number' && !Number.isNaN(p) ? p : Number.MAX_SAFE_INTEGER;
}

function sortExplainResultsByPosition(results: ExplainResultRecord[]): ExplainResultRecord[] {
  return [...results].sort((a, b) => positionForSort(a) - positionForSort(b));
}

async function collectExplainResults(stream: ReadableStream<Uint8Array>): Promise<ParsedExplainPayload> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const allResults: ExplainResultRecord[] = [];
  let latestMeta: SseMeta | null = null;

  while (true) {
    // eslint-disable-next-line no-await-in-loop
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    const { results: batchResults, meta: batchMeta } = parseSseLines(lines);
    allResults.push(...batchResults);
    if (batchMeta) latestMeta = batchMeta;
  }

  if (buffer) {
    const { results: batchResults, meta: batchMeta } = parseSseLines([buffer]);
    allResults.push(...batchResults);
    if (batchMeta) latestMeta = batchMeta;
  }

  return {
    results: sortExplainResultsByPosition(allResults),
    layer_index: latestMeta?.layer_index ?? 0,
    prompt_length: latestMeta?.prompt_length ?? allResults.length,
  };
}

type PriorAggregate = {
  byPosition: Map<number, ExplainResultRecord>;
  layerIndex: number;
  promptLength: number;
  tokens: string[];
};

async function fetchPriorAggregate(args: {
  text: string;
  temperature: number;
  modelId: string;
  nlaSourceId: string;
}): Promise<PriorAggregate> {
  const textHash = nlaExplainTextHash(args.text);
  const rows = await prisma.nlaExplainCache.findMany({
    where: {
      numCompletionTokens: 0,
      temperature: args.temperature,
      modelId: args.modelId,
      nlaSourceId: args.nlaSourceId,
      textHash,
    },
    orderBy: { createdAt: 'desc' },
  });

  console.log('[nla-explain route] fetchPriorAggregate', {
    queriedTextLength: args.text.length,
    queriedTextHead: args.text.slice(0, 80),
    queriedTextTail: args.text.slice(-80),
    matchingCacheRowCount: rows.length,
    matchingCacheRowIds: rows.map((r) => r.id),
    matchingCacheRowSortedPositions: rows.map((r) => r.sortedPositions),
  });

  const byPosition = new Map<number, ExplainResultRecord>();
  let layerIndex = 0;
  let promptLength = 0;
  let tokens: string[] = [];

  for (const row of rows) {
    try {
      const parsed = JSON.parse(row.resultJson) as ParsedExplainPayload;
      (parsed.results || []).forEach((r) => {
        const p = r.position;
        if (typeof p === 'number' && Number.isInteger(p) && p >= 0 && !byPosition.has(p)) {
          byPosition.set(p, r);
        }
      });
      if (parsed.layer_index && !layerIndex) layerIndex = parsed.layer_index;
      if (parsed.prompt_length && !promptLength) promptLength = parsed.prompt_length;
      if (row.tokens.length > tokens.length) tokens = row.tokens;
    } catch {
      // ignore malformed rows
    }
  }

  return { byPosition, layerIndex, promptLength, tokens };
}

async function upsertUnionRow(args: {
  text: string;
  temperature: number;
  modelId: string;
  nlaSourceId: string;
  sortedPositions: number[];
  tokens: string[];
  results: ExplainResultRecord[];
  layerIndex: number;
  promptLength: number;
}) {
  const textHash = nlaExplainTextHash(args.text);
  const resultJson = JSON.stringify({
    results: args.results,
    layer_index: args.layerIndex,
    prompt_length: args.promptLength,
  });
  const existing = await prisma.nlaExplainCache.findFirst({
    where: {
      numCompletionTokens: 0,
      temperature: args.temperature,
      modelId: args.modelId,
      nlaSourceId: args.nlaSourceId,
      sortedPositions: { equals: args.sortedPositions },
      textHash,
    },
  });
  if (existing) {
    return prisma.nlaExplainCache.update({
      where: { id: existing.id },
      data: { text: args.text, textHash, resultJson, tokens: args.tokens },
    });
  }
  return prisma.nlaExplainCache.create({
    data: {
      text: args.text,
      textHash,
      numCompletionTokens: 0,
      temperature: args.temperature,
      modelId: args.modelId,
      nlaSourceId: args.nlaSourceId,
      sortedPositions: args.sortedPositions,
      tokens: args.tokens,
      resultJson,
    },
  });
}

async function fetchPriorFromCacheId(args: {
  priorCacheId: string;
  text: string;
  temperature: number;
  modelId: string;
  nlaSourceId: string;
  requestTokens: string[];
}): Promise<PriorAggregate | null> {
  const row = await prisma.nlaExplainCache.findUnique({ where: { id: args.priorCacheId } });
  if (!row) {
    console.log('[nla-explain route] priorCacheId not found', { priorCacheId: args.priorCacheId });
    return null;
  }

  const matches =
    row.numCompletionTokens === 0 &&
    row.temperature === args.temperature &&
    row.modelId === args.modelId &&
    row.nlaSourceId === args.nlaSourceId &&
    args.text.startsWith(row.text);

  // The new request's token list (if provided) MUST match the prior row's
  // tokens for the overlapping prefix. The model is autoregressive, so
  // identical tokens at the same positions guarantee identical activations
  // — but a different tokenization of the same text (e.g. tokenizer
  // version mismatch) would silently yield wrong-looking explanations.
  let tokensPrefixMatches = true;
  if (args.requestTokens.length > 0 && row.tokens.length > 0) {
    if (args.requestTokens.length < row.tokens.length) {
      tokensPrefixMatches = false;
    } else {
      for (let i = 0; i < row.tokens.length; i += 1) {
        if (args.requestTokens[i] !== row.tokens[i]) {
          tokensPrefixMatches = false;
          break;
        }
      }
    }
  }

  if (!matches || !tokensPrefixMatches) {
    console.log('[nla-explain route] priorCacheId validation failed', {
      priorCacheId: args.priorCacheId,
      checks: {
        numCompletionTokensZero: row.numCompletionTokens === 0,
        temperatureMatches: row.temperature === args.temperature,
        modelIdMatches: row.modelId === args.modelId,
        nlaSourceIdMatches: row.nlaSourceId === args.nlaSourceId,
        textIsPrefix: args.text.startsWith(row.text),
        tokensPrefixMatches,
      },
      priorTextLength: row.text.length,
      newTextLength: args.text.length,
    });
    return null;
  }

  let parsed: ParsedExplainPayload;
  try {
    parsed = JSON.parse(row.resultJson) as ParsedExplainPayload;
  } catch {
    console.log('[nla-explain route] priorCacheId resultJson parse failed', {
      priorCacheId: args.priorCacheId,
    });
    return null;
  }

  const byPosition = new Map<number, ExplainResultRecord>();
  (parsed.results || []).forEach((r) => {
    const p = r.position;
    if (typeof p === 'number' && Number.isInteger(p) && p >= 0) byPosition.set(p, r);
  });

  console.log('[nla-explain route] priorCacheId hit & validated', {
    priorCacheId: args.priorCacheId,
    priorTextLength: row.text.length,
    priorTokenCount: row.tokens.length,
    priorExplainedPositions: Array.from(byPosition.keys()).sort((a, b) => a - b),
  });

  return {
    byPosition,
    layerIndex: parsed.layer_index ?? 0,
    promptLength: parsed.prompt_length ?? row.tokens.length,
    tokens: row.tokens,
  };
}

function mergePriorAggregates(a: PriorAggregate, b: PriorAggregate): PriorAggregate {
  // `a` wins on conflict.
  const byPosition = new Map(a.byPosition);
  b.byPosition.forEach((v, k) => {
    if (!byPosition.has(k)) byPosition.set(k, v);
  });
  return {
    byPosition,
    layerIndex: a.layerIndex || b.layerIndex,
    promptLength: a.promptLength || b.promptLength,
    tokens: a.tokens.length >= b.tokens.length ? a.tokens : b.tokens,
  };
}

/**
 * @swagger
 * /api/nla/explain:
 *   post:
 *     summary: Explain NLA Activations at Token Positions
 *     tags:
 *       - NLA
 *     security:
 *       - apiKey: []
 *       - {}
 *     description: |
 *       Returns natural-language descriptions of the activations at the
 *       requested token positions in a prompt, produced by the Natural
 *       Language Autoencoder (NLA) at a specific source (AV/AR
 *       pair). See the `GET /api/nla/sources` for available NLA sources.
 *
 *       Provide either `text` (a chat-templated string) OR `messages`
 *       (an array of `{role, content}` chat turns that the server will
 *       template using the model's chat template).
 *
 *       Each request may target at most 16 *new* (uncached) positions.
 *       Previously-explained positions for the same `(text, modelId,
 *       nlaSourceId, temperature)` are served from cache and don't count
 *       toward this limit.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - modelId
 *               - nlaSourceId
 *               - positions
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: The Neuronpedia model id (e.g. `gemma-3-27b-it`, or for Llama3, `llama3.3-70b-it`).
 *               nlaSourceId:
 *                 type: string
 *                 description: The NLA source id for this model (e.g. `kitft-l41`, or e.g. `kitft-l53` for Llama3). See `GET /api/nla/sources`.
 *               text:
 *                 type: string
 *                 description: Pre-chat-templated prompt string. Provide this OR `messages`. Max 16384 characters.
 *               messages:
 *                 type: array
 *                 description: Chat-format turns. Server applies the model's chat template. Provide this OR `text`.
 *                 items:
 *                   type: object
 *                   required: [role, content]
 *                   properties:
 *                     role: { type: string, enum: [user, assistant] }
 *                     content: { type: string }
 *               positions:
 *                 type: array
 *                 description: Token positions (0-indexed) to explain. Up to 16 new positions per request.
 *                 items: { type: integer, minimum: 0 }
 *               temperature:
 *                 type: number
 *                 description: Sampling temperature for the NLA explainer. Default `0.7`.
 *             example:
 *               modelId: gemma-3-27b-it   # For Llama3 use: llama3.3-70b-it
 *               nlaSourceId: kitft-l41    # For Llama3 use: kitft-l53
 *               messages:
 *                 - role: user
 *                   content: What is the capital of Canada?
 *               positions: [4, 7, 9]
 *     responses:
 *       200:
 *         description: Explanations for the requested positions, plus the canonical token list of the prompt.
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 results:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       position: { type: integer }
 *                       token: { type: string }
 *                       description: { type: string }
 *                 layer_index: { type: integer }
 *                 prompt_length: { type: integer }
 *       400:
 *         description: Invalid request (missing text/messages, bad positions, too many new positions, etc).
 *       429:
 *         description: Rate-limited. Default cap is 120 requests/hour per IP.
 */
export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  const {
    text: rawText,
    messages,
    temperature,
    modelId,
    nlaSourceId,
    positions,
    tokens,
    priorCacheId,
    stream,
  } = body as {
    text?: string;
    messages?: unknown;
    temperature?: number;
    modelId?: string;
    nlaSourceId?: string;
    positions?: number[];
    tokens?: string[];
    priorCacheId?: string;
    stream?: boolean;
  };

  // Render server-side when the caller passes `messages` instead of a
  // pre-templated `text`. We prefer `text` if both are supplied so the
  // frontend (which always sends the rendered string) is unaffected.
  let text: string | undefined = typeof rawText === 'string' ? rawText : undefined;
  if (!text && isChatMessageArray(messages)) {
    if (!modelId) {
      return NextResponse.json(
        { error: 'modelId is required when using `messages` (so the server can pick the chat template).' },
        { status: 400 },
      );
    }
    text = formatChatForModel(modelId, messages);
  }

  if (!text || typeof text !== 'string') {
    return NextResponse.json({ error: 'Provide either `text` or `messages`.' }, { status: 400 });
  }
  if (text.length > MAX_TEXT_LENGTH) {
    return NextResponse.json({ error: `text must be ${MAX_TEXT_LENGTH} characters or less` }, { status: 400 });
  }
  // Default the API contract to non-streaming JSON. The frontend explicitly
  // passes `stream: true` to opt into the SSE-with-cacheId behavior it was
  // built against; external callers omit it and get a single JSON object.
  const wantsStream = stream === true;

  // Positions are required: every cache row is keyed on (prompt, source,
  // sortedPositions). The client now sends the *cumulative* set of positions
  // it wants explained (already-explained ∪ newly-selected); the route
  // reuses any prior cache rows covering subsets of that set and only
  // forwards the genuinely-missing positions to the upstream NLA server.
  if (!Array.isArray(positions) || positions.length === 0) {
    return NextResponse.json({ error: 'positions is required and must be a non-empty array' }, { status: 400 });
  }
  if (!positions.every((p) => Number.isInteger(p) && p >= 0)) {
    return NextResponse.json({ error: 'positions must contain non-negative integers' }, { status: 400 });
  }

  const sortedPositions = sortPositions(Array.from(new Set(positions)));

  console.log('[nla-explain route] incoming request', {
    textLength: text.length,
    textHead: text.slice(0, 80),
    textTail: text.slice(-80),
    sortedPositions,
    sortedPositionsCount: sortedPositions.length,
    modelId: modelId ?? '',
    nlaSourceId: nlaSourceId ?? '',
    temperature: temperature ?? 0.7,
    priorCacheId: priorCacheId ?? null,
  });

  // 1 token >= 1 character in practice, so positions can't realistically
  // exceed MAX_TEXT_LENGTH unique entries — anything beyond that is
  // malformed and we reject early.
  if (sortedPositions.length > MAX_TEXT_LENGTH) {
    return NextResponse.json(
      { error: `positions must contain at most ${MAX_TEXT_LENGTH} unique entries` },
      { status: 400 },
    );
  }
  const effectiveTemperature = temperature ?? 0.7;
  const effectiveModelId = modelId ?? '';
  const effectiveNlaSourceId = nlaSourceId ?? '';
  // The client tokenizes locally before calling /explain, so it knows the
  // full prompt token list. We persist it on the cache row so a follow-up
  // `?id=...` deep-link can hydrate without re-tokenizing.
  const requestTokens: string[] = Array.isArray(tokens) && tokens.every((t) => typeof t === 'string') ? tokens : [];

  // 1. Fast path: exact cache hit on the cumulative set.
  const requestTextHash = nlaExplainTextHash(text);
  const exact = await prisma.nlaExplainCache.findFirst({
    where: {
      numCompletionTokens: 0,
      temperature: effectiveTemperature,
      modelId: effectiveModelId,
      nlaSourceId: effectiveNlaSourceId,
      sortedPositions: { equals: sortedPositions },
      textHash: requestTextHash,
    },
  });
  if (exact) {
    console.log('[nla-explain route] EXACT cache hit', {
      cacheId: exact.id,
      sortedPositions,
    });
    const parsed = JSON.parse(exact.resultJson);
    return NextResponse.json({ ...parsed, cacheId: exact.id });
  }
  console.log('[nla-explain route] no exact cache hit; aggregating prior rows');

  // 2. Try the priorCacheId hint first: if the client just finished an
  // explain run and is now extending the prompt (typical chat
  // continuation), the cumulative `text` differs from the prior row's
  // text, so the exact-text fetchPriorAggregate query below would miss.
  // The prior row's explanations are still valid for the overlapping
  // prefix because the model is autoregressive — identical tokens at
  // identical positions yield identical activations.
  let prior: PriorAggregate = {
    byPosition: new Map<number, ExplainResultRecord>(),
    layerIndex: 0,
    promptLength: 0,
    tokens: [],
  };
  if (priorCacheId) {
    const priorFromCacheId = await fetchPriorFromCacheId({
      priorCacheId,
      text,
      temperature: effectiveTemperature,
      modelId: effectiveModelId,
      nlaSourceId: effectiveNlaSourceId,
      requestTokens,
    });
    if (priorFromCacheId) prior = priorFromCacheId;
  }

  // 3. Also pull rows whose `text` exactly matches this request — covers
  // the "user clicks Explain again with a different selection on the
  // same prompt" path (no prefix-extension involved). Merge into prior
  // (cacheId-derived entries win to keep the more recent results).
  const exactTextPrior = await fetchPriorAggregate({
    text,
    temperature: effectiveTemperature,
    modelId: effectiveModelId,
    nlaSourceId: effectiveNlaSourceId,
  });
  prior = mergePriorAggregates(prior, exactTextPrior);

  const missing = sortedPositions.filter((p) => !prior.byPosition.has(p));
  const tokensToStore = requestTokens.length > 0 ? requestTokens : prior.tokens;

  console.log('[nla-explain route] prior aggregate', {
    priorCachedPositions: Array.from(prior.byPosition.keys()).sort((a, b) => a - b),
    priorCachedCount: prior.byPosition.size,
    requestedCount: sortedPositions.length,
    missingPositions: missing,
    missingCount: missing.length,
    willCallUpstreamNlaForMissing: missing.length > 0,
  });

  // Cap NEW (non-cached) positions only. The cumulative `positions` set may
  // exceed MAX_TOKENS_TO_EXPLAIN when prior rows already cover the excess
  // (cache hits are free), but anything we'd actually forward to the
  // upstream NLA server must respect the per-request cap.
  if (missing.length > MAX_TOKENS_TO_EXPLAIN) {
    console.log('[nla-explain route] too many new positions to explain', {
      requestedCount: sortedPositions.length,
      priorCachedCount: prior.byPosition.size,
      missingCount: missing.length,
      maxNewPerRequest: MAX_TOKENS_TO_EXPLAIN,
    });
    return NextResponse.json(
      {
        error:
          `Too many new token positions to explain in a single request: ` +
          `received ${missing.length} new positions (${prior.byPosition.size} of ` +
          `${sortedPositions.length} requested were already cached); ` +
          `the limit is ${MAX_TOKENS_TO_EXPLAIN} new positions per request.`,
      },
      { status: 400 },
    );
  }

  // 4. All requested positions already cached across prior rows — just
  // materialize a union row pointing at the same data and return.
  if (missing.length === 0) {
    const results = sortedPositions.map((p) => prior.byPosition.get(p)!) as ExplainResultRecord[];
    const layerIndex = prior.layerIndex || 0;
    const promptLength = prior.promptLength || results.length;
    const row = await upsertUnionRow({
      text,
      temperature: effectiveTemperature,
      modelId: effectiveModelId,
      nlaSourceId: effectiveNlaSourceId,
      sortedPositions,
      tokens: tokensToStore,
      results,
      layerIndex,
      promptLength,
    });
    return NextResponse.json({ results, layer_index: layerIndex, prompt_length: promptLength, cacheId: row.id });
  }

  // 5. Forward only the missing positions to the upstream NLA server.
  // `nlaFetch` shuffles the source's `servers[]` and fails over until
  // one returns 2xx; if every server errors, the last response (e.g.
  // 429 from a busy backend) is returned and we forward its status.
  const nlaResponse = await nlaFetch(modelId, nlaSourceId, '/explain', {
    method: 'POST',
    body: JSON.stringify({
      text,
      temperature: effectiveTemperature,
      stream: true,
      positions: missing,
      max_new_tokens: EXPLAIN_MAX_NEW_TOKENS,
    }),
  });

  if (!nlaResponse.ok) {
    // const errorText = await nlaResponse.text();
    return NextResponse.json({ error: `NLA server error: ${nlaResponse.status}` }, { status: nlaResponse.status });
  }

  if (!nlaResponse.body) {
    return NextResponse.json({ error: 'No response body from NLA server' }, { status: 502 });
  }

  // ── Non-streaming API path ─────────────────────────────────────────
  // Buffer the upstream SSE into a single JSON payload. This is the
  // default for external callers (and any caller that doesn't pass
  // `stream: true`). The cache row is written before responding so the
  // returned `cacheId` is immediately valid for follow-up requests.
  if (!wantsStream) {
    const parsed = await collectExplainResults(nlaResponse.body);
    const merged = new Map(prior.byPosition);
    parsed.results.forEach((r) => {
      const p = r.position;
      if (typeof p === 'number' && Number.isInteger(p) && p >= 0) merged.set(p, r);
    });
    const orderedResults = sortedPositions
      .map((p) => merged.get(p))
      .filter((r): r is ExplainResultRecord => Boolean(r));
    const layerIndex = parsed.layer_index || prior.layerIndex || 0;
    const promptLength = parsed.prompt_length || prior.promptLength || orderedResults.length;
    let cacheId: string | null = null;
    if (orderedResults.length > 0) {
      try {
        const row = await upsertUnionRow({
          text,
          temperature: effectiveTemperature,
          modelId: effectiveModelId,
          nlaSourceId: effectiveNlaSourceId,
          sortedPositions,
          tokens: tokensToStore,
          results: orderedResults,
          layerIndex,
          promptLength,
        });
        cacheId = row.id;
      } catch (err) {
        console.error('Failed to cache NLA explain results:', err);
      }
    }
    return NextResponse.json({
      results: orderedResults,
      layer_index: layerIndex,
      prompt_length: promptLength,
      ...(cacheId ? { cacheId } : {}),
    });
  }

  // ── Streaming path (frontend) ──────────────────────────────────────
  // Tee: one branch streams to the client; the other is consumed in the
  // background to assemble the union cache row.
  const [clientBody, collectBody] = nlaResponse.body.tee();

  const cacheIdPromise: Promise<string | null> = collectExplainResults(collectBody)
    .then(async (parsed) => {
      const merged = new Map(prior.byPosition);
      parsed.results.forEach((r) => {
        const p = r.position;
        if (typeof p === 'number' && Number.isInteger(p) && p >= 0) merged.set(p, r);
      });
      const orderedResults = sortedPositions
        .map((p) => merged.get(p))
        .filter((r): r is ExplainResultRecord => Boolean(r));
      if (orderedResults.length === 0) return null;
      const layerIndex = parsed.layer_index || prior.layerIndex || 0;
      const promptLength = parsed.prompt_length || prior.promptLength || orderedResults.length;
      const row = await upsertUnionRow({
        text,
        temperature: effectiveTemperature,
        modelId: effectiveModelId,
        nlaSourceId: effectiveNlaSourceId,
        sortedPositions,
        tokens: tokensToStore,
        results: orderedResults,
        layerIndex,
        promptLength,
      });
      return row.id;
    })
    .catch((err) => {
      console.error('Failed to cache NLA explain results:', err);
      return null;
    });

  // Wrap the upstream stream so we can intercept its `[DONE]` sentinel,
  // wait for the cache write to land, and emit a final `cacheId` event
  // before signalling done to the client. Any caller can use the cacheId
  // to update the URL so reloads hydrate from the cumulative row.
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = '';

  const transform = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      buffer += decoder.decode(chunk, { stream: true });
      while (true) {
        const eventEnd = buffer.indexOf('\n\n');
        if (eventEnd === -1) break;
        const event = buffer.slice(0, eventEnd);
        buffer = buffer.slice(eventEnd + 2);
        const isDone = event.split('\n').some((line) => line.startsWith('data: ') && line.slice(6).trim() === '[DONE]');
        if (isDone) continue;
        controller.enqueue(encoder.encode(`${event}\n\n`));
      }
    },
    async flush(controller) {
      if (buffer.length > 0) {
        controller.enqueue(encoder.encode(buffer));
        buffer = '';
      }
      const cacheId = await cacheIdPromise;
      if (cacheId) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ cacheId })}\n\n`));
      }
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
    },
  });

  return new NextResponse(clientBody.pipeThrough(transform), {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  });
});
