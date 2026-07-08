import { lensPromptStream } from '@/lib/utils/inference';
import {
  DEFAULT_LENS_COMPLETION_TOKENS,
  DEFAULT_LENS_TEMPERATURE,
  DEFAULT_LENS_TOP_N,
  LENS_TYPES,
  LensChatMessage,
  LensDoneMessage,
  LensMetaMessage,
  LensSteerToken,
  LensTokenMessage,
  LensType,
  MAX_LENS_CHAT_USER_CHARS,
  MAX_LENS_COMPLETION_PROMPT_CHARS,
} from '@/lib/utils/lens';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

// Cap the wall time this route can hold a connection open. Matters most for the
// non-streaming path (`stream: false`), which buffers the full run before
// responding; the run's wall time counts entirely against this limit.
export const maxDuration = 180;

// Non-user chat messages (assistant / system) keep a generous cap: generated or
// edited assistant turns are replayed back through this endpoint for
// re-analysis and are legitimately longer than a user turn. User-supplied
// input (user messages + completion prompt) is capped tightly instead.
const MAX_PROMPT_CHARS = 10000;
const MAX_STEER_STRENGTH = 50;
// Upper bound on client-supplied token-id arrays (`inputTokenIds` /
// `cachedTokenIds`). Bounds worst-case compute + buffered response size; well
// above any realistic prompt (1024 chars) + generated tokens (max 256).
const MAX_TOKEN_IDS = 4096;
// Defense-in-depth caps on otherwise-unbounded array/string inputs. Generous
// relative to real usage; the goal is to reject pathological payloads early.
const MAX_CHAT_MESSAGES = 200;
const MAX_STEER_TOKENS = 16;
const MAX_STEER_LAYERS = 512;
const MAX_STEER_TOKEN_CHARS = 256;
const MAX_MODEL_ID_CHARS = 128;

const chatMessageSchema = yup.object({
  role: yup.string().oneOf(['user', 'assistant', 'system']).required(),
  content: yup
    .string()
    .required()
    .when('role', {
      is: 'user',
      then: (schema) => schema.max(MAX_LENS_CHAT_USER_CHARS),
      otherwise: (schema) => schema.max(MAX_PROMPT_CHARS),
    }),
});

const steerTokenSchema = yup.object({
  token: yup.string().max(MAX_STEER_TOKEN_CHARS).required(),
  type: yup
    .string()
    .oneOf(LENS_TYPES as unknown as string[])
    .required(),
});

const lensPromptRequestSchema = yup.object({
  modelId: yup.string().min(1).max(MAX_MODEL_ID_CHARS).required(),
  // Exactly one of `prompt` (completion) or `chat` (instruct) is required.
  prompt: yup.string().max(MAX_LENS_COMPLETION_PROMPT_CHARS).optional(),
  chat: yup.array().of(chatMessageSchema).max(MAX_CHAT_MESSAGES).optional(),
  type: yup
    .array()
    .of(yup.string().oneOf(LENS_TYPES as unknown as string[]))
    .min(1)
    .max(LENS_TYPES.length)
    .default([...LENS_TYPES]),
  topN: yup.number().integer().min(1).max(8).default(DEFAULT_LENS_TOP_N),
  temperature: yup.number().min(0).max(2).default(DEFAULT_LENS_TEMPERATURE),
  numCompletionTokens: yup.number().integer().min(0).max(256).default(DEFAULT_LENS_COMPLETION_TOKENS),
  prependBos: yup.boolean().default(true),
  enableThinking: yup.boolean().default(false),
  // Token ids the client already has read-outs for (prefix-reuse). The server
  // reuses the longest common token-id prefix and recomputes only new tokens.
  cachedTokenIds: yup.array().of(yup.number().integer()).max(MAX_TOKEN_IDS).optional(),
  // Exact input token ids to read out over, bypassing tokenization (and
  // generation). When provided, `prompt`/`chat` are not required — used to
  // re-run a previously-computed run's read-outs (e.g. when toggling the
  // non-word filter) without re-tokenizing or re-generating.
  inputTokenIds: yup.array().of(yup.number().integer()).max(MAX_TOKEN_IDS).optional(),
  // Whether to drop non-word tokens from each position's top-n read-out
  // server-side (the model's true top-1 per layer is always kept). Defaults to
  // true so API users get the "interesting tokens" behavior by default.
  filterNonWordTokens: yup.boolean().default(true),
  // Steering: readouts to suppress, the layers to inject at, and the strength.
  steerTokens: yup.array().of(steerTokenSchema).max(MAX_STEER_TOKENS).optional(),
  steerLayers: yup.array().of(yup.number().integer()).max(MAX_STEER_LAYERS).optional(),
  steerStrength: yup.number().min(-MAX_STEER_STRENGTH).max(MAX_STEER_STRENGTH).optional(),
  steerAblate: yup.boolean().optional(),
  // SWAP: target readout to replace the source (steerTokens[0]) with.
  swapToken: steerTokenSchema.default(undefined),
  // Apply the steer/swap intervention to generated tokens too (default false).
  steerGeneratedTokens: yup.boolean().optional(),
  // When true (default), stream results as NDJSON (one message per line). When
  // false, the same messages are buffered server-side and returned as a single
  // JSON object ({ meta, tokens, done }). This only affects how this endpoint
  // responds to the caller — the upstream inference call always streams.
  stream: yup.boolean().default(true),
});

/**
 * @swagger
 * /api/lens/prompt:
 *   post:
 *     summary: Run J-lens Over Prompt/Chat
 *     description: |
 *       Runs the Jacobian/Logit lens over a prompt (or chat) and returns the per-position, per-layer top read-out tokens. Results are streamed as newline-delimited JSON (NDJSON) by default, or returned as a single buffered JSON object when `stream: false` (see **Response format** below).
 *
 *       This powers the [Jacobian Lens](https://neuronpedia.org/jlens) tool. The model is run once and the requested lens types share the same residual stream, so requesting both `LOGIT_LENS` and `JACOBIAN_LENS` is essentially free.
 *
 *       Provide **exactly one** of `prompt` (completion / raw text) or `chat` (instruct / chat-formatted turns). This is all most callers need.
 *
 *       _Advanced:_ instead of `prompt`/`chat` you may supply `inputTokenIds` to read out over an exact, pre-tokenized sequence verbatim (no tokenization and no generation). This is intended for faithfully reproducing a previous run's read-outs — e.g. re-running with a different `filterNonWordTokens` setting — by feeding back the token `id` values returned on each `token` message. Most callers should ignore this field.
 *
 *       **Response format:** By default (`stream: true`) the endpoint responds with `Content-Type: application/x-ndjson` and streams one JSON object per line: a `meta` message describing the run, one `token` message per sequence position (prompt tokens then generated tokens), and finally a `done` message. If an error occurs mid-stream, an `error` message is emitted instead. Set `stream: false` to instead receive a single JSON object `{ meta, tokens, done }` once the run completes.
 *     tags:
 *       - Jacobian Lens
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - modelId
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: The Neuronpedia model id to run the lens with.
 *                 maxLength: 128
 *                 example: qwen3.6-27b
 *               prompt:
 *                 type: string
 *                 description: Raw completion prompt. Provide exactly one of `prompt` or `chat` (unless `inputTokenIds` is supplied).
 *                 maxLength: 1024
 *                 example: "The capital of France is"
 *               chat:
 *                 type: array
 *                 description: Chat-formatted conversation turns. Provide exactly one of `prompt` or `chat` (unless `inputTokenIds` is supplied).
 *                 maxItems: 200
 *                 items:
 *                   type: object
 *                   required:
 *                     - role
 *                     - content
 *                   properties:
 *                     role:
 *                       type: string
 *                       enum: [user, assistant, system]
 *                     content:
 *                       type: string
 *                       description: Message content. User messages are capped at 1024 characters; assistant/system messages at 10000.
 *                 example:
 *                   - role: user
 *                     content: "What is the capital of France? Answer in 1 word."
 *               type:
 *                 type: array
 *                 description: Which lens types to compute. Defaults to both.
 *                 maxItems: 2
 *                 items:
 *                   type: string
 *                   enum: [LOGIT_LENS, JACOBIAN_LENS]
 *                 default: [LOGIT_LENS, JACOBIAN_LENS]
 *               topN:
 *                 type: integer
 *                 description: Number of top read-out tokens to return per layer per position.
 *                 minimum: 1
 *                 maximum: 8
 *                 default: 8
 *               temperature:
 *                 type: number
 *                 description: Sampling temperature for generated tokens. 0 = greedy (argmax).
 *                 minimum: 0
 *                 maximum: 2
 *                 default: 0
 *               numCompletionTokens:
 *                 type: integer
 *                 description: Number of tokens to generate after the prompt. 0 = read out over the input only (no generation).
 *                 minimum: 0
 *                 maximum: 256
 *                 default: 128
 *               prependBos:
 *                 type: boolean
 *                 description: Whether to prepend a beginning-of-sequence token.
 *                 default: true
 *               enableThinking:
 *                 type: boolean
 *                 description: Enable "thinking" mode when applying a chat template (chat requests only).
 *                 default: false
 *               filterNonWordTokens:
 *                 type: boolean
 *                 description: Drop non-word tokens (punctuation/whitespace/symbol/special) from each position's read-out before selecting the top-n. The model's true top-1 per layer is always kept.
 *                 default: true
 *               steerTokens:
 *                 type: array
 *                 description: Read-outs to steer on (additively inject, negatively to suppress) into the residual stream.
 *                 maxItems: 16
 *                 items:
 *                   type: object
 *                   required:
 *                     - token
 *                     - type
 *                   properties:
 *                     token:
 *                       type: string
 *                       description: The exact decoded token string (whitespace preserved, e.g. " cat").
 *                       maxLength: 256
 *                     type:
 *                       type: string
 *                       enum: [LOGIT_LENS, JACOBIAN_LENS]
 *               steerLayers:
 *                 type: array
 *                 description: Layers to inject the steering direction at. Empty/omitted = the read-out layers.
 *                 maxItems: 512
 *                 items:
 *                   type: integer
 *               steerStrength:
 *                 type: number
 *                 description: Signed steering strength as a fraction of each position's residual norm (negative suppresses the read-out).
 *                 minimum: -50
 *                 maximum: 50
 *               steerAblate:
 *                 type: boolean
 *                 description: Ablate (project out) the read-out direction instead of additively steering. Mutually exclusive with steerStrength.
 *               swapToken:
 *                 type: object
 *                 description: SWAP target read-out to replace the source read-out (steerTokens[0]) with.
 *                 properties:
 *                   token:
 *                     type: string
 *                     maxLength: 256
 *                   type:
 *                     type: string
 *                     enum: [LOGIT_LENS, JACOBIAN_LENS]
 *               steerGeneratedTokens:
 *                 type: boolean
 *                 description: Apply the steer/swap intervention to generated tokens too.
 *                 default: false
 *               stream:
 *                 type: boolean
 *                 description: When true, stream results as NDJSON. When false, return a single buffered JSON object `{ meta, tokens, done }`.
 *                 default: true
 *               cachedTokenIds:
 *                 type: array
 *                 description: "Advanced. Token ids the client already has read-outs for (prefix reuse). The server reuses the longest common token-id prefix and only recomputes the new positions. Max 4096 ids."
 *                 maxItems: 4096
 *                 items:
 *                   type: integer
 *               inputTokenIds:
 *                 type: array
 *                 description: "Advanced. Exact, pre-tokenized sequence (in order) to read out over, bypassing tokenization and generation. Used to reproduce a previous run's read-outs by feeding back the token `id` values from a prior response. When provided, `prompt`/`chat` are not required. Max 4096 ids."
 *                 maxItems: 4096
 *                 items:
 *                   type: integer
 *           example:
 *             modelId: qwen3.6-27b
 *             chat: [
 *               { role: "user", content: "What is the capital of France? Answer in 1 word." }
 *             ]
 *             type: [LOGIT_LENS, JACOBIAN_LENS]
 *             topN: 8
 *             temperature: 0
 *             numCompletionTokens: 32
 *             stream: false
 *     responses:
 *       200:
 *         description: |
 *           When `stream` is true (default), an NDJSON stream (`application/x-ndjson`) with one JSON message per line: a single `meta` message, then one `token` message per position, then a final `done` message (or an `error` message). When `stream` is false, a single JSON object `{ meta, tokens, done }`.
 *         content:
 *           application/x-ndjson:
 *             schema:
 *               type: string
 *               description: Newline-delimited JSON messages (when `stream` is true).
 *             examples:
 *               stream:
 *                 summary: Example message lines
 *                 value: |
 *                   {"kind":"meta","model":"qwen3.6-27b","types":["LOGIT_LENS","JACOBIAN_LENS"],"layers_by_type":{"LOGIT_LENS":[0,1,2],"JACOBIAN_LENS":[0,1,2]},"top_n":8,"prompt_len":6,"num_completion_tokens":32,"temperature":0,"prepend_bos":true,"reuse_len":0}
 *                   {"kind":"token","position":0,"token":"The","id":464,"is_generated":false,"results":[{"type":"LOGIT_LENS","top_tokens":[["a"," the"]],"top_probs":[[0.12,0.08]]}]}
 *                   {"kind":"done","seq_len":38,"prompt_len":6,"vocab_size":151936,"completion":" Paris."}
 *           application/json:
 *             schema:
 *               type: object
 *               description: Buffered response (when `stream` is false).
 *               properties:
 *                 meta:
 *                   type: object
 *                   description: The run's `meta` message.
 *                 tokens:
 *                   type: array
 *                   description: One `token` message per sequence position.
 *                   items:
 *                     type: object
 *                 done:
 *                   type: object
 *                   description: The final `done` message, including the generated `completion` string.
 *       400:
 *         description: Invalid JSON, validation error, or not exactly one of `prompt`/`chat` provided.
 *       500:
 *         description: Lens request failed or an internal error occurred.
 */
export async function POST(request: Request) {
  try {
    let body;
    try {
      body = await request.json();
    } catch (error) {
      return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
    }

    const validated = await lensPromptRequestSchema.validate(body);

    const hasPrompt = typeof validated.prompt === 'string' && validated.prompt.length > 0;
    const hasChat = Array.isArray(validated.chat) && validated.chat.length > 0;
    const hasInputTokenIds = Array.isArray(validated.inputTokenIds) && validated.inputTokenIds.length > 0;
    // When exact input token ids are supplied we read out over them verbatim
    // (no tokenization/generation), so `prompt`/`chat` aren't required.
    if (!hasInputTokenIds && hasPrompt === hasChat) {
      return NextResponse.json({ error: "Provide exactly one of 'prompt' or 'chat'" }, { status: 400 });
    }

    const inferenceResponse = await lensPromptStream(
      validated.modelId,
      {
        type: validated.type as LensType[],
        prompt: !hasInputTokenIds && hasPrompt ? validated.prompt : undefined,
        chat: !hasInputTokenIds && hasChat ? (validated.chat as LensChatMessage[]) : undefined,
        input_token_ids: hasInputTokenIds ? (validated.inputTokenIds as number[]) : undefined,
        top_n: validated.topN,
        temperature: validated.temperature,
        num_completion_tokens: validated.numCompletionTokens,
        prepend_bos: validated.prependBos,
        enable_thinking: validated.enableThinking,
        cached_token_ids: (validated.cachedTokenIds as number[] | undefined) ?? undefined,
        steer_tokens: (validated.steerTokens as LensSteerToken[] | undefined) ?? undefined,
        steer_layers: (validated.steerLayers as number[] | undefined) ?? undefined,
        steer_strength: validated.steerStrength,
        steer_ablate: validated.steerAblate,
        swap_token: (validated.swapToken as LensSteerToken | undefined) ?? undefined,
        steer_generated_tokens: validated.steerGeneratedTokens,
        filter_non_word_tokens: validated.filterNonWordTokens,
        stream: true,
      },
      request.signal,
    );

    if (!inferenceResponse.ok || !inferenceResponse.body) {
      const errorBody = await inferenceResponse.json().catch(() => ({ error: inferenceResponse.statusText }));
      return NextResponse.json(
        { error: errorBody.error ?? `Lens request failed (${inferenceResponse.status})` },
        { status: inferenceResponse.status >= 400 ? inferenceResponse.status : 500 },
      );
    }

    // Non-streaming callers get the same messages buffered into a single JSON
    // object. The upstream inference call always streams (see lensPromptStream),
    // so we consume that stream here without touching the pass-through path below.
    if (!validated.stream) {
      const text = await inferenceResponse.text();
      let meta: LensMetaMessage | null = null;
      const tokens: LensTokenMessage[] = [];
      let done: LensDoneMessage | null = null;
      for (const line of text.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed) {
          continue;
        }
        let msg;
        try {
          msg = JSON.parse(trimmed);
        } catch {
          continue;
        }
        if (msg.kind === 'meta') {
          meta = msg as LensMetaMessage;
        } else if (msg.kind === 'token') {
          tokens.push(msg as LensTokenMessage);
        } else if (msg.kind === 'done') {
          done = msg as LensDoneMessage;
        } else if (msg.kind === 'error') {
          return NextResponse.json({ error: msg.error || 'Lens stream error' }, { status: 500 });
        }
      }
      if (!meta || !done) {
        return NextResponse.json({ error: 'Lens request produced incomplete data' }, { status: 500 });
      }
      return NextResponse.json({ meta, tokens, done });
    }

    // Pipe the NDJSON stream straight through to the browser.
    return new Response(inferenceResponse.body, {
      status: 200,
      headers: {
        'Content-Type': 'application/x-ndjson; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
      },
    });
  } catch (error) {
    console.error('Error in lens prompt route:', error);
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: 'Validation error', details: error.errors }, { status: 400 });
    }
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}
