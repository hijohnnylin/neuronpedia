import { lensPromptStream } from '@/lib/utils/inference';
import {
  DEFAULT_LENS_COMPLETION_TOKENS,
  DEFAULT_LENS_TEMPERATURE,
  DEFAULT_LENS_TOP_N,
  LENS_TYPES,
  LensChatMessage,
  LensSteerToken,
  LensType,
  MAX_LENS_CHAT_USER_CHARS,
  MAX_LENS_COMPLETION_PROMPT_CHARS,
} from '@/lib/utils/lens';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

// Non-user chat messages (assistant / system) keep a generous cap: generated or
// edited assistant turns are replayed back through this endpoint for
// re-analysis and are legitimately longer than a user turn. User-supplied
// input (user messages + completion prompt) is capped tightly instead.
const MAX_PROMPT_CHARS = 10000;
const MAX_STEER_STRENGTH = 50;

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
  token: yup.string().required(),
  type: yup
    .string()
    .oneOf(LENS_TYPES as unknown as string[])
    .required(),
});

const lensPromptRequestSchema = yup.object({
  modelId: yup.string().min(1).required(),
  // Exactly one of `prompt` (completion) or `chat` (instruct) is required.
  prompt: yup.string().max(MAX_LENS_COMPLETION_PROMPT_CHARS).optional(),
  chat: yup.array().of(chatMessageSchema).optional(),
  type: yup
    .array()
    .of(yup.string().oneOf(LENS_TYPES as unknown as string[]))
    .min(1)
    .default([...LENS_TYPES]),
  topN: yup.number().integer().min(1).max(8).default(DEFAULT_LENS_TOP_N),
  temperature: yup.number().min(0).max(2).default(DEFAULT_LENS_TEMPERATURE),
  numCompletionTokens: yup.number().integer().min(0).max(512).default(DEFAULT_LENS_COMPLETION_TOKENS),
  prependBos: yup.boolean().default(true),
  enableThinking: yup.boolean().default(false),
  // Token ids the client already has read-outs for (prefix-reuse). The server
  // reuses the longest common token-id prefix and recomputes only new tokens.
  cachedTokenIds: yup.array().of(yup.number().integer()).optional(),
  // Exact input token ids to read out over, bypassing tokenization (and
  // generation). When provided, `prompt`/`chat` are not required — used to
  // re-run a previously-computed run's read-outs (e.g. when toggling the
  // non-word filter) without re-tokenizing or re-generating.
  inputTokenIds: yup.array().of(yup.number().integer()).optional(),
  // Whether to drop non-word tokens from each position's top-n read-out
  // server-side (the model's true top-1 per layer is always kept). Defaults to
  // true so API users get the "interesting tokens" behavior by default.
  filterNonWordTokens: yup.boolean().default(true),
  // Steering: readouts to suppress, the layers to inject at, and the strength.
  steerTokens: yup.array().of(steerTokenSchema).optional(),
  steerLayers: yup.array().of(yup.number().integer()).optional(),
  steerStrength: yup.number().min(-MAX_STEER_STRENGTH).max(MAX_STEER_STRENGTH).optional(),
  steerAblate: yup.boolean().optional(),
  // SWAP: target readout to replace the source (steerTokens[0]) with.
  swapToken: steerTokenSchema.default(undefined),
  // Apply the steer/swap intervention to generated tokens too (default false).
  steerGeneratedTokens: yup.boolean().optional(),
});

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
