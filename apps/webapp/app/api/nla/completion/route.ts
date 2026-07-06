import { prisma } from '@/lib/db';
import { nlaFetch } from '@/lib/db/nla-source';
import {
  ChatMessage,
  formatChatForModel,
  formatChatWithAssistantCompletion,
  isChatMessageArray,
} from '@/lib/nla-chat-template';
import { MAX_COMPLETION_TOKENS, MAX_TEXT_LENGTH } from '@/lib/nla-constants';
import { OpenAIClientFactory } from '@/lib/openai-client';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// When true, assistant chat completions are produced via OpenRouter using the
// model row's `openRouterId`. Tokenization and explanations always use the
// NLA inference server (positions / token IDs must match the source model
// the NLA pair was trained on). Flip this for environments that have
// `OPENROUTER_API_KEY` set and want to avoid self-hosting the source model
// for generation.
//
// IMPORTANT: `openRouterId` MUST resolve to the same underlying model the
// NLA server has loaded, otherwise the explanations will be analyzing
// activations of a different model than what generated the text.
const USE_OPENROUTER_FOR_COMPLETION = true;

type TokenInfo = {
  token: string;
  token_id: number;
  position: number;
  fragment_index?: number;
  fragment_count?: number;
};

type TokenizeResponse = {
  tokens: TokenInfo[];
  prompt_length: number;
  text: string;
};

async function fetchNlaTokenize(args: {
  text: string;
  modelId?: string;
  nlaSourceId?: string;
}): Promise<TokenizeResponse> {
  const res = await nlaFetch(args.modelId, args.nlaSourceId, '/tokenize', {
    method: 'POST',
    body: JSON.stringify({ text: args.text }),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`NLA tokenize error: ${res.status} - ${errorText}`);
  }
  return (await res.json()) as TokenizeResponse;
}

/**
 * Non-streaming OpenRouter completion for the documented API path.
 *
 * Calls OpenRouter once with `stream: false`, awaits the full assistant
 * text, then tokenizes the *canonical full chat* (prompt + assistant
 * turn) via NLA's `/tokenize` so the returned positions/token_ids match
 * what a follow-up `/api/nla/explain` request would see. This is the
 * critical step that's not safe to skip — OpenRouter returns text
 * deltas, not BPE tokens, so naively echoing its tokens back would give
 * researchers positions that don't agree with NLA's tokenizer.
 *
 * Response shape (option (a) from the API design discussion — no
 * separate `prompt_length`; callers can identify their own prompt by
 * string-matching against `full_text`):
 *
 *   {
 *     completion: string,       // assistant's text only
 *     full_text:  string,       // canonical chat-templated string
 *     tokens:     TokenInfo[],  // NLA tokens spanning the full_text
 *   }
 */
async function nonStreamingOpenRouterCompletion(args: {
  messages: ChatMessage[];
  openRouterModel: string;
  modelId: string;
  nlaSourceId?: string;
  temperature: number;
  completionTokens: number;
}): Promise<NextResponse> {
  const client = OpenAIClientFactory.createClient({ provider: 'openrouter' });
  const createParams = {
    model: args.openRouterModel,
    messages: args.messages,
    temperature: args.temperature,
    max_tokens: args.completionTokens,
    stream: false as const,
    // provider: { quantizations: ['bf16'] },
  };
  const response = await client.chat.completions.create(
    createParams as unknown as Parameters<typeof client.chat.completions.create>[0],
  );
  // The non-streaming SDK return type is `ChatCompletion`; widen here
  // because the union with the streaming variant doesn't narrow without
  // explicit `stream: false` typing on the call site.
  const completionText = (response as { choices?: { message?: { content?: string } }[] }).choices?.[0]?.message
    ?.content;
  if (typeof completionText !== 'string' || completionText.length === 0) {
    return NextResponse.json({ error: 'OpenRouter returned an empty completion.' }, { status: 502 });
  }

  // Re-render the entire chat-templated string with the assistant's turn
  // appended. The NLA tokenizer will see `prompt + assistant_completion`
  // exactly as a follow-up `/explain` call would, so the positions match.
  const fullText = formatChatWithAssistantCompletion(args.modelId, args.messages, completionText);

  const tokenized = await fetchNlaTokenize({
    text: fullText,
    modelId: args.modelId,
    nlaSourceId: args.nlaSourceId,
  });

  return NextResponse.json({
    completion: completionText,
    full_text: fullText,
    tokens: tokenized.tokens,
  });
}

async function streamOpenRouterCompletion(args: {
  text: string;
  messages: ChatMessage[];
  openRouterModel: string;
  modelId?: string;
  nlaSourceId?: string;
  temperature: number;
  completionTokens: number;
}): Promise<NextResponse> {
  const tokenized = await fetchNlaTokenize({
    text: args.text,
    modelId: args.modelId,
    nlaSourceId: args.nlaSourceId,
  });

  const client = OpenAIClientFactory.createClient({ provider: 'openrouter' });

  // Restrict OpenRouter to bf16 providers only. The NLA pair was trained
  // against the bf16 reference model, so quantized providers (fp8/int4/etc)
  // would generate text whose activations diverge from what the NLA server
  // sees when it re-runs the same prompt for explanations.
  // See https://openrouter.ai/docs/features/provider-routing#quantization
  // `provider` is an OpenRouter-specific extension not in the OpenAI SDK
  // types, so we widen the params object before passing it through.
  const createParams = {
    model: args.openRouterModel,
    messages: args.messages,
    temperature: args.temperature,
    max_tokens: args.completionTokens,
    stream: true as const,
    // provider: { quantizations: ['bf16'] },
  };
  const stream = await client.chat.completions.create(
    createParams as unknown as Parameters<typeof client.chat.completions.create>[0] & { stream: true },
  );

  const encoder = new TextEncoder();

  const sseStream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        // Emit the prompt event first so the client can populate the
        // selection chips with the NLA tokenizer's token strings.
        const promptEvent = {
          type: 'prompt',
          prompt_length: tokenized.prompt_length,
          tokens: tokenized.tokens,
        };
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(promptEvent)}\n\n`));

        // OpenRouter sends OpenAI-shaped chunks with text deltas (not
        // model tokens). The chat panel just concatenates each chunk's
        // `token.token` string into the assistant turn's text and
        // re-tokenizes the canonical chat after the stream completes,
        // so chunked deltas are functionally fine — they just make the
        // live token chips look chunky until the post-stream retokenize.
        //
        // Split each delta on `\n` boundaries so every emitted "token"
        // either contains no newline or ends with exactly one. The chat
        // panel inserts a flex line-break AFTER any chip whose token
        // contains `\n`, so a delta like "foo\nbar" sent as one chip
        // would render the break after "bar" instead of between "foo"
        // and "bar". Real tokenizer tokens already satisfy this
        // invariant; here we enforce it for OpenRouter deltas too.
        let position = tokenized.prompt_length;
        for await (const chunk of stream) {
          const delta = chunk.choices?.[0]?.delta?.content;
          if (typeof delta !== 'string' || delta.length === 0) continue;
          // Split only at `\n` → non-`\n` boundaries: pieces look like
          // optional-text followed by zero-or-more trailing newlines, which
          // matches the shape of real tokenizer tokens (multi-newline runs
          // like `"\n\n"` typically come back as a single BPE token, not
          // two separate `"\n"` tokens). Splitting at every `\n` would
          // produce more chips than the post-stream re-tokenize, so the
          // live view would visibly collapse blank lines once parsing
          // finished.
          const pieces: string[] = [];
          let buf = '';
          let prevWasNewline = false;
          for (const ch of delta) {
            if (prevWasNewline && ch !== '\n') {
              pieces.push(buf);
              buf = '';
            }
            buf += ch;
            prevWasNewline = ch === '\n';
          }
          if (buf.length > 0) pieces.push(buf);
          for (const piece of pieces) {
            const tokenEvent = {
              type: 'token',
              token: { token: piece, token_id: 0, position },
            };
            position += 1;
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(tokenEvent)}\n\n`));
          }
        }

        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      } catch (err) {
        const message = err instanceof Error ? err.message : 'OpenRouter stream error';
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: message })}\n\n`));
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      }
    },
  });

  return new NextResponse(sseStream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  });
}

/**
 * @swagger
 * /api/nla/completion:
 *   post:
 *     summary: Run a Chat Completion (for NLA Analysis)
 *     tags:
 *       - NLA
 *     security:
 *       - apiKey: []
 *       - {}
 *     description: |
 *       Generates an assistant response via OpenRouter using the model
 *       backing the requested NLA pair, then returns the canonical
 *       NLA-tokenized full chat (prompt + assistant turn) so you can
 *       address token positions in a follow-up `/api/nla/explain` call.
 *
 *       Provide `messages` (chat turns). The server applies the model's
 *       chat template, calls OpenRouter, and tokenizes the assembled
 *       full text with the same tokenizer NLA's `/explain` uses — so the
 *       `tokens[*].position` values you receive here are valid inputs
 *       to `/api/nla/explain`'s `positions` array.
 *
 *       See `GET /api/nla/sources` for valid `(modelId, nlaSourceId)` pairs.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - modelId
 *               - messages
 *               - completion_tokens
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: The Neuronpedia model id (e.g. `gemma-3-27b-it`). Must have an OpenRouter mapping.
 *               nlaSourceId:
 *                 type: string
 *                 description: Optional. NLA source id — controls which NLA server tokenizes the full chat. Defaults to a server configured for `modelId`.
 *               messages:
 *                 type: array
 *                 description: Chat turns. Server applies the model's chat template.
 *                 items:
 *                   type: object
 *                   required: [role, content]
 *                   properties:
 *                     role: { type: string, enum: [user, assistant] }
 *                     content: { type: string }
 *               completion_tokens:
 *                 type: integer
 *                 description: Max tokens to generate. Capped server-side at 512.
 *               temperature:
 *                 type: number
 *                 description: Sampling temperature. Default `0.7`.
 *             example:
 *               modelId: gemma-3-27b-it   # For the llama model it's modelId: llama3.3-70b-it, nlaSourceId: kitft-l53
 *               nlaSourceId: kitft-l41
 *               messages:
 *                 - role: user
 *                   content: What is the capital of Canada?
 *               completion_tokens: 32
 *               temperature: 0.7
 *     responses:
 *       200:
 *         description: Chat completion plus the canonical NLA tokenization of the full chat.
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 completion:
 *                   type: string
 *                   description: Assistant's generated text.
 *                 full_text:
 *                   type: string
 *                   description: Chat-templated `prompt + assistant_turn` string. Pass this as `text` to `/api/nla/explain`.
 *                 tokens:
 *                   type: array
 *                   description: NLA tokenizer's token list spanning the full chat.
 *                   items:
 *                     type: object
 *                     properties:
 *                       token: { type: string }
 *                       token_id: { type: integer }
 *                       position: { type: integer }
 *       400:
 *         description: Invalid request (missing messages, bad modelId, etc).
 *       429:
 *         description: Rate-limited. Default cap is 240 requests/hour per IP.
 */
export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  const {
    text: rawText,
    completion_tokens: completionTokens,
    temperature,
    stream,
    modelId,
    nlaSourceId,
    messages,
  } = body as {
    text?: string;
    completion_tokens?: number;
    temperature?: number;
    stream?: boolean;
    modelId?: string;
    nlaSourceId?: string;
    messages?: unknown;
  };

  const effectiveCompletionTokens = Math.min(completionTokens ?? 0, MAX_COMPLETION_TOKENS);
  const effectiveTemperature = temperature ?? 0.7;
  // Streaming is an internal/legacy contract retained for the chat UI; the
  // documented API path defaults to non-streaming (returns a single JSON
  // object). External callers omit `stream` and get the clean response;
  // the chat UI explicitly passes `stream: true`.
  const wantsStream = stream === true;
  const hasMessages = isChatMessageArray(messages);

  // Resolve the chat-templated `text`. The chat UI sends both `text`
  // (pre-rendered client-side) and `messages`; API callers typically send
  // only `messages` and we render server-side.
  let text: string | undefined = typeof rawText === 'string' ? rawText : undefined;
  if (!text && hasMessages && modelId) {
    text = formatChatForModel(modelId, messages);
  }

  if (!text || typeof text !== 'string') {
    return NextResponse.json({ error: 'Provide `messages` (or `text`).' }, { status: 400 });
  }
  if (text.length > MAX_TEXT_LENGTH) {
    return NextResponse.json(
      { error: `Chat exceeds the maximum character length of ${MAX_TEXT_LENGTH}.` },
      { status: 400 },
    );
  }

  // ── Documented API path: non-streaming OpenRouter completion ──────────
  // When the caller wants a generation and supplies structured messages,
  // and we have an OpenRouter mapping, this is the path researchers hit.
  // Always uses OpenRouter so the wire format and provider are consistent
  // across API calls. Returns canonical NLA-tokenized positions.
  if (!wantsStream && USE_OPENROUTER_FOR_COMPLETION && effectiveCompletionTokens > 0 && hasMessages && modelId) {
    const model = await prisma.model.findUnique({ where: { id: modelId } });
    if (!model?.openRouterId) {
      return NextResponse.json({ error: `Model ${modelId} has no openRouterId configured` }, { status: 400 });
    }
    return nonStreamingOpenRouterCompletion({
      messages,
      openRouterModel: model.openRouterId,
      modelId,
      nlaSourceId,
      temperature: effectiveTemperature,
      completionTokens: effectiveCompletionTokens,
    });
  }

  // ── Streaming OpenRouter path (chat UI) ───────────────────────────────
  // Mirrors the previous behavior: the chat UI passes `stream: true` plus
  // `messages` and gets SSE prompt/token events. Same provider-routing
  // rules as the non-streaming API path (must have an openRouterId).
  if (USE_OPENROUTER_FOR_COMPLETION && effectiveCompletionTokens > 0 && wantsStream && hasMessages && modelId) {
    const model = await prisma.model.findUnique({ where: { id: modelId } });
    if (!model?.openRouterId) {
      return NextResponse.json({ error: `Model ${modelId} has no openRouterId configured` }, { status: 400 });
    }
    return streamOpenRouterCompletion({
      text,
      messages,
      openRouterModel: model.openRouterId,
      modelId,
      nlaSourceId,
      temperature: effectiveTemperature,
      completionTokens: effectiveCompletionTokens,
    });
  }

  // ── Legacy direct-NLA completion path ─────────────────────────────────
  // Retained for environments without an OpenRouter mapping (e.g. dev
  // setups using a self-hosted NLA server's /completion endpoint). Not
  // part of the documented API contract.
  if (effectiveCompletionTokens > 0) {
    const nlaResponse = await nlaFetch(modelId, nlaSourceId, '/completion', {
      method: 'POST',
      body: JSON.stringify({
        text,
        completion_tokens: effectiveCompletionTokens,
        temperature: effectiveTemperature,
        stream: wantsStream,
      }),
    });

    if (!nlaResponse.ok) {
      const errorText = await nlaResponse.text();
      return NextResponse.json(
        { error: `NLA server error: ${nlaResponse.status} - ${errorText}` },
        { status: nlaResponse.status },
      );
    }

    if (wantsStream) {
      if (!nlaResponse.body) {
        return NextResponse.json({ error: 'NLA server returned empty stream body' }, { status: 502 });
      }
      return new NextResponse(nlaResponse.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache, no-transform',
          Connection: 'keep-alive',
        },
      });
    }

    const data = await nlaResponse.json();
    return NextResponse.json(data);
  }

  // ── Tokenize-only fallback (completion_tokens === 0) ──────────────────
  // Hit by the chat UI's post-stream re-tokenize pass; also usable as a
  // tokenize-only call. Not part of the documented API surface (researchers
  // should call the dedicated tokenize step that's part of `/api/nla/completion`'s
  // main path, or compute tokens from a /completion response's `tokens`).
  const nlaResponse = await nlaFetch(modelId, nlaSourceId, '/tokenize', {
    method: 'POST',
    body: JSON.stringify({ text }),
  });

  if (!nlaResponse.ok) {
    const errorText = await nlaResponse.text();
    return NextResponse.json(
      { error: `NLA server error: ${nlaResponse.status} - ${errorText}` },
      { status: nlaResponse.status },
    );
  }

  const data = await nlaResponse.json();
  return NextResponse.json(data);
});
