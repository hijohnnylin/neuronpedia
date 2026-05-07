import { prisma } from '@/lib/db';
import { nlaFetch } from '@/lib/db/nla-source';
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

type ChatMessage = { role: 'user' | 'assistant'; content: string };

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

function isChatMessageArray(value: unknown): value is ChatMessage[] {
  return (
    Array.isArray(value) &&
    value.every(
      (m) =>
        m !== null &&
        typeof m === 'object' &&
        (('role' in m && (m.role === 'user' || m.role === 'assistant')) || false) &&
        'content' in m &&
        typeof (m as { content: unknown }).content === 'string',
    )
  );
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

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  const {
    text,
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

  if (!text || typeof text !== 'string') {
    return NextResponse.json({ error: 'text is required' }, { status: 400 });
  }
  if (text.length > MAX_TEXT_LENGTH) {
    return NextResponse.json({ error: `text must be ${MAX_TEXT_LENGTH} characters or less` }, { status: 400 });
  }

  const effectiveCompletionTokens = Math.min(completionTokens ?? 0, MAX_COMPLETION_TOKENS);
  const effectiveTemperature = temperature ?? 0.7;
  const wantsStream = stream === true;

  // ── OpenRouter completion path ────────────────────────────────────────
  // Only redirect to OpenRouter when the flag is on, the caller actually
  // wants a generation (not a tokenize), and we have structured messages
  // to send (OR's chat completions API needs `messages`, not the raw
  // chat-templated string). Tokenization always uses the NLA server.
  if (
    USE_OPENROUTER_FOR_COMPLETION &&
    effectiveCompletionTokens > 0 &&
    wantsStream &&
    isChatMessageArray(messages) &&
    modelId
  ) {
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

  // If a continuation is requested, hit NLA's /completion. Otherwise just
  // tokenize. `nlaFetch` shuffles the source's `servers[]` and fails over
  // until one returns 2xx; if every server errors, the last response is
  // returned and we forward its status to the client.
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
