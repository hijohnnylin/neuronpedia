import { NP_GRAPH_BUCKET } from '@/app/[modelId]/graph/utils';
import { JlensExport, JlensExportSteer } from '@/components/jlens/jlens-export';
import { prisma } from '@/lib/db';
import { lensPromptStream } from '@/lib/utils/inference';
import {
  JLENS_ANONYMOUS_USER_ID,
  JLENS_S3_DIR,
  makeJlensSharePath,
  MAX_JLENS_SHARE_DESCRIPTION_LENGTH,
  MAX_JLENS_SHARE_PUT_REQUESTS_PER_DAY,
  MAX_JLENS_SHARE_UPLOAD_SIZE_BYTES,
} from '@/lib/utils/jlens-share';
import {
  LENS_MODES,
  LENS_TYPE_ORDER,
  LENS_TYPES,
  LensChatMessage,
  LensMetaMessage,
  LensTokenMessage,
  LensType,
  MAX_LENS_STEER_STRENGTH,
} from '@/lib/utils/lens';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import cuid from 'cuid';
import { headers } from 'next/headers';
import { NextResponse } from 'next/server';
import { gzip } from 'node-gzip';
import * as yup from 'yup';

// Defense-in-depth caps on otherwise-unbounded array/string inputs. The final
// S3 blob is size-guarded separately, but several of these (lockedTokens,
// selectedPositions) are persisted directly to the DB, so cap them up front.
// The inference re-run also enforces its own token limit (lens_token_limit).
const MAX_TOKEN_IDS = 4096;
const MAX_CHAT_MESSAGES = 200;
const MAX_MESSAGE_CHARS = 10000;
const MAX_SHARE_PROMPT_CHARS = 1024;
const MAX_LOCKED_TOKENS = 512;
const MAX_LOCKED_TOKEN_KEY_CHARS = 256;
const MAX_SELECTED_POSITIONS = 4096;
const MAX_STEER_LAYERS = 512;
const MAX_MODEL_ID_CHARS = 128;

const chatMessageSchema = yup.object({
  role: yup.string().oneOf(['user', 'assistant', 'system']).required(),
  content: yup.string().max(MAX_MESSAGE_CHARS).required(),
});

const lockedTokenSchema = yup.object({
  key: yup.string().max(MAX_LOCKED_TOKEN_KEY_CHARS).required(),
  type: yup
    .string()
    .oneOf(LENS_TYPES as unknown as string[])
    .required(),
});

// Optional steered run carried with the share: the steer config plus the full
// token-id sequence of the steered run, so we can reproduce its read-outs
// server-side (forced decode over those ids, no generation).
const shareSteerSchema = yup.object({
  token: yup.string().max(MAX_LOCKED_TOKEN_KEY_CHARS).required(),
  type: yup
    .string()
    .oneOf(LENS_TYPES as unknown as string[])
    .required(),
  layers: yup.array().of(yup.number().integer().required()).min(1).max(MAX_STEER_LAYERS).required(),
  strength: yup.number().min(-MAX_LENS_STEER_STRENGTH).max(MAX_LENS_STEER_STRENGTH).required(),
  ablate: yup.boolean().required(),
  // Intervention mode + the target token for a swap (empty for plain steers).
  mode: yup.string().oneOf(['steer', 'swap']).default('steer'),
  swapToken: yup.string().max(MAX_LOCKED_TOKEN_KEY_CHARS).default(''),
  // Whether the intervention was applied to generated tokens too.
  steerGenerated: yup.boolean().default(false),
  inputTokenIds: yup.array().of(yup.number().integer().required()).min(1).max(MAX_TOKEN_IDS).required(),
});

const shareRequestSchema = yup.object({
  modelId: yup.string().min(1).max(MAX_MODEL_ID_CHARS).required(),
  kind: yup.string().oneOf(['chat', 'completion']).default('chat'),
  // Source of the run — provide exactly one:
  //   - `prompt` (completion) or `chat` (instruct): the server tokenizes,
  //     generates `numCompletionTokens`, and stores that fresh run.
  //   - `inputTokenIds` (advanced): forced decode over an exact sequence (no
  //     generation), used by the UI to faithfully reproduce the viewed run.
  // When `inputTokenIds` is present it takes precedence and `prompt`/`chat` (and
  // `messages`) are treated as display metadata only (backward-compatible path).
  inputTokenIds: yup.array().of(yup.number().integer().required()).min(1).max(MAX_TOKEN_IDS).optional(),
  // Chat generation input (instruct), mirrors the `/prompt` endpoint's `chat`.
  chat: yup.array().of(chatMessageSchema).max(MAX_CHAT_MESSAGES).optional(),
  // Chat-kind shares carry the conversation turns; completion-kind shares carry
  // the raw prompt text. For the `inputTokenIds` path these are display data;
  // for the `chat` generation path the turns come from `chat` instead.
  messages: yup.array().of(chatMessageSchema).max(MAX_CHAT_MESSAGES).default([]),
  prompt: yup.string().max(MAX_SHARE_PROMPT_CHARS).default(''),
  topN: yup.number().integer().min(1).max(10).required(),
  temperature: yup.number().min(0).max(2).required(),
  numCompletionTokens: yup.number().integer().min(0).max(256).required(),
  // Number of leading prompt tokens (the remainder were generated). Persisted so
  // a reloaded share can mark the prompt→generated boundary; optional for
  // backward-compat.
  numPromptTokens: yup.number().integer().min(0).optional(),
  activeLensModeTab: yup
    .string()
    .oneOf(LENS_MODES as unknown as string[])
    .required(),
  hideNonWordTokens: yup.boolean().required(),
  lockedTokens: yup.array().of(lockedTokenSchema).max(MAX_LOCKED_TOKENS).default([]),
  selectedPositions: yup.array().of(yup.number().integer().min(0).required()).max(MAX_SELECTED_POSITIONS).default([]),
  description: yup.string().max(MAX_JLENS_SHARE_DESCRIPTION_LENGTH).optional(),
  steer: shareSteerSchema.default(undefined),
});

// Consume the inference NDJSON stream (buffered) into meta + tokens.
async function parseLensNdjson(body: string): Promise<{ meta: LensMetaMessage | null; tokens: LensTokenMessage[] }> {
  let meta: LensMetaMessage | null = null;
  const tokens: LensTokenMessage[] = [];
  for (const line of body.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    const msg = JSON.parse(trimmed);
    if (msg.kind === 'meta') {
      meta = msg as LensMetaMessage;
    } else if (msg.kind === 'token') {
      tokens.push(msg as LensTokenMessage);
    } else if (msg.kind === 'error') {
      throw new Error(msg.error || 'Lens stream error');
    }
  }
  return { meta, tokens };
}

/**
 * @swagger
 * /api/lens/share:
 *   post:
 *     summary: Share J-lens Run
 *     description: |
 *       Creates a shareable, permanent snapshot of a [J-lens](https://neuronpedia.org/jlens) run.
 *
 *       Provide **exactly one** of `prompt` (completion / raw text) or `chat` (instruct / chat-formatted turns) — same input surface as `/api/lens/prompt`. The server tokenizes, generates `numCompletionTokens`, computes the read-outs, gzips the result to S3, and creates a share record. Because the run is computed server-side, the stored data is trusted (not caller-supplied) and the prompt/generated boundary is derived automatically.
 *
 *       _Advanced:_ instead of `prompt`/`chat` you may supply `inputTokenIds` to reproduce an exact, pre-tokenized run verbatim (a forced decode, no generation). This is what the JLens UI uses to share exactly the run being viewed. When `inputTokenIds` is present it takes precedence, and `prompt`/`messages` are stored only as display metadata.
 *
 *       Authentication is optional: authenticated shares are attributed to the user, anonymous shares are attributed to an anonymous owner. Requests are rate-limited per IP address per day.
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
 *               - topN
 *               - temperature
 *               - numCompletionTokens
 *               - activeLensModeTab
 *               - hideNonWordTokens
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: The Neuronpedia model id the run was computed with.
 *                 maxLength: 128
 *                 example: qwen3.6-27b
 *               prompt:
 *                 type: string
 *                 description: Raw completion prompt to run and share. Provide exactly one of `prompt` or `chat` (unless `inputTokenIds` is supplied).
 *                 maxLength: 1024
 *                 example: "The capital of France is"
 *               chat:
 *                 type: array
 *                 description: Chat-formatted conversation turns to run and share. Provide exactly one of `prompt` or `chat` (unless `inputTokenIds` is supplied).
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
 *                       maxLength: 10000
 *               kind:
 *                 type: string
 *                 description: "Advanced. Explicit run kind for the `inputTokenIds` path. For the `prompt`/`chat` path the kind is inferred from the input given."
 *                 enum: [chat, completion]
 *                 default: chat
 *               messages:
 *                 type: array
 *                 description: "Advanced. Conversation turns stored for display on the `inputTokenIds` path. For the `chat` generation path, use `chat` instead."
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
 *                       maxLength: 10000
 *               topN:
 *                 type: integer
 *                 description: Number of top read-out tokens returned per layer per position.
 *                 minimum: 1
 *                 maximum: 10
 *               temperature:
 *                 type: number
 *                 description: Sampling temperature used for the run.
 *                 minimum: 0
 *                 maximum: 2
 *               numCompletionTokens:
 *                 type: integer
 *                 description: Number of generated tokens in the run.
 *                 minimum: 0
 *                 maximum: 256
 *               activeLensModeTab:
 *                 type: string
 *                 description: The lens display mode tab that was active when sharing.
 *                 enum: [JACOBIAN_LENS, LOGIT_LENS, DIFF]
 *               hideNonWordTokens:
 *                 type: boolean
 *                 description: Whether non-word tokens were filtered from the read-outs. The stored snapshot is captured with this filter applied.
 *               lockedTokens:
 *                 type: array
 *                 description: Tokens the user pinned in the UI.
 *                 maxItems: 512
 *                 items:
 *                   type: object
 *                   required:
 *                     - key
 *                     - type
 *                   properties:
 *                     key:
 *                       type: string
 *                       maxLength: 256
 *                     type:
 *                       type: string
 *                       enum: [LOGIT_LENS, JACOBIAN_LENS]
 *               selectedPositions:
 *                 type: array
 *                 description: Token positions the user selected in the UI.
 *                 maxItems: 4096
 *                 items:
 *                   type: integer
 *                   minimum: 0
 *               description:
 *                 type: string
 *                 description: Optional description for the shared run.
 *               inputTokenIds:
 *                 type: array
 *                 description: "Advanced. Exact, pre-tokenized sequence to reproduce verbatim (forced decode, no generation). When provided, takes precedence over `prompt`/`chat` and is the path the JLens UI uses."
 *                 minItems: 1
 *                 maxItems: 4096
 *                 items:
 *                   type: integer
 *               numPromptTokens:
 *                 type: integer
 *                 description: "Advanced. Number of leading prompt tokens for the `inputTokenIds` path (the remainder were generated). Ignored on the `prompt`/`chat` path, where it's derived from the run."
 *                 minimum: 0
 *               steer:
 *                 type: object
 *                 description: "Advanced. Optional steered run to carry with the share. Its read-outs are re-computed server-side over `inputTokenIds`."
 *                 required:
 *                   - token
 *                   - type
 *                   - layers
 *                   - strength
 *                   - ablate
 *                   - inputTokenIds
 *                 properties:
 *                   token:
 *                     type: string
 *                     description: The exact decoded read-out token that was steered on.
 *                     maxLength: 256
 *                   type:
 *                     type: string
 *                     enum: [LOGIT_LENS, JACOBIAN_LENS]
 *                   layers:
 *                     type: array
 *                     minItems: 1
 *                     maxItems: 512
 *                     items:
 *                       type: integer
 *                   strength:
 *                     type: number
 *                     minimum: -2
 *                     maximum: 2
 *                   ablate:
 *                     type: boolean
 *                   mode:
 *                     type: string
 *                     enum: [steer, swap]
 *                     default: steer
 *                   swapToken:
 *                     type: string
 *                     description: Target token for a swap intervention (empty for plain steers).
 *                     maxLength: 256
 *                     default: ""
 *                   steerGenerated:
 *                     type: boolean
 *                     description: Whether the intervention was applied to generated tokens too.
 *                     default: false
 *                   inputTokenIds:
 *                     type: array
 *                     description: Full token-id sequence of the steered run.
 *                     minItems: 1
 *                     maxItems: 4096
 *                     items:
 *                       type: integer
 *           example:
 *             modelId: gemma-3-12b
 *             prompt: "The capital of France is"
 *             topN: 8
 *             temperature: 0
 *             numCompletionTokens: 3
 *             activeLensModeTab: JACOBIAN_LENS
 *             hideNonWordTokens: true
 *     responses:
 *       200:
 *         description: The shared run was created successfully.
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 id:
 *                   type: string
 *                   description: The share id.
 *                 path:
 *                   type: string
 *                   description: The relative path to view the shared run on Neuronpedia.
 *                 url:
 *                   type: string
 *                   description: The S3 url of the stored run data.
 *       400:
 *         description: Invalid JSON body or validation error.
 *       413:
 *         description: The shared run exceeds the maximum upload size.
 *       429:
 *         description: Too many shares created from this IP address today.
 *       500:
 *         description: Lens re-run failed, or uploading/saving the share failed.
 */
export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  let bodyJson;
  try {
    bodyJson = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  let body;
  try {
    body = await shareRequestSchema.validate(bodyJson);
  } catch (error) {
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: 'Validation error', details: error.errors }, { status: 400 });
    }
    return NextResponse.json({ error: 'Validation error' }, { status: 400 });
  }

  const userId = request.user?.id ?? null;

  // Determine the run source. `inputTokenIds` (the UI's faithful-reproduction
  // path) takes precedence; otherwise the server generates from `prompt`/`chat`
  // like the `/prompt` endpoint. Exactly one of `prompt`/`chat` is required in
  // the generation path.
  const hasInputTokenIds = Array.isArray(body.inputTokenIds) && body.inputTokenIds.length > 0;
  const hasPrompt = typeof body.prompt === 'string' && body.prompt.length > 0;
  const hasChat = Array.isArray(body.chat) && body.chat.length > 0;
  if (!hasInputTokenIds && hasPrompt === hasChat) {
    return NextResponse.json({ error: "Provide exactly one of 'prompt', 'chat', or 'inputTokenIds'" }, { status: 400 });
  }
  // For the generation path, the kind follows the input given (chat vs prompt).
  // For the `inputTokenIds` path, honor the caller's `kind` (backward-compatible).
  const effectiveKind: 'chat' | 'completion' = hasInputTokenIds
    ? (body.kind as 'chat' | 'completion')
    : hasChat
      ? 'chat'
      : 'completion';

  // Per-IP/day rate limit (mirrors the graph put-request limit).
  const headersList = await headers();
  const forwardedFor = headersList.get('x-forwarded-for');
  const ip = forwardedFor ? forwardedFor.split(',')[0].trim() : 'unknown';

  const recentPutRequests = await prisma.jlensSharePutRequest.count({
    where: {
      ipAddress: ip,
      createdAt: { gte: new Date(Date.now() - 24 * 60 * 60 * 1000) },
    },
  });
  if (recentPutRequests >= MAX_JLENS_SHARE_PUT_REQUESTS_PER_DAY) {
    return NextResponse.json(
      { error: `Too many shares today. The maximum is ${MAX_JLENS_SHARE_PUT_REQUESTS_PER_DAY}.` },
      { status: 429 },
    );
  }

  // Run inference server-side so the stored data is trusted (computed by us, not
  // the caller) and reproducible. Either a forced decode over the exact token
  // ids (the UI reproduction path, no generation) or a fresh generation from
  // `prompt`/`chat` (the simple API path).
  let meta: LensMetaMessage | null;
  let tokens: LensTokenMessage[];
  try {
    const inferenceResponse = await lensPromptStream(
      body.modelId,
      hasInputTokenIds
        ? {
            type: LENS_TYPE_ORDER,
            input_token_ids: body.inputTokenIds,
            top_n: body.topN,
            num_completion_tokens: 0,
            // Capture the run with the sharer's non-word filter applied, so the
            // stored blob matches what they were viewing (the share is a frozen
            // snapshot for this filter; viewers toggle by re-running live).
            filter_non_word_tokens: body.hideNonWordTokens,
          }
        : {
            type: LENS_TYPE_ORDER,
            prompt: hasChat ? undefined : body.prompt,
            chat: hasChat ? (body.chat as LensChatMessage[]) : undefined,
            top_n: body.topN,
            temperature: body.temperature,
            num_completion_tokens: body.numCompletionTokens,
            filter_non_word_tokens: body.hideNonWordTokens,
          },
    );
    if (!inferenceResponse.ok || !inferenceResponse.body) {
      const errorBody = await inferenceResponse.json().catch(() => ({ error: inferenceResponse.statusText }));
      return NextResponse.json(
        { error: errorBody.error ?? `Lens re-run failed (${inferenceResponse.status})` },
        { status: 500 },
      );
    }
    const text = await inferenceResponse.text();
    ({ meta, tokens } = await parseLensNdjson(text));
  } catch (error) {
    console.error('Error re-running lens for share:', error);
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }

  if (!meta || tokens.length === 0) {
    return NextResponse.json({ error: 'Lens re-run produced no data' }, { status: 500 });
  }

  // Re-run the steered read-out (if a steered run was shared) the same way: a
  // forced decode over the exact steered token ids with the steer injected and
  // no generation, so the stored steered data is trusted + reproducible (the
  // original steered generation is non-deterministic, but the ids are pinned).
  let steerExport: JlensExportSteer | undefined;
  if (body.steer) {
    try {
      const isSwap = body.steer.mode === 'swap' && !!body.steer.swapToken.trim();
      const steerResponse = await lensPromptStream(body.modelId, {
        type: LENS_TYPE_ORDER,
        input_token_ids: body.steer.inputTokenIds,
        top_n: body.topN,
        num_completion_tokens: 0,
        steer_tokens: [{ token: body.steer.token, type: body.steer.type as LensType }],
        steer_layers: body.steer.layers,
        steer_strength: body.steer.strength,
        steer_ablate: body.steer.ablate,
        swap_token: isSwap ? { token: body.steer.swapToken, type: body.steer.type as LensType } : undefined,
        steer_generated_tokens: body.steer.steerGenerated,
        filter_non_word_tokens: body.hideNonWordTokens,
      });
      if (!steerResponse.ok || !steerResponse.body) {
        const errorBody = await steerResponse.json().catch(() => ({ error: steerResponse.statusText }));
        return NextResponse.json(
          { error: errorBody.error ?? `Steered lens re-run failed (${steerResponse.status})` },
          { status: 500 },
        );
      }
      const steerText = await steerResponse.text();
      const { meta: steerMeta, tokens: steerTokens } = await parseLensNdjson(steerText);
      if (!steerMeta || steerTokens.length === 0) {
        return NextResponse.json({ error: 'Steered lens re-run produced no data' }, { status: 500 });
      }
      steerExport = {
        config: {
          token: body.steer.token,
          type: body.steer.type as LensType,
          layers: body.steer.layers,
          strength: body.steer.strength,
          ablate: body.steer.ablate,
          mode: body.steer.mode as 'steer' | 'swap',
          swapToken: body.steer.swapToken,
          steerGenerated: body.steer.steerGenerated,
        },
        meta: steerMeta,
        tokens: steerTokens,
      };
    } catch (error) {
      console.error('Error re-running steered lens for share:', error);
      return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
    }
  }

  // Chat turns for display: the `inputTokenIds` path carries them in `messages`
  // (existing behavior); the `chat` generation path carries them in `chat`.
  const chatTurns = hasInputTokenIds ? body.messages : (body.chat ?? []);
  // For the generation path the prompt-token boundary is known from the run's
  // meta; for the reproduction path honor the caller-supplied value.
  const resolvedNumPromptTokens = hasInputTokenIds ? (body.numPromptTokens ?? null) : (meta.prompt_len ?? null);

  // Assemble the heavy S3 blob (compact, not pretty-printed). The shape depends
  // on the run kind: chat carries the conversation turns, completion the prompt.
  const blob: JlensExport =
    effectiveKind === 'completion'
      ? {
          version: 1,
          kind: 'completion',
          modelId: body.modelId,
          exportedAt: new Date().toISOString(),
          prompt: body.prompt,
          meta,
          tokens,
          steer: steerExport,
        }
      : {
          version: 1,
          kind: 'chat',
          modelId: body.modelId,
          exportedAt: new Date().toISOString(),
          messages: chatTurns.map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content })),
          meta,
          tokens,
          steer: steerExport,
        };
  const json = JSON.stringify(blob);
  const uncompressedBytes = Buffer.byteLength(json, 'utf8');
  if (uncompressedBytes > MAX_JLENS_SHARE_UPLOAD_SIZE_BYTES) {
    return NextResponse.json(
      {
        error: `Shared run is too large (${(uncompressedBytes / (1024 * 1024)).toFixed(1)}MB). The maximum is ${
          MAX_JLENS_SHARE_UPLOAD_SIZE_BYTES / (1024 * 1024)
        }MB.`,
      },
      { status: 413 },
    );
  }

  const shareId = cuid();
  const region = process.env.AWS_REGION || 'us-east-1';
  const key = `${JLENS_S3_DIR}/${userId || JLENS_ANONYMOUS_USER_ID}/${shareId}.json`;
  const url = `https://${NP_GRAPH_BUCKET}.s3.${region}.amazonaws.com/${key}`;

  try {
    // Store gzipped but as application/json with Content-Encoding: gzip, so the
    // object is small in S3 yet served back to the browser as plaintext JSON.
    const compressed = await gzip(json);
    const s3Client = new S3Client({
      region,
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
      },
    });
    await s3Client.send(
      new PutObjectCommand({
        Bucket: NP_GRAPH_BUCKET,
        Key: key,
        Body: compressed,
        ContentType: 'application/json',
        ContentEncoding: 'gzip',
      }),
    );
  } catch (error) {
    console.error('Error uploading jlens share to S3:', error);
    return NextResponse.json({ error: 'Failed to upload shared run' }, { status: 500 });
  }

  try {
    await prisma.jlensShare.create({
      data: {
        id: shareId,
        kind: effectiveKind,
        modelId: body.modelId,
        url,
        description: body.description || null,
        lockedTokens: body.lockedTokens,
        selectedPositions: body.selectedPositions,
        activeLensModeTab: body.activeLensModeTab,
        topN: body.topN,
        hideNonWordTokens: body.hideNonWordTokens,
        temperature: body.temperature,
        numCompletionTokens: body.numCompletionTokens,
        numPromptTokens: resolvedNumPromptTokens,
        steerToken: body.steer?.token ?? null,
        steerType: body.steer?.type ?? null,
        steerLayers: body.steer?.layers ?? [],
        steerStrength: body.steer?.strength ?? null,
        steerAblate: body.steer?.ablate ?? null,
        steerMode: body.steer?.mode ?? null,
        swapToken: body.steer?.swapToken || null,
        steerGenerated: body.steer?.steerGenerated ?? null,
        userId,
      },
    });
    await prisma.jlensSharePutRequest.create({
      data: { ipAddress: ip, filename: `${shareId}.json`, url, userId },
    });
  } catch (error) {
    console.error('Error saving jlens share to db:', error);
    return NextResponse.json({ error: 'Failed to save shared run' }, { status: 500 });
  }

  return NextResponse.json({ id: shareId, path: makeJlensSharePath(shareId), url });
});
