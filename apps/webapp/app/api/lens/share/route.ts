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

const chatMessageSchema = yup.object({
  role: yup.string().oneOf(['user', 'assistant', 'system']).required(),
  content: yup.string().required(),
});

const lockedTokenSchema = yup.object({
  key: yup.string().required(),
  type: yup
    .string()
    .oneOf(LENS_TYPES as unknown as string[])
    .required(),
});

// Optional steered run carried with the share: the steer config plus the full
// token-id sequence of the steered run, so we can reproduce its read-outs
// server-side (forced decode over those ids, no generation).
const shareSteerSchema = yup.object({
  token: yup.string().required(),
  type: yup
    .string()
    .oneOf(LENS_TYPES as unknown as string[])
    .required(),
  layers: yup.array().of(yup.number().integer().required()).min(1).required(),
  strength: yup.number().min(-MAX_LENS_STEER_STRENGTH).max(MAX_LENS_STEER_STRENGTH).required(),
  ablate: yup.boolean().required(),
  // Intervention mode + the target token for a swap (empty for plain steers).
  mode: yup.string().oneOf(['steer', 'swap']).default('steer'),
  swapToken: yup.string().default(''),
  // Whether the intervention was applied to generated tokens too.
  steerGenerated: yup.boolean().default(false),
  inputTokenIds: yup.array().of(yup.number().integer().required()).min(1).required(),
});

const shareRequestSchema = yup.object({
  modelId: yup.string().min(1).required(),
  kind: yup.string().oneOf(['chat', 'completion']).default('chat'),
  // Full token-id sequence of the run, used to faithfully re-run inference.
  inputTokenIds: yup.array().of(yup.number().integer().required()).min(1).required(),
  // Chat-kind shares carry the conversation turns; completion-kind shares carry
  // the raw prompt text.
  messages: yup.array().of(chatMessageSchema).default([]),
  prompt: yup.string().default(''),
  topN: yup.number().integer().min(1).max(10).required(),
  temperature: yup.number().min(0).max(2).required(),
  numCompletionTokens: yup.number().integer().min(0).max(512).required(),
  // Number of leading prompt tokens (the remainder were generated). Persisted so
  // a reloaded share can mark the prompt→generated boundary; optional for
  // backward-compat.
  numPromptTokens: yup.number().integer().min(0).optional(),
  activeLensModeTab: yup
    .string()
    .oneOf(LENS_MODES as unknown as string[])
    .required(),
  hideNonWordTokens: yup.boolean().required(),
  lockedTokens: yup.array().of(lockedTokenSchema).default([]),
  selectedPositions: yup.array().of(yup.number().integer().min(0).required()).default([]),
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

  // Re-run inference server-side over the exact token ids (no generation) so the
  // stored data is trusted (computed by us, not the user) and reproducible.
  let meta: LensMetaMessage | null;
  let tokens: LensTokenMessage[];
  try {
    const inferenceResponse = await lensPromptStream(body.modelId, {
      type: LENS_TYPE_ORDER,
      input_token_ids: body.inputTokenIds,
      top_n: body.topN,
      num_completion_tokens: 0,
      // Capture the run with the sharer's non-word filter applied, so the stored
      // blob matches what they were viewing (the share is a frozen snapshot for
      // this filter; viewers toggle by re-running live).
      filter_non_word_tokens: body.hideNonWordTokens,
    });
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

  // Assemble the heavy S3 blob (compact, not pretty-printed). The shape depends
  // on the run kind: chat carries the conversation turns, completion the prompt.
  const blob: JlensExport =
    body.kind === 'completion'
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
          messages: body.messages.map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content })),
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
        kind: body.kind,
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
        numPromptTokens: body.numPromptTokens ?? null,
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
