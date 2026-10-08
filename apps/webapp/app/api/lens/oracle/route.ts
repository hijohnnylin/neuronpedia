import { lensOracleStream } from '@/lib/utils/inference';
import {
  LensErrorMessage,
  LensOracleDoneMessage,
  LensOracleMetaMessage,
  LensOracleReadMessage,
  LensOracleStreamMessage,
  MAX_ORACLE_POSITIONS,
  ORACLE_BULLETS,
  ORACLE_MAX_TOKENS,
} from '@/lib/utils/lens';
import {
  cachedOracleMessages,
  getOracleEntry,
  mapNdjson,
  oracleCacheKey,
  recordOracleMessage,
} from '@/lib/utils/oracle-cache';
import { oraclePrefixDigest, signOracleRead } from '@/lib/utils/oracle-signature';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

// A read is 5 to 11 short greedy generations in one batch; this bounds a read
// that waits behind a long lens run on the same host.
export const maxDuration = 120;

// Same bound as `/api/lens/prompt`: the whole sequence, chat history included.
const MAX_TOKEN_IDS = 4096;
const MAX_MODEL_ID_CHARS = 128;
const MAX_LAYERS = 64;

const NDJSON_HEADERS = {
  'Content-Type': 'application/x-ndjson; charset=utf-8',
  'Cache-Control': 'no-cache, no-transform',
};

// Adds the signature to each read, with the adapter from the meta before it. A read
// with no `position` is at `positions[0]`.
function readSigner(modelId: string, maxBullets: number, tokenIds: number[], positions: number[]) {
  const digests = new Map(positions.map((p) => [p, oraclePrefixDigest(tokenIds.slice(0, p + 1))]));
  let adapter = '';
  return (message: LensOracleStreamMessage): LensOracleStreamMessage => {
    if (message.kind === 'meta') {
      adapter = message.adapter;
      return message;
    }
    if (message.kind !== 'read') {
      return message;
    }
    const position = message.position ?? positions[0];
    const prefixDigest = digests.get(position);
    if (prefixDigest === undefined) {
      return message;
    }
    const sig = signOracleRead({
      modelId,
      adapter,
      maxBullets,
      position,
      prefixDigest,
      layer: message.layer,
      bullets: message.bullets,
      text: message.text,
      finish: message.finish,
    });
    return sig ? { ...message, sig } : message;
  };
}

// A read from the webapp's cache, in the same form as one from the inference server.
function respondFromCache(messages: LensOracleStreamMessage[], stream: boolean) {
  if (stream) {
    return new Response(messages.map((m) => `${JSON.stringify(m)}\n`).join(''), {
      status: 200,
      headers: NDJSON_HEADERS,
    });
  }
  return NextResponse.json({
    meta: messages[0],
    reads: messages.filter((m) => m.kind === 'read'),
    done: messages[messages.length - 1],
  });
}

const lensOracleRequestSchema = yup.object({
  modelId: yup.string().min(1).max(MAX_MODEL_ID_CHARS).required(),
  tokenIds: yup.array().of(yup.number().integer().min(0).required()).min(1).max(MAX_TOKEN_IDS).required(),
  position: yup.number().integer().min(0).required(),
  positions: yup
    .array()
    .of(yup.number().integer().min(0).required())
    .max(MAX_ORACLE_POSITIONS - 1)
    .default([]),
  layers: yup.array().of(yup.number().integer().min(0).required()).max(MAX_LAYERS).default([]),
  stream: yup.boolean().default(true),
  partial: yup.boolean().default(true),
});

// Keeps what each message says, under the cache key of its position.
function oracleRecorder(keys: Map<number, string>, positions: number[], defaultLayers: boolean) {
  return (message: LensOracleStreamMessage) => {
    if (message.kind === 'meta') {
      positions.forEach((p, i) => {
        const at = message.positions ? message.positions.indexOf(p) : i;
        const token = message.tokens?.[at] ?? message.token;
        const { positions: _positions, tokens: _tokens, ...meta } = message;
        recordOracleMessage(keys.get(p)!, defaultLayers, { ...meta, position: p, token });
      });
    } else if (message.kind === 'read') {
      const key = keys.get(message.position ?? positions[0]);
      if (key) {
        recordOracleMessage(key, defaultLayers, message);
      }
    }
  };
}

/**
 * @swagger
 * /api/lens/oracle:
 *   post:
 *     summary: Read Token Positions With the Oracle Lens
 *     description: |
 *       Describes the model's activation at a token position, in words, at each oracle layer. The oracle lens is a LoRA adapter trained to read one residual vector (see [WorkspaceBench](https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the)). Only models with an oracle adapter support it; the `meta` message of `/api/lens/prompt` lists the layers in `oracle_layers` (empty when there is no oracle).
 *
 *       Send the token `id` values of a lens run as `tokenIds`, and the position to read. Add more positions in `positions` to read them in the same request. Only `tokenIds[0..position]` affect the read at a position. Each layer's read is greedy and stops after one bullet or 32 tokens. A read served before comes from a cache, with `cached: true`. Greedy reads can still change a little between repeats, so each read has a `sig`: send the read with its `sig` to `/api/lens/share` to store that exact text.
 *
 *       **Response format:** By default (`stream: true`) the endpoint responds with NDJSON: one `meta` message, one `read` message per position and layer (in the order the reads end), then one `done` message. Before a read, `partial` messages (`{"kind":"partial","position":12,"layer":20,"text":"- Pa"}`) can give its text so far; each replaces the one before. If an error occurs mid-stream, an `error` message is emitted instead. Set `stream: false` to receive `{ meta, reads, done }` in one JSON object.
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
 *               - tokenIds
 *               - position
 *             properties:
 *               modelId:
 *                 type: string
 *                 maxLength: 128
 *                 example: qwen3.6-27b
 *               tokenIds:
 *                 type: array
 *                 description: The token ids of the sequence, as echoed by `/api/lens/prompt`.
 *                 maxItems: 4096
 *                 items:
 *                   type: integer
 *               position:
 *                 type: integer
 *                 description: The position to read. Must be inside `tokenIds`.
 *               positions:
 *                 type: array
 *                 description: More positions to read in the same request. Each must be inside `tokenIds`.
 *                 maxItems: 63
 *                 items:
 *                   type: integer
 *               layers:
 *                 type: array
 *                 description: Layers to read. Empty (default) reads all the server's oracle layers.
 *                 items:
 *                   type: integer
 *               stream:
 *                 type: boolean
 *                 default: true
 *               partial:
 *                 type: boolean
 *                 description: Send `partial` messages when streaming.
 *                 default: true
 *     responses:
 *       200:
 *         description: The oracle read, streamed as NDJSON or buffered.
 *         content:
 *           application/x-ndjson:
 *             examples:
 *               stream:
 *                 value: |
 *                   {"kind":"meta","model":"Qwen/Qwen3.6-27B","adapter":"neuronpedia/jacobian-lens:qwen3.6-27b/oracle","position":12,"token":" Paris","positions":[12],"tokens":[" Paris"],"layers":[20,24],"max_bullets":1}
 *                   {"kind":"read","position":12,"layer":24,"bullets":["The capital of France"],"text":"- The capital of France\n","finish":"bullets","cached":false,"sig":"v1.3kq0…"}
 *                   {"kind":"done","elapsed_ms":1840,"cached_layers":0}
 *       400:
 *         description: Invalid JSON, a validation error, a position outside `tokenIds`, or a layer the oracle does not read.
 *       404:
 *         description: This model's servers have no oracle lens.
 *       500:
 *         description: The read failed.
 */
export async function POST(request: Request) {
  try {
    let body;
    try {
      body = await request.json();
    } catch (error) {
      return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
    }
    const validated = await lensOracleRequestSchema.validate(body);
    const { modelId, tokenIds } = validated;
    const maxBullets = ORACLE_BULLETS;
    const maxTokens = ORACLE_MAX_TOKENS;
    const positions = Array.from(new Set([validated.position, ...validated.positions]));
    if (positions.some((p) => p >= tokenIds.length)) {
      return NextResponse.json({ error: 'A position is outside tokenIds' }, { status: 400 });
    }

    const keys = new Map(
      positions.map((p) => [p, oracleCacheKey(modelId, maxBullets, maxTokens, tokenIds.slice(0, p + 1))]),
    );
    const record = oracleRecorder(keys, positions, validated.layers.length === 0);
    const sign = readSigner(modelId, maxBullets, tokenIds, positions);
    const hit = cachedOracleMessages(
      positions.map((p) => ({ position: p, entry: getOracleEntry(keys.get(p)!) })),
      validated.layers,
    );
    if (hit) {
      return respondFromCache(hit.map(sign), validated.stream);
    }

    const inferenceResponse = await lensOracleStream(
      modelId,
      {
        tokenIds,
        position: positions[0],
        // Servers that read one position per request do not know this field.
        ...(positions.length > 1 ? { positions: positions.slice(1) } : {}),
        layers: validated.layers,
        maxBullets,
        maxTokens,
        stream: true,
        partial: validated.stream && validated.partial,
      },
      request.signal,
    );

    if (!inferenceResponse.ok || !inferenceResponse.body) {
      const errorBody = await inferenceResponse.json().catch(() => ({ error: inferenceResponse.statusText }));
      return NextResponse.json(
        { error: errorBody.error ?? `Oracle request failed (${inferenceResponse.status})` },
        { status: inferenceResponse.status >= 400 ? inferenceResponse.status : 500 },
      );
    }

    if (!validated.stream) {
      const text = await inferenceResponse.text();
      let meta: LensOracleMetaMessage | null = null;
      const reads: LensOracleReadMessage[] = [];
      let done: LensOracleDoneMessage | null = null;
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
        record(msg);
        msg = sign(msg);
        if (msg.kind === 'meta') {
          meta = msg as LensOracleMetaMessage;
        } else if (msg.kind === 'read') {
          reads.push(msg as LensOracleReadMessage);
        } else if (msg.kind === 'done') {
          done = msg as LensOracleDoneMessage;
        } else if (msg.kind === 'error') {
          return NextResponse.json(
            { error: (msg as LensErrorMessage).error || 'Oracle stream error' },
            { status: 500 },
          );
        }
      }
      if (!meta || !done) {
        return NextResponse.json({ error: 'Oracle request produced incomplete data' }, { status: 500 });
      }
      return NextResponse.json({ meta, reads, done });
    }

    return new Response(
      mapNdjson(inferenceResponse.body, (msg) => {
        record(msg);
        return sign(msg);
      }),
      { status: 200, headers: NDJSON_HEADERS },
    );
  } catch (error) {
    console.error('Error in lens oracle route:', error);
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: 'Validation error', details: error.errors }, { status: 400 });
    }
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}
