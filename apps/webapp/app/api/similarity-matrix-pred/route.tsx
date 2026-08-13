import { INFERENCE_BASE_PATH } from '@/lib/api/inference-types';
import { getOneRandomServerHostForSource } from '@/lib/db/inference-host-source';
import { getTransformerLensModelIdIfExists } from '@/lib/db/model';
import { INFERENCE_SERVER_SECRET, USE_LOCALHOST_INFERENCE } from '@/lib/env';
import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

type RequestBody = {
  modelId: string;
  sourceId: string;
  text: string;
};

const maxTextLength = 700;

export const POST = withOptionalUser(async (request) => {
  let body: RequestBody;
  try {
    body = await request.json();
  } catch (error) {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const { modelId, sourceId, text } = body || ({} as RequestBody);
  if (!modelId || !sourceId || !text) {
    return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
  }

  // if text is longer than maxTextLength, return error
  if (text.length > maxTextLength) {
    return NextResponse.json(
      {
        error: `Text is too long. The maximum allowed is ${maxTextLength} characters. Your text is ${text.length} characters long.`,
      },
      { status: 400 },
    );
  }

  try {
    const serverHost = await getOneRandomServerHostForSource(modelId, sourceId, request.user);
    if (!serverHost) {
      return NextResponse.json({ error: 'No inference host found' }, { status: 500 });
    }

    const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

    const base = (USE_LOCALHOST_INFERENCE ? undefined : serverHost) || serverHost;
    const url = `${base}${INFERENCE_BASE_PATH}/util/similarity-matrix-pred`;

    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
      },
      body: JSON.stringify({
        modelId: transformerLensModelId,
        sourceId,
        index: 0, // this isn't used, can remove it
        text,
      }),
      cache: 'no-cache',
    });

    if (!resp.ok) {
      const errText = await resp.text();
      return NextResponse.json({ error: 'Inference server error', details: errText }, { status: resp.status });
    }

    const data = await resp.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Error in similarity-matrix-pred route:', error);
    return NextResponse.json(
      { error: 'Failed to fetch similarity matrix', message: error instanceof Error ? error.message : String(error) },
      { status: 500 },
    );
  }
});
