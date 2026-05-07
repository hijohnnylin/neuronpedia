import { nlaFetch } from '@/lib/db/nla-source';
import { fetchDecoderLatent } from '@/lib/utils/saelens';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { hfRepoId, hfFolderId, index, modelId, nlaSourceId } = body as {
    hfRepoId?: string;
    hfFolderId?: string;
    index?: number;
    modelId?: string;
    nlaSourceId?: string;
  };

  if (!hfRepoId || !hfFolderId || index === undefined || index === null) {
    return NextResponse.json({ error: 'Missing required fields: hfRepoId, hfFolderId, index' }, { status: 400 });
  }
  if (!Number.isInteger(index) || index < 0) {
    return NextResponse.json({ error: 'index must be a non-negative integer' }, { status: 400 });
  }

  const safetensorsPath = `${hfFolderId}/sae_weights.safetensors`;

  try {
    const latent = await fetchDecoderLatent(hfRepoId, safetensorsPath, index);

    const nlaResponse = await nlaFetch(modelId, nlaSourceId, '/describe', {
      method: 'POST',
      body: JSON.stringify({
        activations: [latent.values],
        temperature: 0.7,
        max_new_tokens: 200,
        stream: true,
      }),
    });

    if (!nlaResponse.ok) {
      const errorText = await nlaResponse.text();
      return NextResponse.json(
        { error: `NLA server error: ${nlaResponse.status} - ${errorText}` },
        { status: nlaResponse.status },
      );
    }

    if (!nlaResponse.body) {
      return NextResponse.json({ error: 'No response body from NLA server' }, { status: 502 });
    }

    return new NextResponse(nlaResponse.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
      },
    });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
