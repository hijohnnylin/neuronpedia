import { prisma } from '@/lib/db';
import { nlaFetch } from '@/lib/db/nla-source';
import { fetchDecoderLatent, getSAEFormatConfig, SAEFormat } from '@/lib/utils/saelens';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { hfRepoId, hfFolderId, index, modelId, nlaSourceId, layerNum, format } = body as {
    hfRepoId?: string;
    hfFolderId?: string;
    index?: number;
    modelId?: string;
    nlaSourceId?: string;
    layerNum?: number;
    format?: string;
  };

  if (!hfRepoId || !hfFolderId || index === undefined || index === null) {
    return NextResponse.json({ error: 'Missing required fields: hfRepoId, hfFolderId, index' }, { status: 400 });
  }
  if (!Number.isInteger(index) || index < 0) {
    return NextResponse.json({ error: 'index must be a non-negative integer' }, { status: 400 });
  }

  // Default to SAELens for backward compatibility (every caller predating
  // the format param implicitly used SAELens layout).
  const saeFormat = format ?? SAEFormat.SAELens;
  if (!Object.values(SAEFormat).includes(saeFormat as SAEFormat)) {
    return NextResponse.json(
      {
        error: `Unsupported SAE format: ${saeFormat}. Allowed values: ${Object.values(SAEFormat).join(', ')}`,
      },
      { status: 400 },
    );
  }

  // Resolve which NlaSource (and therefore which set of GPU servers) to
  // forward the /describe call to. Two ways callers can specify:
  //   (1) explicit `nlaSourceId` — used as-is, paired with `modelId`
  //   (2) `modelId` + `layerNum` — we look up the matching NlaSource row
  // The UI today sends (2) because it doesn't track NlaSource ids; future
  // API callers can use (1) to pin to a specific source row when the same
  // (modelId, layerNum) has multiple configured (different ar/av).
  let resolvedNlaSourceId = nlaSourceId;
  if (!resolvedNlaSourceId && modelId && layerNum !== undefined && layerNum !== null) {
    if (!Number.isInteger(layerNum) || layerNum < 0) {
      return NextResponse.json({ error: 'layerNum must be a non-negative integer' }, { status: 400 });
    }
    // Stable ordering so repeat requests for the same (modelId, layerNum)
    // hit the same row when multiple match (rare; differs only by ar/av).
    const found = await prisma.nlaSource.findFirst({
      where: { modelId, layerNum },
      orderBy: { id: 'asc' },
    });
    if (!found) {
      return NextResponse.json(
        {
          error: `No NLA source configured for modelId="${modelId}", layerNum=${layerNum}. Pass an explicit nlaSourceId, or configure an NlaSource for this (model, layer) pair.`,
        },
        { status: 404 },
      );
    }
    resolvedNlaSourceId = found.id;
  }

  const { weightsFilename } = getSAEFormatConfig(saeFormat as SAEFormat);
  const safetensorsPath = `${hfFolderId}/${weightsFilename}`;

  try {
    const latent = await fetchDecoderLatent(hfRepoId, safetensorsPath, index, saeFormat as SAEFormat);

    const nlaResponse = await nlaFetch(modelId, resolvedNlaSourceId, '/describe', {
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
