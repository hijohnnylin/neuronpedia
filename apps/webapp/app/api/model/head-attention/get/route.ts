import { getAttentionForHead } from '@/lib/utils/inference';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// Runs custom-text attention for a single (layer, head) and returns a
// HeadSequenceData-shaped result so the client can render it with the same
// HeadActivationItem component used for the stored top sequences.
export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  try {
    const body = await request.json();
    const { modelId, layer, headIndex, text } = body ?? {};

    if (
      !modelId ||
      layer === undefined ||
      layer === null ||
      headIndex === undefined ||
      headIndex === null ||
      typeof text !== 'string'
    ) {
      return NextResponse.json({ error: 'Missing required parameters' }, { status: 400 });
    }

    const layerInt = Number(layer);
    const headIndexInt = Number(headIndex);
    if (!Number.isInteger(layerInt) || !Number.isInteger(headIndexInt)) {
      return NextResponse.json({ error: 'Invalid layer or headIndex' }, { status: 400 });
    }

    if (text.trim().length === 0) {
      return NextResponse.json({ error: 'Please enter some text.' }, { status: 400 });
    }

    const result = await getAttentionForHead(modelId, layerInt, headIndexInt, text);

    return NextResponse.json({
      id: `custom-${layerInt}-${headIndexInt}`,
      layer: layerInt,
      headIndex: headIndexInt,
      // Custom input has no dataset interval; use 0 so it's excluded from the
      // interval filter (which only shows intervals present in stored data).
      interval: 0,
      tokens: result.tokens,
      attentionIndices: result.attention_indices,
      attentionValues: result.attention_values,
      maxActivation: result.max_activation,
    });
  } catch (error) {
    console.error('Error running custom head attention:', error);
    const message = error instanceof Error ? error.message : 'Internal server error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
});
