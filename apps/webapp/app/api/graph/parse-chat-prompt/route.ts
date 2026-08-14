import { getModelById } from '@/lib/db/model';
import { getGraphParseChatPrompt } from '@/lib/utils/graph';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

const MAX_PROMPT_CHARS = 10000;

// `modelId` is deliberately not restricted to GRAPH_GENERATION_ENABLED_MODELS.
// The steer modal calls this for `unsteerable_positions` on any graph it can
// steer, which is a wider set than the one we can generate graphs for (see
// `SteerLogitsRequestSchema`, equally unrestricted). Resolving the graph host
// is the real gate — it throws `SourceSet not found.` for a model with none.
const parseChatPromptRequestSchema = yup.object({
  prompt: yup.string().min(1).max(MAX_PROMPT_CHARS).required(),
  modelId: yup.string().min(1).required(),
  sourceSetName: yup.string().nullable(),
});

// Internal: recover the structured chat turns behind an already-rendered graph
// prompt so the Remix flow can put them back into the editor. Undocumented on
// purpose (no @swagger block) — it exists to keep chat-template knowledge out of
// the frontend, not as a public API surface.
export async function POST(request: Request) {
  try {
    let body;
    try {
      body = await request.json();
    } catch {
      return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
    }

    const validatedData = await parseChatPromptRequestSchema.validate(body);

    if (!validatedData.sourceSetName) {
      const model = await getModelById(validatedData.modelId);
      validatedData.sourceSetName = model?.defaultGraphSourceSetName;
      if (!validatedData.sourceSetName) {
        return NextResponse.json(
          {
            error: 'Source Set Missing',
            message: `The model ${validatedData.modelId} has no default graph source set, so you must provide one in the sourceSetName parameter.`,
          },
          { status: 400 },
        );
      }
    }

    const parsed = await getGraphParseChatPrompt(
      validatedData.prompt,
      validatedData.modelId,
      validatedData.sourceSetName,
    );

    return NextResponse.json(parsed, { status: 200 });
  } catch (error) {
    console.error('Error in parse-chat-prompt route:', error);
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: 'Validation error', details: error.errors }, { status: 400 });
    }
    const errorMessage = error instanceof Error ? error.message : String(error);
    const isGraphServerError = errorMessage.includes('Graph server') || errorMessage.includes('External API');
    return NextResponse.json(
      {
        error: isGraphServerError ? 'Graph server error' : 'Failed to parse chat prompt',
        message: errorMessage,
      },
      { status: isGraphServerError ? 502 : 500 },
    );
  }
}
