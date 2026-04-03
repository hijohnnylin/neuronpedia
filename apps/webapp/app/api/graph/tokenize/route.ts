import { getModelById } from '@/lib/db/model';
import {
  getGraphTokenize,
  GRAPH_DESIREDLOGITPROB_DEFAULT,
  GRAPH_DESIREDLOGITPROB_MAX,
  GRAPH_DESIREDLOGITPROB_MIN,
  GRAPH_GENERATION_ENABLED_MODELS,
  GRAPH_MAXNLOGITS_DEFAULT,
  GRAPH_MAXNLOGITS_MAX,
  GRAPH_MAXNLOGITS_MIN,
} from '@/lib/utils/graph';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

const MAX_TOKENIZE_CHARS = 10000;

const tokenizeRequestSchema = yup.object({
  prompt: yup.string().max(MAX_TOKENIZE_CHARS).min(1).required(),
  modelId: yup.string().min(1).required().oneOf(GRAPH_GENERATION_ENABLED_MODELS),
  sourceSetName: yup.string().nullable(),
  maxNLogits: yup
    .number()
    .integer('Must be an integer.')
    .min(GRAPH_MAXNLOGITS_MIN, `Must be at least ${GRAPH_MAXNLOGITS_MIN}.`)
    .max(GRAPH_MAXNLOGITS_MAX, `Must be at most ${GRAPH_MAXNLOGITS_MAX}.`)
    .default(GRAPH_MAXNLOGITS_DEFAULT)
    .required('This field is required.'),
  desiredLogitProb: yup
    .number()
    .min(GRAPH_DESIREDLOGITPROB_MIN, `Must be at least ${GRAPH_DESIREDLOGITPROB_MIN}.`)
    .max(GRAPH_DESIREDLOGITPROB_MAX, `Must be at most ${GRAPH_DESIREDLOGITPROB_MAX}.`)
    .default(GRAPH_DESIREDLOGITPROB_DEFAULT)
    .required('This field is required.'),
});

export async function POST(request: Request) {
  try {
    let body;
    try {
      body = await request.json();
    } catch (error) {
      return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
    }

    const validatedData = await tokenizeRequestSchema.validate(body);

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

    const tokenizedResponse = await getGraphTokenize(
      validatedData.prompt,
      validatedData.maxNLogits,
      validatedData.desiredLogitProb,
      validatedData.modelId,
      validatedData.sourceSetName,
    );

    // console.log('tokenizedResponse', tokenizedResponse);
    return NextResponse.json(tokenizedResponse, { status: 200 });
  } catch (error) {
    console.error('Error in tokenize route:', error);
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: 'Validation error', details: error.errors }, { status: 400 });
    }
    const errorMessage = error instanceof Error ? error.message : String(error);
    const isGraphServerError = errorMessage.includes('Graph server') || errorMessage.includes('External API');
    return NextResponse.json(
      {
        error: isGraphServerError ? 'Graph server error' : 'Failed to tokenize text',
        message: errorMessage,
      },
      { status: isGraphServerError ? 502 : 500 },
    );
  }
}
