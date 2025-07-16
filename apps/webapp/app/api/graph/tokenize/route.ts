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

    const tokenizedResponse = await getGraphTokenize(
      validatedData.prompt,
      validatedData.maxNLogits,
      validatedData.desiredLogitProb,
    );

    // console.log('tokenizedResponse', tokenizedResponse);
    return NextResponse.json(tokenizedResponse, { status: 200 });
  } catch (error) {
    console.error('Error in tokenize route:', error);
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: 'Validation error', details: error.errors }, { status: 400 });
    }
    return NextResponse.json(
      { error: 'Failed to tokenize text', message: error instanceof Error ? error.message : String(error) },
      { status: 500 },
    );
  }
}
