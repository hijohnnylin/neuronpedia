import { tokenizeText } from '@/lib/utils/inference';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

const MAX_TOKENIZE_CHARS = 10000;

const tokenizeRequestSchema = yup.object({
  prompt: yup.string().max(MAX_TOKENIZE_CHARS).min(1).required(),
  modelId: yup.string().min(1).required(),
  prependBos: yup.boolean().default(true),
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

    const tokenizedResponse = await tokenizeText(validatedData.modelId, validatedData.prompt, validatedData.prependBos);

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
