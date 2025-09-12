import { RequestOptionalUser } from '@/lib/types/auth';
import { getCosSimForFeature } from '@/lib/utils/inference';
import { NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

const featureWithVectorSchema = yup.object({
  modelId: yup.string().required('Model ID is required'),
  source: yup.string().required('Source is required'),
  index: yup.number().required('Index is required'),
  targetModelId: yup.string().required('Target model ID is required'),
  targetSourceId: yup.string().required('Target source ID is required'),
});

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();

  try {
    const validatedFeatureWithVector = await featureWithVectorSchema.validate(body);

    const cosSimIndices = await getCosSimForFeature(
      new NeuronIdentifier(
        validatedFeatureWithVector.modelId,
        validatedFeatureWithVector.source,
        validatedFeatureWithVector.index.toString(),
      ),
      validatedFeatureWithVector.targetModelId,
      validatedFeatureWithVector.targetSourceId,
      request.user,
    );

    return NextResponse.json(cosSimIndices);
  } catch (error) {
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: error.errors }, { status: 400 });
    }
    throw error;
  }
});
