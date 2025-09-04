import { getNeurons } from '@/lib/db/neuron';
import { RequestOptionalUser } from '@/lib/types/auth';
import { NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { array, number, object, string, ValidationError } from 'yup';

const arrayOfFeaturesSchema = array().of(
  object({
    modelId: string().required(),
    layer: string().required(),
    index: number().integer().required(),
    maxActsToReturn: number().integer().optional(),
  }).required(),
);

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const bodyJson = await request.json();
  if (!bodyJson) {
    return NextResponse.json({ message: 'Invalid JSON body' }, { status: 400 });
  }

  try {
    const featuresRequest = await arrayOfFeaturesSchema.validate(bodyJson);

    if (!featuresRequest) {
      return NextResponse.json({ message: 'Invalid JSON body' }, { status: 400 });
    }

    const features: NeuronIdentifier[] = [];
    featuresRequest.forEach((feature) => {
      features.push(new NeuronIdentifier(feature.modelId, feature.layer, feature.index.toString()));
    });

    const startTime = Date.now();
    let featuresResponse = await getNeurons(features, request.user, featuresRequest[0].maxActsToReturn);
    const endTime = Date.now();
    console.log(`Time taken to get neurons: ${endTime - startTime}ms`);

    // retain the original order of the features
    featuresResponse = featuresResponse.sort((a, b) => {
      if (featuresRequest) {
        return (
          featuresRequest.findIndex((f) => f.index === parseInt(a.index, 10)) -
          featuresRequest.findIndex((f) => f.index === parseInt(b.index, 10))
        );
      }
      return 0;
    });

    return NextResponse.json(featuresResponse);
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
