import { dangerousGetNeuronRangeInternalUseOnly } from '@/lib/db/neuron';
import { RequestAuthedAdminUser } from '@/lib/types/auth';
import { withAuthedAdminUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const body = await request.json();
  console.log({ body });

  const { modelId } = body;
  const { sourceId } = body;
  const { beginIndex } = body;
  const { endIndex } = body;

  const neuron = await dangerousGetNeuronRangeInternalUseOnly(modelId, sourceId, beginIndex, endIndex);

  console.log('returning:', neuron.length);

  return NextResponse.json(neuron);
});
