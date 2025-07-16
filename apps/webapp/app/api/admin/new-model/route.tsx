import { RequestAuthedAdminUser, withAuthedAdminUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { createModelAdmin } from '../../../../lib/db/model';

type ModelToCreate = {
  id: string;
  displayName: string;
  owner: string;
  layers: number;
};

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const body = (await request.json()) as ModelToCreate;

  const modelId = body.id.toLowerCase();
  const { displayName } = body;

  const model = await createModelAdmin(modelId, displayName, body.layers, body.owner, request.user);

  return NextResponse.json(model);
});
