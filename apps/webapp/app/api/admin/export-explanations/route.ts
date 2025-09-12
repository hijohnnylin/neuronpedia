import { exportExplanations } from '@/lib/db/explanation';
import { RequestAuthedAdminUser } from '@/lib/types/auth';
import { withAuthedAdminUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { number, object, string, ValidationError } from 'yup';

const postSchema = object({
  modelId: string().required(),
  sourceId: string().required(),
  maxResults: number().integer(),
});

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  try {
    const body = await postSchema.validate(await request.json());
    const results = await exportExplanations(body.modelId, body.sourceId, body.maxResults, request.user);
    return NextResponse.json(results);
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
