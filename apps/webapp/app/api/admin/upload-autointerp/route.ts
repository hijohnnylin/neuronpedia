import { createNewExplanationWithoutScore } from '@/lib/db/explanation';
import { RequestAuthedAdminUser } from '@/lib/types/auth';
import { withAuthedAdminUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

type ExplainedFeature = {
  modelId: string;
  layer: string;
  index: string;
  explanation: string;
  creatorId: string;
  skipScore: boolean;
};

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const body = (await request.json()) as ExplainedFeature[];

  const toRun: Promise<any>[] = [];
  body.forEach((element) => {
    if (element.creatorId !== request.user.id) {
      throw new Error("creatorId doesn't match authed user");
    }
    toRun.push(
      createNewExplanationWithoutScore(
        element.modelId,
        element.layer,
        element.index,
        element.explanation,
        element.creatorId,
        request.user,
      ),
    );
  });

  await Promise.all(toRun);
  return NextResponse.json({ message: 'Success' });
});
