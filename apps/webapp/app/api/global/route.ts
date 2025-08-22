import {
  getExplanationModels,
  getExplanationScoreModelTypes,
  getExplanationScoreTypes,
  getExplanationTypes,
} from '@/lib/db/explanation-type';
import { getGlobalModels } from '@/lib/db/model';
import { getGlobalSourceReleases } from '@/lib/db/source';
import { RequestOptionalUser } from '@/lib/types/auth';
import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const [models, explanationTypes, explanationModels, releases, explanationScoreTypes, explanationScoreModelTypes] =
    await Promise.all([
      getGlobalModels(request.user),
      getExplanationTypes(),
      getExplanationModels(),
      getGlobalSourceReleases(request.user),
      getExplanationScoreTypes(),
      getExplanationScoreModelTypes(),
    ]);

  return NextResponse.json({
    models,
    explanationTypes,
    explanationModels,
    releases,
    explanationScoreTypes,
    explanationScoreModelTypes,
  });
});
