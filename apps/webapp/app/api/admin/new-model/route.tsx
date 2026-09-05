import { HF_REPO_ID_ERROR_MESSAGE, isValidHfRepoId } from '@/lib/utils/model';
import { RequestAuthedAdminUser, withAuthedAdminUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { createModelAdmin, getModelByHfRepoIdIgnoringAccess } from '../../../../lib/db/model';

type ModelToCreate = {
  id: string;
  displayName: string;
  owner: string;
  layers: number;
  hfRepoId?: string | null;
};

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const body = (await request.json()) as ModelToCreate;

  const modelId = body.id.toLowerCase();
  const { displayName } = body;

  // Not lowercased: repo ids are case-sensitive on the Hub (`google/gemma-4-E2B`).
  const hfRepoId = body.hfRepoId?.trim() || null;
  if (hfRepoId) {
    if (!isValidHfRepoId(hfRepoId)) {
      return NextResponse.json({ error: HF_REPO_ID_ERROR_MESSAGE }, { status: 400 });
    }
    // Checked here so a duplicate reads as a conflict rather than surfacing as a 500 from the
    // unique index. Racy by nature, which is why the index exists as well.
    const existing = await getModelByHfRepoIdIgnoringAccess(hfRepoId);
    if (existing) {
      return NextResponse.json(
        { error: `HuggingFace repo ${hfRepoId} is already mapped to model ${existing.id}` },
        { status: 409 },
      );
    }
  }

  const model = await createModelAdmin(modelId, displayName, body.layers, body.owner, request.user, hfRepoId);

  return NextResponse.json(model);
});
