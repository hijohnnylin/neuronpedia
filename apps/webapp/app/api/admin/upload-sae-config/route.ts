import { prisma } from '@/lib/db';
import { RequestAuthedAdminUser } from '@/lib/types/auth';
import { withAuthedAdminUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

type SaeConfigUpload = {
  modelId: string;
  saeId: string;
  config: string;
  hfRepoId: string;
  hfFolder: string;
  saelensRelease: string;
  saelensSaeId: string;
};

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const body = (await request.json()) as SaeConfigUpload;

  console.log('=== Upload SAE config for:', body.modelId, body.saeId);

  // ensure evalCfg is a valid json object
  // console.log("config", body.config);
  const config = JSON.parse(body.config);
  // console.log("valid config");

  try {
    await prisma.source.update({
      where: {
        modelId_id: {
          modelId: body.modelId,
          id: body.saeId,
        },
      },
      data: {
        saelensConfig: config,
        hfRepoId: body.hfRepoId,
        hfFolderId: body.hfFolder,
        saelensRelease: body.saelensRelease,
        saelensSaeId: body.saelensSaeId,
      },
    });
    // console.log("Sae config updated: ", body.modelId, body.saeId);
  } catch (e) {
    console.log('Error updating sae config: ', e);
    console.log('Failing config:', body.config);
  }

  return NextResponse.json({ message: 'Success' });
});
