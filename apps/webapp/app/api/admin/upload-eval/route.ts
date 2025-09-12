import { prisma } from '@/lib/db';
import { env } from '@/lib/env';
import { RequestAuthedAdminUser } from '@/lib/types/auth';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { withAuthedAdminUser } from '@/lib/with-user';
import { Visibility } from '@prisma/client';
import { NextResponse } from 'next/server';

const SAEBENCH_RELEASE_NAME = 'sae-bench-0125';

type EvalUpload = {
  modelId: string;
  sourceId: string;
  type: string;
  data: string;
  saelensConfig: string;
  hfFolderId: string;
  hfRepoId: string;
};

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const body = (await request.json()) as EvalUpload[];

  const evalUploads = [];
  const checkedSourceSets = new Set<string>();

  console.log('Checking for existing evals and creating sources');
  // eslint-disable-next-line no-restricted-syntax
  for (const evalUpload of body) {
    // ensure data is a valid json object
    JSON.parse(evalUpload.data);

    // create the sourceSet if it doesn't exist (don't check same sourceset twice)
    if (!checkedSourceSets.has(evalUpload.sourceId)) {
      // eslint-disable-next-line no-await-in-loop
      let sourceSet = await prisma.sourceSet.findFirst({
        where: {
          modelId: evalUpload.modelId,
          name: getSourceSetNameFromSource(evalUpload.sourceId),
        },
      });
      if (!sourceSet) {
        console.log('Creating source set: ', evalUpload.modelId, getSourceSetNameFromSource(evalUpload.sourceId));
        // eslint-disable-next-line no-await-in-loop
        sourceSet = await prisma.sourceSet.create({
          data: {
            modelId: evalUpload.modelId,
            name: getSourceSetNameFromSource(evalUpload.sourceId),
            description: '',
            creatorName: '',
            creatorId: env.DEFAULT_CREATOR_USER_ID,
            hasDashboards: false,
            releaseName: SAEBENCH_RELEASE_NAME,
            visibility: Visibility.PUBLIC,
          },
        });
      }
      checkedSourceSets.add(evalUpload.sourceId);
    }

    // create the source if it doesn't exist
    // eslint-disable-next-line no-await-in-loop
    let source = await prisma.source.findFirst({
      where: {
        modelId: evalUpload.modelId,
        id: evalUpload.sourceId,
      },
    });
    if (!source) {
      console.log('Creating source: ', evalUpload.modelId, evalUpload.sourceId);
      // eslint-disable-next-line no-await-in-loop
      source = await prisma.source.create({
        data: {
          modelId: evalUpload.modelId,
          id: evalUpload.sourceId,
          setName: getSourceSetNameFromSource(evalUpload.sourceId),
          creatorId: env.DEFAULT_CREATOR_USER_ID,
          hasDashboards: false,
          saelensConfig: evalUpload.saelensConfig,
          hfFolderId: evalUpload.hfFolderId,
          hfRepoId: evalUpload.hfRepoId,
          visibility: Visibility.PUBLIC,
        },
      });
    }

    // check that this eval doesn't already exist
    // eslint-disable-next-line no-await-in-loop
    const existingEval = await prisma.eval.findFirst({
      where: {
        modelId: evalUpload.modelId,
        sourceId: evalUpload.sourceId,
        typeName: evalUpload.type,
      },
    });

    if (existingEval) {
      console.log('SKIPPING === Eval already exists: ', evalUpload.modelId, evalUpload.sourceId, evalUpload.type);
    } else {
      evalUploads.push(evalUpload);
    }
  }

  // createmany
  console.log('Creating evals');
  await prisma.eval.createMany({
    data: evalUploads.map((evalUpload) => {
      const data = JSON.parse(evalUpload.data);
      const resultDetails = data.eval_result_details;
      data.eval_result_details = undefined;
      return {
        modelId: evalUpload.modelId,
        sourceId: evalUpload.sourceId,
        typeName: evalUpload.type,
        output: data,
        detailedMetrics: resultDetails,
      };
    }),
  });

  console.log(`${evalUploads.length} Evals created`);

  return NextResponse.json({ message: 'Success' });
});
