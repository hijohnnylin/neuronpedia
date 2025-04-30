/* eslint-disable no-restricted-syntax */
/* eslint-disable no-await-in-loop */

import { prisma } from '@/lib/db';
import { importConfigFromS3, importJsonlString } from '@/lib/db/import';
import { IS_LOCALHOST } from '@/lib/env';
import {
  DATASET_BASE_PATH,
  downloadAndDecompressFile,
  downloadFileJsonlParsedLines,
  getFilesInPath,
} from '@/lib/utils/s3';
import { getAuthedAdminUser, RequestAuthedAdminUser, RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// Hobby plans don't support > 60 seconds
// export const maxDuration = 300;

function enqueueProgress(controller: ReadableStreamDefaultController, progress: number, progressText: string) {
  const encoder = new TextEncoder();
  controller.enqueue(encoder.encode(`data: ${JSON.stringify({ progress, progressText })}\n\n`));
}

export const GET = withOptionalUser(async (request: RequestOptionalUser) => {
  const { searchParams } = new URL(request.url);
  const modelId = searchParams.get('modelId');
  const sourceId = searchParams.get('sourceId');
  if (!modelId || !sourceId) {
    return NextResponse.json({ error: 'modelId and sourceId are required query parameters' }, { status: 400 });
  }
  const path = `${DATASET_BASE_PATH}${modelId}/${sourceId}`;
  console.log('Importing data from', path);

  if (!IS_LOCALHOST && request.user && !(await getAuthedAdminUser(request as RequestAuthedAdminUser))) {
    return NextResponse.json({ error: 'This route is only available on localhost or to admin users' }, { status: 400 });
  }

  // for testing only
  // const activationsCsvString = fs.readFileSync(`${process.cwd()}/app/api/admin/import/temp.csv`, 'utf8');
  // await importCsvString('Activation', activationsCsvString);
  // throw new Error('SUCCESS');

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        enqueueProgress(controller, 0.25, '(0 of 4) Updating Config...');

        await importConfigFromS3();

        enqueueProgress(controller, 0.25, '(1 of 4) Importing Metadata...');
        /* RELEASE */
        try {
          const releaseLines = await downloadFileJsonlParsedLines(`${path}/release.jsonl`);
          for (const line of releaseLines) {
            const existingRelease = await prisma.sourceRelease.findUnique({
              where: { name: line.name },
            });
            if (!existingRelease) {
              await prisma.sourceRelease.create({
                data: line,
              });
            }
          }
        } catch (error) {
          console.error('Error importing release:', error);
        }

        /* MODEL */
        // don't overwrite existing
        const modelLines = await downloadFileJsonlParsedLines(`${path}/model.jsonl`);
        for (const line of modelLines) {
          const existingModel = await prisma.model.findUnique({
            where: { id: line.id },
          });
          if (!existingModel) {
            await prisma.model.create({
              data: line,
            });
          }
        }

        /* SOURCESET */
        const sourceSetLines = await downloadFileJsonlParsedLines(`${path}/sourceset.jsonl`);
        for (const line of sourceSetLines) {
          const existingSourceSet = await prisma.sourceSet.findUnique({
            where: {
              modelId_name: {
                modelId,
                name: line.name,
              },
            },
          });
          if (!existingSourceSet) {
            await prisma.sourceSet.create({
              data: {
                ...line,
                urls: line.urls ?? [],
              },
            });
          }
        }

        /* SOURCE */
        const sourceLines = await downloadFileJsonlParsedLines(`${path}/source.jsonl`);
        for (const line of sourceLines) {
          const existingSource = await prisma.source.findUnique({
            where: {
              modelId_id: {
                modelId,
                id: line.id,
              },
            },
          });
          if (!existingSource) {
            await prisma.source.create({
              data: {
                ...line,
                inferenceHosts: line.inferenceHosts && line.inferenceHosts.length > 0 ? line.inferenceHosts : undefined,
              },
            });
          }
        }

        /* INFERENCE HOST SOURCES */
        try {
          const inferenceHostSourceLines = await downloadFileJsonlParsedLines(`${path}/inference_hosts.jsonl`);
          for (const line of inferenceHostSourceLines) {
            await prisma.inferenceHostSource.upsert({
              where: { id: line.id },
              update: line,
              create: line,
            });
          }
        } catch (error) {
          console.error('Error importing inference host sources:', error);
        }

        /* INFERENCE HOST SOURCE CONNECTIONS */
        try {
          const inferenceHostOnSourceLines = await downloadFileJsonlParsedLines(
            `${path}/inference_hosts_on_source.jsonl`,
          );
          for (const line of inferenceHostOnSourceLines) {
            await prisma.inferenceHostSourceOnSource.upsert({
              where: {
                sourceId_sourceModelId_inferenceHostId: {
                  inferenceHostId: line.inferenceHostId,
                  sourceId: line.sourceId,
                  sourceModelId: line.sourceModelId,
                },
              },
              update: line,
              create: line,
            });
          }
        } catch (error) {
          console.error('Error importing inference host source connections:', error);
        }

        // /* FEATURES */
        const featuresPaths = await getFilesInPath(`${path}/features`, '.jsonl.gz');
        for (const [index, featuresPath] of featuresPaths.entries()) {
          enqueueProgress(controller, index / featuresPaths.length, `(2 of 4) Importing Features...`);
          console.log('Importing features from', featuresPath);
          const featuresJsonlString = await downloadAndDecompressFile(featuresPath);
          await importJsonlString('Neuron', featuresJsonlString);
        }

        /* EXPLANATIONS */
        const explanationsPaths = await getFilesInPath(`${path}/explanations`, '.jsonl.gz');
        for (const [index, explanationsPath] of explanationsPaths.entries()) {
          enqueueProgress(controller, index / explanationsPaths.length, `(3 of 4) Importing Explanations...`);
          console.log('Importing explanations from', explanationsPath);
          const explanationsJsonlString = await downloadAndDecompressFile(explanationsPath);
          await importJsonlString('Explanation', explanationsJsonlString);
        }

        /* ACTIVATIONS */
        const activationsPaths = await getFilesInPath(`${path}/activations`, '.jsonl.gz');
        for (const [index, activationsPath] of activationsPaths.entries()) {
          enqueueProgress(controller, index / activationsPaths.length, `(4 of 4) Importing Activations...`);
          console.log('Importing activations from', activationsPath);
          const activationsJsonlString = await downloadAndDecompressFile(activationsPath);
          await importJsonlString('Activation', activationsJsonlString);
        }

        controller.enqueue(encoder.encode('event: complete\ndata: {}\n\n'));
        controller.close();
      } catch (error) {
        console.error('Error importing data:', error);
        controller.enqueue(encoder.encode(`event: error\ndata: ${JSON.stringify({ message: error })}\n\n`));
        controller.close();
      }
    },
  });

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
});
