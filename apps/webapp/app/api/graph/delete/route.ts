import { prisma } from '@/lib/db';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { DeleteObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { NextResponse } from 'next/server';
import { object, string, ValidationError } from 'yup';

const deleteGraphSchema = object({
  modelId: string().required(),
  slug: string().required(),
});

const s3Client = new S3Client({ region: 'us-east-1' });

/**
 * @swagger
 * /api/graph/delete:
 *   post:
 *     summary: Delete Graph
 *     description: Deletes an existing graph from Neuronpedia. You can only delete graphs you created.
 *     tags:
 *       - Attribution Graphs
 *     security:
 *       - apiKey: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - modelId
 *               - slug
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: ID of the model
 *               slug:
 *                 type: string
 *                 description: Slug of the graph to delete
 *     responses:
 *       200:
 *         description: Graph deleted successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const bodyJson = await request.json();

  try {
    const body = await deleteGraphSchema.validate(bodyJson);

    // Find the graph metadata to get the URL
    const graphMetadata = await prisma.graphMetadata.findUnique({
      where: {
        modelId_slug: {
          modelId: body.modelId,
          slug: body.slug,
        },
      },
    });

    if (!graphMetadata) {
      return NextResponse.json({ message: 'Graph not found' }, { status: 404 });
    }

    // Check if the user is the owner of the graph
    if (graphMetadata.userId !== request.user.id) {
      return NextResponse.json({ message: 'Unauthorized to delete this graph' }, { status: 403 });
    }

    // Parse the URL to get the S3 key
    const { url } = graphMetadata;
    const s3KeyMatch = url.match(/amazonaws\.com\/(.+)$/);
    const s3Key = s3KeyMatch ? s3KeyMatch[1] : '';

    console.log('s3 object to delete', s3Key);

    const bucket = url.match(/https:\/\/([^.]+)/)?.[1];
    console.log('bucket ', bucket);

    const deleteCommand = new DeleteObjectCommand({
      Bucket: bucket,
      Key: s3Key,
    });

    const response = await s3Client.send(deleteCommand);
    // S3 returns 204 No Content on successful deletion
    if (response.$metadata.httpStatusCode === 204) {
      console.log('S3 object deleted successfully');
    } else {
      throw new Error(`S3 object deletion failed: ${response.$metadata.httpStatusCode}`);
    }

    await prisma.graphMetadata.delete({
      where: {
        modelId_slug: {
          modelId: body.modelId,
          slug: body.slug,
        },
      },
    });

    console.log('graph metadata deleted');

    return NextResponse.json({ message: 'Graph deleted successfully' });
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    console.error('Error deleting graph:', error);
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
