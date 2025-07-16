import { MAX_GRAPH_UPLOAD_SIZE_BYTES, NP_GRAPH_BUCKET } from '@/app/[modelId]/graph/utils';
import { prisma } from '@/lib/db';
import { getUserByName } from '@/lib/db/user';
import { GRAPH_S3_USER_GRAPHS_DIR, MAX_PUT_REQUESTS_PER_DAY } from '@/lib/utils/graph';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { headers } from 'next/headers';
import { NextResponse } from 'next/server';
import { number, object, string } from 'yup';

const signedPutRequestSchema = object({
  filename: string().required(),
  contentLength: number().required().min(1024).max(MAX_GRAPH_UPLOAD_SIZE_BYTES),
});

/**
 * @swagger
 * /api/graph/signed-put:
 *   post:
 *     summary: Upload Graph 1/2 - Get Pre-Signed URL
 *     description: Creates a pre-signed URL that allows authenticated users to upload graph files directly to S3. Both this and the second step are necessary for your graph to be saved correctly. Use the returned URL with a PUT request, like this `curl -X PUT -T my-graph.json [returned-url]`. Don't lose your putRequestId, you'll need it for the second part.
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
 *               - filename
 *               - contentLength
 *             properties:
 *               filename:
 *                 type: string
 *                 description: Name of the file to be uploaded
 *               contentLength:
 *                 type: number
 *                 description: Size of the file in bytes
 *                 minimum: 1024
 *                 maximum: 209715200
 *               contentType:
 *                 type: string
 *                 description: MIME type of the file (optional)
 *                 default: application/json
 *     responses:
 *       200:
 *         description: Successfully generated signed URL
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 url:
 *                   type: string
 *                   description: Pre-signed URL for uploading to S3
 *                 putRequestId:
 *                   type: string
 *                   description: ID of the created put request record
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const bodyJson = await request.json();
  if (!bodyJson) {
    return NextResponse.json({ message: 'Invalid JSON body' }, { status: 400 });
  }

  const user = await getUserByName(request.user.name);

  try {
    const body = await signedPutRequestSchema.validate(bodyJson);

    if (!body.filename.endsWith('.json')) {
      return NextResponse.json({ error: 'Invalid file extension - must be .json' }, { status: 400 });
    }

    const userId = user.id;

    const headersList = headers();
    const forwardedFor = headersList.get('x-forwarded-for');
    const ip = forwardedFor ? forwardedFor.split(',')[0].trim() : null;

    if (!ip) {
      throw new Error('IP address is required');
    }

    // look up how many put requests this IP has made in the last 24 hours
    const putRequests = await prisma.graphMetadataDataPutRequest.findMany({
      where: {
        ipAddress: ip,
        createdAt: {
          gte: new Date(Date.now() - 24 * 60 * 60 * 1000),
        },
      },
    });
    if (putRequests.length >= MAX_PUT_REQUESTS_PER_DAY) {
      return NextResponse.json(
        { error: `Too many put requests today. The maximum is ${MAX_PUT_REQUESTS_PER_DAY}.` },
        { status: 429 },
      );
    }
    console.log('putRequests count | user ID', putRequests.length, userId);

    const key = `${GRAPH_S3_USER_GRAPHS_DIR}/${userId}/${body.filename}`;

    // Initialize S3 client
    const s3Client = new S3Client({
      region: process.env.AWS_REGION || 'us-east-1',
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
      },
    });

    // Create the command for putting an object
    const command = new PutObjectCommand({
      Bucket: NP_GRAPH_BUCKET,
      Key: key,
      ContentType: 'application/json',
      ContentLength: body.contentLength,
    });

    // Generate the presigned URL for 1 hour
    const signedUrl = await getSignedUrl(s3Client, command, {
      expiresIn: 3600, // 1 hour
    });

    // add the put request to the database
    const putRequest = await prisma.graphMetadataDataPutRequest.create({
      data: {
        userId: request.user.id,
        ipAddress: ip,
        filename: body.filename,
        url: signedUrl,
      },
    });

    return NextResponse.json({
      url: signedUrl,
      key,
      putRequestId: putRequest.id,
    });
  } catch (error) {
    console.error('Error generating signed URL:', error);
    return NextResponse.json({ error: 'Failed to generate signed URL' }, { status: 500 });
  }
});
