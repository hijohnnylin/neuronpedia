import { CLTGraph } from '@/app/[modelId]/graph/graph-types';
import { ATTRIBUTION_GRAPH_SCHEMA, makeGraphPublicAccessGraphUrl, NP_GRAPH_BUCKET } from '@/app/[modelId]/graph/utils';
import { prisma } from '@/lib/db';
import { getGraphServerRunpodHostForSourceSet } from '@/lib/db/graph-host-source';
import { getModelById } from '@/lib/db/model';
import { env } from '@/lib/env';
import { RequestOptionalUser } from '@/lib/types/auth';
import {
  checkRunpodQueueJobs,
  generateGraphAndUploadToS3,
  getGraphTokenize,
  GRAPH_ANONYMOUS_USER_ID,
  GRAPH_MAX_TOKENS,
  GRAPH_S3_USER_GRAPHS_DIR,
  GRAPH_SLUG_MIN,
  graphGenerateSchemaClient,
  MAX_RUNPOD_JOBS_IN_QUEUE,
  RUNPOD_BUSY_ERROR,
} from '@/lib/utils/graph';
import { withOptionalUser } from '@/lib/with-user';
import { PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import Ajv from 'ajv';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

/**
 * @swagger
 * /api/graph/generate:
 *   post:
 *     summary: Generate New Graph
 *     description: Creates a new graph by analyzing the provided text prompt using the specified model. You'll get a link to access the graph visualization directly on Neuronpedia's UI, and the graph will be saved to S3 and metadata stored in the database.
 *     tags:
 *       - Attribution Graphs
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - prompt
 *               - modelId
 *               - slug
 *             properties:
 *               prompt:
 *                 type: string
 *                 description: The text prompt to analyze and generate a graph from. Capped at 64 max tokens.
 *                 maxLength: 10000
 *                 example: "abc12"
 *               modelId:
 *                 type: string
 *                 description: The ID of the model to use for graph generation. Currently only gemma-2-2b is supported.
 *                 pattern: '^[a-zA-Z0-9_-]+$'
 *                 example: "gemma-2-2b"
 *               sourceSetName:
 *                 type: string
 *                 description: Optional. The name of the source set to use for graph generation. If not provided, the default source set for the model will be used.
 *                 pattern: '^[a-zA-Z0-9_-]+$'
 *                 example: "gemmascope-transcoder-16k"
 *               slug:
 *                 type: string
 *                 description: A unique identifier for this graph (lowercase, alphanumeric, underscores, and hyphens only)
 *                 pattern: '^[a-z0-9_-]+$'
 *               maxNLogits:
 *                 type: number
 *                 description: Maximum number of logits to consider
 *                 minimum: 5
 *                 maximum: 15
 *                 default: 10
 *               desiredLogitProb:
 *                 type: number
 *                 description: Desired logit probability threshold
 *                 minimum: 0.6
 *                 maximum: 0.99
 *                 default: 0.95
 *               nodeThreshold:
 *                 type: number
 *                 description: Threshold for including nodes in the graph
 *                 minimum: 0.5
 *                 maximum: 1.0
 *                 default: 0.8
 *               edgeThreshold:
 *                 type: number
 *                 description: Threshold for including edges in the graph
 *                 minimum: 0.8
 *                 maximum: 1.0
 *                 default: 0.85
 *               maxFeatureNodes:
 *                 type: number
 *                 description: Maximum number of feature nodes to include in the graph
 *                 minimum: 3000
 *                 maximum: 10000
 *                 default: 5000
 *     responses:
 *       200:
 *         description: Graph generated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                   example: "Graph saved to database"
 *                 s3url:
 *                   type: string
 *                   description: The S3 URL where the graph data is stored
 *                 url:
 *                   type: string
 *                   description: The public URL to access the generated graph
 *                 numNodes:
 *                   type: integer
 *                   description: Number of nodes in the generated graph
 *                 numLinks:
 *                   type: integer
 *                   description: Number of links in the generated graph
 *       400:
 *         description: Bad request - validation error, prompt too long, or duplicate slug
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 error:
 *                   type: string
 *                   enum: ["Validation error", "Prompt Too Long", "Model + Slug/ID Exists", "Invalid scan or slug"]
 *                 message:
 *                   type: string
 *                   description: Detailed error message
 *                 details:
 *                   type: string
 *                   description: Additional error details (for validation errors)
 *       500:
 *         description: Internal server error
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 error:
 *                   type: string
 *                   example: "Failed to generate graph"
 *                 message:
 *                   type: string
 *                   description: Error message
 *       503:
 *         description: Service unavailable - GPUs are busy
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 error:
 *                   type: string
 *                   example: "GPUs Busy"
 *                 message:
 *                   type: string
 *                   example: "Sorry, all GPUs are currently busy. Please try again in a minute."
 */

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  try {
    let body = '';
    try {
      body = await request.json();
    } catch {
      throw new Error('Invalid JSON body');
    }
    const validatedData = await graphGenerateSchemaClient.validate(body);

    // if sourceSetName is not provided, use the default source set for the model
    if (!validatedData.sourceSetName) {
      const model = await getModelById(validatedData.modelId);
      validatedData.sourceSetName = model?.defaultGraphSourceSetName;
      if (!validatedData.sourceSetName) {
        return NextResponse.json(
          {
            error: 'Source Set Missing',
            message: `The model ${validatedData.modelId} has no default graph source set, so you must provide one in the sourceSetName parameter.`,
          },
          { status: 400 },
        );
      }
    }

    try {
      const tokenized = await getGraphTokenize(
        validatedData.prompt,
        validatedData.maxNLogits,
        validatedData.desiredLogitProb,
        validatedData.modelId,
        validatedData.sourceSetName,
      );
      if (tokenized.input_tokens.length > GRAPH_MAX_TOKENS) {
        return NextResponse.json(
          {
            error: 'Prompt Too Long',
            message: `Max tokens supported is ${GRAPH_MAX_TOKENS}, your prompt was ${tokenized.input_tokens.length} tokens.`,
          },
          { status: 400 },
        );
      }

      console.log(`Tokens in text: ${tokenized.input_tokens.length}`);
    } catch (error) {
      console.error('Error tokenizing text, continuing:', error);
    }

    if (
      !validatedData.slug ||
      validatedData.slug.trim().length === 0 ||
      validatedData.slug.trim().length < GRAPH_SLUG_MIN
    ) {
      // if slug doesn't exist, generate one based on the prompt
      // first remove all <> and \n (estimated to be special tokens)
      const promptReplaced = validatedData.prompt.replace(/<[^>]*>|\n/g, '');
      // remove "user" and "assistant", and "system"
      const promptReplaced2 = promptReplaced.replace(/user|assistant|system/g, '');
      // keeping only alphanumeric characters, and keeping only the first 16 characters
      const slug = promptReplaced2.replace(/[^a-zA-Z0-9_-]/g, '').substring(0, 16);
      // append the current epoch time
      const timestamp = Date.now();
      validatedData.slug = `${slug}-${timestamp}`.toLowerCase();
      console.log('generated slug', validatedData.slug);
    }

    // if scan or slug has weird characters, return error
    if (/[^a-zA-Z0-9_-]/.test(validatedData.slug)) {
      return NextResponse.json(
        { error: 'Invalid scan or slug. They must be alphanumeric and contain only underscores and hyphens.' },
        { status: 400 },
      );
    }

    // check if the modelId + slug exists in the database already
    const existingGraphMetadata = await prisma.graphMetadata.findUnique({
      where: { modelId_slug: { modelId: validatedData.modelId, slug: validatedData.slug } },
    });

    if (existingGraphMetadata) {
      return NextResponse.json(
        {
          error: 'Model + Slug/ID Exists',
          message: `The model ${validatedData.modelId} already has a graph with slug/id ${validatedData.slug}. Please choose a different slug/ID.`,
        },
        { status: 400 },
      );
    }

    // make a signed put for this user

    const userId = request.user?.id;
    const userName = request.user?.name;

    const key = `${GRAPH_S3_USER_GRAPHS_DIR}/${userId || GRAPH_ANONYMOUS_USER_ID}/${validatedData.slug}.json`;

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
    });

    // Generate the presigned URL for 1 hour
    const signedUrl = await getSignedUrl(s3Client, command, {
      expiresIn: 3600, // 1 hour
    });

    // check the queue
    if (env.USE_RUNPOD_GRAPH) {
      const host = await getGraphServerRunpodHostForSourceSet(validatedData.modelId, validatedData.sourceSetName);
      if (!host) {
        throw new Error('No runpod serverless host found.');
      }
      const queueNumber = await checkRunpodQueueJobs(host);
      if (queueNumber > MAX_RUNPOD_JOBS_IN_QUEUE) {
        // console.log('larger than queue but continuing');
        return NextResponse.json(
          {
            error: RUNPOD_BUSY_ERROR,
            message: RUNPOD_BUSY_ERROR,
          },
          { status: 503 },
        );
      }
    }

    await generateGraphAndUploadToS3(
      validatedData.prompt,
      validatedData.modelId,
      validatedData.sourceSetName,
      validatedData.maxNLogits,
      validatedData.desiredLogitProb,
      validatedData.nodeThreshold,
      validatedData.edgeThreshold,
      validatedData.slug,
      validatedData.maxFeatureNodes,
      signedUrl,
      userName || 'Anonymous (CT)',
    );

    // download the file from S3
    const cleanUrl = signedUrl.split('?')[0];
    const response = await fetch(cleanUrl);
    if (!response.ok) {
      return NextResponse.json(
        {
          error: 'Failed to download graph.',
          message: `Failed to download graph. HTTP ${response.status}: ${response.statusText}`,
        },
        { status: 500 },
      );
    }
    const responseJson = await response.json();
    const graph = responseJson as CLTGraph;

    // Validate the graph against the JSON schema
    const ajv = new Ajv({ allErrors: true, strict: false });
    const validate = ajv.compile(ATTRIBUTION_GRAPH_SCHEMA);

    // Create a deep copy of the data to avoid mutation during validation
    const dataCopy = JSON.parse(JSON.stringify(responseJson));

    const isValid = validate(dataCopy);

    if (!isValid) {
      console.log('invalid: ', validate.errors);
      return NextResponse.json(
        {
          error: 'Invalid Graph Generated',
          message: validate.errors,
        },
        { status: 500 },
      );
    }
    console.log('valid graph');

    // save it to the database
    await prisma.graphMetadata.upsert({
      where: {
        modelId_slug: {
          modelId: graph.metadata.scan,
          slug: graph.metadata.slug,
        },
      },
      update: {
        userId: request.user?.id ? request.user?.id : null,
        modelId: graph.metadata.scan,
        sourceSetName: validatedData.sourceSetName,
        slug: graph.metadata.slug,
        titlePrefix: '',
        promptTokens: graph.metadata.prompt_tokens,
        prompt: graph.metadata.prompt,
        url: cleanUrl,
        isFeatured: false,
      },
      create: {
        userId: request.user?.id ? request.user?.id : null,
        modelId: graph.metadata.scan,
        sourceSetName: validatedData.sourceSetName,
        slug: graph.metadata.slug,
        titlePrefix: '',
        promptTokens: graph.metadata.prompt_tokens,
        prompt: graph.metadata.prompt,
        url: cleanUrl,
        isFeatured: false,
      },
    });

    return NextResponse.json({
      message: 'Graph saved to database',
      s3url: cleanUrl,
      url: makeGraphPublicAccessGraphUrl(graph.metadata.scan, graph.metadata.slug),
      numNodes: graph.nodes.length,
      numLinks: graph.links.length,
    });
  } catch (error) {
    console.error('Error generating graph:', error);

    if (error instanceof yup.ValidationError) {
      return NextResponse.json(
        { error: 'Validation error', details: error.message, path: error.path },
        { status: 400 },
      );
    }
    if (error instanceof Error && error.message.indexOf('503') > -1) {
      return NextResponse.json(
        {
          error: 'GPUs Busy',
          message: 'Sorry, all GPUs are currently busy. Please try again in a minute.',
        },
        { status: 503 },
      );
    }

    return NextResponse.json(
      { error: 'Failed to generate graph', message: error instanceof Error ? error.message : String(error) },
      { status: 500 },
    );
  }
});
