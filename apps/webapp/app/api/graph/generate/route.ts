import {
  ATTRIBUTION_GRAPH_SCHEMA,
  CLTGraph,
  makeGraphPublicAccessGraphUrl,
  NP_GRAPH_BUCKET,
} from '@/app/[modelId]/graph/utils';
import { prisma } from '@/lib/db';
import {
  generateGraph,
  GRAPH_ANONYMOUS_USER_ID,
  GRAPH_BATCH_SIZE,
  GRAPH_MAX_TOKENS,
  GRAPH_S3_USER_GRAPHS_DIR,
  graphGenerateSchemaClient,
} from '@/lib/utils/graph';
import { tokenizeText } from '@/lib/utils/inference';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import Ajv from 'ajv';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

const SCAN_TO_SOURCE_URLS = {
  'gemma-2-2b': [
    'https://neuronpedia.org/gemma-2-2b/gemmascope-transcoder-16k',
    'https://huggingface.co/google/gemma-scope-2b-pt-transcoders',
  ],
};

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
 *               slug:
 *                 type: string
 *                 description: A unique identifier for this graph (alphanumeric, underscores, and hyphens only)
 *                 pattern: '^[a-zA-Z0-9_-]+$'
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
 *                 default: 0.98
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
    const tokenized = await tokenizeText(validatedData.modelId, validatedData.prompt, false);

    if (tokenized.tokens.length > GRAPH_MAX_TOKENS) {
      return NextResponse.json(
        {
          error: 'Prompt Too Long',
          message: `Max tokens supported is ${GRAPH_MAX_TOKENS}, your prompt was ${tokenized.tokens.length} tokens.`,
        },
        { status: 400 },
      );
    }

    console.log(`Tokens in text: ${tokenized.tokens.length} - ${tokenized.tokenStrings}`);

    // if scan or slug has weird characters, return error
    if (/[^a-zA-Z0-9_-]/.test(validatedData.slug)) {
      return NextResponse.json(
        { error: 'Invalid scan or slug. They must be alphanumeric and contain only underscores and hyphens.' },
        { status: 400 },
      );
    }

    // check if the modelId/slug exist in the database already
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

    const data = await generateGraph(
      validatedData.prompt,
      validatedData.modelId,
      validatedData.maxNLogits,
      validatedData.desiredLogitProb,
      validatedData.nodeThreshold,
      validatedData.edgeThreshold,
      validatedData.slug,
      validatedData.maxFeatureNodes,
    );

    console.log('data generated');

    // Validate the graph against the JSON schema
    const ajv = new Ajv({ allErrors: true, strict: false });
    const validate = ajv.compile(ATTRIBUTION_GRAPH_SCHEMA);

    // Create a deep copy of the data to avoid mutation during validation
    const dataCopy = JSON.parse(JSON.stringify(data));
    const isValid = validate(dataCopy);

    if (!isValid) {
      return NextResponse.json({
        error: 'Invalid Graph Generated',
        message: validate.errors,
      });
    }

    // data is valid - parse it as as json string to CLTGraph and add our metadata to it
    const graph = data as CLTGraph;
    graph.metadata = {
      ...graph.metadata,
      info: {
        creator_name: `${request.user?.name || 'Anonymous'} (OSCF)`,
        creator_url: 'https://neuronpedia.org',
        // TODO: other sources when they become available
        source_urls: SCAN_TO_SOURCE_URLS[data.metadata.scan as keyof typeof SCAN_TO_SOURCE_URLS] || [],
        generator: {
          name: 'Open Source Circuit Finding by Piotrowski & Hanna',
          version: '0.1.0 | 042bf4df20be61890110ed22b6024c72337719cf',
          url: 'https://github.com/safety-research/open-source-circuit-finding',
        },
        create_time_ms: Date.now(),
      },
      generation_settings: {
        max_n_logits: validatedData.maxNLogits,
        desired_logit_prob: validatedData.desiredLogitProb,
        batch_size: GRAPH_BATCH_SIZE,
        max_feature_nodes: validatedData.maxFeatureNodes,
      },
      pruning_settings: {
        node_threshold: validatedData.nodeThreshold,
        edge_threshold: validatedData.edgeThreshold,
      },
    };

    // once we have the data, upload it to S3
    const key = `${GRAPH_S3_USER_GRAPHS_DIR}/${request.user?.id || GRAPH_ANONYMOUS_USER_ID}/${validatedData.slug}.json`;

    const s3Client = new S3Client({
      region: process.env.AWS_REGION || 'us-east-1',
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
      },
    });

    const command = new PutObjectCommand({
      Bucket: NP_GRAPH_BUCKET,
      Key: key,
      ContentType: 'application/json',
      ContentLength: Buffer.byteLength(JSON.stringify(data)),
      Body: JSON.stringify(data),
    });

    await s3Client.send(command);

    const cleanUrl = `https://${NP_GRAPH_BUCKET}.s3.${process.env.AWS_REGION || 'us-east-1'}.amazonaws.com/${key}`;

    console.log('S3 file upload complete');

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
