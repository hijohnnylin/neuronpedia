import type { NeuronConnectionsResponse, TraceNode } from '@/lib/api/sparsity-types';
import { prisma } from '@/lib/db';
import { SPARSITY_SERVER, SPARSITY_SERVER_SECRET, USE_LOCALHOST_SPARSITY } from '@/lib/env';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// Explanation type for the response
type ExplanationInfo = {
  id: string;
  description: string;
  explanationModelName: string | null;
  typeName: string | null;
};

// Response types for the API
type NeuronExplanation = {
  modelId: string;
  layer: string;
  index: string;
  explanations: ExplanationInfo[];
};

type ResidChannelExplanation = {
  modelId: string;
  layer: string; // e.g., "0-resid", "2-resid"
  index: string; // channel index
  explanations: ExplanationInfo[];
};

type ConnectedNeuronsResponse = {
  layer: number;
  neuron: number;
  trace_forward: TraceNode[];
  trace_backward: TraceNode[];
  neuronExplanations: NeuronExplanation[];
  residChannelExplanations: ResidChannelExplanation[];
};

/**
 * Fetch top 2 explanations for multiple features in a single query
 * Returns a map keyed by "layer:index" for easy lookup
 */
async function getBulkTopExplanations(
  modelId: string,
  features: { layer: string; index: string }[],
): Promise<Map<string, ExplanationInfo[]>> {
  if (features.length === 0) {
    return new Map();
  }

  // Build OR conditions for all features
  const orConditions = features.map((f) => ({
    layer: f.layer,
    index: f.index,
  }));

  // Fetch all matching explanations in a single query
  const allExplanations = await prisma.explanation.findMany({
    where: {
      modelId,
      typeName: 'python-code',
      OR: orConditions,
    },
    select: {
      id: true,
      description: true,
      explanationModelName: true,
      typeName: true,
      layer: true,
      index: true,
      scoreV2: true,
      scoreV1: true,
    },
    orderBy: [{ scoreV2: 'desc' }, { scoreV1: 'desc' }],
  });

  // Group by layer:index and take top 2 for each
  const resultMap = new Map<string, ExplanationInfo[]>();

  for (const exp of allExplanations) {
    const key = `${exp.layer}:${exp.index}`;
    const existing = resultMap.get(key) || [];

    // Only keep top 2 per feature
    if (existing.length < 2) {
      existing.push({
        id: exp.id,
        description: exp.description ?? '',
        explanationModelName: exp.explanationModelName,
        typeName: exp.typeName,
      });
      resultMap.set(key, existing);
    }
  }

  return resultMap;
}

/**
 * Extract all unique neurons from trace data (forward and backward)
 */
function extractAllNeurons(traceForward: TraceNode[], traceBackward: TraceNode[]): { layer: number; neuron: number }[] {
  const neurons: { layer: number; neuron: number }[] = [];
  const seen = new Set<string>();

  const collectNeurons = (nodes: TraceNode[], direction: 'forward' | 'backward') => {
    nodes.forEach((node) => {
      const key = `${node.layer}-${node.neuron}`;
      if (!seen.has(key)) {
        seen.add(key);
        neurons.push({ layer: node.layer, neuron: node.neuron });
      }
      // Recursively collect from children/parents
      if (direction === 'forward' && node.children) {
        collectNeurons(node.children, direction);
      }
      if (direction === 'backward' && node.parents) {
        collectNeurons(node.parents, direction);
      }
    });
  };

  collectNeurons(traceForward, 'forward');
  collectNeurons(traceBackward, 'backward');

  return neurons;
}

/**
 * Extract all unique resid channels with their relevant layers from trace data.
 * Fills in intermediate layers so we have explanations for each layer a channel passes through.
 *
 * For backward trace: If a neuron at layer X writes to channel Y which the current neuron reads,
 *   we need explanations for layers X, X+1, ..., currentLayer-1 at channel Y.
 *
 * For forward trace: If the current neuron writes to channel Y which a neuron at layer X reads,
 *   we need explanations for layers currentLayer, currentLayer+1, ..., X at channel Y.
 */
function extractResidChannels(
  currentLayer: number,
  traceForward: TraceNode[],
  traceBackward: TraceNode[],
): { layer: number; channel: number }[] {
  const seen = new Set<string>();
  const channels: { layer: number; channel: number }[] = [];

  const addChannel = (layer: number, channel: number) => {
    const key = `${layer}-${channel}`;
    if (!seen.has(key)) {
      seen.add(key);
      channels.push({ layer, channel });
    }
  };

  // For backward trace: collect all channels and track the min layer for each channel
  // Then fill in from minLayer to currentLayer-1
  const backwardChannelMinLayer = new Map<number, number>();

  const collectBackwardChannels = (nodes: TraceNode[]) => {
    nodes.forEach((node) => {
      const channel = node.via_channel;
      const existingMin = backwardChannelMinLayer.get(channel);
      if (existingMin === undefined || node.layer < existingMin) {
        backwardChannelMinLayer.set(channel, node.layer);
      }

      if (node.parents) {
        collectBackwardChannels(node.parents);
      }
    });
  };

  // For forward trace: collect all channels and track the max layer for each channel
  // Then fill in from currentLayer to maxLayer
  const forwardChannelMaxLayer = new Map<number, number>();

  const collectForwardChannels = (nodes: TraceNode[]) => {
    nodes.forEach((node) => {
      const channel = node.via_channel;
      const existingMax = forwardChannelMaxLayer.get(channel);
      if (existingMax === undefined || node.layer > existingMax) {
        forwardChannelMaxLayer.set(channel, node.layer);
      }

      if (node.children) {
        collectForwardChannels(node.children);
      }
    });
  };

  collectBackwardChannels(traceBackward);
  collectForwardChannels(traceForward);

  // Fill in all layers from minLayer to currentLayer-1 for backward channels
  for (const [channel, minLayer] of backwardChannelMinLayer) {
    for (let layer = minLayer; layer < currentLayer; layer += 1) {
      addChannel(layer, channel);
    }
  }

  // Fill in all layers from currentLayer to maxLayer for forward channels
  for (const [channel, maxLayer] of forwardChannelMaxLayer) {
    for (let layer = currentLayer; layer <= maxLayer; layer += 1) {
      addChannel(layer, channel);
    }
  }

  return channels;
}

/**
 * @swagger
 * {
 *   "/api/sparsity/connected-neurons": {
 *     "get": {
 *       "tags": ["Sparsity"],
 *       "summary": "Get Connected Neurons",
 *       "description": "Returns connected neurons from the sparsity server, along with top 2 explanations for each neuron and residual stream channel.",
 *       "parameters": [
 *         {
 *           "name": "modelId",
 *           "in": "query",
 *           "description": "Model ID",
 *           "required": true,
 *           "schema": { "type": "string" }
 *         },
 *         {
 *           "name": "layer",
 *           "in": "query",
 *           "description": "Layer index",
 *           "required": true,
 *           "schema": { "type": "integer" }
 *         },
 *         {
 *           "name": "index",
 *           "in": "query",
 *           "description": "Neuron index",
 *           "required": true,
 *           "schema": { "type": "integer" }
 *         },
 *         {
 *           "name": "traceDepth",
 *           "in": "query",
 *           "description": "Depth of circuit trace (default: 1)",
 *           "required": false,
 *           "schema": { "type": "integer", "default": 1 }
 *         },
 *         {
 *           "name": "traceK",
 *           "in": "query",
 *           "description": "Top K channels/neurons per step in trace (default: 5)",
 *           "required": false,
 *           "schema": { "type": "integer", "default": 5 }
 *         }
 *       ],
 *       "responses": {
 *         "200": { "description": "Connected neurons with explanations" },
 *         "400": { "description": "Missing required parameters" },
 *         "500": { "description": "Error fetching from sparsity server" }
 *       }
 *     }
 *   }
 * }
 */
export const GET = withOptionalUser(async (request: RequestOptionalUser) => {
  const { searchParams } = new URL(request.url);
  const modelId = searchParams.get('modelId');
  const layerParam = searchParams.get('layer');
  const indexParam = searchParams.get('index');
  const traceDepth = searchParams.get('traceDepth') || '1';
  const traceK = searchParams.get('traceK') || '5';

  if (!modelId || !layerParam || !indexParam) {
    return NextResponse.json({ error: 'Missing required parameters: modelId, layer, index' }, { status: 400 });
  }

  const layer = parseInt(layerParam, 10);
  const index = parseInt(indexParam, 10);

  if (Number.isNaN(layer) || Number.isNaN(index)) {
    return NextResponse.json({ error: 'layer and index must be valid integers' }, { status: 400 });
  }

  try {
    // Build the sparsity server URL
    const sparsityUrl = USE_LOCALHOST_SPARSITY ? 'http://localhost:5005' : SPARSITY_SERVER;

    // Fetch connected neurons from sparsity server
    const headers: HeadersInit = {};
    if (SPARSITY_SERVER_SECRET) {
      headers['X-SECRET-KEY'] = SPARSITY_SERVER_SECRET;
    }

    const sparsityResponse = await fetch(
      `${sparsityUrl}/neuron/${layer}/${index}?trace_depth=${traceDepth}&trace_k=${traceK}`,
      { headers },
    );

    if (!sparsityResponse.ok) {
      const errorText = await sparsityResponse.text();
      return NextResponse.json(
        { error: `Sparsity server error: ${sparsityResponse.status} - ${errorText}` },
        { status: sparsityResponse.status },
      );
    }

    const sparsityData: NeuronConnectionsResponse = await sparsityResponse.json();

    // Extract all neurons from trace data
    const allNeurons = extractAllNeurons(sparsityData.trace_forward, sparsityData.trace_backward);
    allNeurons.push({ layer, neuron: index });

    // Extract all resid channels with their layers
    const allResidChannels = extractResidChannels(
      sparsityData.layer,
      sparsityData.trace_forward,
      sparsityData.trace_backward,
    );

    // Build list of all features (neurons and resid channels) to fetch explanations for
    const neuronFeatures = allNeurons.map((n) => ({
      layer: `${n.layer}-mlp`,
      index: n.neuron.toString(),
    }));

    const residFeatures = allResidChannels.map((rc) => ({
      layer: `${rc.layer}-resid`,
      index: rc.channel.toString(),
    }));

    // Combine all features and fetch explanations in a single query
    const allFeatures = [...neuronFeatures, ...residFeatures];

    const explanationsMap = await getBulkTopExplanations(modelId, allFeatures);

    // Map results back to neuron explanations
    const neuronExplanations: NeuronExplanation[] = neuronFeatures.map((f) => ({
      modelId,
      layer: f.layer,
      index: f.index,
      explanations: explanationsMap.get(`${f.layer}:${f.index}`) || [],
    }));

    // Map results back to resid channel explanations
    const residChannelExplanations: ResidChannelExplanation[] = residFeatures.map((f) => ({
      modelId,
      layer: f.layer,
      index: f.index,
      explanations: explanationsMap.get(`${f.layer}:${f.index}`) || [],
    }));

    const response: ConnectedNeuronsResponse = {
      layer: sparsityData.layer,
      neuron: sparsityData.neuron,
      trace_forward: sparsityData.trace_forward,
      trace_backward: sparsityData.trace_backward,
      neuronExplanations,
      residChannelExplanations,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Error fetching connected neurons:', error);
    return NextResponse.json(
      { error: `Failed to fetch connected neurons: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 },
    );
  }
});
