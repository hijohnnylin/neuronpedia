import ATTRIBUTION_GRAPH_SCHEMA from '@/app/api/graph/graph-schema.json';
import { env } from '@/lib/env';
import { GraphMetadata } from '@/prisma/generated/zod';
import cuid from 'cuid';
import d3 from './d3-jetpack';
import {
  AnthropicGraphMetadata,
  CLTGraph,
  CLTGraphLink,
  CLTGraphNode,
  CltVisState,
  GraphClerps,
  GraphClerpsSchema,
  GraphSupernodes,
  GraphSupernodesSchema,
  ModelToGraphMetadatasMap,
} from './graph-types';

// TODO: make this an env variable
export const NP_GRAPH_BUCKET = 'neuronpedia-attrib';
export const ANT_BUCKET_URL = 'https://transformer-circuits.pub/2025/attribution-graphs';
export const MAX_GRAPH_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024;

// ============ Neuronpedia Specific =============

export const DEFAULT_GRAPH_MODEL_ID = 'gemma-2-2b';
export const ADDITIONAL_MODELS_TO_LOAD = new Set(['qwen3-4b']);
export const MODELS_WITH_NP_DASHBOARDS = new Set(['gemma-2-2b', 'qwen3-4b']);
export const MODELS_TO_CALCULATE_REPLACEMENT_SCORES = MODELS_WITH_NP_DASHBOARDS;
export const ANTHROPIC_MODELS = new Set(['jackl-circuits-runs-1-4-sofa-v3_0']);
export const ANTHROPIC_MODEL_TO_NUM_LAYERS = {
  'jackl-circuits-runs-1-4-sofa-v3_0': 18,
};
export const ANTHROPIC_MODEL_TO_DISPLAY_NAME = new Map<string, string>([
  ['jackl-circuits-runs-1-4-sofa-v3_0', 'Haiku'],
]);
export const ANTHROPIC_DUMMY_SOURCESET = { name: 'jackl-circuits-runs-1-4-sofa-v3_0', creatorName: 'Anthropic' };
export const GRAPH_BASE_URL_TO_NAME = {
  'https://transformer-circuits.pub/2025/attribution-graphs': 'Ameisen et al.',
  'https://d1fk9w8oratjix.cloudfront.net': 'Piotrowski & Hanna',
};

export const graphModelHasNpDashboards = (graph: CLTGraph) =>
  graph.metadata.feature_details?.neuronpedia_source_set !== undefined ||
  MODELS_WITH_NP_DASHBOARDS.has(graph.metadata.scan);

export const isOldQwenGraph = (graph: CLTGraph) =>
  graph.metadata.scan === 'qwen3-4b' &&
  (graph.metadata.schema_version === undefined || graph.metadata.schema_version === null);

export function isHideLayer(scan: string) {
  return scan === 'jackl-circuits-runs-1-4-sofa-v3_0';
}

export const ERROR_MODEL_DOES_NOT_EXIST = 'ERR_MODEL_DOES_NOT_EXIST';

// ============ End of Neuronpedia Specific =============

export function getLayerFromOldSchema0Feature(modelId: string, featureNode: CLTGraphNode) {
  if (modelId === 'qwen3-4b') {
    return parseInt(featureNode.layer, 10);
  }
  // otherwise this is gemma-2-2b schema 0 which has 5 digits in the feature
  const gemma2FeatureDigits = 5;
  // remove dash and everything after it
  const layer = featureNode.feature.toString().replace(/-.*$/, '');
  // the layer is the number before the last digitsInNumFeatures digits
  const layerStr = layer.slice(0, -gemma2FeatureDigits);
  if (layerStr.length === 0) {
    return 0;
  }
  return parseInt(layerStr, 10);
}

function getIndexFromOldSchema0Feature(modelId: string, featureNode: CLTGraphNode) {
  if (modelId === 'qwen3-4b') {
    return featureNode.feature;
  }
  // otherwise this is gemma-2-2b schema 0 which has 5 digits in the feature
  const gemma2FeatureDigits = 5;
  // remove dash and everything before it
  const index = featureNode.feature.toString().replace(/-.*$/, '');
  // the index is the last digitsInNumFeatures digits
  const indexStr = index.slice(-gemma2FeatureDigits);
  if (indexStr.length === 0) {
    return 0;
  }
  return parseInt(indexStr, 10);
}

export function getIndexFromCantorValue(feature: number): number {
  const w = Math.floor((Math.sqrt(8 * feature + 1) - 1) / 2);
  const t = (w * w + w) / 2;
  const y = feature - t;
  return y;
}

export function getLayerFromCantorValue(feature: number): number {
  const w = Math.floor((Math.sqrt(8 * feature + 1) - 1) / 2);
  const t = (w * w + w) / 2;
  const x = w - (feature - t);
  return x;
}

export function getLayerFromFeatureAndGraph(modelId: string, node: CLTGraphNode, selectedGraph: CLTGraph | null) {
  if (selectedGraph?.metadata.schema_version === 1) {
    return getLayerFromCantorValue(node.feature);
  }
  // cases
  // - gemma-2-2b (new schema = handled above)
  // - gemma-2-2b old schema = handled here
  // - qwen3-4b = new schema = handled above
  // - qwen3-4b old schema = we aren't steering those
  if (modelId === 'gemma-2-2b') {
    return getLayerFromOldSchema0Feature(modelId, node);
  }
  console.error(
    `LayerFromFeature: ${modelId} - failed to get layer from feature. Returning 0. Graph: ${selectedGraph?.metadata.scan}`,
  );
  return 0;
}

export function getIndexFromFeatureAndGraph(modelId: string, node: CLTGraphNode, selectedGraph: CLTGraph | null) {
  if (selectedGraph?.metadata.schema_version === 1) {
    return getIndexFromCantorValue(node.feature);
  }
  // cases
  // - gemma-2-2b (new schema = handled above)
  // - gemma-2-2b old schema = handled here
  // - qwen3-4b = new schema = handled above
  // - qwen3-4b old schema = we aren't steering those
  if (modelId === 'gemma-2-2b') {
    return getIndexFromOldSchema0Feature(modelId, node);
  }
  console.error(
    `IndexFromFeature: ${modelId} - failed to get index from feature. Returning 0. Graph: ${selectedGraph?.metadata.scan}`,
  );
  return 0;
}

export function getOldSchema0Gemma2FeatureIdFromLayerAndIndex(layer: number, index: number) {
  const digitsInNumFeatures = 5;
  const paddedIndex = index.toString().padStart(digitsInNumFeatures, '0');
  return parseInt(`${layer}${paddedIndex}`, 10);
}

function getCantorValueFromLayerAndIndex(layer: number, index: number) {
  // return the cantor value using the cantor pairing function
  const cantorValue = ((layer + index) * (layer + index + 1)) / 2 + index;
  return cantorValue;
}

export function getFeatureIdFromLayerAndIndex(
  modelId: string,
  layer: number,
  index: number,
  selectedGraph: CLTGraph | null,
) {
  if (selectedGraph?.metadata.schema_version === 1) {
    return getCantorValueFromLayerAndIndex(layer, index);
  }
  if (modelId === 'gemma-2-2b') {
    return getOldSchema0Gemma2FeatureIdFromLayerAndIndex(layer, index);
  }
  console.error(
    `FeatureIdFromLayerAndIndex: ${modelId} - failed to get feature id from layer and index. Returning 0. Graph: ${selectedGraph?.metadata.scan}`,
  );
  return 0;
}

export function getGraphBaseUrlToName(url: string) {
  // if url starts with one of the keys in GRAPH_BASE_URL_TO_NAME, return the value
  const key = Object.keys(GRAPH_BASE_URL_TO_NAME).find((k) => url.startsWith(k));
  if (key) {
    return GRAPH_BASE_URL_TO_NAME[key as keyof typeof GRAPH_BASE_URL_TO_NAME];
  }
  return null;
}

export function makeGraphPublicAccessGraphUri(modelId: string, slug: string) {
  return `/${modelId}/graph?slug=${slug}`;
}

export function makeGraphPublicAccessGraphUrl(modelId: string, slug: string) {
  return `${env.NEXT_PUBLIC_URL}${makeGraphPublicAccessGraphUri(modelId, slug)}`;
}

export function nodeTypeHasFeatureDetail(node: CLTGraphNode): boolean {
  return (
    node.feature_type !== 'embedding' &&
    node.feature_type !== 'mlp reconstruction error' &&
    node.feature_type !== 'logit'
  );
}

// Anthropic graph metadata are ones where the metadata is stored in the bucket in graph-metadata.json
// We store our metadata in the database
export async function getGraphMetadatasFromBucket(baseUrl: string): Promise<ModelToGraphMetadatasMap> {
  // first get featured graphs
  const featuredResponse = await fetch(`${baseUrl}/data/graph-metadata.json`);
  const anthropicFeaturedGraphs: { graphs: AnthropicGraphMetadata[] } = await featuredResponse.json();

  // convert to our GraphMetadata type to read locally
  // by default all anthropic graphs are featured
  const featuredGraphs: GraphMetadata[] = anthropicFeaturedGraphs.graphs.map((graph) => ({
    modelId: graph.scan,
    sourceSetName: null,
    slug: graph.slug,
    promptTokens: graph.prompt_tokens,
    prompt: graph.prompt,
    titlePrefix: graph.title_prefix,
    url: `${baseUrl}/graph_data/${graph.slug}.json`,
    userId: env.DEFAULT_CREATOR_USER_ID,
    id: cuid(),
    createdAt: new Date(),
    updatedAt: new Date(),
    isFeatured: true,
  }));

  // add all graphs to the map
  const graphsByModelId = featuredGraphs.reduce((acc, graph) => {
    acc[graph.modelId] = [...(acc[graph.modelId] || []), graph];
    return acc;
  }, {} as ModelToGraphMetadatasMap);

  return graphsByModelId;
}

// ========= util-cg.js formatData equivalent =========
// TODO: we changed == to === in many places. Ensure this does not break anything.
//
// Adds virtual logit node showing A-B logit difference based on url param logitDiff=⍽tokenA⍽__vs__⍽tokenB⍽
function addVirtualDiff(data: CLTGraph, logitDiff: string | null) {
  // Filter out any previous virtual nodes/links
  const nodes = data.nodes.filter((d) => !d.isJsVirtual);
  const links = data.links.filter((d) => !d.isJsVirtual);
  // @ts-ignore
  // eslint-disable-next-line
  nodes.forEach((d) => (d.logitToken = d.clerp?.split(`"`)[1]?.split(`" k(p=`)[0]));

  const [logitAStr, logitBStr] = logitDiff?.split('__vs__') || [];
  if (!logitAStr || !logitBStr) return { nodes, links };
  const logitANode = nodes.find((d) => d.logitToken === logitAStr);
  const logitBNode = nodes.find((d) => d.logitToken === logitBStr);
  if (!logitANode || !logitBNode) return { nodes, links };

  const virtualId = `virtual-diff-${logitAStr}-vs-${logitBStr}`;
  const diffNode = {
    ...logitANode,
    node_id: virtualId,
    jsNodeId: virtualId,
    feature: parseInt(virtualId, 10), // TODO: check if this is correct - the original code passed it as a string but the type doesn't match
    isJsVirtual: true,
    logitToken: `${logitAStr} - ${logitBStr}`,
    clerp: `Logit diff: ${logitAStr} - ${logitBStr}`,
  };
  nodes.push(diffNode);

  const targetLinks = links.filter((d) => d.target === logitANode.node_id || d.target === logitBNode.node_id);
  // eslint-disable-next-line
  d3.nestBy(targetLinks, (d) => d.source).map((sourceLinks) => {
    const linkA = sourceLinks.find((d) => d.target === logitANode.node_id);
    const linkB = sourceLinks.find((d) => d.target === logitBNode.node_id);

    links.push({
      source: sourceLinks[0].source,
      target: diffNode.node_id,
      weight: (linkA?.weight || 0) - (linkB?.weight || 0),
      isJsVirtual: true,
    });
  });

  return { nodes, links };
}

function layerLocationLabel(layer: string, location: number) {
  if (layer === 'E') return 'Emb';
  if (layer === 'E1') return 'Lgt';
  if (location === -1) return 'logit';

  // TODO: is stream probe_location_idx no longer be saved out?
  // NOTE: For now, location is literally ProbePointLocation
  return `L${layer}`;
}

/* eslint-disable no-param-reassign */
export function formatCLTGraphData(data: CLTGraph, logitDiff: string | null): CLTGraph {
  const { metadata } = data;
  let { nodes, links } = addVirtualDiff(data, logitDiff);

  const pyNodeIdToNode: Record<string, CLTGraphNode> = {};
  const idToNode: Record<string, CLTGraphNode> = {};
  const maxLayer = d3.max(
    nodes.filter((d) => d.feature_type !== 'logit'),
    (d) => +d.layer,
  );
  if (!maxLayer) throw new Error('No layer found');
  nodes.forEach((d, i) => {
    // To make hover state work across prompts, drop ctx from node id

    // we assume we want each occurrence of a feature to be a unique node, so we append ctx_idx to the featureId
    d.featureId = `${d.layer}_${d.feature}_${d.ctx_idx}`;

    d.active_feature_idx = d.feature;
    d.nodeIndex = i;

    if (d.feature_type === 'logit') d.layer = (maxLayer + 1).toString(); // TODO: check - we added typecast

    // TODO: does this handle error nodes correctly?
    // TODO: this comparison is not valid
    // @ts-ignore
    if (d.feature_type === 'unexplored node' && !d.layer !== 'E') {
      d.feature_type = 'cross layer transcoder';
    }

    // count from end to align last token on diff prompts
    d.ctx_from_end = data.metadata.prompt_tokens.length - d.ctx_idx;
    // add clerp to embed and error nodes
    if (d.feature_type.includes('error')) {
      d.isError = true;

      if (!d.featureId.includes('__err_idx_')) d.featureId = `${d.featureId}__err_idx_${d.ctx_from_end}`;

      if (d.feature_type === 'mlp reconstruction error') {
        d.clerp = `Err: mlp " ${data.metadata.prompt_tokens[d.ctx_idx]}"`; // deleted ppToken, it doesn't do anything
      }
    } else if (d.feature_type === 'embedding') {
      d.clerp = `Emb: " ${data.metadata.prompt_tokens[d.ctx_idx]}"`; // deleted ppToken, it doesn't do anything
    }

    d.url = d.vis_link;
    d.isFeature = true;

    d.targetLinks = [];
    d.sourceLinks = [];

    // TODO: switch to featureIndex in graphgen
    d.featureIndex = d.feature;

    // anthropic model subgraphs use jsNodeId as the nodeId
    if (ANTHROPIC_MODELS.has(data.metadata.scan)) {
      d.nodeId = d.jsNodeId;
    } else {
      d.nodeId = d.node_id;
    }
    if (d.feature_type === 'logit' && d.clerp) d.logitPct = +d.clerp.split('(p=')[1].split(')')[0];
    idToNode[d.nodeId] = d;
    pyNodeIdToNode[d.node_id] = d;
  });

  nodes = d3.sort(nodes, (d) => +d.layer);

  links = links.filter((d) => pyNodeIdToNode[d.source] && pyNodeIdToNode[d.target]);

  // connect links to nodes
  links.forEach((link) => {
    link.sourceNode = pyNodeIdToNode[link.source];
    link.targetNode = pyNodeIdToNode[link.target];

    link.linkId = `${link.sourceNode.nodeId}__${link.targetNode.nodeId}`;

    link.sourceNode?.targetLinks?.push(link);
    link.targetNode?.sourceLinks?.push(link);
    link.absWeight = Math.abs(link.weight);
  });
  links = d3.sort(links, (d) => d.absWeight);

  nodes.forEach((d) => {
    d.inputAbsSum = d3.sum(d.sourceLinks || [], (e) => Math.abs(e.weight));
    // @ts-ignore
    // eslint-disable-next-line
    d.sourceLinks?.forEach((e) => (e.pctInput = e.weight / d.inputAbsSum));
    d.inputError = d3.sum(
      // @ts-ignore
      d.sourceLinks?.filter((e) => e.sourceNode.isError),
      (e) => Math.abs(e.weight),
    );
    d.pctInputError = d.inputError / d.inputAbsSum;
  });

  // convert layer/probe_location_idx to a streamIdx used to position nodes
  let byStream = d3.nestBy(nodes, (d) => `${[d.layer, d.probe_location_idx]}`);
  byStream = d3.sort(byStream, (d) => d[0].probe_location_idx);
  byStream = d3.sort(byStream, (d) => (d[0].layer === 'E' ? -1 : +d[0].layer));
  byStream.forEach((stream, streamIdx) => {
    stream.forEach((d) => {
      d.streamIdx = streamIdx;
      // @ts-ignore
      d.layerLocationLabel = layerLocationLabel(d.layer, d.probe_location_idx);

      // @ts-ignore
      // eslint-disable-next-line
      if (!isHideLayer(metadata.scan)) d.streamIdx = isFinite(d.layer) ? +d.layer : 0;
    });
  });

  // add target_logit_effect__ columns for each logit
  const logitNodeMap = new Map(nodes.filter((d) => d.isLogit).map((d) => [d.node_id, d.logitToken]));
  nodes.forEach((node) => {
    node.targetLinks?.forEach((link) => {
      if (!logitNodeMap.has(link.target)) return;
      // @ts-ignore
      node[`target_logit_effect__${logitNodeMap.get(link.target)}`] = link.weight;
    });
  });

  // add ppClerp
  // TODO: this seems to do nothing
  nodes.forEach((d) => {
    if (!d.clerp) d.clerp = '';
    d.remoteClerp = '';
  });

  // condense nodes into features, using last occurence of feature if necessary to point to a node
  const features = d3
    .nestBy(
      nodes.filter((d) => d.isFeature),
      (d) => d.featureId || '',
    )
    .map((d) => ({
      featureId: d[0].featureId,
      feature_type: d[0].feature_type,
      clerp: d[0].clerp,
      remoteClerp: d[0].remoteClerp,
      layer: d[0].layer,
      streamIdx: d[0].streamIdx,
      probe_location_idx: d[0].probe_location_idx,
      featureIndex: d[0].featureIndex,
      top_logit_effects: d[0].top_logit_effects,
      bottom_logit_effects: d[0].bottom_logit_effects,
      top_embedding_effects: d[0].top_embedding_effects,
      bottom_embedding_effects: d[0].bottom_embedding_effects,
      url: d[0].url,
      lastNodeId: d.at(-1)?.nodeId,
      isLogit: d[0].isLogit,
      isError: d[0].isError,
    }));

  // TODO: these don't sense, nodes/features/links are arrays
  // @ts-ignore
  nodes.idToNode = idToNode;
  // @ts-ignore
  features.idToFeature = Object.fromEntries(features.map((d) => [d.featureId, d]));
  // @ts-ignore
  links.idToLink = Object.fromEntries(links.map((d) => [d.linkId, d]));

  Object.assign(data, { nodes, features, links, byStream });
  return data;
}

export function hideTooltip() {
  d3.select('.tooltip').classed('tooltip-hidden', true);
}

export function showTooltip(ev: MouseEvent, d: CLTGraphNode, overrideClerp?: string) {
  const tooltipSel = d3.select('.tooltip');
  const x = ev.clientX;
  const y = ev.clientY;
  // @ts-ignore
  const bb = tooltipSel.node()?.getBoundingClientRect();
  const left = d3.clamp(20, x - bb.width / 2, window.innerWidth - bb.width - 20);
  const top = window.innerHeight > y + 20 + bb.height ? y + 20 : y - bb.height - 20;

  const clerp = overrideClerp || d.ppClerp || `F#${d.feature}`;

  const tooltipHtml = `<div className="text-center flex flex-col items-center justify-center">
  ${clerp}
  ${ev.metaKey ? `<div className="text-slate-400 text-center mt-1 text-[7px]" > Holding CMD/Ctrl: Click to Pin to Subgraph</div>` : ''}
</div>`;
  // if (ev.metaKey) {
  //   tooltipHtml += `<div className="text-slate-400 text-center mt-1 text-[7px]">Holding CMD/Ctrl: Click to Pin to Subgraph</div>`;
  // Object.keys(d)
  // // @ts-ignore
  // .filter((str) => typeof d[str] !== 'object' && typeof d[str] !== 'function' && !keysToSkip.has(str) && d[str])
  // .map((str) => {
  //   // @ts-ignore
  //   let val = d[str];
  //   if (typeof val === 'number' && !Number.isInteger(val)) val = val.toFixed(6);
  //   return `<div>${str}: <b>${val}</b></div>`;
  // })
  // .join('');
  // }

  tooltipSel.style('left', `${left}px`).style('top', `${top}px`).html(tooltipHtml).classed('tooltip-hidden', false);
}

// Helper function to convert feature type to display text
export function featureTypeToText(type: string): string {
  if (type === 'logit') return '■';
  if (type === 'embedding') return '■';
  if (type === 'mlp reconstruction error') return '◆';
  return '●';
}

export function featureTypeToTextSize(isMobile: boolean, type: string): number {
  if (isMobile && type === 'mlp reconstruction error') {
    return 7;
  }
  return 14;
}

export { ATTRIBUTION_GRAPH_SCHEMA };

// filtering utils for influence and density

export function shouldShowNodeForInfluenceThreshold(
  node: CLTGraphNode,
  visState: CltVisState,
  clickedId: string | null,
): boolean {
  // always show embeddings and logits
  if (node.feature_type === 'embedding' || node.feature_type === 'logit') {
    return true;
  }

  // always show pinned nodes
  if (node.nodeId !== undefined && visState.pinnedIds.includes(node.nodeId)) {
    return true;
  }

  // always show clicked nodes
  if (clickedId !== null && node.nodeId === clickedId) {
    return true;
  }

  // if we have influence and pruning threshold, show if influence is less than pruning threshold
  if (
    node.influence !== undefined &&
    node.influence !== null &&
    visState.pruningThreshold !== undefined &&
    visState.pruningThreshold !== null
  ) {
    if (node.influence <= visState.pruningThreshold) {
      return true;
    }
    return false;
  }
  // no influence and pruning threshold. show all.
  return true;
}

export function shouldShowNodeForDensityThreshold(
  isNPDashboard: boolean,
  d: CLTGraphNode,
  visState: CltVisState,
  clickedId: string | null,
): boolean {
  if (!isNPDashboard) {
    return true;
  }

  // always show embeddings and logits
  if (d.feature_type === 'embedding' || d.feature_type === 'logit') {
    return true;
  }

  // always show pinned nodes
  if (d.nodeId !== undefined && visState.pinnedIds.includes(d.nodeId)) {
    return true;
  }

  if (clickedId !== null && d.nodeId === clickedId) {
    return true;
  }

  // show if density threshold is met
  if (
    d.featureDetailNP?.frac_nonzero !== undefined &&
    d.featureDetailNP?.frac_nonzero !== null &&
    visState.densityThreshold !== undefined &&
    visState.densityThreshold !== null
  ) {
    const shouldShow = d.featureDetailNP?.frac_nonzero <= visState.densityThreshold;
    return shouldShow;
  }
  // no density threshold. show all.
  return true;
}

// Extended type for custom CLTGraph properties
export interface CLTGraphExtended extends CLTGraph {
  byStream?: Array<any>;
  features?: Array<any>;
}

export function filterNodes(
  data: CLTGraphExtended,
  nodes: CLTGraphNode[],
  selectedGraph: CLTGraph,
  visState: CltVisState,
  clickedId: string | null,
  hideMlpErrors: boolean = false,
) {
  if (data.metadata.node_threshold !== undefined && data.metadata.node_threshold > 0) {
    nodes = nodes.filter((d) => shouldShowNodeForInfluenceThreshold(d, visState, clickedId));
  }
  // if we have neuronpedia dashboards, then we use density threshold
  nodes = nodes.filter((d) =>
    shouldShowNodeForDensityThreshold(graphModelHasNpDashboards(selectedGraph), d, visState, clickedId),
  );
  if (hideMlpErrors) {
    nodes = nodes.filter((d) => d.feature_type !== 'mlp reconstruction error');
  }
  return nodes;
}

export function clientCheckIsEmbed() {
  return typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('embed') === 'true';
}

// various optimizations for Claude to get less confused. currently: hides MLP reconstruction errors
export function clientCheckClaudeMode() {
  return typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('claudeMode') === 'true';
}

export function parseGraphSupernodes(supernodes?: string): GraphSupernodes {
  if (!supernodes) return [];
  return GraphSupernodesSchema.parse(JSON.parse(supernodes));
}

export function parseGraphClerps(clerps?: string): GraphClerps {
  if (!clerps) return [];
  return GraphClerpsSchema.parse(JSON.parse(clerps));
}

// ********* Replacement & Completeness Scores *********

function normalizeMatrix(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    const absRow = row.map((val) => Math.abs(val));
    const sum = absRow.reduce((acc, val) => acc + val, 0);
    const clampedSum = Math.max(sum, 1e-10);
    return absRow.map((val) => val / clampedSum);
  });
}

function computeInfluence(A: number[][], logitWeights: number[], maxIter: number = 1000): number[] {
  // Normally we calculate total influence B using A + A^2 + ... or (I - A)^-1 - I,
  // and do logit_weights @ B
  // But it's faster / more efficient to compute logit_weights @ A + logit_weights @ A^2
  // as follows:

  // Matrix-vector multiplication: logitWeights @ A
  let currentInfluence = new Array(A[0].length).fill(0);
  for (let j = 0; j < A[0].length; j += 1) {
    for (let i = 0; i < logitWeights.length; i += 1) {
      currentInfluence[j] += logitWeights[i] * A[i][j];
    }
  }

  const influence = [...currentInfluence];
  let iterations = 0;

  while (currentInfluence.some((val) => Math.abs(val) > 1e-10)) {
    if (iterations >= maxIter) {
      throw new Error(`Influence computation failed to converge after ${iterations} iterations`);
    }

    // currentInfluence @ A
    const newInfluence = new Array(A[0].length).fill(0);
    for (let j = 0; j < A[0].length; j += 1) {
      for (let i = 0; i < currentInfluence.length; i += 1) {
        newInfluence[j] += currentInfluence[i] * A[i][j];
      }
    }

    currentInfluence = newInfluence;
    for (let i = 0; i < influence.length; i += 1) {
      influence[i] += currentInfluence[i];
    }
    iterations += 1;
  }

  return influence;
}

function reconstructAdjacencyMatrix(
  nodes: CLTGraphNode[],
  edges: CLTGraphLink[],
): {
  matrix: number[][];
  sortedNodes: CLTGraphNode[];
} {
  // Define the expected order of feature types
  const featureTypeOrder = [
    'cross layer transcoder', // Feature nodes (indices 0 to n_features-1)
    'mlp reconstruction error', // Error nodes
    'embedding', // Token/embedding nodes
    'logit', // Logit nodes (last n_logits positions)
  ];

  // Sort nodes by feature_type order, then by any secondary criteria if needed
  function getSortKey(node: CLTGraphNode) {
    let typePriority;
    try {
      typePriority = featureTypeOrder.indexOf(node.feature_type);
    } catch {
      // Handle any unexpected feature types
      typePriority = featureTypeOrder.length;
    }
    if (typePriority === -1) {
      typePriority = featureTypeOrder.length;
    }

    // Secondary sort criteria
    const layerNum = node.layer === 'E' ? 0 : Number.isNaN(parseInt(node.layer, 10)) ? 999 : parseInt(node.layer, 10);
    const secondary = [layerNum, node.ctx_idx, node.feature || 0];

    return [typePriority, ...secondary];
  }

  // Sort nodes to match the expected adjacency matrix ordering
  const sortedNodes = [...nodes].sort((a, b) => {
    const keyA = getSortKey(a);
    const keyB = getSortKey(b);
    for (let i = 0; i < Math.max(keyA.length, keyB.length); i += 1) {
      const valA = keyA[i] || 0;
      const valB = keyB[i] || 0;
      if (valA !== valB) return valA - valB;
    }
    return 0;
  });

  // Create mapping from node_id to matrix index
  const nodeIdToIdx: Record<string, number> = {};
  sortedNodes.forEach((node, idx) => {
    nodeIdToIdx[node.node_id] = idx;
  });

  const nNodes = sortedNodes.length;

  // Verify the ordering matches expectations
  const featureCounts: Record<string, number> = {};
  featureTypeOrder.forEach((ft) => {
    featureCounts[ft] = 0;
  });
  sortedNodes.forEach((node) => {
    if (node.feature_type in featureCounts) {
      featureCounts[node.feature_type] += 1;
    }
  });

  // console.log(`Node counts by type: ${JSON.stringify(featureCounts)}`);

  // Initialize adjacency matrix
  const adjacencyMatrix: number[][] = Array(nNodes)
    .fill(null)
    .map(() => Array(nNodes).fill(0));

  // Fill in the edges
  edges.forEach((edge) => {
    const srcIdx = nodeIdToIdx[edge.source];
    const dstIdx = nodeIdToIdx[edge.target];
    const { weight } = edge;

    if (srcIdx !== undefined && dstIdx !== undefined) {
      // Convention: adjacency_matrix[dst, src] = weight (edge from src to dst)
      // Rows represent target nodes, columns represent source nodes
      adjacencyMatrix[dstIdx][srcIdx] = weight;
    }
  });

  return { matrix: adjacencyMatrix, sortedNodes };
}

export function computeGraphScoresFromGraphData(
  graphData: CLTGraph,
  pinnedIds: string[] = [],
): {
  replacementScore: number;
  completenessScore: number;
} {
  let graphNodesToUse = graphData.nodes;
  // if we have pinned IDs, then we need to filter the features to only include the pinned IDs, and the mlp errors to only include ones connected to pinned IDs
  if (pinnedIds.length > 0) {
    // get all the pinned nodes, we'll use this to find the mlp errors to keep
    const pinnedNodes = graphData.nodes.filter((node) => pinnedIds.includes(node.node_id));
    // make filteredGraphNodes
    const filteredGraphNodes: CLTGraphNode[] = [];
    // iterate through all nodes
    for (const node of graphData.nodes) {
      // if it's pinnedId and it's a cross layer transcoder, then add it to filteredGraphNodes
      if (node.feature_type === 'cross layer transcoder') {
        if (pinnedIds.includes(node.node_id)) {
          filteredGraphNodes.push(node);
        }
      }
      // if it's a mlp error and connected to a pinned ID, then add it to filteredGraphNodes
      else if (node.feature_type === 'mlp reconstruction error') {
        // check each pinned node
        for (const pinnedNode of pinnedNodes) {
          // check in the graphData.links if there's a link from the pinned node to the mlp error
          const link = graphData.links.find(
            (l) =>
              (l.source === node.node_id && l.target === pinnedNode.node_id) ||
              (l.source === pinnedNode.node_id && l.target === node.node_id),
          );
          if (link) {
            filteredGraphNodes.push(node);
            break;
          }
        }
      } else {
        // the remaining are embed and logits, add them
        filteredGraphNodes.push(node);
      }
    }
    graphNodesToUse = filteredGraphNodes;
    // if the graphNodes don't have any cross layer transcoders, then return 0, 0
    if (graphNodesToUse.filter((node) => node.feature_type === 'cross layer transcoder').length === 0) {
      return { replacementScore: 0, completenessScore: 0 };
    }
  }

  // Get the adjacency matrix
  const { matrix: adjacencyMatrix, sortedNodes } = reconstructAdjacencyMatrix(graphNodesToUse, graphData.links);

  // Filter by feature_type "logit" in nodes and get its "token_prob" to make logit_probabilities
  const logitProbabilities = graphNodesToUse
    .filter((node) => node.feature_type === 'logit')
    .map((node) => node.token_prob);
  const nLogits = logitProbabilities.length;

  // Filter by feature_type "embedding" in nodes to get n_tokens
  const nTokens = graphNodesToUse.filter((node) => node.feature_type === 'embedding').length;

  // Filter by feature_type "cross layer transcoder" in nodes to get n_features
  const nFeatures = graphNodesToUse.filter((node) => node.feature_type === 'cross layer transcoder').length;

  const errorStart = nFeatures;

  // error_end is the end of the error nodes
  // find the index where the feature_type is "embedding"
  const embeddingIndex = sortedNodes.findIndex((node) => node.feature_type === 'embedding');
  const errorEnd = embeddingIndex !== -1 ? embeddingIndex : errorStart;
  const tokenEnd = errorEnd + nTokens;

  const logitWeights = new Array(adjacencyMatrix.length).fill(0);
  for (let i = 0; i < nLogits; i += 1) {
    logitWeights[adjacencyMatrix.length - nLogits + i] = logitProbabilities[i];
  }

  const normalizedMatrix = normalizeMatrix(adjacencyMatrix);
  const nodeInfluence = computeInfluence(normalizedMatrix, logitWeights);

  const tokenInfluence = nodeInfluence.slice(errorEnd, tokenEnd).reduce((sum, val) => sum + val, 0);
  const errorInfluence = nodeInfluence.slice(errorStart, errorEnd).reduce((sum, val) => sum + val, 0);
  const replacementScore = tokenInfluence / (tokenInfluence + errorInfluence);

  const nonErrorFractions = normalizedMatrix.map(
    (row) => 1 - row.slice(errorStart, errorEnd).reduce((sum, val) => sum + val, 0),
  );

  const outputInfluence = nodeInfluence.map((val, i) => val + logitWeights[i]);
  const completenessScore =
    nonErrorFractions.map((fraction, i) => fraction * outputInfluence[i]).reduce((sum, val) => sum + val, 0) /
    outputInfluence.reduce((sum, val) => sum + val, 0);

  return {
    replacementScore: Number.isNaN(replacementScore) ? 0 : replacementScore,
    completenessScore: Number.isNaN(completenessScore) ? 0 : completenessScore,
  };
}
