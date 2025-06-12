import ATTRIBUTION_GRAPH_SCHEMA from '@/app/api/graph/graph-schema.json';
import { DEFAULT_CREATOR_USER_ID, NEXT_PUBLIC_URL } from '@/lib/env';
import { GraphMetadata, GraphMetadataWithPartialRelations, NeuronWithPartialRelations } from '@/prisma/generated/zod';
import cuid from 'cuid';
import { z } from 'zod';
import d3 from './d3-jetpack';

// TODO: make this an env variable
export const NP_GRAPH_BUCKET = 'neuronpedia-attrib';

export const ANT_BUCKET_URL = 'https://transformer-circuits.pub/2025/attribution-graphs';

export const MAX_GRAPH_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024;

// ============ Neuronpedia Specific =============

// TODO: make this a DB column
// models not in this list can only get FeatureDetails from the bucket
export const MODEL_WITH_NP_DASHBOARDS_NOT_YET_CANTOR = new Set(['gemma-2-2b']);
// TODO: remove the MODEL_WITH_NP_DASHBOARDS_NOT_YET_CANTOR once fellows graph gets on cantor
export const graphModelHasNpDashboards = (graph: CLTGraph) =>
  MODEL_WITH_NP_DASHBOARDS_NOT_YET_CANTOR.has(graph.metadata.scan) ||
  graph.metadata.feature_details?.neuronpedia_source_set !== undefined;

// has dashboards in the bucket
export const MODEL_HAS_S3_DASHBOARDS = new Set([
  'llama-3.2-1b',
  'llama-3-131k-relu',
  'jackl-circuits-runs-1-4-sofa-v3_0',
  'jackl-circuits-runs-1-1-druid-cp_0',
  'jackl-circuits-runs-12-19-valet-m_0',
  'jackl-circuits-runs-1-12-rune-cp3_0',
]);

export const ANT_MODELS_TO_LOAD = new Set(['jackl-circuits-runs-1-4-sofa-v3_0']);

// if neither, then no dashboards yet for them

export const MODEL_DO_NOT_FILTER_NODES = new Set(['gelu-4l-x128k64-v0']);

// TODO: this should be by model and source, not just model
// we use this to figure out the scheme for the feature IDs - how many digits is the layer vs feature id
export const MODEL_DIGITS_IN_FEATURE_ID = {
  'gemma-2-2b': Number(16384).toString().length,
  'llama-3-131k-relu': Number(131072).toString().length,
};

export const MODEL_TO_SOURCESET_ID = {
  'gemma-2-2b': 'gemmascope-transcoder-16k',
  'llama-3-131k-relu': 'skip-transcoder-mntss',
};

export const ANT_MODEL_ID_TO_NEURONPEDIA_MODEL_ID = {
  'gemma-2-2b': 'gemma-2-2b',
  'llama-3-131k-relu': 'llama-3.2-1b',
};

export const ERROR_MODEL_DOES_NOT_EXIST = 'ERR_MODEL_DOES_NOT_EXIST';

// ============ End of Neuronpedia Specific =============

export function getLayerFromAnthropicFeatureId(modelId: keyof typeof MODEL_DIGITS_IN_FEATURE_ID, featureId: number) {
  // remove dash and everything after it
  const layer = featureId.toString().replace(/-.*$/, '');
  // the layer is the number before the last digitsInNumFeatures digits
  const layerStr = layer.slice(0, -MODEL_DIGITS_IN_FEATURE_ID[modelId]);
  if (layerStr.length === 0) {
    return 0;
  }
  return parseInt(layerStr, 10);
}

export function getIndexFromAnthropicFeatureId(modelId: keyof typeof MODEL_DIGITS_IN_FEATURE_ID, featureId: number) {
  // remove dash and everything before it
  const index = featureId.toString().replace(/-.*$/, '');
  // the index is the last digitsInNumFeatures digits
  const indexStr = index.slice(-MODEL_DIGITS_IN_FEATURE_ID[modelId]);
  if (indexStr.length === 0) {
    return 0;
  }
  return parseInt(indexStr, 10);
}

export function convertAnthropicFeatureIdToNeuronpediaSourceSet(
  modelId: keyof typeof MODEL_DIGITS_IN_FEATURE_ID,
  featureId: number,
) {
  return `${getLayerFromAnthropicFeatureId(modelId, featureId)}-${MODEL_TO_SOURCESET_ID[modelId]}`;
}

export const GRAPH_BASE_URL_TO_NAME = {
  'https://transformer-circuits.pub/2025/attribution-graphs': 'Ameisen et al.',
  'https://d1fk9w8oratjix.cloudfront.net': 'Piotrowski & Hanna',
};

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
  return `${NEXT_PUBLIC_URL}${makeGraphPublicAccessGraphUri(modelId, slug)}`;
}

export type AnthropicGraphMetadata = {
  slug: string;
  scan: string;
  prompt_tokens: string[];
  prompt: string;
  title_prefix: string;
};

export type ModelToGraphMetadatasMap = {
  [modelId: string]: GraphMetadataWithPartialRelations[];
};

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
    slug: graph.slug,
    promptTokens: graph.prompt_tokens,
    prompt: graph.prompt,
    titlePrefix: graph.title_prefix,
    url: `${baseUrl}/graph_data/${graph.slug}.json`,
    userId: DEFAULT_CREATOR_USER_ID,
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

export type CltSubgraphState = {
  sticky: boolean;
  dagrefy: boolean;
  supernodes: string[][];
  activeGrouping: {
    isActive: boolean;
    selectedNodeIds: Set<string>;
  };
};

// https://github.com/anthropics/attribution-graphs-frontend/blob/main/attribution_graph/init-cg.js
export type CltVisState = {
  pinnedIds: string[];
  hiddenIds: string[];
  hoveredNodeId: string | null;
  hoveredCtxIdx: number | null;
  clickedCtxIdx: number | null;
  linkType: string;
  isShowAllLinks: string;
  isSyncEnabled: string;
  subgraph: CltSubgraphState | null;
  isEditMode: number;
  isHideLayer: boolean;
  sg_pos: string;
  isModal: boolean;
  isGridsnap: boolean;
  supernodes: string[][]; // this is from qParams

  og_sg_pos?: string;

  clerps: string[][];

  pruningThreshold?: number;

  // only for neuronpedia dashboards
  densityThreshold?: number;
};

export const modelIdToModelDisplayName = new Map<string, string>([
  ['jackl-circuits-runs-1-4-sofa-v3_0', 'Haiku'],
  ['jackl-circuits-runs-1-1-druid-cp_0', '18L'],
  ['jackl-circuits-runs-12-19-valet-m_0', 'Model Organism'],
  ['jackl-circuits-runs-1-12-rune-cp3_0', '18L PLTs'],
  ['llama-3-131k-relu', 'Llama 3.2 1B - Relu'],
]);

export const anthropicModels = [
  'jackl-circuits-runs-1-4-sofa-v3_0',
  'jackl-circuits-runs-1-1-druid-cp_0',
  'jackl-circuits-runs-12-19-valet-m_0',
  'jackl-circuits-runs-1-12-rune-cp3_0',
];

export const cltModelToNumLayers = {
  'jackl-circuits-runs-1-4-sofa-v3_0': 18,
  'jackl-circuits-runs-1-1-druid-cp_0': 18,
  'jackl-circuits-runs-12-19-valet-m_0': 16,
  'jackl-circuits-runs-1-12-rune-cp3_0': 18,
  'llama-3-131k-relu': 16,
};

export type CLTGraphInnerMetadata = {
  slug: string;
  scan: string;
  prompt_tokens: string[];
  prompt: string;

  // dynamic pruning - mntss/hanna
  // default value for cltVisState.pruningThreshold
  // filters out > node.influence values
  node_threshold?: number;

  // add the extra metadata from graph-schema.json
  feature_details?: {
    feature_json_base_url?: string;
    neuronpedia_source_set?: string;
  };
  info?: {
    description?: string;
    creator_name?: string;
    creator_url?: string;
    source_urls?: string[];
    generator?: {
      name?: string;
      version?: string;
      url?: string;
      email?: string;
    };
    create_time_ms?: number;
  };
  generation_settings?: {
    max_n_logits?: number;
    desired_logit_prob?: number;
    batch_size?: number;
    max_feature_nodes?: number;
  };
  pruning_settings?: {
    node_threshold?: number;
    edge_threshold?: number;
  };

  // we add these ourselves
  // subset of our DB Model
  neuronpedia_internal_model?: {
    id: string;
    displayName: string;
    layers: number;
  };
};

export type CLTGraphQParams = {
  linkType: string;
  pinnedIds: string[];
  clickedId: string;
  supernodes: string[][];
  sg_pos: string;
};

export type CLTGraphNode = {
  node_id: string;
  feature: number;
  layer: string;
  ctx_idx: number;
  feature_type: string;
  token_prob: number;
  is_target_logit: boolean;
  run_idx: number;
  reverse_ctx_idx: number;
  jsNodeId: string;
  clerp: string;

  // feature details
  featureDetail?: AnthropicFeatureDetail;
  featureDetailNP?: NeuronWithPartialRelations;

  // following ones are added after formatData
  active_feature_idx?: number;
  ctx_from_end?: number;
  featureId?: string;
  featureIndex?: number;
  idToNode?: Record<string, CLTGraphNode>;
  inputAbsSum?: number;
  inputError?: number;
  isError?: boolean;
  isFeature?: boolean;
  isJsVirtual?: boolean;
  isLogit?: boolean;
  layerLocationLabel?: string;
  localClerp?: string;
  logitToken?: string;
  logitPct?: number;
  nodeColor?: string;
  nodeId?: string; // TODO: check - why is this value different from node[underscore]id?
  nodeIndex?: number;
  pctInputError?: number;
  pos?: number[];
  ppClerp?: string;
  probe_location_idx?: number;
  remoteClerp?: string;
  sourceLinks?: CLTGraphLink[];
  streamIdx?: number;
  supernodeId?: string;
  targetLinks?: CLTGraphLink[];
  tmpClickedLink?: CLTGraphLink;
  tmpClickedSourceLink?: CLTGraphLink;
  tmpClickedTargetLink?: CLTGraphLink;
  top_logit_effects?: Record<string, number>;
  bottom_logit_effects?: Record<string, number>;
  top_embedding_effects?: Record<string, number>;
  bottom_embedding_effects?: Record<string, number>;
  url?: string;
  vis_link?: string;
  xOffset?: number;
  yOffset?: number;

  memberNodes?: CLTGraphNode[];
  memberSet?: Set<string>;

  // Added for subgraph visualization
  inputAbsSumExternalSn?: number;
  sgSnInputWeighting?: number;
  isSuperNode?: boolean;
  memberNodeIds?: string[];
  textHeight?: number;
  tmpClickedSgSource?: CLTGraphLink;
  tmpClickedSgTarget?: CLTGraphLink;

  // added for dynamic pruning
  influence?: number;

  // for fellows graphs only
  activation?: number;

  // test hover links
  tmpHoveredLink?: CLTGraphLink;
  tmpHoveredSourceLink?: CLTGraphLink;
  tmpHoveredTargetLink?: CLTGraphLink;
};

export type CLTGraphLink = {
  source: string;
  target: string;
  weight: number;

  // following are after formatData
  absWeight?: number;
  color?: string;
  isJsVirtual?: boolean;
  linkId?: string;
  pctInput?: number;
  pctInputColor?: string;
  sourceNode?: CLTGraphNode;
  strokeWidth?: number;
  targetNode?: CLTGraphNode;

  tmpClickedCtxOffset?: number;
  tmpColor?: string;

  tmpHoveredCtxOffset?: number;
};

export type CLTGraph = {
  metadata: CLTGraphInnerMetadata;
  qParams: CLTGraphQParams;
  nodes: CLTGraphNode[];
  links: CLTGraphLink[];
};

export enum FilterGraphType {
  Featured = 'featured',
  Community = 'community',
  Mine = 'mine',
}

export function isHideLayer(scan: string) {
  return scan === 'jackl-circuits-runs-1-4-sofa-v3_0';
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
    if (anthropicModels.includes(data.metadata.scan)) {
      d.nodeId = d.jsNodeId;
    } else {
      d.nodeId = d.node_id;
    }
    if (d.feature_type === 'logit' && d.clerp) d.logitPct = +d.clerp.split('(p=')[1].split(')')[0];
    idToNode[d.nodeId] = d;
    pyNodeIdToNode[d.node_id] = d;
  });

  // // delete features that occur in than 2/3 of tokens
  // // SPECIAL CASE: for eleuther graphs don't do the filtering out
  // if (!MODEL_DO_NOT_FILTER_NODES.has(metadata.scan)) {
  //   const deletedFeatures: CLTGraphNode[][] = [];
  //   const byFeatureId = d3.nestBy(nodes, (d) => d.featureId || '');
  //   byFeatureId.forEach((feature) => {
  //     if (feature.length > (metadata.prompt_tokens.length * 2) / 3) {
  //       deletedFeatures.push(feature);
  //       feature.forEach((d) => {
  //         if (d.nodeId) delete idToNode[d.nodeId];
  //         if (d.node_id) delete pyNodeIdToNode[d.node_id];
  //       });
  //     }
  //   });
  //   //   if (deletedFeatures.length) console.log({ deletedFeatures });

  //   nodes = nodes.filter((d) => (d.nodeId ? idToNode[d.nodeId] : false));
  // }

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

export type AnthropicFeatureExample = {
  'ha-haiku35_resampled'?: boolean;
  is_repeated_datapoint: boolean;
  train_token_ind: number;
  tokens: string[];
  tokens_acts_list: number[];
};

export type AnthropicFeatureExampleQuantile = {
  examples: AnthropicFeatureExample[];
  quantile_name: string;
};

export type AnthropicFeatureDetail = {
  bottom_logits: string[];
  top_logits: string[];
  index: number;
  examples_quantiles: AnthropicFeatureExampleQuantile[];
};

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
) {
  if (data.metadata.node_threshold !== undefined && data.metadata.node_threshold > 0) {
    nodes = nodes.filter((d) => shouldShowNodeForInfluenceThreshold(d, visState, clickedId));
  }
  // if we have neuronpedia dashboards, then we use density threshold
  nodes = nodes.filter((d) =>
    shouldShowNodeForDensityThreshold(graphModelHasNpDashboards(selectedGraph), d, visState, clickedId),
  );
  return nodes;
}

export function clientCheckIsEmbed() {
  return typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('embed') === 'true';
}

// various optimizations for Claude to get less confused. currently: hides MLP reconstruction errors
export function clientCheckClaudeMode() {
  return typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('claudeMode') === 'true';
}

const GraphSupernodesSchema = z.array(z.array(z.string()));
const GraphClerpsSchema = z.array(z.array(z.string()));

export type GraphSupernodes = z.infer<typeof GraphSupernodesSchema>;
export type GraphClerps = z.infer<typeof GraphClerpsSchema>;

export function parseGraphSupernodes(supernodes?: string): GraphSupernodes {
  if (!supernodes) return [];
  return GraphSupernodesSchema.parse(JSON.parse(supernodes));
}

export function parseGraphClerps(clerps?: string): GraphClerps {
  if (!clerps) return [];
  return GraphClerpsSchema.parse(JSON.parse(clerps));
}

export const SaveSubgraphRequestSchema = z.object({
  modelId: z.string(),
  slug: z.string(),
  displayName: z.string().optional(),
  pinnedIds: z.array(z.string()),
  supernodes: GraphSupernodesSchema,
  clerps: GraphClerpsSchema,
  pruningThreshold: z.number().nullable(),
  densityThreshold: z.number().nullable(),
  overwriteId: z.string().optional(),
});

export type SaveSubgraphRequest = z.infer<typeof SaveSubgraphRequestSchema>;

export const DeleteSubgraphRequestSchema = z.object({
  subgraphId: z.string(),
});
