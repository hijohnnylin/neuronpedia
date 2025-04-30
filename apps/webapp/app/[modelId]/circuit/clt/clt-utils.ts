import d3 from './d3-jetpack';

export const CLT_BASE_URLS = [
  'https://transformer-circuits.pub/2025/attribution-graphs',
  'https://d1fk9w8oratjix.cloudfront.net',
];

export type CLTMetadataGraph = {
  slug: string;
  scan: string;
  prompt_tokens: string[];
  prompt: string;
  title_prefix: string;
};

export type ModelToCLTMetadataGraphsMap = {
  [scanId: string]: CLTMetadataGraph[];
};

export function makeCltFetchUrl(baseUrl: string, path: string): string {
  return `${baseUrl}/${path}`;
}

export function nodeHasFeatureDetail(node: CLTGraphNode): boolean {
  return (
    node.feature_type !== 'embedding' &&
    node.feature_type !== 'mlp reconstruction error' &&
    node.feature_type !== 'logit'
  );
}

// no proxy needed since this is server to server
export async function getCLTMetadata(baseUrl: string): Promise<ModelToCLTMetadataGraphsMap> {
  const fetchUrl = makeCltFetchUrl(baseUrl, 'data/graph-metadata.json');
  const response = await fetch(fetchUrl);
  const data: { graphs: CLTMetadataGraph[] } = await response.json();

  // break up the graphs by scan (model)
  const graphsByScan = data.graphs.reduce((acc, graph) => {
    acc[graph.scan] = [...(acc[graph.scan] || []), graph];
    return acc;
  }, {} as ModelToCLTMetadataGraphsMap);

  return graphsByScan;
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
  hoveredId: string | null;
  hoveredNodeId: string | null;
  hoveredCtxIdx: number | null;
  clickedId: string | null;
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
};

export const metadataScanToModelDisplayName = new Map<string, string>([
  ['jackl-circuits-runs-1-4-sofa-v3_0', 'Haiku'],
  ['jackl-circuits-runs-1-1-druid-cp_0', '18L'],
  ['jackl-circuits-runs-12-19-valet-m_0', 'Model Organism'],
  ['jackl-circuits-runs-1-12-rune-cp3_0', '18L PLTs'],
  ['gemma-2-2b', 'Gemma 2 2B'],
  ['llama-3-131k-relu', 'Llama 3.2 1B - Relu'],
  // ['llama-hf-3-nobos', 'Llama 3.2 1B - NoBos'],
  // ['llama-hf-3', 'Llama 3.2 1B - Other'],
]);

export const metadataScansToDisplay = new Set([
  'jackl-circuits-runs-1-4-sofa-v3_0',
  'jackl-circuits-runs-1-1-druid-cp_0',
  'jackl-circuits-runs-12-19-valet-m_0',
  'jackl-circuits-runs-1-12-rune-cp3_0',
  'gemma-2-2b',
  'llama-3-131k-relu',
]);

export const scanSlugToName = {
  h35: 'jackl-circuits-runs-1-4-sofa-v3_0',
  '18l': 'jackl-circuits-runs-1-1-druid-cp_0',
  moc: 'jackl-circuits-runs-12-19-valet-m_0',
};

export const cltModelToLayers = {
  'jackl-circuits-runs-1-4-sofa-v3_0': 18,
  'jackl-circuits-runs-1-1-druid-cp_0': 18,
  'jackl-circuits-runs-12-19-valet-m_0': 16,
  'jackl-circuits-runs-1-12-rune-cp3_0': 18,
  'gemma-2-2b': 26,
  'llama-3-131k-relu': 16,
};

export type CLTGraphMetadata = {
  slug: string;
  scan: string;
  prompt_tokens: string[];
  prompt: string;
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
  featureDetail?: CLTFeature;

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
};

export type CLTGraph = {
  metadata: CLTGraphMetadata;
  qParams: CLTGraphQParams;
  nodes: CLTGraphNode[];
  links: CLTGraphLink[];
};

export function isHideLayer(scan: string) {
  return scan === scanSlugToName.h35 || scan === scanSlugToName.moc;
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
    d.featureId = `${d.layer}_${d.feature}`;

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

    d.nodeId = d.jsNodeId;
    if (d.feature_type === 'logit' && d.clerp) d.logitPct = +d.clerp.split('(p=')[1].split(')')[0];
    idToNode[d.nodeId] = d;
    pyNodeIdToNode[d.node_id] = d;
  });

  // delete features that occur in than 2/3 of tokens
  // TODO: more principled way of filtering them out — maybe by feature density?
  const deletedFeatures: CLTGraphNode[][] = [];
  const byFeatureId = d3.nestBy(nodes, (d) => d.featureId || '');
  byFeatureId.forEach((feature) => {
    if (feature.length > (metadata.prompt_tokens.length * 2) / 3) {
      deletedFeatures.push(feature);
      feature.forEach((d) => {
        if (d.nodeId) delete idToNode[d.nodeId];
        if (d.node_id) delete pyNodeIdToNode[d.node_id];
      });
    }
  });
  //   if (deletedFeatures.length) console.log({ deletedFeatures });

  nodes = nodes.filter((d) => (d.nodeId ? idToNode[d.nodeId] : false));
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

const keysToSkip = new Set([
  'node_id',
  'jsNodeId',
  'nodeId',
  'layerLocationLabel',
  'remoteClerp',
  'localClerp',
  'tmpClickedTargetLink',
  'tmpClickedLink',
  'tmpClickedSourceLink',
  'pos',
  'xOffset',
  'yOffset',
  'sourceLinks',
  'targetLinks',
  'url',
  'vis_link',
  'run_idx',
  'featureId',
  'active_feature_idx',
  'nodeIndex',
  'isFeature',
  'Distribution',
  'clerp',
  'ppClerp',
  'is_target_logit',
  'token_prob',
  'reverse_ctx_idx',
  'ctx_from_end',
  'feature',
  'logitToken',
  'featureIndex',
  'streamIdx',
  'nodeColor',
  'umap_enc_x',
  'umap_enc_y',
  'umap_dec_x',
  'umap_dec_y',
  'umap_concat_x',
  'umap_concat_y',
]);

export function showTooltip(ev: MouseEvent, d: CLTGraphNode) {
  const tooltipSel = d3.select('.tooltip');
  const x = ev.clientX;
  const y = ev.clientY;
  // @ts-ignore
  const bb = tooltipSel.node()?.getBoundingClientRect();
  const left = d3.clamp(20, x - bb.width / 2, window.innerWidth - bb.width - 20);
  const top = window.innerHeight > y + 20 + bb.height ? y + 20 : y - bb.height - 20;

  const tooltipHtml = !ev.metaKey
    ? d.ppClerp || `F#${d.feature}`
    : Object.keys(d)
        // @ts-ignore
        .filter((str) => typeof d[str] !== 'object' && typeof d[str] !== 'function' && !keysToSkip.has(str))
        .map((str) => {
          // @ts-ignore
          let val = d[str];
          if (typeof val === 'number' && !Number.isInteger(val)) val = val.toFixed(6);
          return `<div>${str}: <b>${val}</b></div>`;
        })
        .join('');

  tooltipSel.style('left', `${left}px`).style('top', `${top}px`).html(tooltipHtml).classed('tooltip-hidden', false);
}

// Helper function to convert feature type to display text
export function featureTypeToText(type: string): string {
  if (type === 'logit') return '■';
  if (type === 'embedding') return '■';
  if (type === 'mlp reconstruction error') return '◆';
  return '●';
}

export type CLTFeatureExample = {
  'ha-haiku35_resampled'?: boolean;
  is_repeated_datapoint: boolean;
  train_token_ind: number;
  tokens: string[];
  tokens_acts_list: number[];
};

export type CLTFeatureExampleQuantile = {
  examples: CLTFeatureExample[];
  quantile_name: string;
};

export type CLTFeature = {
  bottom_logits: string[];
  top_logits: string[];
  index: number;
  examples_quantiles: CLTFeatureExampleQuantile[];
};
