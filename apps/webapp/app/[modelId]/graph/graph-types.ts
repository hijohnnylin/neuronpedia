import { GraphMetadataWithPartialRelations, NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { z } from 'zod';

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
    neuronpedia_lorsa_source_set?: string;
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
    neuronpedia_link?: string;
    neuronpedia_source_set?: string;
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

  // we calculate this on the frontend
  replacement_score?: number;
  completeness_score?: number;

  // determines cantor or not for feature ID
  schema_version?: number;

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

  // QK tracing results (only present on lorsa feature nodes)
  qk_tracing_results?: {
    pair_wise_contributors: [string, string, number][];
    top_q_marginal_contributors: [string, number][];
    top_k_marginal_contributors: [string, number][];
  };
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
export const GraphSupernodesSchema = z.array(z.array(z.string()));
export const GraphClerpsSchema = z.array(z.array(z.string()));

export type GraphSupernodes = z.infer<typeof GraphSupernodesSchema>;
export type GraphClerps = z.infer<typeof GraphClerpsSchema>;
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
