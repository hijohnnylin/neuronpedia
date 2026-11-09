import type { CLTGraphLink, CLTGraphNode } from './graph-types';

export const MAX_SUPERNODE_CANDIDATES = 200;
export const GROUPABLE_FEATURE_TYPES: ReadonlySet<string> = new Set(['cross layer transcoder', 'lorsa']);

export type SupernodeProfileMode = 'both' | 'incoming' | 'outgoing';

export type SupernodeSuggestionConfig = {
  profileMode: SupernodeProfileMode;
  similarityThreshold: number;
  minSharedNeighbors: number;
  maxGroupSize: number;
  sameFeatureTypeOnly: boolean;
};

export const DEFAULT_SUPERNODE_SUGGESTION_CONFIG: SupernodeSuggestionConfig = {
  profileMode: 'both',
  similarityThreshold: 0.75,
  minSharedNeighbors: 2,
  maxGroupSize: 8,
  sameFeatureTypeOnly: true,
};

export type SupernodeSuggestion = {
  id: string;
  label: string;
  memberNodeIds: string[];
  minCohesion: number;
  meanCohesion: number;
  minSharedIncomingNeighbors: number;
  minSharedOutgoingNeighbors: number;
  featureTypes: string[];
  layers: string[];
  ctxIdxRange: [number, number];
};

type SuggestionGraph = {
  nodes: readonly CLTGraphNode[];
  links: readonly CLTGraphLink[];
};

type SparseProfile = Map<string, number>;

type NodeProfiles = {
  incoming: SparseProfile;
  outgoing: SparseProfile;
};

type Candidate = {
  id: string;
  node: CLTGraphNode;
  profiles: NodeProfiles;
};

type PairEvidence = {
  eligible: boolean;
  score: number;
  sharedIncomingNeighbors: number;
  sharedOutgoingNeighbors: number;
  selectedSharedNeighbors: number;
};

type ScoredPair = PairEvidence & {
  leftId: string;
  rightId: string;
};

export function suggestSupernodes(
  graph: SuggestionGraph,
  pinnedNodeIds: readonly string[],
  existingSupernodes: readonly (readonly string[])[] = [],
  configOverrides: Partial<SupernodeSuggestionConfig> = {},
): SupernodeSuggestion[] {
  const config = normalizeConfig(configOverrides);
  const { displayedIdByAlias, nodeByDisplayedId } = buildNodeIdentityIndex(graph.nodes);
  const existingMemberIds = new Set(existingSupernodes.flatMap((group) => group.slice(1)));
  const candidateIds = [...new Set(pinnedNodeIds.slice(0, MAX_SUPERNODE_CANDIDATES))]
    .filter((id) => {
      const node = nodeByDisplayedId.get(id);
      return node && GROUPABLE_FEATURE_TYPES.has(node.feature_type) && !existingMemberIds.has(id);
    })
    .sort(compareStrings);
  const profilesByNodeId = buildProfiles(graph.links, candidateIds, displayedIdByAlias);
  const candidates = candidateIds
    .map((id) => {
      const node = nodeByDisplayedId.get(id);
      const profiles = profilesByNodeId.get(id);
      return node && profiles && (profiles.incoming.size > 0 || profiles.outgoing.size > 0)
        ? { id, node, profiles }
        : null;
    })
    .filter((candidate): candidate is Candidate => candidate !== null);

  if (candidates.length < 2) return [];

  const evidence = buildPairEvidence(candidates, config);
  const clusters = clusterCandidates(candidates, evidence, config);
  const candidateById = new Map(candidates.map((candidate) => [candidate.id, candidate]));

  const suggestions = clusters.map((memberNodeIds) => summarizeCluster(memberNodeIds, candidateById, evidence));
  suggestions.sort(
    (left, right) =>
      right.minCohesion - left.minCohesion ||
      compareStrings(left.memberNodeIds.join('|'), right.memberNodeIds.join('|')),
  );

  return suggestions.map((suggestion, index) => ({ ...suggestion, label: `Suggested group ${index + 1}` }));
}

export function mergeAcceptedSupernodeSuggestions({
  existingSupernodes,
  suggestions,
  acceptedSuggestionIds,
  pinnedNodeIds,
}: {
  existingSupernodes: readonly (readonly string[])[];
  suggestions: readonly SupernodeSuggestion[];
  acceptedSuggestionIds: ReadonlySet<string>;
  pinnedNodeIds: readonly string[];
}): string[][] {
  const merged = existingSupernodes.map((group) => [...group]);
  const usedNodeIds = new Set(existingSupernodes.flatMap((group) => group.slice(1)));
  const pinnedSet = new Set(pinnedNodeIds.slice(0, MAX_SUPERNODE_CANDIDATES));

  suggestions.forEach((suggestion) => {
    if (!acceptedSuggestionIds.has(suggestion.id)) return;

    const availableMembers = [...new Set(suggestion.memberNodeIds)].filter(
      (nodeId) => pinnedSet.has(nodeId) && !usedNodeIds.has(nodeId),
    );
    if (availableMembers.length < 2) return;

    merged.push([suggestion.label, ...availableMembers]);
    availableMembers.forEach((nodeId) => usedNodeIds.add(nodeId));
  });

  return merged;
}

function normalizeConfig(overrides: Partial<SupernodeSuggestionConfig>): SupernodeSuggestionConfig {
  const config = { ...DEFAULT_SUPERNODE_SUGGESTION_CONFIG, ...overrides };
  return {
    ...config,
    similarityThreshold: clamp(finiteOr(config.similarityThreshold, 0.75), -1, 1),
    minSharedNeighbors: Math.max(0, Math.floor(finiteOr(config.minSharedNeighbors, 2))),
    maxGroupSize: clamp(Math.floor(finiteOr(config.maxGroupSize, 8)), 2, MAX_SUPERNODE_CANDIDATES),
  };
}

function buildNodeIdentityIndex(nodes: readonly CLTGraphNode[]) {
  const displayedIdByAlias = new Map<string, string>();
  const nodeByDisplayedId = new Map<string, CLTGraphNode>();

  nodes.forEach((node) => {
    const displayedId = node.nodeId || node.node_id || node.jsNodeId;
    if (!displayedId) return;
    nodeByDisplayedId.set(displayedId, node);
    [node.node_id, node.nodeId, node.jsNodeId].forEach((alias) => {
      if (alias) displayedIdByAlias.set(alias, displayedId);
    });
  });

  return { displayedIdByAlias, nodeByDisplayedId };
}

function buildProfiles(
  links: readonly CLTGraphLink[],
  candidateIds: readonly string[],
  displayedIdByAlias: ReadonlyMap<string, string>,
): Map<string, NodeProfiles> {
  const profilesByNodeId = new Map(
    candidateIds.map((id) => [id, { incoming: new Map<string, number>(), outgoing: new Map<string, number>() }]),
  );

  links.forEach((link) => {
    if (!Number.isFinite(link.weight) || link.weight === 0) return;
    const sourceId = resolveEndpointId(link.sourceNode, link.source, displayedIdByAlias);
    const targetId = resolveEndpointId(link.targetNode, link.target, displayedIdByAlias);
    if (!sourceId || !targetId || sourceId === targetId) return;

    const sourceProfiles = profilesByNodeId.get(sourceId);
    const targetProfiles = profilesByNodeId.get(targetId);
    if (sourceProfiles) addProfileWeight(sourceProfiles.outgoing, targetId, link.weight);
    if (targetProfiles) addProfileWeight(targetProfiles.incoming, sourceId, link.weight);
  });

  return profilesByNodeId;
}

function resolveEndpointId(
  attachedNode: CLTGraphNode | undefined,
  rawId: string,
  displayedIdByAlias: ReadonlyMap<string, string>,
): string | undefined {
  const attachedId = attachedNode && (attachedNode.nodeId || attachedNode.node_id || attachedNode.jsNodeId);
  if (attachedId) return displayedIdByAlias.get(attachedId) || attachedId;
  return displayedIdByAlias.get(rawId);
}

function addProfileWeight(profile: SparseProfile, neighborId: string, weight: number) {
  const nextWeight = (profile.get(neighborId) || 0) + weight;
  if (nextWeight === 0) profile.delete(neighborId);
  else profile.set(neighborId, nextWeight);
}

function buildPairEvidence(
  candidates: readonly Candidate[],
  config: SupernodeSuggestionConfig,
): Map<string, Map<string, PairEvidence>> {
  const evidence = new Map<string, Map<string, PairEvidence>>();
  candidates.forEach((candidate) => evidence.set(candidate.id, new Map()));

  for (let leftIndex = 0; leftIndex < candidates.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < candidates.length; rightIndex += 1) {
      const left = candidates[leftIndex];
      const right = candidates[rightIndex];
      const pairEvidence = scorePair(left, right, config);
      evidence.get(left.id)?.set(right.id, pairEvidence);
      evidence.get(right.id)?.set(left.id, pairEvidence);
    }
  }
  return evidence;
}

function scorePair(left: Candidate, right: Candidate, config: SupernodeSuggestionConfig): PairEvidence {
  if (config.sameFeatureTypeOnly && left.node.feature_type !== right.node.feature_type) {
    return emptyEvidence();
  }

  const incomingAvailable = left.profiles.incoming.size > 0 && right.profiles.incoming.size > 0;
  const outgoingAvailable = left.profiles.outgoing.size > 0 && right.profiles.outgoing.size > 0;
  const incomingScore = incomingAvailable
    ? sparseCosineSimilarity(left.profiles.incoming, right.profiles.incoming)
    : null;
  const outgoingScore = outgoingAvailable
    ? sparseCosineSimilarity(left.profiles.outgoing, right.profiles.outgoing)
    : null;
  const sharedIncomingNeighbors = countSharedNeighbors(left.profiles.incoming, right.profiles.incoming);
  const sharedOutgoingNeighbors = countSharedNeighbors(left.profiles.outgoing, right.profiles.outgoing);
  const selectedScores: number[] = [];
  let selectedSharedNeighbors = 0;

  if (config.profileMode !== 'outgoing' && incomingScore !== null) {
    selectedScores.push(incomingScore);
    selectedSharedNeighbors += sharedIncomingNeighbors;
  }
  if (config.profileMode !== 'incoming' && outgoingScore !== null) {
    selectedScores.push(outgoingScore);
    selectedSharedNeighbors += sharedOutgoingNeighbors;
  }

  return {
    eligible: selectedScores.length > 0,
    score: selectedScores.length > 0 ? average(selectedScores) : 0,
    sharedIncomingNeighbors,
    sharedOutgoingNeighbors,
    selectedSharedNeighbors,
  };
}

function sparseCosineSimilarity(left: SparseProfile, right: SparseProfile): number {
  let leftNormSquared = 0;
  let rightNormSquared = 0;
  let dotProduct = 0;
  left.forEach((value, key) => {
    leftNormSquared += value * value;
    dotProduct += value * (right.get(key) || 0);
  });
  right.forEach((value) => {
    rightNormSquared += value * value;
  });
  if (leftNormSquared === 0 || rightNormSquared === 0) return 0;
  return clamp(dotProduct / Math.sqrt(leftNormSquared * rightNormSquared), -1, 1);
}

function countSharedNeighbors(left: SparseProfile, right: SparseProfile): number {
  const smaller = left.size <= right.size ? left : right;
  const larger = smaller === left ? right : left;
  let count = 0;
  smaller.forEach((value, key) => {
    if (value !== 0 && larger.has(key) && larger.get(key) !== 0) count += 1;
  });
  return count;
}

function clusterCandidates(
  candidates: readonly Candidate[],
  evidence: ReadonlyMap<string, ReadonlyMap<string, PairEvidence>>,
  config: SupernodeSuggestionConfig,
): string[][] {
  const scoredPairs: ScoredPair[] = [];
  for (let leftIndex = 0; leftIndex < candidates.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < candidates.length; rightIndex += 1) {
      const leftId = candidates[leftIndex].id;
      const rightId = candidates[rightIndex].id;
      const pair = getEvidence(evidence, leftId, rightId);
      if (pair && pairPasses(pair, config)) scoredPairs.push({ leftId, rightId, ...pair });
    }
  }
  scoredPairs.sort(
    (left, right) =>
      right.score - left.score ||
      compareStrings(left.leftId, right.leftId) ||
      compareStrings(left.rightId, right.rightId),
  );

  const clusterById = new Map(candidates.map((candidate) => [candidate.id, [candidate.id]]));
  const clusterIdByNodeId = new Map(candidates.map((candidate) => [candidate.id, candidate.id]));

  scoredPairs.forEach((pair) => {
    const leftClusterId = clusterIdByNodeId.get(pair.leftId);
    const rightClusterId = clusterIdByNodeId.get(pair.rightId);
    if (!leftClusterId || !rightClusterId || leftClusterId === rightClusterId) return;
    const leftMembers = clusterById.get(leftClusterId);
    const rightMembers = clusterById.get(rightClusterId);
    if (!leftMembers || !rightMembers || leftMembers.length + rightMembers.length > config.maxGroupSize) return;
    if (!allCrossPairsPass(leftMembers, rightMembers, evidence, config)) return;

    const mergedMembers = [...leftMembers, ...rightMembers].sort(compareStrings);
    const mergedClusterId = mergedMembers[0];
    clusterById.delete(leftClusterId);
    clusterById.delete(rightClusterId);
    clusterById.set(mergedClusterId, mergedMembers);
    mergedMembers.forEach((nodeId) => clusterIdByNodeId.set(nodeId, mergedClusterId));
  });

  return [...clusterById.values()].filter((members) => members.length >= 2);
}

function allCrossPairsPass(
  leftMembers: readonly string[],
  rightMembers: readonly string[],
  evidence: ReadonlyMap<string, ReadonlyMap<string, PairEvidence>>,
  config: SupernodeSuggestionConfig,
): boolean {
  return leftMembers.every((leftId) =>
    rightMembers.every((rightId) => {
      const pair = getEvidence(evidence, leftId, rightId);
      return pair !== undefined && pairPasses(pair, config);
    }),
  );
}

function pairPasses(pair: PairEvidence, config: SupernodeSuggestionConfig): boolean {
  return (
    pair.eligible &&
    pair.score >= config.similarityThreshold &&
    pair.selectedSharedNeighbors >= config.minSharedNeighbors
  );
}

function summarizeCluster(
  memberNodeIds: string[],
  candidateById: ReadonlyMap<string, Candidate>,
  evidence: ReadonlyMap<string, ReadonlyMap<string, PairEvidence>>,
): SupernodeSuggestion {
  const pairEvidence: PairEvidence[] = [];
  for (let leftIndex = 0; leftIndex < memberNodeIds.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < memberNodeIds.length; rightIndex += 1) {
      const pair = getEvidence(evidence, memberNodeIds[leftIndex], memberNodeIds[rightIndex]);
      if (pair) pairEvidence.push(pair);
    }
  }
  const members = memberNodeIds
    .map((id) => candidateById.get(id))
    .filter((candidate): candidate is Candidate => !!candidate);
  const ctxIndices = members.map((candidate) => candidate.node.ctx_idx);
  const layers = [...new Set(members.map((candidate) => candidate.node.layer))].sort(compareStrings);

  return {
    id: `auto:${memberNodeIds.map(encodeURIComponent).join(',')}`,
    label: '',
    memberNodeIds: [...memberNodeIds],
    minCohesion: Math.min(...pairEvidence.map((pair) => pair.score)),
    meanCohesion: average(pairEvidence.map((pair) => pair.score)),
    minSharedIncomingNeighbors: Math.min(...pairEvidence.map((pair) => pair.sharedIncomingNeighbors)),
    minSharedOutgoingNeighbors: Math.min(...pairEvidence.map((pair) => pair.sharedOutgoingNeighbors)),
    featureTypes: [...new Set(members.map((candidate) => candidate.node.feature_type))].sort(compareStrings),
    layers,
    ctxIdxRange: [Math.min(...ctxIndices), Math.max(...ctxIndices)],
  };
}

function getEvidence(
  evidence: ReadonlyMap<string, ReadonlyMap<string, PairEvidence>>,
  leftId: string,
  rightId: string,
): PairEvidence | undefined {
  return evidence.get(leftId)?.get(rightId);
}

function emptyEvidence(): PairEvidence {
  return {
    eligible: false,
    score: 0,
    sharedIncomingNeighbors: 0,
    sharedOutgoingNeighbors: 0,
    selectedSharedNeighbors: 0,
  };
}

function average(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function finiteOr(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
