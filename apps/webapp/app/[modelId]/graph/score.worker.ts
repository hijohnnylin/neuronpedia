/* eslint-disable no-restricted-globals */
// Lightweight worker to compute subgraph scores off the main thread.
// Duplicates the core scoring logic to avoid bringing in DOM/d3 dependencies.

import { CLTGraphLink, CLTGraphNode } from './graph-types';

type WorkerGraph = {
  nodes: CLTGraphNode[];
  links: CLTGraphLink[];
};

function normalizeMatrix(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    const absRow = row.map((val) => Math.abs(val));
    const sum = absRow.reduce((acc, val) => acc + val, 0);
    const clampedSum = Math.max(sum, 1e-10);
    return absRow.map((val) => val / clampedSum);
  });
}

function computeInfluence(A: number[][], logitWeights: number[], maxIter: number = 1000): number[] {
  let currentInfluence = new Array(A[0].length).fill(0);
  for (let j = 0; j < A[0].length; j += 1) {
    for (let i = 0; i < logitWeights.length; i += 1) {
      currentInfluence[j] += logitWeights[i] * A[i][j];
    }
  }

  const influence = [...currentInfluence];
  let iterations = 0;

  while (currentInfluence.some((val) => val !== 0)) {
    if (iterations >= maxIter) {
      throw new Error(`Influence computation failed to converge after ${iterations} iterations`);
    }

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
): { matrix: number[][]; sortedNodes: CLTGraphNode[] } {
  const featureTypeOrder = [
    'cross layer transcoder',
    'lorsa',
    'mlp reconstruction error',
    'lorsa error',
    'embedding',
    'logit',
  ];

  function getSortKey(node: CLTGraphNode) {
    let typePriority = featureTypeOrder.indexOf(node.feature_type);
    if (typePriority === -1) typePriority = featureTypeOrder.length;
    const layerNum = node.layer === 'E' ? 0 : Number.isNaN(parseInt(node.layer, 10)) ? 999 : parseInt(node.layer, 10);
    return [typePriority, layerNum, node.ctx_idx, node.feature || 0];
  }

  const sortedNodes = [...nodes].sort((a, b) => {
    const keyA = getSortKey(a);
    const keyB = getSortKey(b);
    for (let i = 0; i < Math.max(keyA.length, keyB.length); i += 1) {
      const valA = (keyA as number[])[i] || 0;
      const valB = (keyB as number[])[i] || 0;
      if (valA !== valB) return valA - valB;
    }
    return 0;
  });

  const nodeIdToIdx: Record<string, number> = {};
  sortedNodes.forEach((node, idx) => {
    nodeIdToIdx[node.node_id] = idx;
  });

  const nNodes = sortedNodes.length;
  const adjacencyMatrix: number[][] = Array(nNodes)
    .fill(null)
    .map(() => Array(nNodes).fill(0));

  edges.forEach((edge) => {
    const srcIdx = nodeIdToIdx[edge.source];
    const dstIdx = nodeIdToIdx[edge.target];
    if (srcIdx !== undefined && dstIdx !== undefined) {
      adjacencyMatrix[dstIdx][srcIdx] = edge.weight;
    }
  });

  return { matrix: adjacencyMatrix, sortedNodes };
}

function computeGraphScoresFromGraphData(
  graphData: WorkerGraph,
  pinnedIds: string[] = [],
): { replacementScore: number; completenessScore: number } {
  const hasLorsa = graphData.nodes.some(
    (n) => n.feature_type === 'lorsa' || n.feature_type === 'lorsa error',
  );
  if (hasLorsa) {
    console.warn(
      'Score computation skipped: graph contains lorsa nodes which are not yet supported by the scoring algorithm.',
    );
    return { replacementScore: -1, completenessScore: -1 };
  }

  const graphNodesToUse = graphData.nodes;

  const { matrix: adjacencyMatrix, sortedNodes } = reconstructAdjacencyMatrix(graphNodesToUse, graphData.links);

  // Helper to zero a row and its corresponding column by index
  const zeroRowAndColumn = (idx: number) => {
    for (let c = 0; c < adjacencyMatrix.length; c += 1) {
      adjacencyMatrix[idx][c] = 0;
    }
    for (let r = 0; r < adjacencyMatrix.length; r += 1) {
      adjacencyMatrix[r][idx] = 0;
    }
  };

  // If pinnedIds provided, merge non-pinned feature nodes into their corresponding error nodes
  if (pinnedIds.length > 0) {
    const pinnedSet = new Set(pinnedIds);

    // Build a lookup from (layer, ctx_idx) to error node index in sortedNodes
    const errorIndexByKey: Record<string, number> = {};
    sortedNodes.forEach((node, idx) => {
      if (node.feature_type === 'mlp reconstruction error') {
        const key = `${node.layer}|${node.ctx_idx}`;
        errorIndexByKey[key] = idx;
      }
    });

    sortedNodes.forEach((node, featureIdx) => {
      if (node.feature_type !== 'cross layer transcoder') return;
      if (pinnedSet.has(node.node_id)) return; // keep pinned features

      const key = `${node.layer}|${node.ctx_idx}`;
      const errorIdx = errorIndexByKey[key];
      if (errorIdx === undefined) {
        // No matching error node found; just zero out the feature node
        zeroRowAndColumn(featureIdx);
        return;
      }

      // Merge outgoing edges (column) of feature into error node's column
      for (let r = 0; r < adjacencyMatrix.length; r += 1) {
        adjacencyMatrix[r][errorIdx] += adjacencyMatrix[r][featureIdx];
      }

      // Zero both incoming and outgoing edges for the feature node
      zeroRowAndColumn(featureIdx);
    });
  }

  // Derive counts from sortedNodes to align with adjacency order
  const nFeatures = sortedNodes.filter((n) => n.feature_type === 'cross layer transcoder').length;
  const nErrorNodes = sortedNodes.filter((n) => n.feature_type === 'mlp reconstruction error').length;
  const nTokens = sortedNodes.filter((n) => n.feature_type === 'embedding').length;
  const nLogits = sortedNodes.filter((n) => n.feature_type === 'logit').length;

  const errorStart = nFeatures;
  const errorEnd = errorStart + nErrorNodes;
  const tokenEnd = errorEnd + nTokens;

  // Align logit weights with sortedNodes order so that the last indices match the correct logits
  const logitWeights = new Array(adjacencyMatrix.length).fill(0);
  const sortedLogitProbs = sortedNodes
    .filter((n) => n.feature_type === 'logit')
    .map((n) => n.token_prob);
  for (let i = 0; i < nLogits; i += 1) {
    logitWeights[adjacencyMatrix.length - nLogits + i] = sortedLogitProbs[i];
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

console.log('Worker script loaded');

self.onmessage = (ev: MessageEvent) => {
  console.log('Worker received message:', ev.data ? 'has data' : 'null data');

  try {
    // Handle Chrome's stricter Web Worker message handling
    if (!ev.data) {
      // Chrome may send null data during worker initialization - ignore these messages
      return;
    }

    const { data } = ev;

    // Validate that we have the expected message structure
    // eslint-disable-next-line no-prototype-builtins
    if (typeof data !== 'object' || !data.hasOwnProperty('graph') || !data.hasOwnProperty('requestId')) {
      console.error('Worker: Invalid message format', data);
      // @ts-ignore
      self.postMessage({ error: 'Invalid message format received by worker' });
      return;
    }

    console.log('Worker: starting computation');
    const { requestId, graph, pinnedIds } = data as { requestId: number; graph: WorkerGraph; pinnedIds: string[] };
    const scores = computeGraphScoresFromGraphData(graph, pinnedIds);
    console.log('Worker: computation complete, sending results:', scores);
    // @ts-ignore
    self.postMessage({ requestId, ...scores });
  } catch (err) {
    console.error('Worker error:', err);
    // @ts-ignore
    self.postMessage({ error: (err as Error)?.message || 'Unknown worker error' });
  }
};
