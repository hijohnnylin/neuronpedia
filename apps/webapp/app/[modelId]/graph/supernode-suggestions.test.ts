import { describe, expect, it } from 'vitest';
import { CLTGraphLink, CLTGraphNode } from './graph-types';
import { mergeAcceptedSupernodeSuggestions, suggestSupernodes, SupernodeSuggestion } from './supernode-suggestions';

function makeNode(
  id: string,
  {
    featureType = 'cross layer transcoder',
    rawId = id,
    jsNodeId = id,
    layer = '1',
    ctxIdx = 0,
  }: {
    featureType?: string;
    rawId?: string;
    jsNodeId?: string;
    layer?: string;
    ctxIdx?: number;
  } = {},
): CLTGraphNode {
  return {
    node_id: rawId,
    nodeId: id,
    feature: 0,
    layer,
    ctx_idx: ctxIdx,
    feature_type: featureType,
    token_prob: 0,
    is_target_logit: false,
    run_idx: 0,
    reverse_ctx_idx: 0,
    jsNodeId,
    clerp: id,
  };
}

function makeLink(source: CLTGraphNode, target: CLTGraphNode, weight: number, attachNodes = true): CLTGraphLink {
  return {
    source: source.node_id,
    target: target.node_id,
    weight,
    ...(attachNodes ? { sourceNode: source, targetNode: target } : {}),
  };
}

function graph(nodes: CLTGraphNode[], links: CLTGraphLink[]) {
  return { nodes, links };
}

function incomingGraph(
  candidateSpecs: Array<{ node: CLTGraphNode; weights: number[] }>,
  upstreamNodes: CLTGraphNode[],
) {
  const candidates = candidateSpecs.map(({ node }) => node);
  const links = candidateSpecs.flatMap(({ node, weights }) =>
    weights.map((weight, index) => makeLink(upstreamNodes[index], node, weight)),
  );
  return graph([...upstreamNodes, ...candidates], links);
}

describe('suggestSupernodes', () => {
  it('groups nodes with identical signed incoming profiles', () => {
    const upstream = [makeNode('up-1', { featureType: 'embedding' }), makeNode('up-2', { featureType: 'embedding' })];
    const left = makeNode('left');
    const right = makeNode('right');
    const result = suggestSupernodes(
      incomingGraph(
        [
          { node: left, weights: [1, 0.5] },
          { node: right, weights: [2, 1] },
        ],
        upstream,
      ),
      ['left', 'right'],
    );

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      label: 'Suggested group 1',
      memberNodeIds: ['left', 'right'],
      minCohesion: 1,
      minSharedIncomingNeighbors: 2,
    });
  });

  it('does not group profiles with opposite attribution signs', () => {
    const upstream = [makeNode('up-1', { featureType: 'embedding' }), makeNode('up-2', { featureType: 'embedding' })];
    const result = suggestSupernodes(
      incomingGraph(
        [
          { node: makeNode('positive'), weights: [1, 1] },
          { node: makeNode('negative'), weights: [-1, -1] },
        ],
        upstream,
      ),
      ['positive', 'negative'],
    );

    expect(result).toEqual([]);
  });

  it('supports incoming, outgoing, and combined profile modes independently', () => {
    const upstream = [makeNode('up-1', { featureType: 'embedding' }), makeNode('up-2', { featureType: 'embedding' })];
    const downstream = [makeNode('down-1', { featureType: 'logit' }), makeNode('down-2', { featureType: 'logit' })];
    const left = makeNode('left');
    const right = makeNode('right');
    const links = [
      ...upstream.flatMap((node) => [makeLink(node, left, 1), makeLink(node, right, 1)]),
      makeLink(left, downstream[0], 1),
      makeLink(left, downstream[1], 1),
      makeLink(right, downstream[0], -1),
      makeLink(right, downstream[1], -1),
    ];
    const input = graph([...upstream, left, right, ...downstream], links);

    expect(suggestSupernodes(input, ['left', 'right'], [], { profileMode: 'incoming' })).toHaveLength(1);
    expect(suggestSupernodes(input, ['left', 'right'], [], { profileMode: 'outgoing' })).toEqual([]);
    expect(suggestSupernodes(input, ['left', 'right'], [], { profileMode: 'both' })).toEqual([]);
  });

  it('requires the configured amount of shared-neighbor evidence', () => {
    const upstream = [makeNode('shared', { featureType: 'embedding' })];
    const input = incomingGraph(
      [
        { node: makeNode('left'), weights: [1] },
        { node: makeNode('right'), weights: [1] },
      ],
      upstream,
    );

    expect(suggestSupernodes(input, ['left', 'right'])).toEqual([]);
    expect(suggestSupernodes(input, ['left', 'right'], [], { minSharedNeighbors: 1 })).toHaveLength(1);
  });

  it('resolves raw edge IDs to displayed Anthropic node IDs without attached nodes', () => {
    const upstream = [
      makeNode('display-up-1', { featureType: 'embedding', rawId: 'raw-up-1', jsNodeId: 'display-up-1' }),
      makeNode('display-up-2', { featureType: 'embedding', rawId: 'raw-up-2', jsNodeId: 'display-up-2' }),
    ];
    const left = makeNode('display-left', { rawId: 'raw-left', jsNodeId: 'display-left' });
    const right = makeNode('display-right', { rawId: 'raw-right', jsNodeId: 'display-right' });
    const links = upstream.flatMap((node) => [makeLink(node, left, 1, false), makeLink(node, right, 1, false)]);

    expect(suggestSupernodes(graph([...upstream, left, right], links), ['display-left', 'display-right'])).toHaveLength(
      1,
    );
  });

  it('recovers a human-authored Dallas subgroup from a reduced public graph fixture', () => {
    // Reduced from the Response Dallas supernode in:
    // https://transformer-circuits.pub/2025/attribution-graphs/graph_data/capital-state-dallas.json
    const left = makeNode('2_16536263_-0', { rawId: '2_16536263_10', layer: '2', ctxIdx: 10 });
    const right = makeNode('2_23133920_-0', { rawId: '2_23133920_10', layer: '2', ctxIdx: 10 });
    const austinDecoy = makeNode('10_8513782_-0', { rawId: '10_8513782_11', layer: '10', ctxIdx: 11 });
    const incoming = [
      ['E_28948388_10', 2.2459492683410645, 1.5262408256530762],
      ['1_21862382_10', 1.710408091545105, 1.098325490951538],
      ['1_4418612_10', 0.6796928644180298, 0.5530434846878052],
      ['1_13686348_10', 0.2643650770187378, 0.29762279987335205],
    ] as const;
    const outgoing = [
      ['3_27636071_10', 1.0517706871032715, 0.6656948328018188],
      ['3_22886075_10', 0.8075348138809204, 0.5080311894416809],
      ['4_1348861_10', 0.5275150537490845, 0.29129698872566223],
      ['15_21568836_11', 0.6396480202674866, 0.13760587573051453],
    ] as const;
    const contextNodes = [...incoming, ...outgoing].map(([id]) => makeNode(id, { featureType: 'embedding' }));
    const decoyContext = [
      makeNode('6_24231446_11', { featureType: 'embedding' }),
      makeNode('15_2811388_11', { featureType: 'embedding' }),
    ];
    const contextById = new Map(contextNodes.map((node) => [node.node_id, node]));
    const links = [
      ...incoming.flatMap(([id, leftWeight, rightWeight]) => [
        makeLink(contextById.get(id)!, left, leftWeight, false),
        makeLink(contextById.get(id)!, right, rightWeight, false),
      ]),
      ...outgoing.flatMap(([id, leftWeight, rightWeight]) => [
        makeLink(left, contextById.get(id)!, leftWeight, false),
        makeLink(right, contextById.get(id)!, rightWeight, false),
      ]),
      makeLink(decoyContext[0], austinDecoy, 0.4334683120250702, false),
      makeLink(austinDecoy, decoyContext[1], 1.7817903757095337, false),
    ];
    const input = graph([...contextNodes, ...decoyContext, left, right, austinDecoy], links);

    const result = suggestSupernodes(input, [left.nodeId!, right.nodeId!, austinDecoy.nodeId!]);

    expect(result).toHaveLength(1);
    expect(result[0].memberNodeIds).toEqual([left.nodeId, right.nodeId]);
    expect(result[0].minCohesion).toBeGreaterThan(0.9);
  });

  it('excludes unsupported, missing, zero-degree, and already grouped nodes', () => {
    const upstream = [makeNode('up-1', { featureType: 'embedding' }), makeNode('up-2', { featureType: 'embedding' })];
    const eligible = makeNode('eligible');
    const unsupported = makeNode('unsupported', { featureType: 'logit' });
    const grouped = makeNode('grouped');
    const zeroDegree = makeNode('zero-degree');
    const input = incomingGraph(
      [
        { node: eligible, weights: [1, 1] },
        { node: unsupported, weights: [1, 1] },
        { node: grouped, weights: [1, 1] },
      ],
      upstream,
    );
    input.nodes.push(zeroDegree);

    expect(
      suggestSupernodes(
        input,
        ['eligible', 'unsupported', 'grouped', 'zero-degree', 'missing'],
        [['manual', 'grouped']],
      ),
    ).toEqual([]);
  });

  it('keeps feature types separate by default and can relax that constraint', () => {
    const upstream = [makeNode('up-1', { featureType: 'embedding' }), makeNode('up-2', { featureType: 'embedding' })];
    const input = incomingGraph(
      [
        { node: makeNode('mlp'), weights: [1, 1] },
        { node: makeNode('attention', { featureType: 'lorsa' }), weights: [1, 1] },
      ],
      upstream,
    );

    expect(suggestSupernodes(input, ['mlp', 'attention'])).toEqual([]);
    expect(suggestSupernodes(input, ['mlp', 'attention'], [], { sameFeatureTypeOnly: false })).toHaveLength(1);
  });

  it('uses complete-link validation to prevent chain clusters', () => {
    const upstream = [makeNode('x', { featureType: 'embedding' }), makeNode('y', { featureType: 'embedding' })];
    const input = incomingGraph(
      [
        { node: makeNode('a'), weights: [1, 0] },
        { node: makeNode('b'), weights: [1, 1] },
        { node: makeNode('c'), weights: [0, 1] },
      ],
      upstream,
    );
    const result = suggestSupernodes(input, ['a', 'b', 'c'], [], {
      similarityThreshold: 0.7,
      minSharedNeighbors: 1,
    });

    expect(result).toHaveLength(1);
    expect(result[0].memberNodeIds).toEqual(['a', 'b']);
  });

  it('enforces maximum group size without assigning a node twice', () => {
    const upstream = [makeNode('up-1', { featureType: 'embedding' }), makeNode('up-2', { featureType: 'embedding' })];
    const nodes = ['a', 'b', 'c', 'd'].map((id) => makeNode(id));
    const input = incomingGraph(
      nodes.map((node) => ({ node, weights: [1, 1] })),
      upstream,
    );
    const result = suggestSupernodes(
      input,
      nodes.map((node) => node.nodeId || ''),
      [],
      { maxGroupSize: 2 },
    );
    const allMembers = result.flatMap((suggestion) => suggestion.memberNodeIds);

    expect(result.every((suggestion) => suggestion.memberNodeIds.length === 2)).toBe(true);
    expect(new Set(allMembers).size).toBe(allMembers.length);
  });

  it('is invariant to node, link, and pinned-ID ordering and does not mutate inputs', () => {
    const upstream = [makeNode('up-1', { featureType: 'embedding' }), makeNode('up-2', { featureType: 'embedding' })];
    const nodes = ['a', 'b', 'c'].map((id) => makeNode(id));
    const input = incomingGraph(
      nodes.map((node) => ({ node, weights: [1, 1] })),
      upstream,
    );
    const originalInput = structuredClone(input);
    const forward = suggestSupernodes(input, ['a', 'b', 'c']);
    const reversed = suggestSupernodes(graph([...input.nodes].reverse(), [...input.links].reverse()), ['c', 'b', 'a']);

    expect(reversed).toEqual(forward);
    expect(input).toEqual(originalInput);
  });
});

describe('mergeAcceptedSupernodeSuggestions', () => {
  function suggestion(id: string, label: string, memberNodeIds: string[]): SupernodeSuggestion {
    return {
      id,
      label,
      memberNodeIds,
      minCohesion: 1,
      meanCohesion: 1,
      minSharedIncomingNeighbors: 2,
      minSharedOutgoingNeighbors: 0,
      featureTypes: ['cross layer transcoder'],
      layers: ['1'],
      ctxIdxRange: [0, 0],
    };
  }

  it('preserves manual groups and revalidates stale, overlapping, and unpinned members', () => {
    const existing = [['manual', 'a']];
    const suggestions = [
      suggestion('first', 'Suggested group 1', ['a', 'b', 'c']),
      suggestion('second', 'Suggested group 2', ['c', 'd']),
      suggestion('third', 'Suggested group 3', ['e', 'f']),
    ];
    const result = mergeAcceptedSupernodeSuggestions({
      existingSupernodes: existing,
      suggestions,
      acceptedSuggestionIds: new Set(['first', 'second', 'third']),
      pinnedNodeIds: ['a', 'b', 'c', 'd', 'e'],
    });

    expect(result).toEqual([
      ['manual', 'a'],
      ['Suggested group 1', 'b', 'c'],
    ]);
    expect(existing).toEqual([['manual', 'a']]);
    expect(suggestions[0].memberNodeIds).toEqual(['a', 'b', 'c']);
  });

  it('applies only explicitly accepted suggestions', () => {
    const suggestions = [
      suggestion('accepted', 'Accepted', ['a', 'b']),
      suggestion('rejected', 'Rejected', ['c', 'd']),
    ];
    const result = mergeAcceptedSupernodeSuggestions({
      existingSupernodes: [],
      suggestions,
      acceptedSuggestionIds: new Set(['accepted']),
      pinnedNodeIds: ['a', 'b', 'c', 'd'],
    });

    expect(result).toEqual([['Accepted', 'a', 'b']]);
  });
});
