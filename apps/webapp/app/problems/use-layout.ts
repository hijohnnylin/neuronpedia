import { type Edge, type Node } from '@xyflow/react';
// eslint-disable-next-line import/no-extraneous-dependencies
import ELK, { type ElkNode } from 'elkjs/lib/elk.bundled';
import { DRAFT_NODE_HEIGHT_POPULATED, DRAFT_NODE_HEIGHT_URL } from './draft-node';
import { DEFAULT_NODE_WIDTH, ROOT_NODE_WIDTH } from './problem-node';

const NODE_HEIGHT_TOPIC = 42;
const NODE_HEIGHT_DEFAULT = 45;
const SIBLING_GAP = 4;

const elk = new ELK();

function getNodeDimensions(node: Node): { width: number; height: number } {
  if (node.type === 'draft') {
    const hasTitleData = !!(node.data as any)?.draftTitle;
    return { width: DEFAULT_NODE_WIDTH, height: hasTitleData ? DRAFT_NODE_HEIGHT_POPULATED : DRAFT_NODE_HEIGHT_URL };
  }
  const isRoot = (node.data as any)?.isRoot;
  const width = isRoot ? ROOT_NODE_WIDTH : DEFAULT_NODE_WIDTH;
  const nodeTypes: string[] = (node.data as any)?.nodeTypes || [];
  const isTopicOnly = nodeTypes.length > 0 && nodeTypes.every((t: string) => t === 'topic');
  const height = isTopicOnly ? NODE_HEIGHT_TOPIC : NODE_HEIGHT_DEFAULT;
  return { width, height };
}

// Sort nodes so that children of the same parent are grouped and alphabetically ordered.
// This preserves the relative order across different parent groups while sorting within each.
function sortChildrenByParent(nodes: Node[], edges: Edge[]): Node[] {
  const nodeTitle = new Map<string, string>();
  nodes.forEach((n) => nodeTitle.set(n.id, ((n.data as any)?.label || '').toLowerCase()));

  // Group child node IDs by parent
  const childrenByParent = new Map<string, string[]>();
  edges.forEach((e) => {
    const group = childrenByParent.get(e.source) || [];
    group.push(e.target);
    childrenByParent.set(e.source, group);
  });

  // Sort each parent's children alphabetically
  childrenByParent.forEach((children) => {
    children.sort((a, b) => (nodeTitle.get(a) || '').localeCompare(nodeTitle.get(b) || ''));
  });

  // Build ordered list: roots first, then BFS inserting sorted children after their parent
  const childSet = new Set(edges.map((e) => e.target));
  const roots = nodes.filter((n) => !childSet.has(n.id));
  const result: Node[] = [];
  const nodeMap = new Map<string, Node>();
  nodes.forEach((n) => nodeMap.set(n.id, n));

  const queue = [...roots];
  while (queue.length > 0) {
    const node = queue.shift()!;
    result.push(node);
    const children = childrenByParent.get(node.id);
    if (children) {
      children.forEach((childId) => {
        const child = nodeMap.get(childId);
        if (child) queue.push(child);
      });
    }
  }

  // Add any remaining nodes not reached by BFS
  const added = new Set(result.map((n) => n.id));
  nodes.forEach((n) => {
    if (!added.has(n.id)) result.push(n);
  });

  return result;
}

export async function getLayoutedElements(nodes: Node[], edges: Edge[]) {
  const graph: ElkNode = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.spacing.nodeNode': String(SIBLING_GAP),
      'elk.layered.spacing.nodeNodeBetweenLayers': '20',
      'elk.layered.crossingMinimization.forceNodeModelOrder': 'true',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.margins': '[top=40,left=40,bottom=40,right=40]',
    },
    children: sortChildrenByParent(nodes, edges).map((node) => {
      const { width, height } = getNodeDimensions(node);
      return { id: node.id, width, height };
    }),
    edges: edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  };

  const layoutedGraph = await elk.layout(graph);

  const layoutedNodes = nodes.map((node) => {
    const elkNode = layoutedGraph.children?.find((n) => n.id === node.id);
    return {
      ...node,
      position: { x: elkNode?.x ?? 0, y: elkNode?.y ?? 0 },
    };
  });

  return { nodes: layoutedNodes, edges };
}
