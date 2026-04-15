'use client';

import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import { useGlobalContext } from '@/components/provider/global-provider';
import { BreadcrumbLink } from '@/components/shadcn/breadcrumbs';
import { Button } from '@/components/shadcn/button';
import {
  ConnectionLineType,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Edge,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useSession } from 'next-auth/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DraftEditSidebar } from './draft-edit-sidebar';
import DraftNodeComponent from './draft-node';
import { DraftPreviewSidebar } from './draft-preview-sidebar';
import ProblemNodeComponent, { TYPE_COLORS as NODE_TYPE_COLORS } from './explorer-node';
import { DRAFT_ID, TYPE_HEX_COLORS, type DetailNode, type ProblemNodeData } from './explorer-shared';
import { NodeSidebar } from './node-sidebar';
import { NODE_HEIGHT, getLayoutedElements } from './use-layout';

const nodeTypes = { problem: ProblemNodeComponent, draft: DraftNodeComponent };

// Padding for fitView — adjust these to control spacing around the graph
const FIT_VIEW_PADDING_TOP = 0.04;
const FIT_VIEW_PADDING_BOTTOM = 0.03;

function getVisibleNodesWithCollapse(
  allNodes: ProblemNodeData[],
  selectedId: number | null,
  expandForIds?: Set<number>,
): { visibleNodes: ProblemNodeData[]; hiddenChildCounts: Map<number, number> } {
  const nodeById = new Map(allNodes.map((n) => [n.id, n]));

  const childrenOf = new Map<number, number[]>();
  allNodes.forEach((n) => {
    if (n.parentId != null && nodeById.has(n.parentId)) {
      const siblings = childrenOf.get(n.parentId) || [];
      siblings.push(n.id);
      childrenOf.set(n.parentId, siblings);
    }
  });

  const depths = new Map<number, number>();
  const roots = allNodes.filter((n) => !n.parentId || !nodeById.has(n.parentId));
  const queue: [number, number][] = roots.map((r) => [r.id, 0]);
  while (queue.length > 0) {
    const [id, depth] = queue.shift()!;
    if (!depths.has(id)) {
      depths.set(id, depth);
      (childrenOf.get(id) || []).forEach((childId) => queue.push([childId, depth + 1]));
    }
  }

  const ancestorIds = new Set<number>();
  if (selectedId != null && selectedId !== DRAFT_ID) {
    let cur: number | null | undefined = selectedId;
    while (cur != null) {
      ancestorIds.add(cur);
      cur = nodeById.get(cur)?.parentId;
    }
  }

  // Also expand ancestors of filter-matched nodes so they're visible
  if (expandForIds) {
    for (const id of expandForIds) {
      let cur: number | null | undefined = id;
      while (cur != null) {
        ancestorIds.add(cur);
        cur = nodeById.get(cur)?.parentId;
      }
    }
  }

  // A node is expanded (shows its children) if it's a root, on the ancestor path, or the selected node itself
  const expandedIds = new Set<number>([...ancestorIds]);
  roots.forEach((r) => expandedIds.add(r.id));
  if (selectedId != null && selectedId !== DRAFT_ID) {
    expandedIds.add(selectedId);
  }

  const visibleIds = new Set<number>();
  allNodes.forEach((n) => {
    const depth = depths.get(n.id) ?? 0;
    if (depth <= 1) {
      visibleIds.add(n.id);
    } else if (n.parentId != null && expandedIds.has(n.parentId)) {
      visibleIds.add(n.id);
    }
  });

  const hiddenChildCounts = new Map<number, number>();
  allNodes.forEach((n) => {
    if (!visibleIds.has(n.id)) return;
    const children = childrenOf.get(n.id) || [];
    const hiddenCount = children.filter((cId) => !visibleIds.has(cId)).length;
    if (hiddenCount > 0) {
      hiddenChildCounts.set(n.id, hiddenCount);
    }
  });

  return {
    visibleNodes: allNodes.filter((n) => visibleIds.has(n.id)),
    hiddenChildCounts,
  };
}

type FlowBuildParams = {
  problemNodes: ProblemNodeData[];
  onAddChild?: (parentId: number) => void;
  onEditNode?: (nodeId: number) => void;
  draftCallbacks?: {
    onUpdateDraft: (fields: Partial<ProblemNodeData>) => void;
    onSubmitDraft: () => void;
    onStartEdit: () => void;
    onCancel: () => void;
    saving: boolean;
  };
  hiddenChildCounts?: Map<number, number>;
  dimmedIds?: Set<number>;
  hoverCallbacks?: { onHoverNode: (id: number) => void; onHoverLeave: () => void };
};

function buildFlowNodesAndEdges(params: FlowBuildParams): { nodes: Node[]; edges: Edge[] } {
  const { problemNodes, onAddChild, onEditNode, draftCallbacks, hiddenChildCounts, dimmedIds, hoverCallbacks } = params;

  const nodes: Node[] = problemNodes.map((pn) => {
    if (pn.id === DRAFT_ID) {
      return {
        id: String(pn.id),
        type: 'draft' as const,
        position: { x: 0, y: 0 },
        data: {
          draftTitle: pn.title,
          draftTypes: pn.nodeTypes,
          draftDescription: pn.description,
          draftUrl: pn.mainUrl,
          currentAdditionalUrls: pn.additionalUrls,
          currentNodeTypes: pn.nodeTypes,
          onUpdateDraft: draftCallbacks
            ? (fields: Partial<ProblemNodeData>) => draftCallbacks.onUpdateDraft(fields)
            : undefined,
          onSubmitDraft: draftCallbacks?.onSubmitDraft,
          onStartEdit: draftCallbacks?.onStartEdit,
          onCancel: draftCallbacks?.onCancel,
          saving: draftCallbacks?.saving ?? false,
        },
      };
    }
    return {
      id: String(pn.id),
      type: 'problem' as const,
      position: { x: 0, y: 0 },
      data: {
        label: pn.title,
        author: pn.author,
        nodeTypes: pn.nodeTypes,
        description: pn.description,
        childCount: pn.children?.length ?? 0,
        children: pn.children || [],
        approvalState: pn.approvalState,
        mainUrl: pn.mainUrl,
        isRoot: !pn.parentId || pn.parentId === 1,
        isDraft: false,
        hiddenChildCount: hiddenChildCounts?.get(pn.id) ?? 0,
        dimmed: dimmedIds?.has(pn.id) ?? false,
        onAddChild: onAddChild ? () => onAddChild(pn.id) : undefined,
        onEditNode: onEditNode ? () => onEditNode(pn.id) : undefined,
        onHoverNode: hoverCallbacks ? () => hoverCallbacks.onHoverNode(pn.id) : undefined,
        onHoverLeave: hoverCallbacks?.onHoverLeave,
      },
    };
  });

  const nodeIdSet = new Set(nodes.map((n) => n.id));
  const edges: Edge[] = [...problemNodes]
    .filter((pn) => pn.parentId && nodeIdSet.has(String(pn.id)) && nodeIdSet.has(String(pn.parentId)))
    .sort((a, b) => (a.title || '').localeCompare(b.title || ''))
    .map((pn) => ({
      id: `e-${pn.parentId}-${pn.id}`,
      source: String(pn.parentId!),
      target: String(pn.id),
      type: 'bezier',
      style: { stroke: '#cbd5e1', strokeWidth: 1 },
      animated: false,
    }));

  return { nodes, edges };
}

async function buildFlowGraph(params: FlowBuildParams) {
  const { nodes, edges } = buildFlowNodesAndEdges(params);
  return getLayoutedElements(nodes, edges);
}

function ProblemsGraphInner({
  initialNodes,
  canEdit,
  initialSelectedId,
}: {
  initialNodes: ProblemNodeData[];
  canEdit: boolean;
  initialSelectedId?: number;
}) {
  const { data: session } = useSession();
  const { setSignInModalOpen, showToastMessage } = useGlobalContext();
  const { fitView, getNodes, setViewport } = useReactFlow();
  const flowContainerRef = useRef<HTMLDivElement>(null);

  const fitViewCustom = useCallback(() => {
    const container = flowContainerRef.current;
    const allNodes = getNodes();
    if (!container || allNodes.length === 0) {
      fitView({ padding: FIT_VIEW_PADDING_TOP, duration: 300 });
      return;
    }
    // Calculate bounds of all nodes
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    allNodes.forEach((n) => {
      const w = n.measured?.width ?? n.width ?? 270;
      const h = n.measured?.height ?? n.height ?? 33;
      minX = Math.min(minX, n.position.x);
      minY = Math.min(minY, n.position.y);
      maxX = Math.max(maxX, n.position.x + w);
      maxY = Math.max(maxY, n.position.y + h);
    });
    const graphW = maxX - minX;
    const graphH = maxY - minY;
    const rect = container.getBoundingClientRect();
    // Calculate zoom with asymmetric padding
    const zoomX = rect.width / (graphW * (1 + FIT_VIEW_PADDING_TOP * 2));
    const zoomY = rect.height / (graphH * (1 + FIT_VIEW_PADDING_TOP + FIT_VIEW_PADDING_BOTTOM));
    const zoom = Math.min(zoomX, zoomY, 1.5);
    // Center horizontally, apply asymmetric vertical padding
    const scaledW = graphW * zoom;
    const scaledH = graphH * zoom;
    const x = (rect.width - scaledW) / 2 - minX * zoom;
    const topPad = (rect.height - scaledH) * (FIT_VIEW_PADDING_TOP / (FIT_VIEW_PADDING_TOP + FIT_VIEW_PADDING_BOTTOM));
    const y = topPad - minY * zoom;
    setViewport({ x, y, zoom }, { duration: 200 });
  }, [fitView, getNodes, setViewport]);

  const panToNode = useCallback(
    (nodeX: number, nodeY: number, nodeW: number, nodeH: number, opts?: { zoom?: number; duration?: number }) => {
      const container = flowContainerRef.current;
      if (!container) return;
      const zoom = opts?.zoom ?? 1.5;
      const rect = container.getBoundingClientRect();
      const x = -((nodeX + nodeW / 2) * zoom) + rect.width * 0.45;
      const y = -((nodeY + nodeH / 2) * zoom) + rect.height / 2;
      setViewport({ x, y, zoom }, { duration: opts?.duration });
    },
    [setViewport],
  );

  const [problemNodes, setProblemNodes] = useState<ProblemNodeData[]>(initialNodes);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(initialSelectedId ?? null);
  const selectedNodeIdRef = useRef<number | null>(null);
  selectedNodeIdRef.current = selectedNodeId;
  const [detailNode, setDetailNode] = useState<DetailNode | null>(null);
  const [draftNode, setDraftNode] = useState<ProblemNodeData | null>(null);
  const [editOnSelect, setEditOnSelect] = useState(false);
  const [showUnapproved, setShowUnapproved] = useState(false);
  const [filteredNodeCount, setFilteredNodeCount] = useState(0);
  const [totalNodeCount, setTotalNodeCount] = useState(0);
  const [hoveredNodeId, setHoveredNodeId] = useState<number | null>(null);
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const onHoverNode = useCallback(
    (id: number) => {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      setHoveredNodeId(id);
      setNodes((nds) =>
        nds.map((n) => ({
          ...n,
          zIndex: n.id === String(id) ? 1000 : 0,
          data: { ...n.data, hoverDimmed: n.id !== String(id) },
        })),
      );
    },
    [setNodes],
  );

  const onHoverLeave = useCallback(() => {
    hoverTimerRef.current = setTimeout(() => {
      setHoveredNodeId(null);
      setNodes((nds) =>
        nds.map((n) => ({
          ...n,
          zIndex: 0,
          data: { ...n.data, hoverDimmed: false },
        })),
      );
    }, 75);
  }, [setNodes]);

  const selectedNode = useMemo(() => {
    if (selectedNodeId === DRAFT_ID && draftNode) return draftNode;
    return problemNodes.find((n) => n.id === selectedNodeId) || null;
  }, [problemNodes, selectedNodeId, draftNode]);

  const ancestryChain = useMemo(() => {
    if (!selectedNodeId || selectedNodeId === DRAFT_ID) return [];
    const nodeById = new Map(problemNodes.map((n) => [n.id, n]));
    const chain: ProblemNodeData[] = [];
    let cur = nodeById.get(selectedNodeId);
    while (cur) {
      chain.unshift(cur);
      cur = cur.parentId ? nodeById.get(cur.parentId) : undefined;
    }
    return chain;
  }, [problemNodes, selectedNodeId]);

  const breadcrumbs = useMemo(() => {
    const crumbs: React.ReactNode[] = [];
    if (ancestryChain.length === 0) {
      crumbs.push(<span key="explorer">Field Explorer</span>);
    } else {
      crumbs.push(
        <BreadcrumbLink key="explorer" href="/explorer">
          Field Explorer
        </BreadcrumbLink>,
      );
      ancestryChain.forEach((node, i) => {
        const isLast = i === ancestryChain.length - 1;
        if (isLast) {
          crumbs.push(
            <span key={node.id} className="truncate">
              {node.title || '(untitled)'}
            </span>,
          );
        } else {
          crumbs.push(
            <BreadcrumbLink key={node.id} href={`/explorer/${node.id}`}>
              {node.title || '(untitled)'}
            </BreadcrumbLink>,
          );
        }
      });
    }
    return crumbs;
  }, [ancestryChain]);

  // Fetch full detail (comments, edges) only when a node is selected
  useEffect(() => {
    if (!selectedNodeId || selectedNodeId === DRAFT_ID) {
      setDetailNode(null);
      return;
    }
    setDetailNode(null);
    fetch('/api/problem/node/get', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: selectedNodeId }),
    })
      .then((res) => res.json())
      .then((data) => setDetailNode(data))
      .catch(() => setDetailNode(null));
  }, [selectedNodeId]);

  const fetchNodes = useCallback(async () => {
    try {
      const res = await fetch('/api/problem/node/list', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ includeUnapproved: showUnapproved }),
      });
      const data = await res.json();
      setProblemNodes(data);
    } catch {
      // fetch failed
    }
  }, [showUnapproved]);

  // Re-fetch when showUnapproved changes (but not on mount — we have initialNodes)
  const [didMount, setDidMount] = useState(false);
  useEffect(() => {
    if (!didMount) {
      setDidMount(true);
      return;
    }
    fetchNodes();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showUnapproved]);

  const [draftEditing, setDraftEditing] = useState(false);
  const [draftSaving, setDraftSaving] = useState(false);

  const updateDraftFields = useCallback(
    (fields: Partial<ProblemNodeData>) => {
      if (!draftNode) return;
      setDraftNode({ ...draftNode, ...fields });
    },
    [draftNode],
  );

  const submitDraft = useCallback(async () => {
    if (!draftNode?.title?.trim()) return;
    setDraftSaving(true);
    try {
      const res = await fetch('/api/problem/node/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nodeTypes: draftNode.nodeTypes,
          title: draftNode.title.trim(),
          description: draftNode.description?.trim() || null,
          mainUrl: draftNode.mainUrl?.trim() || null,
          author: draftNode.author?.trim() || null,
          additionalUrls: draftNode.additionalUrls || [],
          parentId: draftNode.parentId || null,
        }),
      });
      if (res.ok) {
        const savedNode = await res.json();
        const newId = savedNode.id;
        // Select the new node and fetch the updated list before clearing
        // the draft, so the saved node is already in the graph when the
        // draft disappears — no flash or zoom reset.
        selectedNodeIdRef.current = newId;
        setSelectedNodeId(newId);
        await fetchNodes();
        setDraftNode(null);
        setDraftEditing(false);
      } else {
        const data = await res.json().catch(() => ({ error: 'Failed to save node' }));
        showToastMessage(data.error || 'Failed to save node');
      }
    } catch {
      showToastMessage('Network error saving node');
    } finally {
      setDraftSaving(false);
    }
  }, [draftNode, fetchNodes, showToastMessage]);

  // Auto-open edit sidebar when draft gets populated (title set)
  useEffect(() => {
    if (draftNode?.title && !draftEditing) {
      setDraftEditing(true);
    }
  }, [draftNode?.title, draftEditing]);

  const startDraftEdit = useCallback(() => {
    setDraftEditing(true);
    setSelectedNodeId(DRAFT_ID);
  }, []);

  const handleDraftCancelled = useCallback(() => {
    setDraftNode(null);
    setDraftEditing(false);
    if (selectedNodeIdRef.current === DRAFT_ID) {
      setSelectedNodeId(null);
      window.history.replaceState(null, '', '/explorer');
    }
  }, []);

  const addChildToNode = useCallback(
    (parentId: number) => {
      if (!session?.user) {
        setSignInModalOpen(true);
        return;
      }
      // eslint-disable-next-line no-alert
      if (draftNode && !window.confirm('This will discard your current node creation. Are you sure?')) return;
      const parentNode = problemNodes.find((n) => n.id === parentId);
      const draft: ProblemNodeData = {
        id: DRAFT_ID,
        nodeTypes: ['topic'],
        parentId,
        title: null,
        description: null,
        mainUrl: null,
        additionalUrls: [],
        applicationTags: [],
        approvalState: 'PENDING',
        children: [],
        parent: parentNode ? { id: parentNode.id, title: parentNode.title, nodeTypes: parentNode.nodeTypes } : null,
        author: null,
        createdBy: { id: session.user.id, name: session.user.name || '' },
        approver: null,
      };
      setDraftNode(draft);
      setSelectedNodeId(DRAFT_ID);
    },
    [session, setSignInModalOpen, problemNodes, draftNode],
  );

  const draftCallbacks = useMemo(
    () => ({
      onUpdateDraft: updateDraftFields,
      onSubmitDraft: submitDraft,
      onStartEdit: startDraftEdit,
      onCancel: handleDraftCancelled,
      saving: draftSaving,
    }),
    [updateDraftFields, submitDraft, startDraftEdit, handleDraftCancelled, draftSaving],
  );

  const typeFilterOptions = useMemo(() => {
    const base = ['all', 'tool', 'replication', 'paper', 'dataset', 'eval', 'model'];
    if (canEdit) base.push('unapproved');
    return base;
  }, [canEdit]);
  const [typeFilter, setTypeFilter] = useState('all');

  const positionCacheRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const prevProblemNodesRef = useRef(problemNodes);
  const prevTypeFilterRef = useRef(typeFilter);
  const prevDraftNodeRef = useRef(draftNode);

  const handleTypeFilter = useCallback(
    (t: string) => {
      setTypeFilter(t);
      setShowUnapproved(t === 'unapproved');
      setSelectedNodeId(null);
      window.history.replaceState(null, '', '/explorer');
    },
    [setShowUnapproved],
  );

  const selectNode = useCallback(
    (id: number) => {
      setSelectedNodeId(id);
      setTypeFilter('all');
      setShowUnapproved(false);
      setNodes((nds) => nds.map((n) => ({ ...n, selected: n.id === String(id) })));
      window.history.replaceState(null, '', `/explorer/${id}`);
      setTimeout(() => {
        const currentNodes = getNodes();
        const target = currentNodes.find((n) => n.id === String(id));
        if (target) {
          const width = target.measured?.width ?? 200;
          const height = target.measured?.height ?? 40;
          panToNode(target.position.x, target.position.y, width, height, { duration: 200 });
        }
      }, 100);
    },
    [setNodes, getNodes, panToNode],
  );

  const editNode = useCallback(
    (id: number) => {
      if (id !== selectedNodeId) {
        selectNode(id);
      }
      setEditOnSelect(true);
    },
    [selectNode, selectedNodeId],
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const clickedId = Number(node.id);
      if (draftNode && clickedId !== DRAFT_ID) {
        // eslint-disable-next-line no-alert
        if (!window.confirm('This will discard your current node creation. Are you sure?')) return;
        handleDraftCancelled();
      }
      if (clickedId === selectedNodeId) {
        setSelectedNodeId(null);
        setNodes((nds) => nds.map((n) => ({ ...n, selected: false })));
        window.history.replaceState(null, '', '/explorer');
      } else {
        selectNode(clickedId);
      }
    },
    [selectNode, draftNode, handleDraftCancelled, selectedNodeId, setNodes],
  );

  const handleClose = useCallback(() => {
    setSelectedNodeId(null);
    window.history.replaceState(null, '', '/explorer');
  }, []);

  const onPaneClick = useCallback(() => {
    if (hoveredNodeId != null) {
      setHoveredNodeId(null);
      setNodes((nds) => nds.map((n) => ({ ...n, zIndex: 0, data: { ...n.data, hoverDimmed: false } })));
    }
    if (draftNode) return;
    if (selectedNodeId != null) {
      handleClose();
      setTimeout(() => fitViewCustom(), 50);
    } else {
      fitViewCustom();
    }
  }, [draftNode, selectedNodeId, handleClose, fitViewCustom, hoveredNodeId, setNodes]);

  const handleAddNode = useCallback(() => {
    if (!session?.user) {
      setSignInModalOpen(true);
      return;
    }
    // if (!selectedNodeId || selectedNodeId === DRAFT_ID) {
    //   // eslint-disable-next-line no-alert
    //   window.alert('Before adding a node, click an existing node to choose its parent node.');
    //   return;
    // }
    // eslint-disable-next-line no-alert
    if (draftNode && !window.confirm('This will discard your current node creation. Are you sure?')) return;
    const draft: ProblemNodeData = {
      id: DRAFT_ID,
      nodeTypes: ['topic'],
      parentId: selectedNodeId,
      title: null,
      description: null,
      mainUrl: null,
      additionalUrls: [],
      applicationTags: [],
      approvalState: 'PENDING',
      children: [],
      parent: selectedNode
        ? { id: selectedNode.id, title: selectedNode.title, nodeTypes: selectedNode.nodeTypes }
        : null,
      author: null,
      createdBy: { id: session.user.id, name: session.user.name || '' },
      approver: null,
    };
    setDraftNode(draft);
    setSelectedNodeId(DRAFT_ID);
  }, [session, setSignInModalOpen, selectedNodeId, selectedNode, draftNode]);

  const handleUpdated = useCallback(() => {
    fetchNodes();
    if (selectedNodeId) {
      fetch('/api/problem/node/get', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: selectedNodeId }),
      })
        .then((res) => res.json())
        .then((data) => setDetailNode(data))
        .catch(() => {});
    }
  }, [fetchNodes, selectedNodeId]);

  const handleDelete = useCallback(async () => {
    // eslint-disable-next-line no-alert
    if (!selectedNodeId || !window.confirm('Delete this node?')) return;
    const parentId = selectedNode?.parentId ?? null;
    const res = await fetch('/api/problem/node/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: selectedNodeId }),
    });
    if (res.ok) {
      if (parentId) {
        selectedNodeIdRef.current = parentId;
        setSelectedNodeId(parentId);
        window.history.replaceState(null, '', `/explorer/${parentId}`);
      } else {
        handleClose();
      }
      fetchNodes();
    }
  }, [selectedNodeId, selectedNode, handleClose, fetchNodes]);

  useEffect(() => {
    if (problemNodes.length === 0 && !draftNode) return;
    const filtered =
      typeFilter === 'all'
        ? problemNodes
        : typeFilter === 'unapproved'
          ? problemNodes.filter((n) => n.approvalState !== 'APPROVED')
          : problemNodes.filter((n) => n.nodeTypes.includes(typeFilter));
    setFilteredNodeCount(filtered.length);
    setTotalNodeCount(problemNodes.length);
    const includeIds = new Set(filtered.map((n) => n.id));
    const nodeById = new Map(problemNodes.map((n) => [n.id, n]));
    filtered.forEach((n) => {
      let cur = n;
      while (cur.parentId) {
        if (includeIds.has(cur.parentId)) break;
        includeIds.add(cur.parentId);
        const parent = nodeById.get(cur.parentId);
        if (!parent) break;
        cur = parent;
      }
    });
    const withParents = problemNodes.filter((n) => includeIds.has(n.id));

    // Dim nodes not related to the selected node (ancestors, selected, descendants stay bright)
    let selectionDimmedIds: Set<number> | undefined;
    if (selectedNodeId != null && selectedNodeId !== DRAFT_ID) {
      const focusedIds = new Set<number>();
      const nbm = new Map(withParents.map((n) => [n.id, n]));

      // Walk up to collect ancestors
      let cur: number | null | undefined = selectedNodeId;
      while (cur != null) {
        focusedIds.add(cur);
        cur = nbm.get(cur)?.parentId;
      }

      // Walk down to collect all descendants
      const descQueue = [selectedNodeId];
      while (descQueue.length > 0) {
        const id = descQueue.shift()!;
        focusedIds.add(id);
        withParents.forEach((n) => {
          if (n.parentId === id && !focusedIds.has(n.id)) {
            descQueue.push(n.id);
          }
        });
      }

      selectionDimmedIds = new Set(withParents.filter((n) => !focusedIds.has(n.id)).map((n) => n.id));
    }

    const filterMatchIds = typeFilter !== 'all' ? new Set(filtered.map((n) => n.id)) : undefined;
    const { visibleNodes, hiddenChildCounts } = getVisibleNodesWithCollapse(
      withParents,
      selectedNodeId,
      filterMatchIds,
    );
    const filterDimmedIds =
      filterMatchIds && filterMatchIds.size > 0
        ? new Set(visibleNodes.filter((n) => !filterMatchIds.has(n.id)).map((n) => n.id))
        : undefined;
    const dimmedIds =
      selectionDimmedIds || filterDimmedIds
        ? new Set([...(selectionDimmedIds || []), ...(filterDimmedIds || [])])
        : undefined;
    const allNodes = draftNode ? [...visibleNodes, draftNode] : visibleNodes;
    const buildParams: FlowBuildParams = {
      problemNodes: allNodes,
      onAddChild: addChildToNode,
      onEditNode: canEdit ? editNode : undefined,
      draftCallbacks: draftNode ? draftCallbacks : undefined,
      hiddenChildCounts,
      dimmedIds,
      hoverCallbacks: selectedNodeId == null ? { onHoverNode, onHoverLeave } : undefined,
    };

    const needsRelayout =
      positionCacheRef.current.size === 0 ||
      problemNodes !== prevProblemNodesRef.current ||
      typeFilter !== prevTypeFilterRef.current ||
      draftNode !== prevDraftNodeRef.current;

    prevProblemNodesRef.current = problemNodes;
    prevTypeFilterRef.current = typeFilter;
    prevDraftNodeRef.current = draftNode;

    if (needsRelayout) {
      buildFlowGraph(buildParams).then(({ nodes: layoutedNodes, edges: layoutedEdges }) => {
        const cache = new Map<string, { x: number; y: number }>();
        layoutedNodes.forEach((n) => cache.set(n.id, { ...n.position }));
        positionCacheRef.current = cache;

        const sid = selectedNodeIdRef.current;
        setNodes(layoutedNodes.map((n) => ({ ...n, selected: sid != null && n.id === String(sid) })));
        setEdges(layoutedEdges);
        if (draftNode) {
          const focusIds = [String(DRAFT_ID)];
          if (draftNode.parentId) focusIds.push(String(draftNode.parentId));
          setTimeout(() => fitView({ padding: 0.3, nodes: layoutedNodes.filter((n) => focusIds.includes(n.id)) }), 50);
        } else if (sid != null) {
          const target = layoutedNodes.find((n) => n.id === String(sid));
          if (target) {
            const w = target.measured?.width ?? 200;
            const h = target.measured?.height ?? 40;
            setTimeout(() => panToNode(target.position.x, target.position.y, w, h, { duration: 200 }), 50);
          } else {
            setTimeout(() => fitViewCustom(), 50);
          }
        } else {
          setTimeout(() => fitViewCustom(), 50);
        }
      });
    } else {
      const { nodes: builtNodes, edges: builtEdges } = buildFlowNodesAndEdges(buildParams);

      const sid = selectedNodeIdRef.current;
      const focusedNodeIds = new Set<string>();
      if (sid != null) {
        focusedNodeIds.add(String(sid));
        const nbm = new Map(buildParams.problemNodes.map((pn) => [pn.id, pn]));
        let cur: number | null | undefined = sid;
        while (cur != null) {
          focusedNodeIds.add(String(cur));
          cur = nbm.get(cur)?.parentId;
        }
        const dq = [sid];
        while (dq.length > 0) {
          const id = dq.shift()!;
          focusedNodeIds.add(String(id));
          buildParams.problemNodes.forEach((pn) => {
            if (pn.parentId === id && !focusedNodeIds.has(String(pn.id))) dq.push(pn.id);
          });
        }
      }

      const newChildOffsets = new Map<string, number>();
      const positioned = builtNodes.map((n) => {
        const isFocused = focusedNodeIds.has(n.id);
        const cached = positionCacheRef.current.get(n.id);
        if (cached) return { ...n, position: { ...cached }, zIndex: isFocused ? 100 : 0 };

        const parentEdge = builtEdges.find((e) => e.target === n.id);
        if (parentEdge) {
          const parentPos = positionCacheRef.current.get(parentEdge.source);
          if (parentPos) {
            const offset = newChildOffsets.get(parentEdge.source) || 0;
            newChildOffsets.set(parentEdge.source, offset + 1);
            const pos = { x: parentPos.x + 300, y: parentPos.y + offset * (NODE_HEIGHT + 4) };
            positionCacheRef.current.set(n.id, pos);
            return { ...n, position: pos, zIndex: 100 };
          }
        }
        return n;
      });

      setNodes(positioned.map((n) => ({ ...n, selected: sid != null && n.id === String(sid) })));
      setEdges(builtEdges);

      if (sid != null) {
        const target = positioned.find((n) => n.id === String(sid));
        if (target) {
          const w = target.measured?.width ?? 200;
          const h = target.measured?.height ?? 40;
          setTimeout(() => panToNode(target.position.x, target.position.y, w, h, { duration: 200 }), 50);
        } else {
          setTimeout(() => fitViewCustom(), 50);
        }
      } else {
        setTimeout(() => fitViewCustom(), 50);
      }
    }
  }, [
    typeFilter,
    selectedNodeId,
    problemNodes,
    draftNode,
    addChildToNode,
    canEdit,
    editNode,
    draftCallbacks,
    setNodes,
    setEdges,
    fitView,
    fitViewCustom,
    panToNode,
    onHoverNode,
    onHoverLeave,
  ]);

  return (
    <>
      <BreadcrumbsComponent crumbsArray={breadcrumbs} />
      <div className="flex h-[calc(100vh-48px)] items-center justify-center sm:hidden">
        <p className="px-8 text-center text-sm text-slate-500">
          This interface is not yet optimized for mobile. Go to{' '}
          <a href="/explorer" className="text-sky-600 underline">
            neuronpedia.org/explorer
          </a>{' '}
          on a larger screen.
        </p>
      </div>
      <div className="hidden h-[calc(100vh-84px)] w-full sm:flex">
        <div ref={flowContainerRef} className="flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            nodesDraggable={false}
            defaultEdgeOptions={{ type: 'bezier' }}
            connectionLineType={ConnectionLineType.Bezier}
            panOnScroll
            zoomOnScroll={false}
            zoomOnDoubleClick={false}
            minZoom={0.7}
            maxZoom={2}
            proOptions={{ hideAttribution: true }}
          >
            {/* <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#e2e8f0" /> */}
            <Panel position="top-right" className="flex items-start gap-2">
              <Controls
                showInteractive={false}
                showFitView
                onFitView={fitViewCustom}
                position="top-right"
                className="!static !m-0 !shadow-none"
                style={{ border: '1px solid #cbd5e1', borderRadius: 8, overflow: 'hidden' }}
              />
              <MiniMap
                pannable
                zoomable
                position="top-right"
                className="!static !m-0"
                nodeColor={(node) => {
                  const types = (node.data as Record<string, string[]>)?.nodeTypes;
                  return TYPE_HEX_COLORS[types?.[0]] || '#94a3b8';
                }}
                maskColor="rgba(100, 116, 139, 0.35)"
                style={{ height: 100, width: 160, border: '1px solid #cbd5e1', borderRadius: 8, overflow: 'hidden' }}
              />
            </Panel>

            <Panel position="top-left">
              <div className="flex flex-col gap-2">
                <div className="rounded-lg border border-slate-200 bg-white px-4 pb-3.5 pt-2.5 shadow">
                  <h1 className="mb-1 w-full text-center text-[15px] font-semibold text-slate-700">
                    Interpretability Field Explorer
                  </h1>
                  <div className="mb-1 w-full text-center text-[8px] font-medium uppercase text-slate-400">
                    Filter by type
                  </div>
                  <div className="flex flex-wrap justify-center gap-1 pt-0">
                    {typeFilterOptions.map((t) => {
                      const colors = NODE_TYPE_COLORS[t];
                      return (
                        <button
                          type="button"
                          key={t}
                          onClick={() => handleTypeFilter(t)}
                          className={`rounded-full border px-3 py-1 text-center text-[9px] font-bold uppercase transition-colors ${
                            typeFilter === t
                              ? `${colors ? colors.icon : 'bg-slate-800'} border-transparent text-white`
                              : `${colors ? `${colors.border} ${colors.label}` : 'border-slate-200 text-slate-600'} bg-white hover:bg-slate-50`
                          }`}
                        >
                          {t}
                        </button>
                      );
                    })}
                  </div>
                  {typeFilter !== 'all' && (
                    <div className="mt-2.5 flex items-center justify-center gap-3 rounded bg-slate-100 py-2.5">
                      <span className="text-[12px] text-slate-600">
                        Filtering to {filteredNodeCount} of {totalNodeCount} nodes
                      </span>
                      <Button
                        size="xs"
                        onClick={() => handleTypeFilter('all')}
                        className="rounded px-3 py-1.5 text-[10px] font-semibold uppercase"
                      >
                        Clear Filter
                      </Button>
                    </div>
                  )}
                  {selectedNodeId != null && selectedNodeId !== DRAFT_ID && (
                    <div className="mt-2.5 flex items-center justify-center gap-3 rounded bg-slate-100 py-2.5">
                      <span className="text-[12px] text-slate-600">Selected Single Node</span>
                      <Button
                        size="xs"
                        onClick={handleClose}
                        className="rounded px-3 py-1.5 text-[10px] font-semibold uppercase"
                      >
                        Show All Nodes
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </Panel>

            <Panel position="bottom-right">
              <button
                type="button"
                onClick={handleAddNode}
                className="rounded-lg bg-sky-700 px-4 py-2 text-xs font-medium text-white shadow-sm transition-colors hover:bg-sky-800"
              >
                + Add Node
              </button>
            </Panel>
          </ReactFlow>
        </div>

        {selectedNode && selectedNodeId === DRAFT_ID && draftNode ? (
          draftEditing ? (
            <DraftEditSidebar
              draftNode={draftNode}
              setDraftNode={setDraftNode}
              existingNodes={problemNodes}
              onSaved={submitDraft}
              onCancel={handleDraftCancelled}
            />
          ) : (
            <DraftPreviewSidebar draftNode={draftNode} onCancel={handleDraftCancelled} />
          )
        ) : selectedNode ? (
          <NodeSidebar
            selectedNode={selectedNode}
            detailNode={detailNode}
            canEdit={canEdit}
            session={session}
            allNodes={problemNodes}
            onClose={handleClose}
            onDelete={handleDelete}
            onUpdated={handleUpdated}
            onSelectNode={selectNode}
            editOnSelect={editOnSelect}
            onEditOnSelectConsumed={() => setEditOnSelect(false)}
          />
        ) : null}

        {problemNodes.length === 0 && !draftNode && (
          <div className="flex h-full w-full items-center justify-center">
            <div className="text-center">
              <p className="text-sm text-slate-500">No problem nodes yet.</p>
              <button
                type="button"
                onClick={handleAddNode}
                className="mt-2 rounded-lg bg-sky-700 px-4 py-2 text-xs font-medium text-white hover:bg-sky-800"
              >
                Create the first one
              </button>
            </div>
          </div>
        )}
        {problemNodes.length > 0 && nodes.length === 0 && !draftNode && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
            <div className="pointer-events-auto text-center">
              <p className="text-sm text-slate-500">No nodes matched the current filter.</p>
              <button
                type="button"
                onClick={() => handleTypeFilter('all')}
                className="mt-2 rounded-lg border border-slate-200 px-4 py-2 text-xs font-medium text-slate-600 hover:bg-slate-50"
              >
                Reset filter
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

export default function ProblemsGraph({
  initialNodes,
  canEdit,
  initialSelectedId,
}: {
  initialNodes: ProblemNodeData[];
  canEdit: boolean;
  initialSelectedId?: number;
}) {
  return (
    <ReactFlowProvider>
      <ProblemsGraphInner initialNodes={initialNodes} canEdit={canEdit} initialSelectedId={initialSelectedId} />
    </ReactFlowProvider>
  );
}
