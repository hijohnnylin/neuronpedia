'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
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
import { NodeSidebar } from './node-sidebar';
import ProblemNodeComponent, { TYPE_COLORS as NODE_TYPE_COLORS } from './problem-node';
import { DRAFT_ID, type DetailNode, type ProblemNodeData } from './problems-shared';
import { getLayoutedElements } from './use-layout';

const nodeTypes = { problem: ProblemNodeComponent, draft: DraftNodeComponent };

// Padding for fitView — adjust these to control spacing around the graph
const FIT_VIEW_PADDING_TOP = 0.09;
const FIT_VIEW_PADDING_BOTTOM = 0.03;

function getVisibleNodesWithCollapse(
  allNodes: ProblemNodeData[],
  selectedId: number | null,
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

  // A node is expanded (shows its children) if it's a root, on the ancestor path, or a descendant of the selected node
  const expandedIds = new Set<number>([...ancestorIds]);
  roots.forEach((r) => expandedIds.add(r.id));
  if (selectedId != null && selectedId !== DRAFT_ID) {
    const descQueue = [selectedId];
    while (descQueue.length > 0) {
      const id = descQueue.shift()!;
      expandedIds.add(id);
      (childrenOf.get(id) || []).forEach((childId) => descQueue.push(childId));
    }
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

async function buildFlowGraph(
  problemNodes: ProblemNodeData[],
  onAddChild?: (parentId: number) => void,
  onEditNode?: (nodeId: number) => void,
  draftCallbacks?: {
    onUpdateDraft: (fields: Partial<ProblemNodeData>) => void;
    onSubmitDraft: () => void;
    onStartEdit: () => void;
    onCancel: () => void;
    saving: boolean;
  },
  hiddenChildCounts?: Map<number, number>,
) {
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
        approvalState: pn.approvalState,
        mainUrl: pn.mainUrl,
        isRoot: !pn.parentId || pn.parentId === 1,
        isDraft: false,
        hiddenChildCount: hiddenChildCounts?.get(pn.id) ?? 0,
        onAddChild: onAddChild ? () => onAddChild(pn.id) : undefined,
        onEditNode: onEditNode ? () => onEditNode(pn.id) : undefined,
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
      // Position node at 20% from left, vertically centered
      const x = -((nodeX + nodeW / 2) * zoom) + rect.width * 0.3;
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

  const selectedNode = useMemo(() => {
    if (selectedNodeId === DRAFT_ID && draftNode) return draftNode;
    return problemNodes.find((n) => n.id === selectedNodeId) || null;
  }, [problemNodes, selectedNodeId, draftNode]);

  // Fetch full detail (comments, edges) only when a node is selected
  useEffect(() => {
    if (!selectedNodeId || selectedNodeId === DRAFT_ID) {
      setDetailNode(null);
      return;
    }
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
      window.history.replaceState(null, '', '/problems');
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

  const selectNode = useCallback(
    (id: number) => {
      setSelectedNodeId(id);
      setNodes((nds) => nds.map((n) => ({ ...n, selected: n.id === String(id) })));
      window.history.replaceState(null, '', `/problems/${id}`);
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
      selectNode(id);
      setEditOnSelect(true);
    },
    [selectNode],
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
        window.history.replaceState(null, '', '/problems');
      } else {
        selectNode(clickedId);
      }
    },
    [selectNode, draftNode, handleDraftCancelled, selectedNodeId, setNodes],
  );

  const handleClose = useCallback(() => {
    setSelectedNodeId(null);
    window.history.replaceState(null, '', '/problems');
  }, []);

  const onPaneClick = useCallback(() => {
    if (draftNode) return;
    if (selectedNodeId != null) {
      handleClose();
      // Delay zoom reset until sidebar is gone and container has resized
      setTimeout(() => fitViewCustom(), 50);
    } else {
      fitViewCustom();
    }
  }, [draftNode, selectedNodeId, handleClose, fitViewCustom]);

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
        window.history.replaceState(null, '', `/problems/${parentId}`);
      } else {
        handleClose();
      }
      fetchNodes();
    }
  }, [selectedNodeId, selectedNode, handleClose, fetchNodes]);

  const typeFilterOptions = useMemo(
    () => ['all', 'topic', 'paper', 'tool', 'dataset', 'eval', 'replication', 'model'],
    [],
  );
  const [typeFilter, setTypeFilter] = useState('all');

  useEffect(() => {
    if (problemNodes.length === 0 && !draftNode) return;
    const filtered = typeFilter === 'all' ? problemNodes : problemNodes.filter((n) => n.nodeTypes.includes(typeFilter));
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
    const { visibleNodes, hiddenChildCounts } = getVisibleNodesWithCollapse(withParents, selectedNodeId);
    const allNodes = draftNode ? [...visibleNodes, draftNode] : visibleNodes;
    buildFlowGraph(
      allNodes,
      addChildToNode,
      canEdit ? editNode : undefined,
      draftNode ? draftCallbacks : undefined,
      hiddenChildCounts,
    ).then(({ nodes: layoutedNodes, edges: layoutedEdges }) => {
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
  ]);

  return (
    <>
      <div className="flex h-[calc(100vh-48px)] items-center justify-center sm:hidden">
        <p className="px-8 text-center text-sm text-slate-500">
          This interface is not yet optimized for mobile. Go to{' '}
          <a href="/problems" className="text-sky-600 underline">
            {typeof window !== 'undefined' ? window.location.host : ''}/problems
          </a>{' '}
          on a larger screen.
        </p>
      </div>
      <div className="hidden h-[calc(100vh-48px)] w-full sm:flex">
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
                  const colorMap: Record<string, string> = {
                    topic: '#3b82f6',
                    paper: '#10b981',
                    tool: '#4f46e5',
                    dataset: '#f59e0b',
                    eval: '#f43f5e',
                    replication: '#c026d3',
                    model: '#78716c',
                  };
                  return colorMap[(node.data as any)?.nodeTypes?.[0]] || '#94a3b8';
                }}
                maskColor="rgba(100, 116, 139, 0.35)"
                style={{ height: 100, width: 160, border: '1px solid #cbd5e1', borderRadius: 8, overflow: 'hidden' }}
              />
            </Panel>

            <div className="pointer-events-none absolute left-1/2 top-3 z-10 -translate-x-1/2 rounded-lg bg-white/50 px-4 py-2 text-center backdrop-blur-[3px]">
              <h1 className="text-sm font-semibold text-slate-800">Mech Interp Problems</h1>
              <p className="mt-0.5 text-[11px] text-slate-400">Seeded from Sharkey et al. 2025</p>
            </div>
            <Panel position="bottom-left">
              <div className="flex flex-col gap-2">
                <div className="rounded-lg border border-slate-200 bg-white px-3 py-3 shadow-xl">
                  <div className="grid grid-cols-4 gap-1 pt-0">
                    {typeFilterOptions.map((t) => {
                      const colors = NODE_TYPE_COLORS[t];
                      return (
                        <button
                          type="button"
                          key={t}
                          onClick={() => setTypeFilter(t)}
                          className={`rounded-full border px-1.5 py-1 text-center text-[8px] font-bold uppercase transition-colors ${
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
                  <div className="mt-2 flex items-center justify-center gap-2">
                    <button
                      type="button"
                      role="switch"
                      aria-label="Show unapproved"
                      aria-checked={showUnapproved}
                      onClick={() => setShowUnapproved((v) => !v)}
                      className={`relative inline-flex h-4 w-7 shrink-0 cursor-pointer rounded-full transition-colors ${
                        showUnapproved ? 'bg-sky-700' : 'bg-slate-300'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-3 w-3 translate-y-0.5 rounded-full bg-white shadow transition-transform ${
                          showUnapproved ? 'translate-x-3.5' : 'translate-x-0.5'
                        }`}
                      />
                    </button>
                    <span className="text-[11px] text-slate-500">Show unapproved</span>
                  </div>
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
