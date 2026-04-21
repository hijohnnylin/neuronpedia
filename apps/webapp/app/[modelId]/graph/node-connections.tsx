import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { useGraphStateContext } from '@/components/provider/graph-state-provider';
import { cn } from '@/lib/utils/ui';
import { ChevronRight, Circle, Diamond, Expand, Minimize2, Triangle } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Virtuoso } from 'react-virtuoso';
import { CLTGraphNode } from './graph-types';
import GraphFeatureLink from './np-feature-link';
import {
  clientCheckClaudeMode,
  clientCheckIsEmbed,
  featureTypeToText,
  graphModelHasNpDashboards,
  shouldShowNodeForDensityThreshold,
  shouldShowNodeForInfluenceThreshold,
} from './utils';

function FeatureList({
  title,
  nodes,
  linkType,
  hasNPDashboards,
  visState,
  isEditingLabel,
  getNodeSupernodeAndOverrideLabel,
  hoveredId,
  onHoverNode,
  onClearHover,
  onClickNode,
}: {
  title: string;
  nodes: CLTGraphNode[];
  linkType: 'source' | 'target';
  hasNPDashboards: boolean;
  visState: any;
  isEditingLabel: boolean;
  getNodeSupernodeAndOverrideLabel: (node: CLTGraphNode) => string;
  hoveredId: string | null;
  onHoverNode: (node: CLTGraphNode) => void;
  onClearHover: () => void;
  onClickNode: (node: CLTGraphNode, addToPinned: boolean) => void;
}) {
  const linkProp = linkType === 'source' ? 'tmpClickedSourceLink' : 'tmpClickedTargetLink';
  const filteredNodes = nodes
    ?.toSorted((a, b) => (b[linkProp]?.weight ?? 0) - (a[linkProp]?.weight ?? 0))
    .filter((node) => {
      // no input = don't show
      if (node[linkProp]?.weight === null || node[linkProp]?.weight === undefined) {
        return false;
      }

      if (
        clientCheckClaudeMode() &&
        (node.feature_type === 'mlp reconstruction error' || node.feature_type === 'lorsa error')
      ) {
        return false;
      }

      // otherwise there is some input, but check if we should check both influence and density filters
      return (
        shouldShowNodeForInfluenceThreshold(node, visState, null) &&
        shouldShowNodeForDensityThreshold(hasNPDashboards, node, visState, null)
      );
    });

  const itemContent = (index: number, node: CLTGraphNode) => (
    <div className={`mb-0.5 ${index === 0 ? 'pt-1' : ''}`}>
      <div className="px-1">
        <button
          type="button"
          className={`flex w-full cursor-pointer flex-row items-center justify-between gap-x-1.5 rounded bg-slate-50 px-2 py-[3px] text-[10px] hover:bg-sky-100 ${
            node.nodeId === hoveredId ? 'z-20 outline-dotted outline-[3px] outline-[#f0f]' : ''
          } ${(node[linkProp]?.pctInput ?? 0) > 0.25 || (node[linkProp]?.pctInput ?? 0) < -0.25 ? 'text-white' : ''}`}
          style={{ backgroundColor: node[linkProp]?.tmpColor }}
          onMouseEnter={() => {
            if (!isEditingLabel) {
              onHoverNode(node);
            }
          }}
          onMouseLeave={() => {
            onClearHover();
          }}
          onClick={(e) => {
            const addToPinned = e.ctrlKey || e.metaKey;
            onClickNode(node, addToPinned);
          }}
        >
          <svg width={10} height={14} className="mr-0 inline-block">
            <g>
              <g
                className={`default-icon block fill-none ${node.feature_type === 'mlp reconstruction error' || node.feature_type === 'lorsa error' ? 'opacity-35' : ''} ${(node[linkProp]?.pctInput ?? 0) > 0.25 || (node[linkProp]?.pctInput ?? 0) < -0.25 ? 'stroke-white' : 'stroke-slate-800'} ${node.nodeId && visState.pinnedIds?.includes(node.nodeId) ? 'stroke-[1.7]' : 'stroke-[0.7]'}`}
              >
                <text fontSize={15} textAnchor="middle" dominantBaseline="central" dx={5} dy={5}>
                  {featureTypeToText(node.feature_type)}
                </text>
              </g>
            </g>
          </svg>
          <div className="flex-1 text-left leading-snug">{getNodeSupernodeAndOverrideLabel(node)}</div>
          {node[linkProp]?.tmpClickedCtxOffset !== undefined &&
            (node[linkProp]?.tmpClickedCtxOffset > 0 ? (
              <div>→</div>
            ) : node[linkProp]?.tmpClickedCtxOffset < 0 ? (
              <div>←</div>
            ) : (
              ''
            ))}
          <div className="flex flex-col items-center justify-center">
            <div className="font-mono">
              {node[linkProp]?.weight !== null && node[linkProp]?.weight !== undefined
                ? node[linkProp]?.weight > 0
                  ? `+${node[linkProp]?.weight?.toFixed(2)}`
                  : node[linkProp]?.weight?.toFixed(2)
                : ''}
            </div>
            <div className="font-mono">{node.layer !== 'E' ? `L${node.layer}` : ''}</div>
          </div>
        </button>
      </div>
    </div>
  );

  return (
    <div className="flex flex-1 flex-col overflow-hidden pb-1 text-slate-800">
      <div className="sticky top-0 bg-white text-[10px] font-medium uppercase text-slate-500">
        {title} ({filteredNodes.length})
      </div>
      <div className="flex-1 overflow-hidden">
        <Virtuoso
          style={{ height: '100%' }}
          data={filteredNodes}
          itemContent={(index, node) => itemContent(index, node)}
        />
      </div>
    </div>
  );
}

function QKTracingSection({
  clickedNode,
  nodesById,
  hoveredId,
  getNodeSupernodeAndOverrideLabel,
  onHoverNode,
  onClearHover,
  onClickNode,
}: {
  clickedNode: CLTGraphNode;
  nodesById: Map<string, CLTGraphNode>;
  hoveredId: string | null;
  getNodeSupernodeAndOverrideLabel: (node: CLTGraphNode) => string;
  onHoverNode: (node: CLTGraphNode) => void;
  onClearHover: () => void;
  onClickNode: (node: CLTGraphNode, addToPinned: boolean) => void;
}) {
  const results = clickedNode.qk_tracing_results;
  if (!results) return null;

  const handleNodeClick = (nodeId: string, addToPinned: boolean) => {
    const node = nodesById.get(nodeId);
    if (node) onClickNode(node, addToPinned);
  };

  const handleNodeHover = (nodeId: string) => {
    const node = nodesById.get(nodeId);
    if (node) onHoverNode(node);
  };

  const renderNodeLabel = (nodeId: string) => {
    const node = nodesById.get(nodeId);
    if (node) {
      const label = getNodeSupernodeAndOverrideLabel(node);
      return (
        <div className="flex min-w-0 flex-1 items-center gap-x-1">
          <svg width={10} height={14} className="shrink-0">
            <g className="fill-none stroke-slate-800 stroke-[0.7]">
              <text fontSize={15} textAnchor="middle" dominantBaseline="central" dx={5} dy={5}>
                {featureTypeToText(node.feature_type)}
              </text>
            </g>
          </svg>
          <span className="truncate text-left">{label}</span>
          {node.layer !== 'E' && <span className="shrink-0 font-mono text-[9px] text-slate-500">L{node.layer}</span>}
        </div>
      );
    }
    return <span className="truncate text-left">{nodeId}</span>;
  };

  const { pair_wise_contributors, top_q_marginal_contributors, top_k_marginal_contributors } = results;

  return (
    <div className="mt-0.5 flex h-1/2 min-h-0 flex-col gap-y-1 overflow-y-auto">
      <div className="flex flex-row gap-x-2">
        <div className="flex min-w-0 flex-1 flex-col gap-y-1">
          <div className="sticky top-0 z-30 bg-white text-[10px] font-medium uppercase text-slate-500">
            K Marginal (QK Trace)
          </div>
          <div className="flex flex-col gap-y-0.5 pl-1">
            {top_k_marginal_contributors.map(([nodeId, score], idx) => (
              <button
                key={`k-${nodeId}-${idx}`}
                type="button"
                className={cn(
                  'flex cursor-pointer items-center justify-between gap-x-1.5 rounded bg-slate-50 px-2 py-[3px] text-[10px] hover:bg-sky-100',
                  nodeId === hoveredId && 'z-20 outline-dotted outline-[3px] outline-[#f0f]',
                )}
                onClick={(e) => handleNodeClick(nodeId, e.metaKey || e.ctrlKey)}
                onMouseEnter={() => handleNodeHover(nodeId)}
                onMouseLeave={onClearHover}
              >
                {renderNodeLabel(nodeId)}
                <div className="shrink-0 font-mono">{score.toFixed(2)}</div>
              </button>
            ))}
            {top_k_marginal_contributors.length === 0 && (
              <div className="px-2 py-1 text-[10px] italic text-slate-400">None</div>
            )}
          </div>
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-y-1">
          <div className="sticky top-0 z-30 bg-white text-[10px] font-medium uppercase text-slate-500">
            Q Marginal (QK Trace)
          </div>
          <div className="flex flex-col gap-y-0.5 pr-1">
            {top_q_marginal_contributors.map(([nodeId, score], idx) => (
              <button
                key={`q-${nodeId}-${idx}`}
                type="button"
                className={cn(
                  'flex cursor-pointer items-center justify-between gap-x-1.5 rounded bg-slate-50 px-2 py-[3px] text-[10px] hover:bg-sky-100',
                  nodeId === hoveredId && 'z-20 outline-dotted outline-[3px] outline-[#f0f]',
                )}
                onClick={(e) => handleNodeClick(nodeId, e.metaKey || e.ctrlKey)}
                onMouseEnter={() => handleNodeHover(nodeId)}
                onMouseLeave={onClearHover}
              >
                {renderNodeLabel(nodeId)}
                <div className="shrink-0 font-mono">{score.toFixed(2)}</div>
              </button>
            ))}
            {top_q_marginal_contributors.length === 0 && (
              <div className="px-2 py-1 text-[10px] italic text-slate-400">None</div>
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-y-1">
        <div className="text-[10px] font-medium uppercase text-slate-500">Top Pairwise Contributors</div>
        <div className="flex flex-col gap-y-0.5">
          {pair_wise_contributors.map(([qId, kId, score], idx) => (
            <div
              key={`pw-${idx}`}
              className="flex items-stretch overflow-hidden rounded bg-slate-50 py-1 pl-1 text-[10px]"
            >
              <button
                type="button"
                className={cn(
                  'flex min-w-0 flex-1 cursor-pointer items-center gap-x-1 rounded px-2 py-[3px] hover:bg-sky-100',
                  kId === hoveredId && 'z-20 outline-dotted outline-[3px] outline-[#f0f]',
                )}
                onClick={(e) => handleNodeClick(kId, e.metaKey || e.ctrlKey)}
                onMouseEnter={() => handleNodeHover(kId)}
                onMouseLeave={onClearHover}
              >
                <span className="shrink-0 text-[9px] font-bold text-slate-400">K:</span>
                {renderNodeLabel(kId)}
              </button>
              <div className="flex shrink-0 items-center justify-center pl-1">
                <ChevronRight className="h-3 w-3 text-slate-400/60" />
              </div>
              <button
                type="button"
                className={cn(
                  'flex min-w-0 flex-1 cursor-pointer items-center gap-x-1 rounded px-2 py-[3px] hover:bg-sky-100',
                  qId === hoveredId && 'z-20 outline-dotted outline-[3px] outline-[#f0f]',
                )}
                onClick={(e) => handleNodeClick(qId, e.metaKey || e.ctrlKey)}
                onMouseEnter={() => handleNodeHover(qId)}
                onMouseLeave={onClearHover}
              >
                <span className="shrink-0 text-[9px] font-bold text-slate-400">Q:</span>
                {renderNodeLabel(qId)}
              </button>
              <div className="flex shrink-0 items-center border-l border-slate-200 bg-white/40 px-2 font-mono">
                {score.toFixed(2)}
              </div>
            </div>
          ))}
          {pair_wise_contributors.length === 0 && <div className="px-2 py-1 text-[10px] text-slate-400">None</div>}
        </div>
      </div>
    </div>
  );
}

export default function GraphNodeConnections() {
  const {
    visState,
    selectedGraph,
    togglePin,
    isEditingLabel,
    getNodeSupernodeAndOverrideLabel,
    setFullNPFeatureDetail,
  } = useGraphContext();

  const { registerHoverCallback, updateHoverState, clearHoverState, registerClickedCallback, updateClickedState } =
    useGraphStateContext();

  const { openWelcomeModalToStep } = useGraphModalContext();

  const [clickedNode, setClickedNode] = useState<CLTGraphNode | null>(null);
  const [localHoveredId, setLocalHoveredId] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  // Register for hover change notifications from other components
  useEffect(() => {
    const unregister = registerHoverCallback((hoveredId) => {
      setLocalHoveredId(hoveredId);
    });

    return unregister; // Cleanup on unmount
  }, [registerHoverCallback]);

  // Register for clicked change notifications from other components
  useEffect(() => {
    const unregister = registerClickedCallback((clickedId) => {
      if (clickedId && selectedGraph) {
        const cNode = selectedGraph.nodes.find((e) => e.nodeId === clickedId);
        if (cNode) {
          setClickedNode(cNode);
        }
      } else {
        setClickedNode(null);
      }
    });

    return unregister; // Cleanup on unmount
  }, [registerClickedCallback, selectedGraph]);

  // Handlers for triggering hover state changes (bidirectional)
  const handleHoverNode = (node: CLTGraphNode) => {
    updateHoverState(node);
  };

  const handleClearHover = () => {
    clearHoverState();
  };

  // Handler for triggering clicked state changes (bidirectional)
  const handleClickNode = (node: CLTGraphNode, addToPinned: boolean) => {
    if (addToPinned && node.nodeId) {
      // If control or command key is pressed, add to pinnedIds
      // if pinned, remove it
      togglePin(node.nodeId);
    } else if (node.nodeId) {
      updateClickedState(node);
    }
  };

  useEffect(() => {
    if (clickedNode) {
      setFullNPFeatureDetail(setClickedNode, clickedNode);
    }
  }, [clickedNode]);

  const nodesById = useMemo(() => {
    const map = new Map<string, CLTGraphNode>();
    selectedGraph?.nodes.forEach((n) => {
      if (n.nodeId) map.set(n.nodeId, n);
      if (n.node_id && !map.has(n.node_id)) map.set(n.node_id, n);
      if (n.jsNodeId && !map.has(n.jsNodeId)) map.set(n.jsNodeId, n);
    });
    return map;
  }, [selectedGraph]);

  const hasQKTracing = Boolean(clickedNode?.qk_tracing_results);

  return (
    <>
      {/* Backdrop for dismissing overlay on background click */}
      {isExpanded && (
        <div
          className="fixed inset-0 z-[9998] bg-black/50 backdrop-blur-sm"
          onClick={() => setIsExpanded(false)}
          onKeyDown={(e) => e.key === 'Escape' && setIsExpanded(false)}
          role="button"
          tabIndex={0}
          aria-label="Close expanded view"
        />
      )}
      <div
        className={`node-connections relative overflow-y-hidden rounded-lg border border-slate-200 bg-white px-2 py-2 shadow-sm transition-all ${
          isExpanded ? '' : 'mt-2 hidden max-w-[420px] flex-1 flex-row sm:flex'
        }`}
        style={{
          ...(isExpanded && {
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 'calc(100vw - 80px)',
            height: 'calc(100vh - 60px)',
            maxWidth: '800px',
            maxHeight: '800px',
            zIndex: 9999,
            border: '20px solid white',
            borderRadius: '12px',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          }),
        }}
      >
        {/* <div className="absolute top-0 mx-auto h-3 w-24 rounded-b bg-[#f0f] text-[6px] text-white">Clicked Node</div> */}

        <div className="flex h-full w-full flex-col text-slate-700">
          {clickedNode ? (
            <div className="flex flex-row items-center gap-x-1.5 text-xs font-medium text-slate-600">
              {/* {!clickedNode?.featureDetailNP && <div className="">F#{clickedNode?.feature}</div>} */}
              {clickedNode.feature_type === 'lorsa' ? (
                <Triangle className="h-3.5 max-h-3.5 min-h-3.5 w-3.5 min-w-3.5 max-w-3.5 fill-[#f0f] text-[#f0f]" />
              ) : clickedNode.feature_type === 'mlp reconstruction error' ||
                clickedNode.feature_type === 'lorsa error' ? (
                <Diamond className="h-3.5 max-h-3.5 min-h-3.5 w-3.5 min-w-3.5 max-w-3.5 text-[#f0f]" />
              ) : (
                <Circle className="h-3.5 max-h-3.5 min-h-3.5 w-3.5 min-w-3.5 max-w-3.5 text-[#f0f]" />
              )}
              <div className="flex-1 leading-tight">{getNodeSupernodeAndOverrideLabel(clickedNode)}</div>
              <GraphFeatureLink selectedGraph={selectedGraph} node={clickedNode} />
              {!clientCheckIsEmbed() && (
                <div className="flex flex-col gap-y-0.5">
                  <button
                    type="button"
                    onClick={() => {
                      if (isExpanded) {
                        setIsExpanded(false);
                      }
                      openWelcomeModalToStep(3);
                    }}
                    className="flex h-[20px] w-[20px] items-center justify-center gap-x-1 self-start rounded-full bg-slate-200 px-0 py-0.5 text-[11px] font-medium transition-colors hover:bg-slate-300"
                    aria-label="Open User Guide"
                  >
                    ?
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="flex h-[20px] w-[20px] items-center justify-center gap-x-1 self-start rounded-full bg-slate-200 px-0 py-0.5 text-[12px] font-medium transition-colors hover:bg-slate-300"
                    aria-label={isExpanded ? 'Exit fullscreen' : 'Enter fullscreen'}
                    title={isExpanded ? 'Exit fullscreen' : 'Enter fullscreen'}
                  >
                    {isExpanded ? (
                      <Minimize2 className="h-2.5 w-2.5 text-slate-600" />
                    ) : (
                      <Expand className="h-2.5 w-2.5 text-slate-600" />
                    )}
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="relative flex h-[100%] flex-col items-center justify-center text-center text-sm font-medium text-slate-700">
              <div className="mb-2 text-lg font-bold">Node Connections</div>
              <div className="">Click a node on the left to see its connections.</div>
              {!clientCheckIsEmbed() && (
                <div className="absolute right-0 top-0 flex flex-col gap-y-0.5">
                  <button
                    type="button"
                    onClick={() => {
                      if (isExpanded) {
                        setIsExpanded(false);
                      }
                      openWelcomeModalToStep(3);
                    }}
                    className="flex h-[20px] w-[20px] items-center justify-center gap-x-1 rounded-full bg-slate-200 py-0.5 text-[11px] font-medium transition-colors hover:bg-slate-300"
                    aria-label="Open User Guide"
                  >
                    ?
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="flex h-[20px] w-[20px] items-center justify-center gap-x-1 rounded-full bg-slate-200 py-0.5 text-[11px] font-medium transition-colors hover:bg-slate-300"
                    aria-label={isExpanded ? 'Exit fullscreen' : 'Enter fullscreen'}
                    title={isExpanded ? 'Exit fullscreen' : 'Enter fullscreen'}
                  >
                    {isExpanded ? (
                      <Minimize2 className="h-2.5 w-2.5 text-slate-600" />
                    ) : (
                      <Expand className="h-2.5 w-2.5 text-slate-600" />
                    )}
                  </button>
                </div>
              )}
            </div>
          )}
          {clickedNode && (
            <div
              className={`mt-1.5 flex h-full min-h-0 w-full flex-1 flex-col gap-y-0 ${clickedNode?.featureDetailNP ? 'pb-0' : 'pb-0'}`}
            >
              <div className={`flex min-h-[50%] w-full flex-row gap-x-0 ${hasQKTracing ? 'h-1/2' : 'h-full flex-1'}`}>
                <FeatureList
                  title="Input Features"
                  nodes={selectedGraph?.nodes || []}
                  linkType="source"
                  hasNPDashboards={selectedGraph ? graphModelHasNpDashboards(selectedGraph) : false}
                  visState={visState}
                  isEditingLabel={isEditingLabel}
                  getNodeSupernodeAndOverrideLabel={getNodeSupernodeAndOverrideLabel}
                  hoveredId={localHoveredId}
                  onHoverNode={handleHoverNode}
                  onClearHover={handleClearHover}
                  onClickNode={handleClickNode}
                />
                <FeatureList
                  title="Output Features"
                  nodes={selectedGraph?.nodes || []}
                  linkType="target"
                  hasNPDashboards={selectedGraph ? graphModelHasNpDashboards(selectedGraph) : false}
                  visState={visState}
                  isEditingLabel={isEditingLabel}
                  getNodeSupernodeAndOverrideLabel={getNodeSupernodeAndOverrideLabel}
                  hoveredId={localHoveredId}
                  onHoverNode={handleHoverNode}
                  onClearHover={handleClearHover}
                  onClickNode={handleClickNode}
                />
              </div>
              {hasQKTracing && (
                <QKTracingSection
                  clickedNode={clickedNode}
                  nodesById={nodesById}
                  hoveredId={localHoveredId}
                  getNodeSupernodeAndOverrideLabel={getNodeSupernodeAndOverrideLabel}
                  onHoverNode={handleHoverNode}
                  onClearHover={handleClearHover}
                  onClickNode={handleClickNode}
                />
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
