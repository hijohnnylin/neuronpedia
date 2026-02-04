import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { useGraphStateContext } from '@/components/provider/graph-state-provider';
import { Circle, Expand, Minimize2 } from 'lucide-react';
import { useEffect, useState } from 'react';
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

      // mlp reconstruction error = don't show
      if (clientCheckClaudeMode() && node.feature_type === 'mlp reconstruction error') {
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
            node.featureId === hoveredId ? 'z-20 outline-dotted outline-[3px] outline-[#f0f]' : ''
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
                className={`default-icon block fill-none ${node.feature_type === 'mlp reconstruction error' ? 'opacity-35' : ''} ${(node[linkProp]?.pctInput ?? 0) > 0.25 || (node[linkProp]?.pctInput ?? 0) < -0.25 ? 'stroke-white' : 'stroke-slate-800'} ${node.nodeId && visState.pinnedIds?.includes(node.nodeId) ? 'stroke-[1.7]' : 'stroke-[0.7]'}`}
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
              <Circle className="h-3.5 max-h-3.5 min-h-3.5 w-3.5 min-w-3.5 max-w-3.5 text-[#f0f]" />
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
              className={`mt-2 flex h-full w-full flex-1 flex-row gap-x-0 ${clickedNode?.featureDetailNP ? 'pb-0' : 'pb-0'}`}
            >
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
          )}
        </div>
      </div>
    </>
  );
}
