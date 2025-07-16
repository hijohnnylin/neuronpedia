import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { useGraphStateContext } from '@/components/provider/graph-state-provider';
import { Button } from '@/components/shadcn/button';
import { Circle } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { GroupedVirtuoso } from 'react-virtuoso';
import FeatureDashboard from '../[layer]/[index]/feature-dashboard';
import GraphFeatureDetailItem from './feature-detail-item';
import GraphFeatureLink from './np-feature-link';
import { clientCheckIsEmbed, CLTGraphNode, nodeTypeHasFeatureDetail } from './utils';

export default function GraphFeatureDetail() {
  const {
    visState,
    selectedGraph,
    updateVisStateField,
    isEditingLabel,
    setIsEditingLabel,
    getOverrideClerpForNode,
    getNodeSupernodeAndOverrideLabel,
    getOriginalClerpForNode,
    setFullNPFeatureDetail,
  } = useGraphContext();

  const { hoveredIdRef, clickedIdRef, registerHoverCallback, registerClickedCallback } = useGraphStateContext();

  const [node, setNode] = useState<CLTGraphNode | null>(null);
  const [overallMaxActivationValue, setOverallMaxActivationValue] = useState<number>(1);
  const [tempLabel, setTempLabel] = useState('');
  const activationContainerRef = useRef<HTMLDivElement>(null);
  const groupRef = useRef<any>(null);

  const { openWelcomeModalToStep } = useGraphModalContext();

  // Track current displayed node to avoid unnecessary updates
  const currentDisplayedNodeRef = useRef<string | null>(null);

  // Register for both hover and clicked changes and update node display
  useEffect(() => {
    const updateDisplayedNode = () => {
      // Determine which node to show: hovered node takes priority over clicked
      const nodeIdToShow = hoveredIdRef.current || clickedIdRef.current;

      // Only update if the node actually changed
      if (currentDisplayedNodeRef.current === nodeIdToShow) return;

      currentDisplayedNodeRef.current = nodeIdToShow;

      if (nodeIdToShow && selectedGraph) {
        const targetNode = selectedGraph.nodes.find((e) => e.nodeId === nodeIdToShow || e.featureId === nodeIdToShow);
        if (targetNode) {
          // Get the max activation value across all nodes for scaling
          const maxActValue = Math.max(
            1,
            ...selectedGraph.nodes.map((n) => (n.activation !== undefined ? n.activation : 0)),
          );
          setNode(targetNode);
          setOverallMaxActivationValue(maxActValue);
        }
      } else {
        setNode(null);
      }
    };

    const unregisterHover = registerHoverCallback(() => {
      updateDisplayedNode();
    });

    const unregisterClicked = registerClickedCallback(() => {
      updateDisplayedNode();
    });

    // Initial call to set the current state
    updateDisplayedNode();

    return () => {
      unregisterHover();
      unregisterClicked();
    };
  }, [registerHoverCallback, registerClickedCallback, selectedGraph, clickedIdRef, hoveredIdRef]);

  // Separate useEffect for scrolling when 'feature' changes
  useEffect(() => {
    groupRef.current?.scrollToIndex(0);
    // console.log('node set with activations:', node?.featureDetailNP?.activations?.length);
    if (node) {
      setFullNPFeatureDetail(setNode, node);
    }
  }, [node]);

  const groupCounts = useMemo(() => {
    if (!node) return [];
    return node.featureDetail?.examples_quantiles.map((e) => e.examples.length) || [];
  }, [node]);

  function getIndexInGroup(index: number, groupIndex: number) {
    return index - groupCounts.slice(0, groupIndex).reduce((acc, curr) => acc + curr, 0);
  }

  function saveLabelToVisState() {
    if (visState && node && node.featureId) {
      // check if the node already has an overrideClerp
      const existingOverrideClerp = visState.clerps?.find((e) => e[0] === node.featureId);
      if (existingOverrideClerp) {
        updateVisStateField(
          'clerps',
          visState.clerps?.map((e) => (e[0] === node.featureId ? [node.featureId, tempLabel] : e)),
        );
      } else {
        updateVisStateField('clerps', [...(visState.clerps || []), [node.featureId, tempLabel]]);
      }
    }
    if (tempLabel.length === 0) {
      updateVisStateField(
        'clerps',
        (visState.clerps || []).filter((e) => e[0] !== node?.featureId),
      );
    }
  }

  // Memoize the feature detail content to prevent unnecessary re-rendering
  const memoizedFeatureDetail = useMemo(() => {
    if (!node)
      return (
        <div className="relative hidden h-[100%] flex-col items-center justify-center text-center text-sm font-medium text-slate-700 sm:flex">
          <div className="mb-2 text-lg font-bold">Feature Details</div>
          <div className="">Hover over a node in the graph to see its details and edit its label.</div>
          {!clientCheckIsEmbed() && (
            <button
              type="button"
              onClick={() => openWelcomeModalToStep(5)}
              className="absolute right-0 top-1 flex h-[24px] w-[24px] items-center justify-center gap-x-1 rounded-full bg-slate-200 py-0.5 text-[12px] font-medium transition-colors hover:bg-slate-300"
              aria-label="Open User Guide"
            >
              ?
            </button>
          )}
        </div>
      );

    return (
      <div
        className={`max-h-full w-full flex-col overflow-y-scroll sm:flex ${
          clickedIdRef.current ? 'absolute left-0 top-0 z-10 flex sm:relative' : ''
        }`}
      >
        <div className="flex flex-row items-center justify-between gap-x-1.5 py-1 pl-3 pt-1 text-sm font-medium text-slate-600 sm:py-0">
          <div className="flex flex-1 flex-row items-center gap-x-2">
            <Circle className="h-3.5 max-h-3.5 min-h-3.5 w-3.5 min-w-3.5 max-w-3.5 text-[#f0f]" />
            {isEditingLabel ? (
              <div className="flex flex-1 flex-col items-start justify-start gap-y-1">
                <input
                  type="text"
                  placeholder="Custom feature label"
                  value={tempLabel}
                  onChange={(e) => {
                    setTempLabel(e.target.value);
                  }}
                  // eslint-disable-next-line jsx-a11y/no-autofocus
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.code === 'Enter' || e.key === 'Enter') {
                      saveLabelToVisState();
                      setIsEditingLabel(false);
                      e.preventDefault();
                    }
                  }}
                  className="temp-edit-feat-detail w-full flex-1 rounded border border-slate-300 px-2 py-0 text-xs placeholder:text-slate-300"
                />
                {getOriginalClerpForNode(node) && (
                  <div className="text-xs">
                    <span className="rounded bg-slate-200 px-1 py-0.5 text-[8px] font-bold text-slate-600">
                      ORIGINAL
                    </span>{' '}
                    {getOriginalClerpForNode(node)}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-start justify-start text-[10px] leading-snug sm:text-xs sm:leading-normal">
                {getNodeSupernodeAndOverrideLabel(node)}
                {node.activation && (
                  <span className="text-[9px] font-medium uppercase text-slate-400">
                    Activation: {node.activation.toFixed(2)}
                  </span>
                )}
              </div>
            )}
          </div>
          <div className="flex flex-row gap-x-1">
            {nodeTypeHasFeatureDetail(node) && (
              <Button
                variant="outline"
                size="xs"
                className="w-18 hidden sm:inline-flex"
                onClick={() => {
                  if (isEditingLabel) {
                    if (selectedGraph) {
                      saveLabelToVisState();
                      setIsEditingLabel(false);
                    }
                  } else {
                    setTempLabel(getOverrideClerpForNode(node) || '');
                    setIsEditingLabel(true);
                  }
                }}
              >
                {isEditingLabel ? 'Save' : 'Edit Label'}
              </Button>
            )}
            {isEditingLabel && (
              <Button
                variant="outline"
                size="xs"
                className="w-18"
                onClick={() => {
                  updateVisStateField(
                    'clerps',
                    (visState.clerps || []).filter((e) => e[0] !== node?.featureId),
                  );
                  setTempLabel('');
                  setIsEditingLabel(false);
                }}
              >
                Original
              </Button>
            )}
            {isEditingLabel && (
              <Button
                variant="outline"
                size="xs"
                className="w-18"
                onClick={() => {
                  setTempLabel(node?.ppClerp || '');
                  setIsEditingLabel(false);
                }}
              >
                Cancel
              </Button>
            )}
          </div>
          {!(node.feature_type === 'embedding' || node.feature_type === 'logit') && (
            <GraphFeatureLink selectedGraph={selectedGraph} node={node} />
          )}

          {!clientCheckIsEmbed() && (
            <button
              type="button"
              onClick={() => openWelcomeModalToStep(5)}
              className="hidden h-[24px] w-[24px] items-center justify-center gap-x-1 self-start rounded-full bg-slate-200 py-0.5 text-[12px] font-medium transition-colors hover:bg-slate-300 sm:flex"
              aria-label="Open User Guide"
            >
              ?
            </button>
          )}
        </div>
        {node.featureDetailNP ? (
          <div className="ml-3 flex flex-1 overflow-y-scroll overscroll-y-none rounded-b-md border-b border-slate-200">
            <FeatureDashboard
              forceMiniStats
              key={`${node.featureDetailNP.index}-${node.featureDetailNP.activations?.length || 0}`}
              initialNeuron={node.featureDetailNP}
              activationMarkerValue={node.activation}
              embed
            />
          </div>
        ) : (
          node.featureDetail && (
            <div className="pl-4 pt-1">
              {node.featureDetail?.top_logits && node.featureDetail?.bottom_logits && (
                <>
                  <div className="mb-1.5 mt-2 border-b pb-1 text-sm font-bold text-slate-600">Token Predictions</div>
                  <div className="flex h-5 w-full items-center justify-start gap-x-1 gap-y-0.5 overflow-x-scroll font-mono text-[10px] text-slate-400">
                    <div className="sticky left-0 mr-1 flex h-5 items-center justify-center">Top:</div>
                    {node.featureDetail?.top_logits?.map((logit, idx) => (
                      <span key={idx} className="cursor-default rounded bg-slate-100 px-1 py-[1px] text-slate-700">
                        {logit}
                      </span>
                    ))}
                  </div>
                  <div className="flex h-5 w-full items-center justify-start gap-x-1 gap-y-0.5 overflow-x-scroll font-mono text-[10px] text-slate-400">
                    <div className="sticky left-0 mr-1 flex h-5 items-center justify-center">Bottom:</div>
                    {node?.featureDetail?.bottom_logits?.map((logit, idx) => (
                      <span key={idx} className="cursor-default rounded bg-slate-100 px-1 py-[1px] text-slate-700">
                        {logit}
                      </span>
                    ))}
                  </div>
                </>
              )}
              <div
                ref={activationContainerRef}
                className="flex max-h-[310px] w-full flex-col overflow-y-scroll overscroll-none"
              >
                <GroupedVirtuoso
                  ref={groupRef}
                  className="min-h-[310px] w-full"
                  groupCounts={groupCounts}
                  // eslint-disable-next-line react/no-unstable-nested-components
                  groupContent={(index) => (
                    <div className="h-8 border-b bg-slate-50 pb-1 pt-2 text-sm font-bold text-slate-600">
                      {node?.featureDetail?.examples_quantiles[index].quantile_name}
                    </div>
                  )}
                  // eslint-disable-next-line react/no-unstable-nested-components
                  itemContent={(index, groupIndex) => {
                    const example =
                      node?.featureDetail?.examples_quantiles[groupIndex]?.examples[getIndexInGroup(index, groupIndex)];

                    return (
                      <GraphFeatureDetailItem
                        example={example}
                        overallMaxActivationValue={overallMaxActivationValue}
                        itemKey={index}
                      />
                    );
                  }}
                />
              </div>
            </div>
          )
        )}
      </div>
    );
  }, [node, node?.featureDetailNP?.activations, overallMaxActivationValue, isEditingLabel, tempLabel, visState.clerps]);

  return <div className="flex h-full w-full flex-1 flex-col overflow-y-scroll">{memoizedFeatureDetail}</div>;
}
