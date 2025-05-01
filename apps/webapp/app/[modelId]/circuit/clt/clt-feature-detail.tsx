import { useCircuitCLT } from '@/components/provider/circuit-clt-provider';
import { Circle } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { GroupedVirtuoso, GroupedVirtuosoHandle } from 'react-virtuoso';
import CLTFeatureDetailItem from './clt-feature-detail-item';
import { CLTGraphNode } from './clt-utils';

export default function CLTFeatureDetail() {
  const { visState, selectedGraph } = useCircuitCLT();
  const [node, setNode] = useState<CLTGraphNode | null>(null);
  const [overallMaxActivationValue, setOverallMaxActivationValue] = useState<number>(0);
  const activationContainerRef = useRef<HTMLDivElement>(null);
  const groupRef = useRef<GroupedVirtuosoHandle>(null);

  useEffect(() => {
    const clickedNode = visState.clickedId
      ? selectedGraph?.nodes.find((e) => e.nodeId === visState.clickedId) || null
      : null;

    let maxActValue = Math.max(
      ...(clickedNode?.featureDetail?.examples_quantiles[0].examples.flatMap((e) => e.tokens_acts_list) || []),
    );

    if (visState.hoveredId) {
      const hoveredNode = selectedGraph?.nodes.find((e) => e.featureId === visState.hoveredId);
      if (hoveredNode && hoveredNode.feature) {
        setNode(hoveredNode);
        if (hoveredNode.featureDetail) {
          maxActValue = Math.max(
            ...hoveredNode.featureDetail.examples_quantiles[0].examples.flatMap((e) => e.tokens_acts_list),
          );
        }
      } else {
        setNode(clickedNode);
      }
    } else {
      setNode(clickedNode);
    }

    setOverallMaxActivationValue(maxActValue);
  }, [visState.hoveredId, visState.clickedId, selectedGraph]);

  // Separate useEffect for scrolling when 'feature' changes
  useEffect(() => {
    groupRef.current?.scrollToIndex(0);
  }, [node]);

  const groupCounts = useMemo(() => {
    if (!node) return [];
    return node.featureDetail?.examples_quantiles.map((e) => e.examples.length) || [];
  }, [node]);

  function getIndexInGroup(index: number, groupIndex: number) {
    return index - groupCounts.slice(0, groupIndex).reduce((acc, curr) => acc + curr, 0);
  }

  // Memoize the feature detail content to prevent unnecessary re-rendering
  const memoizedFeatureDetail = useMemo(() => {
    if (!node) return null;

    return (
      <>
        <div className="mb-2 flex flex-row items-center gap-x-2 text-sm font-medium text-slate-600">
          <div className="">F#{node.feature}</div>
          <Circle className="h-3.5 w-3.5 text-[#f0f]" />
          <div>{node.ppClerp}</div>
        </div>
        {node.featureDetail && (
          <>
            <div className="mb-1.5 border-b pb-1 text-sm font-bold text-slate-600">Token Predictions</div>
            <div className="flex h-5 w-full items-center justify-start gap-x-1 gap-y-0.5 overflow-x-scroll font-mono text-[10px] text-slate-400">
              <div className="sticky left-0 mr-1 flex h-5 items-center justify-center bg-white">Top:</div>
              {node.featureDetail?.top_logits.map((logit, idx) => (
                <span key={idx} className="cursor-default rounded bg-slate-100 px-1 py-[1px] text-slate-700">
                  {logit}
                </span>
              ))}
            </div>
            <div className="flex h-5 w-full items-center justify-start gap-x-1 gap-y-0.5 overflow-x-scroll font-mono text-[10px] text-slate-400">
              <div className="sticky left-0 mr-1 flex h-5 items-center justify-center bg-white">Bottom:</div>
              {node?.featureDetail?.bottom_logits.map((logit, idx) => (
                <span key={idx} className="cursor-default rounded bg-slate-100 px-1 py-[1px] text-slate-700">
                  {logit}
                </span>
              ))}
            </div>
            <div
              ref={activationContainerRef}
              className="flex max-h-[320px] w-full flex-col overflow-y-scroll overscroll-none"
            >
              <GroupedVirtuoso
                ref={groupRef}
                className="min-h-[320px] w-full"
                groupCounts={groupCounts}
                // eslint-disable-next-line react/no-unstable-nested-components
                groupContent={(index) => (
                  <div className="h-8 border-b bg-white pb-1 pt-2 text-sm font-bold text-slate-600">
                    {node?.featureDetail?.examples_quantiles[index].quantile_name}
                  </div>
                )}
                // eslint-disable-next-line react/no-unstable-nested-components
                itemContent={(index, groupIndex) => {
                  const example =
                    node?.featureDetail?.examples_quantiles[groupIndex]?.examples[getIndexInGroup(index, groupIndex)];

                  return (
                    <CLTFeatureDetailItem
                      example={example}
                      overallMaxActivationValue={overallMaxActivationValue}
                      itemKey={index}
                    />
                  );
                }}
              />
            </div>
          </>
        )}
      </>
    );
  }, [node, overallMaxActivationValue]);

  return <div className="flex min-h-[400px] w-full flex-1 flex-col gap-y-1">{memoizedFeatureDetail}</div>;
}
