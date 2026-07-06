import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import { getGraphFeatureDisplayLocation } from '@/lib/utils/graph-feature-display';
import { ArrowUpRightFromSquare } from 'lucide-react';
import { CLTGraph, CLTGraphNode } from './graph-types';
import { graphModelHasNpDashboards } from './utils';

export default function GraphFeatureLink({
  selectedGraph,
  node,
}: {
  selectedGraph: CLTGraph | null;
  node: CLTGraphNode;
}) {
  const { setFeatureModalFeature, setFeatureModalOpen, getSource } = useGlobalContext();

  if (!selectedGraph) {
    return null;
  }

  // Bias leaves (and other non-feature source nodes) have no per-feature
  // dashboard or index, so the link would be meaningless.
  if (node.feature_type === 'bias' || node.feature_type === 'unknown') {
    return null;
  }

  const displayLocation = getGraphFeatureDisplayLocation(selectedGraph.metadata.scan, node, selectedGraph);
  return selectedGraph?.metadata.scan && graphModelHasNpDashboards(selectedGraph) && node.featureDetailNP ? (
    <div className="ml-1 flex flex-col gap-x-1 pl-0">
      <Button
        onClick={() => {
          if (node.featureDetailNP) {
            setFeatureModalFeature({
              ...node.featureDetailNP,
              source: getSource(node.featureDetailNP.modelId, node.featureDetailNP.layer),
            });
            setFeatureModalOpen(true);
          }
        }}
        className="flex min-w-[110px] shrink-0 flex-row items-center gap-x-1 whitespace-nowrap rounded-md bg-slate-200 px-[8px] py-[6px] text-[9px] font-medium leading-none text-slate-600 shadow-none hover:bg-sky-200 hover:text-sky-700 sm:mr-0 sm:px-2.5 sm:py-1.5 sm:text-[9px]"
      >
        {node.feature_type === 'lorsa' && <div className="rounded bg-slate-300 px-1 leading-normal">LORSA</div>}
        <div className="flex flex-col gap-y-[3px] font-mono font-medium">
          <div className="">LAYER {displayLocation.layer}</div>
          <div className="">INDEX {displayLocation.index}</div>
        </div>
        <ArrowUpRightFromSquare className="ml-0.5 h-3 w-3" />
      </Button>
    </div>
  ) : node.feature ? (
    <div className="text-xs">F# {node.feature}</div>
  ) : null;
}
