import FeatureDashboard from '@/app/[modelId]/[layer]/[index]/feature-dashboard';
import { CLTGraphNode } from '@/app/[modelId]/graph/utils';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/shadcn/hover-card';
import { LoadingSquare } from '@/components/svg/loading-square';
import { SteeredPositionIdentifier } from '@/lib/utils/graph';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { InfoCircledIcon } from '@radix-ui/react-icons';
import { useState } from 'react';

export default function MiniDashboardHoverCard({
  sourceId,
  nodeSteerIdentifier,
  node,
}: {
  sourceId: string;
  nodeSteerIdentifier: SteeredPositionIdentifier;
  node: CLTGraphNode;
}) {
  const { setFeatureModalFeature, setFeatureModalOpen } = useGlobalContext();
  const [hoveredFeature, setHoveredFeature] = useState<NeuronWithPartialRelations | undefined>();
  return (
    <HoverCard openDelay={0} closeDelay={0}>
      <HoverCardTrigger asChild>
        <Button
          onMouseEnter={() => {
            // if hoveredFeature has same source and index, don't load it again
            if (
              hoveredFeature &&
              hoveredFeature.layer === sourceId &&
              hoveredFeature.index === nodeSteerIdentifier.index.toString()
            ) {
              return;
            }

            // if it's not the same, reset it
            setHoveredFeature(undefined);

            fetch(`/api/feature/${nodeSteerIdentifier.modelId}/${sourceId}/${nodeSteerIdentifier.index}`, {
              method: 'GET',
              headers: { 'Content-Type': 'application/json' },
            })
              .then((response) => response.json())
              .then((n: NeuronWithPartialRelations) => {
                setHoveredFeature(n);
              })
              .catch((error) => {
                console.error(`error submitting getting rest of neuron: ${error}`);
              });
          }}
          onClick={() => {
            if (
              hoveredFeature &&
              hoveredFeature.layer === sourceId &&
              hoveredFeature.index === nodeSteerIdentifier.index.toString() &&
              hoveredFeature.activations
            ) {
              // if we already loaded this from hover, then just set it and open it
              setFeatureModalFeature(hoveredFeature);
            } else {
              setFeatureModalFeature({
                modelId: nodeSteerIdentifier.modelId,
                layer: sourceId,
                index: nodeSteerIdentifier.index.toString(),
              } as NeuronWithPartialRelations);
            }
            setFeatureModalOpen(true);
          }}
          className="group relative mr-0 flex h-7 min-w-[95px] shrink-0 flex-row items-center justify-start whitespace-nowrap rounded-md border border-transparent bg-slate-200 px-0 py-[6px] text-[8.5px] font-medium leading-none text-slate-500 shadow-none hover:border-sky-700 hover:bg-sky-200 hover:text-sky-700"
        >
          <div className="flex flex-col items-start justify-start gap-y-[2px] pl-[11px] text-left font-mono font-medium">
            <div className="">LAYER {nodeSteerIdentifier.layer}</div>
            <div className="">INDEX {nodeSteerIdentifier.index}</div>
          </div>
          <InfoCircledIcon className="absolute right-2 h-3 w-3 text-slate-500 group-hover:text-sky-700" />
        </Button>
      </HoverCardTrigger>
      <HoverCardContent className="h-[386px] max-h-[386px] min-h-[386px] w-[512px] min-w-[512px] max-w-[512px] overflow-y-hidden border-0 bg-white p-0">
        {hoveredFeature?.activations && hoveredFeature?.activations?.length > 0 ? (
          <div className="-mt-2 h-full w-full">
            <FeatureDashboard
              key={`${hoveredFeature?.modelId}-${hoveredFeature?.layer}-${hoveredFeature?.index}`}
              initialNeuron={hoveredFeature}
              embed
              forceMiniStats
              activationMarkerValue={node.activation}
            />
          </div>
        ) : (
          <div className="flex h-full w-full items-center justify-center rounded-lg border">
            <LoadingSquare className="h-6 w-6" />
          </div>
        )}
      </HoverCardContent>
    </HoverCard>
  );
}
