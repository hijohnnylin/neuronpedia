'use client';

import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import ModelsDropdown from '@/components/nav/models-dropdown';
import ModelHeadMetricsPane, { ModelHeadMetricsRow } from '@/components/panes/model-head-metrics-pane';
import { BreadcrumbItem, BreadcrumbLink, BreadcrumbPage } from '@/components/shadcn/breadcrumbs';
import { useState } from 'react';

// Client wrapper so the breadcrumbs reflect the live head selection. The pane updates the URL
// client-side (without navigating) for same-model selections, so it reports the selected head via
// onHeadChange and we mirror it into the breadcrumbs here.
export default function HeadPageContent({
  modelId,
  modelDisplayName,
  initialLayer,
  initialHeadIndex,
  metrics,
  inferenceEnabled = false,
}: {
  modelId: string;
  modelDisplayName: string;
  initialLayer: number;
  initialHeadIndex: number;
  metrics: ModelHeadMetricsRow[];
  inferenceEnabled?: boolean;
}) {
  const [head, setHead] = useState({ layer: initialLayer, headIndex: initialHeadIndex });

  return (
    <div className="flex w-full flex-col items-center pb-10">
      <BreadcrumbsComponent
        crumbsArray={[
          <BreadcrumbPage key={0}>
            <ModelsDropdown isInBreadcrumb />
          </BreadcrumbPage>,
          <BreadcrumbLink href={`/${modelId}`} key={1}>
            {modelDisplayName}
          </BreadcrumbLink>,
          <BreadcrumbItem key={2}>Attention Heads - HeadVis</BreadcrumbItem>,
          <BreadcrumbItem key={3}>Layer {head.layer}</BreadcrumbItem>,
          <BreadcrumbLink href={`/${modelId}/head/${head.layer}/${head.headIndex}`} key={4}>
            Head {head.headIndex}
          </BreadcrumbLink>,
        ]}
      />
      <div className="mt-2 w-full">
        <div className="flex w-full flex-col items-center justify-center">
          <ModelHeadMetricsPane
            modelId={modelId}
            metrics={metrics}
            showCard={false}
            initialLayer={initialLayer}
            initialHeadIndex={initialHeadIndex}
            onHeadChange={({ layer, headIndex }) => setHead({ layer, headIndex })}
            inferenceEnabled={inferenceEnabled}
          />
        </div>
      </div>
    </div>
  );
}
