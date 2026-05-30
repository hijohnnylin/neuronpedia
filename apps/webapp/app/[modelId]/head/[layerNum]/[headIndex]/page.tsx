import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import ModelsDropdown from '@/components/nav/models-dropdown';
import ModelHeadMetricsPane from '@/components/panes/model-head-metrics-pane';
import { BreadcrumbLink, BreadcrumbPage } from '@/components/shadcn/breadcrumbs';
import { prisma } from '@/lib/db';
import { getModelByIdWithSourceSets } from '@/lib/db/model';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/db/user';
import { Metadata } from 'next';
import { notFound } from 'next/navigation';

type PageParams = { modelId: string; layerNum: string; headIndex: string };

export async function generateMetadata(props: { params: Promise<PageParams> }): Promise<Metadata> {
  const params = await props.params;
  const title = `${params.modelId.toUpperCase()} · Layer ${params.layerNum} Head ${params.headIndex}`;
  return {
    title,
    openGraph: {
      title,
      url: `/${params.modelId}/head/${params.layerNum}/${params.headIndex}`,
    },
  };
}

export default async function Page(props: { params: Promise<PageParams> }) {
  const params = await props.params;
  const { modelId } = params;
  const layer = Number(params.layerNum);
  const headIndex = Number(params.headIndex);

  if (!Number.isInteger(layer) || !Number.isInteger(headIndex) || layer < 0 || headIndex < 0) {
    notFound();
  }

  const model = await getModelByIdWithSourceSets(modelId, await makeAuthedUserFromSessionOrReturnNull());
  if (!model) {
    notFound();
  }

  const modelHeadMetrics = await prisma.modelHeadMetrics.findMany({
    where: {
      modelId: model.id,
    },
    // Mirror the model page: only the lightweight, always-displayed metrics are loaded up
    // front. Heavy per-head detail is fetched on click via /api/model/head-metrics/get.
    select: {
      layer: true,
      headIndex: true,
      inductionScore: true,
      prevTokenScore: true,
      patternEntropy: true,
    },
    orderBy: {
      updatedAt: 'desc',
    },
  });

  if (modelHeadMetrics.length === 0) {
    notFound();
  }

  return (
    <div className="flex w-full flex-col items-center pb-10">
      <BreadcrumbsComponent
        crumbsArray={[
          <BreadcrumbPage key={0}>
            <ModelsDropdown isInBreadcrumb />
          </BreadcrumbPage>,
          <BreadcrumbLink href={`/${model.id}`} key={1}>
            {model.displayName}
          </BreadcrumbLink>,
          <BreadcrumbLink href={`/${model.id}/head`} key={2}>
            Attention Heads - HeadVis
          </BreadcrumbLink>,
          <BreadcrumbLink href={`/${model.id}/head/${layer}`} key={3}>
            Layer {layer}
          </BreadcrumbLink>,
          <BreadcrumbLink href={`/${model.id}/head/${layer}/${headIndex}`} key={4}>
            Head {headIndex}
          </BreadcrumbLink>,
        ]}
      />
      <div className="mt-2 w-full">
        <div className="flex w-full flex-col items-center justify-center">
          <ModelHeadMetricsPane
            modelId={model.id}
            metrics={modelHeadMetrics}
            showCard={false}
            initialLayer={layer}
            initialHeadIndex={headIndex}
          />
        </div>
      </div>
    </div>
  );
}
