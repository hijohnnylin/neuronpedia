import { prisma } from '@/lib/db';
import { getModelByIdWithSourceSets } from '@/lib/db/model';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/db/user';
import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import HeadPageContent from './head-page-content';

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
      selfAttentionScore: true,
    },
    orderBy: {
      updatedAt: 'desc',
    },
  });

  if (modelHeadMetrics.length === 0) {
    notFound();
  }

  return (
    <HeadPageContent
      modelId={model.id}
      modelDisplayName={model.displayName ?? model.id}
      initialLayer={layer}
      initialHeadIndex={headIndex}
      metrics={modelHeadMetrics}
    />
  );
}
