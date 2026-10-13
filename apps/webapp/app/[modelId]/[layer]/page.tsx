import { getSource, getSourceSet } from '@/lib/db/source';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/db/user';
import { getLayerNumAsStringFromSource, getSourceSetNameFromSource } from '@/lib/utils/source';
import { SourceWithRelations } from '@/prisma/generated/zod';
import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import PageSource from './page-source';
import PageSourceSet from './page-sourceset';

export async function generateMetadata(props: {
  params: Promise<{ modelId: string; layer: string }>;
}): Promise<Metadata> {
  const params = await props.params;
  let title = `${params.modelId.toUpperCase()} · ${params.layer.toUpperCase()}`;
  let description = '';

  const sourceSet = await getSourceSet(params.modelId, params.layer);
  if (sourceSet) {
    title = sourceSet.description;
    description = `${sourceSet?.modelId.toUpperCase()} · ${sourceSet?.type} · ${sourceSet?.creatorName}`;
  } else {
    const layer = getLayerNumAsStringFromSource(params.layer);
    const source = getSourceSetNameFromSource(params.layer);
    const sourceSetFromSource = await getSourceSet(params.modelId, source);
    if (sourceSetFromSource) {
      title = `${params.modelId.toUpperCase()} · ${params.layer.toUpperCase()}`;
      description = `Layer ${layer} in Source Set ${source.toUpperCase()} by ${sourceSetFromSource?.creatorName}`;
    } else {
      // neither source nor sourceset
      title = '';
      description = '';
    }
  }

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      url: `/${params.modelId}/${params.layer}`,
    },
  };
}

export default async function Page(props: {
  params: Promise<{ modelId: string; layer: string }>;
  searchParams: Promise<{ simMatrix?: string; simMatrixDemo?: string }>;
}) {
  const searchParams = await props.searchParams;
  const params = await props.params;
  // CASE: SourceSet
  // if it's a model + sourceSet (gpt2-small/res-jb)
  const sourceSet = await getSourceSet(params.modelId, params.layer, await makeAuthedUserFromSessionOrReturnNull());
  if (sourceSet) {
    return <PageSourceSet sourceSet={sourceSet} />;
  }

  // CASE: Source
  // if it's a model + source (gpt2-small/0-res-jb)
  const source = await getSource(params.modelId, params.layer, await makeAuthedUserFromSessionOrReturnNull());
  if (source) {
    return (
      <PageSource
        source={source as SourceWithRelations}
        simMatrix={searchParams.simMatrix}
        simMatrixDemo={searchParams.simMatrixDemo}
      />
    );
  }

  // else not found
  notFound();
}
