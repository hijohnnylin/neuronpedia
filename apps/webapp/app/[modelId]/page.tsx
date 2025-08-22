/* eslint-disable no-nested-ternary */

import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import ModelsDropdown from '@/components/nav/models-dropdown';
import SearchInferenceModelPane from '@/components/panes/search-inference-model-pane';
import { BreadcrumbItem, BreadcrumbLink, BreadcrumbPage } from '@/components/shadcn/breadcrumbs';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/auth/utils';
import { getModelById, getModelByIdWithSourceSets } from '@/lib/db/model';
import { getSourceRelease } from '@/lib/db/source';
import { getLayerNumAsStringFromSource } from '@/lib/utils/source';
import { SourceReleaseWithRelations } from '@/prisma/generated/zod';
import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import GemmaScopeHome from './gemmascope/home';
import PageModel from './page-model';
import PageRelease from './page-release';

export async function generateMetadata({
  params,
  searchParams,
}: {
  params: { modelId: string };
  searchParams: {
    q?: string;
    selectedLayers?: string;
    filter?: string;
    sortIndex?: string;
    sortIndexes?: string;
    sourceSet?: string;
    ignoreBos?: string;
  };
}): Promise<Metadata> {
  if (params.modelId === 'gemma-scope') {
    const title = 'Gemma Scope';
    const description = 'Exploring the Inner Workings of Gemma 2 2B';
    return {
      title,
      description,
      openGraph: {
        title,
        description,
        url: `/${params.modelId}`,
        images: [{ url: 'https://neuronpedia.org/gs-og.png' }],
      },
    };
  }

  let title = `${params.modelId.toUpperCase()}`;
  let description = '';

  const model = await getModelById(params.modelId);
  if (model) {
    if (searchParams?.q) {
      title = `🔍 ${searchParams.q}`;
      description =
        params.modelId.toUpperCase() +
        (searchParams?.sourceSet ? ` · ${searchParams.sourceSet.toUpperCase()}` : '') +
        (searchParams?.selectedLayers && JSON.parse(searchParams?.selectedLayers).length > 0
          ? ` · Layer${JSON.parse(searchParams?.selectedLayers).length !== 1 ? 's' : ''}${JSON.parse(
              searchParams.selectedLayers,
            ).map((l: string) => ` ${getLayerNumAsStringFromSource(l)}`)}`
          : ' · All Layers');
    }
  } else {
    const release = await getSourceRelease(params.modelId);
    if (release) {
      title = `${release.description}`;
      description = release.creatorName;
    } else {
      // neither model nor release
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
      url: `/${params.modelId}`,
    },
  };
}

export default async function Page({
  params,
  searchParams,
}: {
  params: { modelId: string };
  searchParams: {
    q?: string;
    selectedLayers?: string;
    filter?: string;
    sortIndex?: string;
    sortIndexes?: string;
    sourceSet?: string;
    ignoreBos?: string;
  };
}) {
  const { modelId } = params;
  const model = await getModelByIdWithSourceSets(modelId, await makeAuthedUserFromSessionOrReturnNull());

  if (model) {
    // CASE: Search via Inference results
    if (searchParams?.q) {
      return (
        <div className="flex w-full flex-col items-center gap-y-6">
          <BreadcrumbsComponent
            crumbsArray={[
              <BreadcrumbPage key={0}>
                <ModelsDropdown isInBreadcrumb />
              </BreadcrumbPage>,
              <BreadcrumbLink href={`/${model.id}`} key={1}>
                {model.displayName}
              </BreadcrumbLink>,
              <BreadcrumbItem key={2}>Search via Inference</BreadcrumbItem>,
            ]}
          />
          <SearchInferenceModelPane model={model} searchParams={searchParams} />
        </div>
      );
    }
    // CASE: Model page
    return <PageModel model={model} />;
  }

  const release = await getSourceRelease(modelId, await makeAuthedUserFromSessionOrReturnNull());
  if (release) {
    // CASE: Gemma Scope
    if (release.isNewUi) {
      return <GemmaScopeHome release={release} />;
    }
    // CASE: Release page
    return <PageRelease release={release as SourceReleaseWithRelations} />;
  }

  // CASE: Not a model, not a release
  notFound();
}
