import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import ModelsDropdown from '@/components/nav/models-dropdown';
import BrowserPane from '@/components/panes/browser-pane/browser-pane';
import GraphModelPane from '@/components/panes/graph-model-pane';
import JumpToPane from '@/components/panes/jump-to-pane';
import ModelHeadMetricsPane from '@/components/panes/model-head-metrics-pane';
import ModelReleases from '@/components/panes/model-releases-pane';
import SearchExplanationsPane from '@/components/panes/search-explanations-pane';
import SearchInferenceModelPane from '@/components/panes/search-inference-model-pane';
import { BreadcrumbLink, BreadcrumbPage } from '@/components/shadcn/breadcrumbs';
import { getVisibilityBadge } from '@/components/visibility-badge';
import { prisma } from '@/lib/db';
import { SearchExplanationsType } from '@/lib/utils/general';
import { getFirstSourceForModel, getFirstSourceSetForModel } from '@/lib/utils/source';
import { ModelWithPartialRelations } from '@/prisma/generated/zod';
import { Visibility } from '@prisma/client';

export default async function PageModel({ model }: { model: ModelWithPartialRelations }) {
  const [graphMetadatas, modelHeadMetrics] = await Promise.all([
    prisma.graphMetadata.findMany({
      where: {
        modelId: model.id,
        isFeatured: true,
      },
    }),
    prisma.modelHeadMetrics.findMany({
      where: {
        modelId: model.id,
      },
      select: {
        layer: true,
        headIndex: true,
        inductionScore: true,
        prevTokenScore: true,
        patternEntropy: true,
        selfAttentionScore: true,
        qkDistance: true,
        qkDistanceVariance: true,
        activationHistogram: true,
        qkDistanceHistogram: true,
        topQueryTokens: true,
        topKeyTokens: true,
      },
      orderBy: {
        updatedAt: 'desc',
      },
    }),
  ]);

  const firstSourceSet = getFirstSourceSetForModel(model, Visibility.PUBLIC, false, false);
  const firstSource = getFirstSourceForModel(model, Visibility.PUBLIC, false, false);
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
        ]}
      />

      <div className="flex w-full flex-row items-center justify-center border-b border-slate-200 py-6">
        <div className="flex w-full max-w-screen-lg flex-row items-center justify-between">
          <div className="flex flex-col items-start">
            {model.visibility !== Visibility.PUBLIC && (
              <div className="pb-1">{getVisibilityBadge(model.visibility)}</div>
            )}
            <div className="text-3xl font-bold text-slate-900">{model.id}</div>
            <div className="mt-2 text-sm font-normal text-slate-500">{model.owner}</div>
          </div>
        </div>
      </div>

      <div className="mt-6 w-full max-w-screen-lg">
        <div className="flex w-full flex-col items-center justify-center">
          <ModelReleases model={model} onlyFeatured={false} includeUnlisted={false} />
        </div>
      </div>

      {modelHeadMetrics.length > 0 && (
        <div className="mt-6 hidden w-full max-w-screen-lg sm:block">
          <div className="flex w-full flex-col items-center justify-center">
            <ModelHeadMetricsPane modelId={model.id} metrics={modelHeadMetrics} />
          </div>
        </div>
      )}

      {graphMetadatas.length > 0 && (
        <div className="mt-6 w-full max-w-screen-lg">
          <GraphModelPane model={model} graphMetadatas={graphMetadatas} />
        </div>
      )}

      <div className="mt-6 flex w-full max-w-screen-lg flex-col items-start justify-center gap-x-3 gap-y-5 sm:flex-row">
        <JumpToPane
          defaultModelId={model.id}
          defaultSourceSetName={firstSourceSet?.name || ''}
          vertical
          filterToFeaturedReleases={false}
        />
        <SearchExplanationsPane
          initialModelId={model.id}
          initialSourceSetName={model.sourceSets && model.sourceSets.length > 0 ? model.sourceSets[0].name : ''}
          defaultTab={SearchExplanationsType.BY_MODEL}
          showTabs
        />
      </div>
      {firstSourceSet?.allowInferenceSearch && (
        <div className="mt-6 w-full max-w-screen-lg">
          <SearchInferenceModelPane model={model} />
        </div>
      )}

      <div className="mt-6 flex w-full max-w-screen-lg flex-col items-center text-slate-700 xl:max-w-screen-xl 2xl:max-w-screen-2xl">
        <BrowserPane
          modelId={model.id}
          sourceSet={firstSourceSet?.name || ''}
          layer={firstSource?.id || ''}
          showModel
        />
      </div>
    </div>
  );
}
