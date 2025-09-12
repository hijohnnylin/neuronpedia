import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import BrowserPane from '@/components/panes/browser-pane/browser-pane';
import JumpToPane from '@/components/panes/jump-to-pane';
import SAEEvalsPane from '@/components/panes/sae-evals-pane/sae-evals-pane';
import SearchExplanationsPane from '@/components/panes/search-explanations-pane';
import SearchInferenceSourcePane from '@/components/panes/search-inference-source-pane';
import SearchTopkByTokenPane from '@/components/panes/search-topk-by-token-pane';
import UmapPane from '@/components/panes/umap-pane';
import { BreadcrumbItem, BreadcrumbLink } from '@/components/shadcn/breadcrumbs';
import { getVisibilityBadge } from '@/components/visibility-badge';
import { env } from '@/lib/env';
import { getLayerNumAsStringFromSource } from '@/lib/utils/source';
import {
  EvalWithPartialRelations,
  SourceReleaseWithRelations,
  SourceWithPartialRelations,
  SourceWithRelations,
} from '@/prisma/generated/zod';
import { Visibility } from '@prisma/client';
import Link from 'next/link';
import SaeLensConfigPane from '../../../components/panes/saelens-config-pane';

export default function PageSource({ source }: { source: SourceWithRelations }) {
  const release = source.set && source.set.releases ? source.set.releases : null;

  return (
    <div className="flex w-full flex-col items-center pb-10">
      {release && (
        <BreadcrumbsComponent
          crumbsArray={[
            [
              ...(release
                ? [
                    <BreadcrumbLink key={0} href={`/${release?.name}`}>
                      {`${release?.creatorNameShort} · ${release?.descriptionShort}`}
                    </BreadcrumbLink>,
                  ]
                : []),
            ],
            <BreadcrumbLink href={`/${source.modelId}`} key={1}>
              {source.set?.model.displayName}
            </BreadcrumbLink>,
            source.hasDashboards ? (
              <BreadcrumbLink key={2} href={`/${source.modelId}/${source.set?.name}`}>
                {source.set?.type}
              </BreadcrumbLink>
            ) : (
              <BreadcrumbItem key={2}>{source.set?.type}</BreadcrumbItem>
            ),
            <BreadcrumbLink key={3} href={`/${source.modelId}/${source.id}`}>
              {source.id.toUpperCase()}
            </BreadcrumbLink>,
          ]}
        />
      )}

      <div className="flex w-full flex-row items-center justify-center border-b border-slate-200 py-6">
        <div className="flex w-full max-w-screen-lg flex-row items-center justify-between">
          <div className="flex flex-col items-start">
            {source.visibility !== Visibility.PUBLIC && (
              <div className="pb-1">{getVisibilityBadge(source.visibility)}</div>
            )}
            <div className="text-2xl font-bold text-slate-900">
              {source.modelId} · {source.id}
            </div>
            <div className="mt-2 text-sm font-normal text-slate-500">
              Source from{' '}
              <Link
                prefetch={false}
                href={`/${source.set?.releases?.name}`}
                className="text-sky-700 hover:text-sky-600 hover:underline"
              >
                {source.set?.releases?.name}
              </Link>{' '}
              ·{' '}
              {source.hasDashboards ? (
                <Link
                  prefetch={false}
                  href={`/${source.modelId}/${source.set?.name}`}
                  className="text-sky-700 hover:text-sky-600 hover:underline"
                >
                  {source.set?.type}
                </Link>
              ) : (
                <span className="">{source.set?.type} </span>
              )}{' '}
              · Layer {getLayerNumAsStringFromSource(source.id)}
            </div>
          </div>
          {source.hasDashboards && (
            <div className="flex flex-row justify-end gap-x-3">
              <JumpToPane
                defaultModelId={source.modelId}
                defaultSourceSetName={source.set?.name}
                defaultSourceId={source.id}
                vertical
                filterToFeaturedReleases={false}
                showRandomFeature={false}
                showTitleAndCard={false}
                showModel={false}
              />
            </div>
          )}
        </div>
      </div>

      <div className="flex w-full max-w-screen-xl flex-col items-center pb-5 pt-6 text-slate-700">
        <div className="flex w-full flex-col items-stretch justify-center gap-x-3 gap-y-6">
          <div className="flex w-full flex-col items-center justify-center">
            <div className="w-full max-w-screen-lg">
              <SaeLensConfigPane inSAEPage sae={source as SourceWithPartialRelations} />
            </div>
          </div>

          {source.inferenceEnabled && (
            <SearchInferenceSourcePane
              model={source.set?.model}
              sourceSetName={source.set?.name || ''}
              initialSelectedLayers={[source.id]}
            />
          )}

          {source.inferenceEnabled && (
            <div className="flex w-full items-center justify-center">
              <div className="flex w-full max-w-screen-lg">
                <SearchTopkByTokenPane modelId={source.modelId} source={source.id} showResultsInNewPage />
              </div>
            </div>
          )}

          <div className="flex w-full items-center justify-center">
            <div className="flex w-full max-w-screen-lg">
              <SearchExplanationsPane
                initialModelId={source.modelId}
                initialSourceSetName={source.set?.name || ''}
                initialSelectedLayers={[source.id]}
                filterToRelease={release as SourceReleaseWithRelations}
                defaultTab="bySource"
                showTabs={false}
              />
            </div>
          </div>

          {source.hasUmap && (
            <div className="w-full">
              <UmapPane
                showModel
                defaultModelId={source.modelId}
                defaultSourceSet={source.set?.name || ''}
                defaultLayer={source.id || ''}
                release={release as SourceReleaseWithRelations}
                filterToRelease={release?.name || ''}
                releaseMultipleUmapSAEs={release?.defaultUmapSourceIds || []}
                newWindowOnSaeChange={false}
              />
            </div>
          )}

          {source.hasDashboards && (
            <BrowserPane
              modelId={source.modelId}
              sourceSet={source.set?.name || ''}
              layer={source.id || ''}
              showModel
              filterToRelease={release?.name || ''}
            />
          )}

          {!env.DEMO_MODE && (
            <SAEEvalsPane
              sae={source as SourceWithPartialRelations}
              evals={source.evals as EvalWithPartialRelations[]}
            />
          )}
        </div>
      </div>
    </div>
  );
}
