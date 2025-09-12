'use client';

import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import BrowserPane from '@/components/panes/browser-pane/browser-pane';
import JumpToPane from '@/components/panes/jump-to-pane';
import SearchExplanationsPane from '@/components/panes/search-explanations-pane';
import SearchInferenceSourcePane from '@/components/panes/search-inference-source-pane';
import SearchTopkByTokenPane from '@/components/panes/search-topk-by-token-pane';
import UmapPane from '@/components/panes/umap-pane';
import { BreadcrumbItem, BreadcrumbLink } from '@/components/shadcn/breadcrumbs';
import { getVisibilityBadge } from '@/components/visibility-badge';
import { Visibility } from '@prisma/client';
import Link from 'next/link';
import {
  ModelWithPartialRelations,
  SourceReleaseWithRelations,
  SourceSetWithPartialRelations,
} from 'prisma/generated/zod';

export default function PageSourceSet({ sourceSet }: { sourceSet: SourceSetWithPartialRelations }) {
  let defaultSourceId: string | undefined;
  if (sourceSet.releases?.defaultSourceId) {
    defaultSourceId = sourceSet.releases?.defaultSourceId;
  } else {
    sourceSet.sources?.every((s) => {
      if (s.hasUmap) {
        defaultSourceId = s.id;
        return false;
      }
      return true;
    });
  }
  if (!defaultSourceId) {
    defaultSourceId = sourceSet.sources?.sort((a, b) => a?.id?.localeCompare(b?.id || '') || 0)[0]?.id;
  }

  return (
    <div className="flex w-full flex-col items-center pb-10">
      {sourceSet.releaseName && (
        <BreadcrumbsComponent
          crumbsArray={[
            [
              <BreadcrumbLink key={0} href={`/${sourceSet.releaseName}`}>
                {`${sourceSet.creatorName} · ${sourceSet.description}`}
              </BreadcrumbLink>,
            ],
            <BreadcrumbLink href={`/${sourceSet.modelId}`} key={1}>
              {sourceSet.model?.displayName}
            </BreadcrumbLink>,
            <BreadcrumbItem key={2}>{sourceSet.name.toUpperCase()}</BreadcrumbItem>,
          ]}
        />
      )}

      <div className="flex w-full flex-row items-center justify-center border-b border-slate-200 py-6">
        <div className="flex w-full max-w-screen-lg flex-col items-center justify-between gap-y-3 sm:flex-row">
          <div className="flex flex-col items-start">
            {sourceSet.visibility !== Visibility.PUBLIC && (
              <div className="pb-1">{getVisibilityBadge(sourceSet.visibility)}</div>
            )}
            <div className="text-lg font-bold text-slate-900 sm:text-2xl">
              {sourceSet.modelId} · {sourceSet.name}
            </div>
            <div className="mt-2 flex flex-row items-center gap-x-2 text-sm font-normal text-slate-500">
              Source Set from{' '}
              <Link
                prefetch={false}
                href={`/${sourceSet?.releases?.name}`}
                className="text-sky-700 hover:text-sky-600 hover:underline"
              >
                {sourceSet?.releases?.name}
              </Link>{' '}
              · {sourceSet.creatorName}
              {sourceSet.creatorEmail && (
                <>
                  {' · '}
                  <Link
                    className="text-sky-700 hover:text-sky-600 hover:underline"
                    href={`mailto:${
                      sourceSet.creatorEmail
                    }?subject=SAE%20Detail%20Request%3A%20${sourceSet.name.toUpperCase()}&body=I'd%20like%20to%20contact%20the%20researcher%20who%20created%20the%20${sourceSet.name.toUpperCase()}%20SAEs.%20Please%20put%20me%20in%20touch%20-%20thanks!`}
                  >
                    Admin Support Contact
                  </Link>
                </>
              )}
            </div>
            {sourceSet.urls && sourceSet.urls.length > 0 && (
              <div className="mt-2 flex flex-row items-center justify-center gap-x-1.5">
                {sourceSet.urls
                  .filter((url) => url.length > 0)
                  .map((url, i) => (
                    <a
                      // eslint-disable-next-line react/no-array-index-key
                      key={i}
                      href={url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-block cursor-pointer break-all rounded-full border-slate-200 bg-slate-200/70 px-3.5 py-1 font-sans text-[11px] text-slate-500 hover:bg-slate-300"
                    >
                      {new URL(url).hostname.replace('www.', '')} ↗
                    </a>
                  ))}
              </div>
            )}
          </div>
          <div className="flex flex-row justify-end gap-x-3">
            <JumpToPane
              defaultModelId={sourceSet.modelId}
              defaultSourceSetName={sourceSet.name}
              defaultSourceId={defaultSourceId}
              vertical
              filterToFeaturedReleases={false}
              showRandomFeature={false}
              showTitleAndCard={false}
              showModel={false}
            />
          </div>
        </div>
      </div>

      <div className="flex w-full max-w-screen-xl flex-col items-center pb-5 pt-6 text-slate-700">
        <div className="mb-6 flex w-full flex-col items-stretch gap-x-3 gap-y-6">
          {sourceSet.allowInferenceSearch && sourceSet.model && (
            <SearchInferenceSourcePane
              model={sourceSet.model as ModelWithPartialRelations}
              sourceSetName={sourceSet.name}
            />
          )}

          {sourceSet.allowInferenceSearch && sourceSet.model && (
            <div className="flex w-full items-center justify-center">
              <div className="flex w-full max-w-screen-lg">
                <SearchTopkByTokenPane
                  modelId={sourceSet.modelId}
                  source={defaultSourceId || ''}
                  showResultsInNewPage
                />
              </div>
            </div>
          )}

          <div className="flex w-full items-center justify-center">
            <div className="flex w-full max-w-screen-lg">
              <SearchExplanationsPane
                initialModelId={sourceSet.modelId}
                initialSourceSetName={sourceSet.name}
                initialSelectedLayers={[defaultSourceId || '']}
                filterToRelease={sourceSet.releases as SourceReleaseWithRelations}
                defaultTab="bySource"
                showTabs={false}
              />
            </div>
          </div>

          {sourceSet.showUmap && (
            <div className="w-full pb-6">
              <UmapPane
                showModel
                defaultModelId={sourceSet.modelId}
                defaultSourceSet={sourceSet.name || ''}
                defaultLayer={defaultSourceId || ''}
                release={sourceSet.releases as SourceReleaseWithRelations}
                filterToRelease={sourceSet.releases?.name || ''}
                releaseMultipleUmapSAEs={sourceSet.releases?.defaultUmapSourceIds || []}
                newWindowOnSaeChange={false}
              />
            </div>
          )}
          {sourceSet.hasDashboards && (
            <div className="flex w-full flex-col gap-y-6">
              <BrowserPane
                modelId={sourceSet.modelId}
                sourceSet={sourceSet.name}
                layer={defaultSourceId || ''}
                showModel
                filterToRelease={sourceSet.releases?.name || ''}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
