import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import ReleasesDropdown from '@/components/nav/releases-dropdown';
import BrowserPane from '@/components/panes/browser-pane/browser-pane';
import JumpToPane from '@/components/panes/jump-to-pane';
import SearchExplanationsPane from '@/components/panes/search-explanations-pane';
import SearchInferenceReleasePane from '@/components/panes/search-inference-release-pane';
import SourceSimilarityMatrixPane from '@/components/panes/source-similarity-matrix-pane';
import UmapPane from '@/components/panes/umap-pane';
import { BreadcrumbLink, BreadcrumbPage } from '@/components/shadcn/breadcrumbs';
import { SearchExplanationsType } from '@/lib/utils/general';
import { getDefaultSourceSetAndSourceForRelease } from '@/lib/utils/source';
import { SourceReleaseWithRelations } from '@/prisma/generated/zod';
import Hero from './gemmascope/hero';

export default function PageRelease({ release }: { release: SourceReleaseWithRelations }) {
  const { defaultSourceSet, defaultSource } = getDefaultSourceSetAndSourceForRelease(release);
  const defaultModelId = defaultSourceSet?.modelId || '';

  let defaultUmapSourceSetName: string | undefined;
  let defaultUmapSourceId: string | undefined;
  release.sourceSets.every((ss) => {
    if (ss.showUmap) {
      defaultUmapSourceSetName = ss.name;
      ss.sources.every((s) => {
        if (s.hasUmap) {
          defaultUmapSourceId = s.id;
          return false;
        }
        return true;
      });
      return false;
    }
    return true;
  });

  const isTemporalSaeRelease = defaultSourceSet?.similarityMatrixEnabled;

  const isGemmaScope2Release = release.name === 'gemma-scope-2';

  return (
    <div className="flex w-full flex-col items-center pb-10">
      <BreadcrumbsComponent
        crumbsArray={[
          <BreadcrumbPage key={0}>
            <ReleasesDropdown breadcrumb />
          </BreadcrumbPage>,
          <BreadcrumbLink href={`/${release.name}`} key={1}>
            {release.description}
          </BreadcrumbLink>,
        ]}
      />
      <Hero release={release} />

      <div className="flex w-full max-w-screen-lg flex-col items-center pb-5 pt-5 text-slate-700 xl:max-w-screen-xl 2xl:max-w-screen-2xl">
        {isGemmaScope2Release && (
          <div className="mt-0 flex w-full max-w-screen-lg flex-col gap-y-6">
            <div className="flex items-start gap-x-3 rounded-lg border border-amber-200 bg-amber-50 px-5 py-4">
              <div className="flex w-full flex-col gap-y-1 text-center text-[13px]">
                <div className="mb-1.5 text-center text-sm font-bold text-slate-800">⚠️ Rolling Release</div>
                <div>
                  An initial release of artifacts and datasets are available now, with more being actively released on a
                  rolling basis.
                </div>
                <div>
                  All artifacts are expected to be finalized by <strong>December 31, 2025</strong>.
                </div>
                <div>
                  Some data may be replaced or updated during this final verification and fine-tuning process. Please
                  check the{' '}
                  <a
                    href="https://huggingface.co/google/gemma-scope-2"
                    target="_blank"
                    rel="noreferrer noopener"
                    className="text-sky-700"
                  >
                    HuggingFace
                  </a>{' '}
                  for details.
                </div>
              </div>
            </div>
          </div>
        )}
        {isTemporalSaeRelease && defaultSource && <SourceSimilarityMatrixPane source={defaultSource} />}
        {defaultUmapSourceId && defaultUmapSourceSetName && (
          <UmapPane
            showModel
            defaultModelId={defaultModelId}
            defaultSourceSet={defaultSourceSet?.name || ''}
            defaultLayer={defaultSource?.id || ''}
            release={release}
            filterToRelease={release.name}
            releaseMultipleUmapSAEs={release.defaultUmapSourceIds}
            newWindowOnSaeChange={false}
          />
        )}

        <div className="mt-6 flex w-full max-w-screen-lg flex-col gap-y-6">
          <JumpToPane
            release={release}
            defaultModelId={defaultModelId}
            defaultSourceSetName={defaultSourceSet?.name || ''}
            defaultSourceId={defaultSource?.id || ''}
            vertical
            filterToFeaturedReleases={false}
          />

          <SearchExplanationsPane
            filterToRelease={release}
            initialModelId={defaultModelId}
            initialSourceSetName={defaultSourceSet?.name || ''}
            defaultTab={SearchExplanationsType.BY_RELEASE}
            showTabs
          />
        </div>
        {defaultSourceSet?.allowInferenceSearch && (
          <div className="mt-6 w-full max-w-screen-lg pb-0">
            <SearchInferenceReleasePane release={release} />
          </div>
        )}

        <div className="mb-6 mt-6 w-full pb-0">
          <BrowserPane
            modelId={defaultModelId}
            sourceSet={defaultSourceSet?.name || ''}
            layer={defaultSource?.id || ''}
            showModel
            filterToRelease={release.name}
          />
        </div>
      </div>
    </div>
  );
}
