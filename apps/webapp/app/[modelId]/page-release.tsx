import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import ReleasesDropdown from '@/components/nav/releases-dropdown';
import BrowserPane from '@/components/panes/browser-pane/browser-pane';
import ConnectedNeuronsPane from '@/components/panes/connected-neurons-pane';
import JumpToPane from '@/components/panes/jump-to-pane';
import SearchExplanationsPane from '@/components/panes/search-explanations-pane';
import SearchInferenceReleasePane from '@/components/panes/search-inference-release-pane';
import SourceSimilarityMatrixPane from '@/components/panes/source-similarity-matrix-pane';
import UmapPane from '@/components/panes/umap-pane';
import { BreadcrumbLink, BreadcrumbPage } from '@/components/shadcn/breadcrumbs';
import {
  CIRCUIT_SPARSITY_DEFAULT_INDEX,
  CIRCUIT_SPARSITY_DEFAULT_LAYER,
  CIRCUIT_SPARSITY_MODELS,
} from '@/lib/utils/circuit-sparsity';
import { SearchExplanationsType } from '@/lib/utils/general';
import { getDefaultSourceSetAndSourceForRelease } from '@/lib/utils/source';
import { NeuronWithPartialRelations, SourceReleaseWithRelations } from '@/prisma/generated/zod';
import Hero from './gemmascope/hero';

export const LLAMA_SCOPE_2 = 'llama-scope-2';

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
        {release.name === LLAMA_SCOPE_2 && (
          <div className="flex w-full flex-col items-center justify-center">
            <div className="mb-4 flex max-w-screen-lg flex-col gap-y-2 text-sm text-slate-600">
              <p>
                OpenMOSS extended Anthropic&apos;s circuit tracing work to add interpretable attention in addition to
                MLP transcoders, calling them
                <a
                  href="https://interp.open-moss.com/posts/complete-replacement"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-1 text-sky-700 underline hover:text-sky-900"
                >
                  Complete Replacement Models
                </a>
                {` `}
                (CRMs). Neuronpedia now supports generating CRM graphs on Qwen3-1.7B.
              </p>
              <p>
                CRM graphs have a new node type to represent attention called
                <a
                  href="https://arxiv.org/abs/2504.20938"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-1 text-sky-700 underline hover:text-sky-900"
                >
                  LORSA
                </a>
                {` `}
                (Low-Rank Sparse Attention), which are displayed as triangle ▲ nodes to visually distinguish them from
                transcoder circle ⏺ nodes.
              </p>
              <p>
                Since CRM graphs incorporate both transcoders and LORSA, they refer to two sets of dashboards. When
                selecting LORSA (triangle) nodes, you&apos;ll see the LORSA dashboard, which shows attention Z patterns
                when hovering over top activation tokens.
              </p>
              <p>
                Additionally, LORSA nodes show QK tracing results under the Node Connections panel — including top
                marginal and pairwise (query-feature, key-feature) contributors. These tell us why a LORSA feature
                attends from one position to another.
              </p>
            </div>
            <iframe
              title="Circuit Tracing with Attention: Complete Replacement Models"
              src="/qwen3-1.7b/graph?slug=dallas-austin&pruningThreshold=0.6&densityThreshold=0.20&embed=true"
              className="h-[800px] w-full rounded-lg border border-slate-200 shadow-md"
            />
          </div>
        )}
        {CIRCUIT_SPARSITY_MODELS.includes(defaultModelId || '') && (
          <div className="w-full max-w-screen-md">
            <ConnectedNeuronsPane
              currentNeuron={
                {
                  modelId: defaultModelId,
                  layer: CIRCUIT_SPARSITY_DEFAULT_LAYER,
                  index: CIRCUIT_SPARSITY_DEFAULT_INDEX.toString(),
                } as NeuronWithPartialRelations
              }
              showSelectors={true}
            />
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
