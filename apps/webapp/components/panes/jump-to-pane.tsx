'use client';

import FeatureSelector from '@/components/feature-selector/feature-selector';
import JumpToSAE from '@/components/jump-to-sae';
import { useGlobalContext } from '@/components/provider/global-provider';
import RandomModelFeature from '@/components/random-model-feature';
import RandomReleaseFeature from '@/components/random-release-feature';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { getFirstSourceForModel, getFirstSourceForRelease } from '@/lib/utils/source';
import { SourceReleaseWithPartialRelations } from '@/prisma/generated/zod';
import { Visibility } from '@prisma/client';

export default function JumpToPane({
  defaultModelId,
  release,
  defaultSourceSetName,
  defaultSourceId,
  filterToFeaturedReleases = true,
  vertical = false,
  showRandomFeature = true,
  showTitleAndCard = true,
  showModel = true,
}: {
  defaultModelId: string;
  release?: SourceReleaseWithPartialRelations;
  defaultSourceSetName: string;
  defaultSourceId?: string;
  vertical?: boolean;
  filterToFeaturedReleases?: boolean;
  showRandomFeature?: boolean;
  showTitleAndCard?: boolean;
  showModel?: boolean;
}) {
  const { globalModels } = useGlobalContext();

  const content = (
    <div className={`flex w-full ${vertical ? 'flex-col' : 'flex-row'} gap-y-3`}>
      <JumpToSAE
        showModel={showModel}
        modelId={defaultModelId}
        layer={
          // eslint-disable-next-line no-nested-ternary
          release
            ? release.defaultSourceId || getFirstSourceForRelease(release, Visibility.PUBLIC, false, false)?.id
            : defaultSourceId || defaultModelId in globalModels
              ? defaultSourceId ||
                getFirstSourceForModel(globalModels[defaultModelId], Visibility.PUBLIC, false, false)?.id
              : undefined
        }
        filterToRelease={release?.name}
        filterToFeaturedReleases={filterToFeaturedReleases}
      />
      <div className="flex w-full flex-col items-start">
        <div className="mb-0.5 font-sans text-[10px] font-medium uppercase text-slate-500">Jump to Feature</div>
        <FeatureSelector
          showModel={showModel}
          filterToRelease={release?.name}
          defaultModelId={defaultModelId}
          defaultSourceSet={defaultSourceSetName}
          defaultSource={defaultSourceId}
          filterToPublic
          filterToFeaturedReleases={false}
          openInNewTab={false}
        />
      </div>
      {showRandomFeature && (
        <div className="flex flex-1 flex-col">
          <div className="mb-1 whitespace-pre font-sans text-[10px] font-normal uppercase text-slate-500">
            Random Feature
          </div>
          {release ? <RandomReleaseFeature release={release} /> : <RandomModelFeature modelId={defaultModelId} />}
        </div>
      )}
    </div>
  );

  return showTitleAndCard ? (
    <Card className="w-full bg-white">
      <CardHeader className="w-full pb-0 pt-5">
        <div className="flex w-full flex-row items-center justify-between">
          <CardTitle>Jump To</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="flex flex-row items-start justify-between pb-6 pt-4">{content}</CardContent>
    </Card>
  ) : (
    content
  );
}
