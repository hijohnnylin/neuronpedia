'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { useRouter } from '@bprogress/next';
import { Button } from './shadcn/button';

const MAX_RANDOM_FEATURE = 10000;

export default function RandomModelFeature({
  modelId,
  callback,
}: {
  modelId: string;
  callback?: (modelId: string, sourceId: string, featureId: number) => void;
}) {
  const router = useRouter();
  const { getSourceSetsForModelId } = useGlobalContext();
  const sourceSets = getSourceSetsForModelId(modelId);

  return (
    <Button
      onClick={() => {
        // pick a random sourceSet
        if (sourceSets && sourceSets.length > 0) {
          const sourceSet = sourceSets[Math.floor(Math.random() * sourceSets.length)];
          if (sourceSet && sourceSet.sources && sourceSet.sources.length > 0) {
            // pick a random source
            const source = sourceSet.sources[Math.floor(Math.random() * sourceSet.sources.length)];
            if (source && source.modelId && source.id) {
              const randomFeature = Math.floor(Math.random() * MAX_RANDOM_FEATURE) + 1;
              router.push(`/${source.modelId}/${source.id}/${randomFeature}`);
              callback?.(source.modelId, source.id, randomFeature);
            }
          }
        }
      }}
      variant="outline"
      className="h-10 max-h-[40px] min-h-[40px] text-xs"
    >
      Random
    </Button>
  );
}
