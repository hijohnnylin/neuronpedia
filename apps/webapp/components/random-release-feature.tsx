'use client';

import { SourceReleaseWithPartialRelations } from '@/prisma/generated/zod';
import { useRouter } from '@bprogress/next';
import { Button } from './shadcn/button';

const MAX_RANDOM_FEATURE = 10000;

export default function RandomReleaseFeature({
  release,
  callback,
}: {
  release: SourceReleaseWithPartialRelations;
  callback?: () => void;
}) {
  const router = useRouter();
  return (
    <Button
      onClick={() => {
        // pick a random sourceSet
        if (release.sourceSets && release.sourceSets.length > 0) {
          const sourceSet = release.sourceSets[Math.floor(Math.random() * release.sourceSets.length)];
          if (sourceSet && sourceSet.sources && sourceSet.sources.length > 0) {
            // pick a random source
            const source = sourceSet.sources[Math.floor(Math.random() * sourceSet.sources.length)];
            if (source && source.modelId && source.id) {
              const randomFeature = Math.floor(Math.random() * MAX_RANDOM_FEATURE) + 1;
              router.push(`/${source.modelId}/${source.id}/${randomFeature}`);
              callback?.();
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
