'use client';

import { useRouter } from '@bprogress/next';
import { Button } from './shadcn/button';

interface RandomFeatureLinkProps {
  modelId: string;
  source: string;
}

export default function RandomFeatureLink({ modelId, source }: RandomFeatureLinkProps) {
  const router = useRouter();

  return (
    <Button
      onClick={() => {
        const randomFeature = Math.floor(Math.random() * 10000);
        router.push(`/${modelId}/${source}/${randomFeature}`);
      }}
      variant="outline"
      className="h-10 max-h-[40px] min-h-[40px] text-xs"
    >
      Random
    </Button>
  );
}
