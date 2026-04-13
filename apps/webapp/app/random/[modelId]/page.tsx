'use client';

import { useRouter } from '@bprogress/next';

export default function Random({ params }: { params: { modelId: string } }) {
  const router = useRouter();

  const randomLayer = Math.floor(Math.random() * 48);
  const randomIndex = Math.floor(Math.random() * 6400);

  router.push(`/${params.modelId}/${randomLayer}/${randomIndex}`);

  return <p />;
}
