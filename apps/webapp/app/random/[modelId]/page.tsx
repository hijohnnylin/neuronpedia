'use client';;
import { use } from "react";

import { useRouter } from '@bprogress/next';

export default function Random(props: { params: Promise<{ modelId: string }> }) {
  const params = use(props.params);
  const router = useRouter();

  const randomLayer = Math.floor(Math.random() * 48);
  const randomIndex = Math.floor(Math.random() * 6400);

  router.push(`/${params.modelId}/${randomLayer}/${randomIndex}`);

  return <p />;
}
