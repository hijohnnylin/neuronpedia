import { prisma } from '@/lib/db';
import { env } from '@/lib/env';
import { notFound, permanentRedirect } from 'next/navigation';

// this page url is formatted as: /steer/[steer-model] or /steer/[steer-output-id]
export default async function Page({ params }: { params: { slug: string } }) {
  const isInferenceEnabledModel = await prisma.model.findFirst({
    where: {
      id: params.slug,
      inferenceEnabled: true,
    },
  });
  // CASE: this is /steer/[steer-model-id] - redirect to the new steer page
  if (isInferenceEnabledModel || env.NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS.includes(params.slug)) {
    permanentRedirect(`/${params.slug}/steer`);
  } else {
    // CASE: this is /steer/[steerOutputId] - redirect to the new steer page and load the steer
    const steerOutput = await prisma.steerOutput.findUnique({
      where: {
        id: params.slug,
      },
    });
    if (steerOutput) {
      permanentRedirect(`/${steerOutput.modelId}/steer/?saved=${params.slug}`);
    } else {
      notFound();
    }
  }
}
