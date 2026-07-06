import { prisma } from '@/lib/db';
import { notFound, redirect } from 'next/navigation';

// A shared jlens link. Resolve the share's model and redirect to the
// model-scoped page that actually loads and renders it.
export default async function Page({ params }: { params: Promise<{ sharedId: string }> }) {
  const { sharedId } = await params;
  const share = await prisma.jlensShare.findUnique({ where: { id: sharedId }, select: { modelId: true } });
  if (!share) {
    notFound();
  }
  redirect(`/${share.modelId}/jlens?shareId=${sharedId}`);
}
