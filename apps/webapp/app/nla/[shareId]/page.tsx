import { prisma } from '@/lib/db';
import { ASSET_BASE_URL } from '@/lib/env';
import { NLA_METADATA_PATH } from '@/lib/nla-constants';
import { Metadata } from 'next';
import { notFound, redirect } from 'next/navigation';

export async function generateMetadata({ params }: { params: Promise<{ shareId: string }> }): Promise<Metadata> {
  const { shareId } = await params;
  const title = `NLA share ${shareId.slice(0, 8)}…`;
  const description = 'Open this Natural Language Autoencoder share link.';
  return {
    title,
    description,
    openGraph: {
      title,
      description,
      images: [`${ASSET_BASE_URL}${NLA_METADATA_PATH}`],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: [`${ASSET_BASE_URL}${NLA_METADATA_PATH}`],
    },
  };
}

export default async function NlaShareRedirectPage({
  params,
  searchParams,
}: {
  params: Promise<{ shareId: string }>;
  searchParams: Promise<{ embed?: string }>;
}) {
  const { shareId } = await params;
  const { embed } = await searchParams;

  const share = await prisma.nlaExplainShare.findUnique({
    where: { id: shareId },
    include: { cache: { select: { modelId: true } } },
  });

  if (!share) {
    notFound();
  }

  const qs = new URLSearchParams();
  qs.set('id', share.cacheId);
  if (share.position !== null) {
    qs.set('position', String(share.position));
  }
  if (share.highlightStart !== null && share.highlightEnd !== null) {
    qs.set('highlightStart', String(share.highlightStart));
    qs.set('highlightEnd', String(share.highlightEnd));
  } else if (share.paragraph !== null) {
    qs.set('paragraph', String(share.paragraph));
  }
  if (share.comment.length > 0) {
    qs.set('comment', share.comment);
  }
  if (embed === 'true') {
    qs.set('embed', 'true');
  }

  redirect(`/${share.cache.modelId}/nla?${qs.toString()}`);
}
