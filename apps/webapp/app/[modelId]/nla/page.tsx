import { NLAModalProvider } from '@/components/provider/nla-modal-provider';
import { NLAProvider, type NlaFeaturedDemo } from '@/components/provider/nla-provider';
import { prisma } from '@/lib/db';
import { ASSET_BASE_URL } from '@/lib/env';
import { NLA_METADATA_PATH } from '@/lib/nla-constants';
import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import NLAExplainer from './nla-explainer';

export async function generateMetadata({ params }: { params: Promise<{ modelId: string }> }): Promise<Metadata> {
  const { modelId } = await params;
  const model = await prisma.model.findUnique({ where: { id: modelId }, select: { displayName: true } });
  const modelName = model?.displayName || modelId;

  const title = `Natural Language Autoencoders – ${modelName}`;
  const description = `Natural Language Autoencoder for ${modelName}. Translate a model's internal thoughts into natural language explanations.`;

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

export default async function Page({
  params,
  searchParams,
}: {
  params: Promise<{ modelId: string }>;
  searchParams: Promise<{ embed?: string }>;
}) {
  const { modelId } = await params;
  const { embed } = await searchParams;
  const isEmbed = embed === 'true';

  const model = await prisma.model.findUnique({ where: { id: modelId }, select: { id: true } });
  if (!model) {
    notFound();
  }

  const nlaSources = await prisma.nlaSource.findMany({
    orderBy: [{ modelId: 'asc' }, { createdAt: 'desc' }],
    include: {
      model: {
        select: {
          id: true,
          displayName: true,
          owner: true,
        },
      },
    },
  });

  const featuredShareRows = await prisma.nlaExplainShare.findMany({
    where: { featured: true },
    orderBy: [{ createdAt: 'desc' }, { id: 'asc' }],
    select: {
      id: true,
      cacheId: true,
      position: true,
      paragraph: true,
      highlightStart: true,
      highlightEnd: true,
      comment: true,
      featuredDisplayName: true,
      cache: {
        select: {
          modelId: true,
          model: { select: { displayName: true, owner: true } },
        },
      },
    },
  });

  const featuredDemos: NlaFeaturedDemo[] = featuredShareRows.map((row) => ({
    shareId: row.id,
    cacheId: row.cacheId,
    modelId: row.cache.modelId,
    modelDisplayName: row.cache.model.displayName || row.cache.modelId,
    modelOwner: row.cache.model.owner,
    position: row.position,
    paragraph: row.paragraph,
    highlightStart: row.highlightStart,
    highlightEnd: row.highlightEnd,
    comment: row.comment,
    featuredDisplayName: row.featuredDisplayName,
  }));

  return (
    <div
      className={`${isEmbed ? 'bg-slate-50' : 'bg-white'} ${
        isEmbed
          ? 'h-screen max-h-screen min-h-screen'
          : 'h-[calc(100vh_-_75px)] max-h-[calc(100vh_-_75px)] min-h-[calc(100vh_-_75px)]'
      } flex w-full max-w-full flex-col overflow-x-hidden`}
    >
      <NLAProvider modelId={modelId} nlaSources={nlaSources} featuredDemos={featuredDemos} isEmbed={isEmbed}>
        <NLAModalProvider>
          <NLAExplainer />
        </NLAModalProvider>
      </NLAProvider>
    </div>
  );
}
