import { prisma } from '@/lib/db';
import { getAllServerHostsForModel } from '@/lib/db/inference-host-source';
import { ASSET_BASE_URL, DEFAULT_CREATOR_USER_ID, USE_LOCALHOST_INFERENCE } from '@/lib/env';
import { JlensShareLockedToken } from '@/lib/utils/jlens-share';
import { JLENS_METADATA_PATH } from '@/lib/utils/lens';
import { Metadata } from 'next';
import { notFound, redirect } from 'next/navigation';
import JlensPageClient, { JlensShareData } from './jlens-page-client';

export async function generateMetadata({ params }: { params: Promise<{ modelId: string }> }): Promise<Metadata> {
  const { modelId } = await params;
  const model = await prisma.model.findUnique({ where: { id: modelId }, select: { displayName: true } });
  const modelName = model?.displayName || modelId;

  const title = `Jacobian Lens – ${modelName}`;
  const description = `Revealing a Global Workspace in Language Models`;

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      images: [`${ASSET_BASE_URL}${JLENS_METADATA_PATH}`],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: [`${ASSET_BASE_URL}${JLENS_METADATA_PATH}`],
    },
  };
}

export default async function Page({
  params,
  searchParams,
}: {
  params: Promise<{ modelId: string }>;
  searchParams: Promise<{ shareId?: string }>;
}) {
  const { modelId } = await params;
  const { shareId } = await searchParams;

  const model = await prisma.model.findUnique({ where: { id: modelId }, select: { id: true } });
  if (!model) {
    notFound();
  }

  // Whether any inference host currently serves this model. When none is
  // available, the page still renders cached/shared results, but live actions
  // (sending messages, steering/swapping, toggling the server-side non-word
  // filter) are gated behind a "model unavailable" notice.
  // In localhost mode all inference is routed to the local host regardless of
  // the DB host table, so treat the model as available without a DB lookup.
  let inferenceAvailable = USE_LOCALHOST_INFERENCE;
  if (!inferenceAvailable) {
    try {
      const hosts = await getAllServerHostsForModel(modelId);
      inferenceAvailable = hosts.length > 0;
    } catch {
      inferenceAvailable = false;
    }
  }

  // When `?shareId=` is present, resolve the shared run server-side and pass it
  // to the page client (which fetches the heavy S3 blob client-side).
  let share: JlensShareData | null = null;
  if (shareId) {
    const row = await prisma.jlensShare.findUnique({
      where: { id: shareId },
      include: { user: { select: { name: true } } },
    });
    if (!row) {
      notFound();
    }
    // Keep the URL canonical: a share belongs to exactly one model.
    if (row.modelId !== modelId) {
      redirect(`/${row.modelId}/jlens?shareId=${shareId}`);
    }

    // Attribution line shown beneath the description. The default creator's
    // shares show the description with no attribution; everyone else is credited
    // (by username if logged in, otherwise as an anonymous sharer).
    let descriptionAttribution: string | null = null;
    if (row.description && row.userId !== DEFAULT_CREATOR_USER_ID) {
      const sharerName = row.userId ? row.user?.name : null;
      descriptionAttribution = sharerName ? `@${sharerName}` : 'Anonymous';
    }

    share = {
      url: row.url,
      description: row.description,
      descriptionAttribution,
      lockedTokens: (row.lockedTokens as unknown as JlensShareLockedToken[]) ?? [],
      selectedPositions: row.selectedPositions ?? [],
      activeLensModeTab: row.activeLensModeTab,
      topN: row.topN,
      hideNonWordTokens: row.hideNonWordTokens,
      temperature: row.temperature,
      numCompletionTokens: row.numCompletionTokens,
      numPromptTokens: row.numPromptTokens ?? null,
    };
  }

  return <JlensPageClient modelId={modelId} share={share} inferenceAvailable={inferenceAvailable} />;
}
