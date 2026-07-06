import { ASSET_BASE_URL } from '@/lib/env';
import { DEFAULT_JLENS_MODEL_ID, JLENS_METADATA_PATH } from '@/lib/utils/lens';
import { Metadata } from 'next';
import { redirect } from 'next/navigation';

export async function generateMetadata(): Promise<Metadata> {
  const title = 'Jacobian Lens';
  const description = 'Revealing a Global Workspace in Language Models';

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

export default function Page() {
  redirect(`/${DEFAULT_JLENS_MODEL_ID}/jlens`);
}
