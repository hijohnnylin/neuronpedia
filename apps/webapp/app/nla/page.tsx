import { ASSET_BASE_URL } from '@/lib/env';
import { NLA_METADATA_PATH } from '@/lib/nla-constants';
import { Metadata } from 'next';
import { redirect } from 'next/navigation';

const DEFAULT_NLA_MODEL_ID = 'llama3.3-70b-it';

export async function generateMetadata(): Promise<Metadata> {
  const title = 'Natural Language Autoencoders';
  const description = 'Translate activation vectors from language models into natural language descriptions and back.';

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

export default function Page() {
  redirect(`/${DEFAULT_NLA_MODEL_ID}/nla`);
}
