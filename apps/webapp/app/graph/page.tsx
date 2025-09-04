import { ASSET_BASE_URL } from '@/lib/constants';
import { Metadata } from 'next';
import { redirect } from 'next/navigation';
import { ANTHROPIC_MODEL_TO_DISPLAY_NAME, DEFAULT_GRAPH_MODEL_ID } from '../[modelId]/graph/utils';

export async function generateMetadata({
  params,
  searchParams,
}: {
  params: { modelId: string };
  searchParams: { [key: string]: string | string[] | undefined };
}): Promise<Metadata> {
  const { modelId } = params;
  const slug = searchParams.slug as string | undefined;

  // use modelIdToModelDisplayName to get the model name if it's there. othewise use it directly
  const modelName = ANTHROPIC_MODEL_TO_DISPLAY_NAME.get(modelId) || modelId;

  const title = `${slug ? `${slug} - ` : ''}${modelName || 'Attribution'} Graph | Neuronpedia`;
  const description = `Attribution Graph for ${modelName}`;
  let url = `/${modelId}/graph`;

  if (slug) {
    url = `/${modelId}/graph?slug=${slug}`;
  }

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      images: [`${ASSET_BASE_URL}/graph-preview.jpg`],
      url,
    },
  };
}

export default function GraphPage() {
  redirect(`/${DEFAULT_GRAPH_MODEL_ID}/graph`);
}
