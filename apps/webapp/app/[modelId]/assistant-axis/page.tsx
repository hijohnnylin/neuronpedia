import { FEATURE_PRESETS } from '@/app/api/steer/presets/route';
import { AssistantAxisModalProvider } from '@/components/provider/assistant-axis-modal-provider';
import { REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT } from '@/lib/db/model';
import { notFound, redirect } from 'next/navigation';
import AssistantAxisSteerer from './assistant-axis-steerer';
import { Metadata } from 'next';
import { ASSET_BASE_URL } from '@/lib/env';

const SUPPORTED_MODELS = ['llama3.3-70b-it'];
const ASSISTANT_CAP_PRESET = 'assistant-cap';

export async function generateMetadata(): Promise<Metadata> {
  const title = `Assistant Axis`;
  const description = `Demo of monitoring and capping language models, based on work by Lu et al.`;

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      images: [`${ASSET_BASE_URL}/cap/monitor.png`]
    },
  };
}

export default async function Page({
  params,
  searchParams,
}: {
  params: { modelId: string };
  searchParams: {
    saved?: string;
    source?: string;
    index?: string;
    strength?: string;
    hideInitialSettingsOnMobile?: string;
    preset?: string;
  };
}) {
  if (!SUPPORTED_MODELS.includes(params.modelId)) {
    notFound();
  }
  // TODO: this is a temporary map since there is a bug in our lesswrong plugin that breaks when dots are in modelIds for hoverover links
  if (params.modelId in REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT) {
    // redirect to the new model id
    const queryString = new URLSearchParams(searchParams as Record<string, string>).toString();
    const redirectUrl = `/${REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT[params.modelId as keyof typeof REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT]}/steer${queryString ? `?${queryString}` : ''}`;
    redirect(redirectUrl);
  }
  const assistantCapPreset = FEATURE_PRESETS['llama3.3-70b-it'].find((p) => p.alias === ASSISTANT_CAP_PRESET);
  return (
    <AssistantAxisModalProvider>
      <div className="flex h-full w-full flex-col items-center overflow-y-scroll bg-white">
        <AssistantAxisSteerer
          initialSavedId={searchParams.saved}
          hideInitialSettingsOnMobile={searchParams.hideInitialSettingsOnMobile === 'true'}
          initialSteerFeatures={assistantCapPreset?.features}
        />
      </div>
    </AssistantAxisModalProvider>
  );
}
