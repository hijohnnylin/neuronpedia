import { FEATURE_PRESETS } from '@/app/api/steer/presets/route';
import { AssistantAxisModalProvider } from '@/components/provider/assistant-axis-modal-provider';
import { ASSET_BASE_URL } from '@/lib/env';
import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import AssistantAxisSteerer from './assistant-axis-steerer';

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
      images: [`${ASSET_BASE_URL}/cap/monitor.png`],
    },
  };
}

export default async function Page(props: {
  params: Promise<{ modelId: string }>;
  searchParams: Promise<{
    saved?: string;
    source?: string;
    index?: string;
    strength?: string;
    hideInitialSettingsOnMobile?: string;
    preset?: string;
  }>;
}) {
  const searchParams = await props.searchParams;
  const params = await props.params;
  if (!SUPPORTED_MODELS.includes(params.modelId)) {
    notFound();
  }
  const assistantCapPreset = FEATURE_PRESETS['llama3.3-70b-it'].find((p) => p.alias === ASSISTANT_CAP_PRESET);
  return (
    <AssistantAxisModalProvider>
      <div className="flex h-full w-full flex-col items-center overflow-y-scroll bg-white">
        <AssistantAxisSteerer initialSavedId={searchParams.saved} initialSteerFeatures={assistantCapPreset?.features} />
      </div>
    </AssistantAxisModalProvider>
  );
}
