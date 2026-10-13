import Steerer from '@/components/steer/steerer';

// this page url is formatted as: /gemma-2-9b-it/steer?saved=cm1l96eeu000nooqkir7bq9lw
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
  return (
    <div className="flex h-full w-full flex-col items-center overflow-y-scroll bg-white">
      <Steerer
        initialModelId={params.modelId}
        initialSavedId={searchParams.saved}
        initialSource={searchParams.source}
        initialIndex={searchParams.index}
        initialStrength={searchParams.strength}
        hideInitialSettingsOnMobile={searchParams.hideInitialSettingsOnMobile === 'true'}
        initialPreset={searchParams.preset}
      />
    </div>
  );
}
