import Steerer from '@/components/steer/steerer';
import { REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT } from '@/lib/db/model';
import { redirect } from 'next/navigation';

// this page url is formatted as: /gemma-2-9b-it/steer?saved=cm1l96eeu000nooqkir7bq9lw
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
  };
}) {
  // TODO: this is a temporary map since there is a bug in our lesswrong plugin that breaks when dots are in modelIds for hoverover links
  if (params.modelId in REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT) {
    // redirect to the new model id
    const queryString = new URLSearchParams(searchParams as Record<string, string>).toString();
    const redirectUrl = `/${REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT[params.modelId as keyof typeof REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT]}/steer${queryString ? `?${queryString}` : ''}`;
    redirect(redirectUrl);
  }
  return (
    <div className="flex h-full w-full flex-col items-center overflow-y-scroll bg-white">
      <Steerer
        initialModelId={params.modelId}
        initialSavedId={searchParams.saved}
        initialSource={searchParams.source}
        initialIndex={searchParams.index}
        initialStrength={searchParams.strength}
        hideInitialSettingsOnMobile={searchParams.hideInitialSettingsOnMobile === 'true'}
      />
    </div>
  );
}
