import SearchExplanationsPane from '@/components/panes/search-explanations-pane';
import { SearchExplanationsType } from '@/lib/utils/general';
import { getSourceSetNameFromSource } from '@/lib/utils/source';

export default async function Page({
  searchParams,
}: {
  searchParams: {
    q: string;
    releaseName: string;
    modelId: string;
    sources: string;
    embed: string;
  };
}) {
  const isEmbed = searchParams.embed === 'true';
  let selectedTab: SearchExplanationsType | undefined;
  if (searchParams.releaseName) {
    selectedTab = SearchExplanationsType.BY_RELEASE;
  } else if (searchParams.sources) {
    selectedTab = SearchExplanationsType.BY_SOURCE;
  } else if (searchParams.modelId) {
    selectedTab = SearchExplanationsType.BY_MODEL;
  } else {
    selectedTab = SearchExplanationsType.BY_ALL;
  }

  const selectedLayers = searchParams.sources ? JSON.parse(searchParams.sources) : [];
  let sourceSet: string | undefined;
  if (selectedLayers.length > 0) {
    sourceSet = getSourceSetNameFromSource(selectedLayers[0]);
  }

  return (
    <div className={`mb-10 flex w-full max-w-screen-lg flex-col items-center ${isEmbed ? '' : 'mt-6 pt-5'}`}>
      <SearchExplanationsPane
        searchQuery={searchParams.q}
        initialReleaseName={searchParams.releaseName}
        initialSelectedLayers={selectedLayers}
        initialModelId={searchParams.modelId}
        initialSourceSetName={sourceSet}
        defaultTab={selectedTab}
      />
    </div>
  );
}
