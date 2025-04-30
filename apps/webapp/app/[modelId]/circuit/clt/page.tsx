import { CircuitCLTProvider } from '@/components/provider/circuit-clt-provider';
import { CLT_BASE_URLS, getCLTMetadata, metadataScansToDisplay, ModelToCLTMetadataGraphsMap } from './clt-utils';
import CLTWrapper from './wrapper';

export default async function Page({
  // params,
  searchParams,
}: {
  // params: {};
  searchParams: { clickedId?: string; logitDiff?: string; modelId: string; slug: string };
}) {
  // TODO: update this to use modelid from url
  // const { modelId } = params;
  // const model = await getModelByIdWithSourceSets(modelId, await makeAuthedUserFromSessionOrReturnNull());

  // if (!model) {
  //   notFound();
  // }

  let metadata: ModelToCLTMetadataGraphsMap | null = null;
  const modelToBaseUrl: Record<string, string> = {};
  // eslint-disable-next-line
  for (const baseUrl of CLT_BASE_URLS) {
    try {
      // eslint-disable-next-line
      const result = await getCLTMetadata(baseUrl);
      if (result) {
        if (!metadata) {
          metadata = result;
        } else {
          // Merge the results
          // eslint-disable-next-line
          Object.keys(result).forEach((scanId) => {
            if (metadata && metadataScansToDisplay.has(scanId)) {
              if (!metadata[scanId]) {
                metadata[scanId] = result[scanId];
              } else {
                metadata[scanId] = [...metadata[scanId], ...result[scanId]];
              }
            }
          });
        }
        Object.keys(result).forEach((scanId) => {
          modelToBaseUrl[scanId] = baseUrl;
        });
      }
    } catch (error) {
      console.error(`Failed to fetch metadata from ${baseUrl}:`, error);
    }
  }
  if (!metadata) {
    return <div>No metadata found</div>;
  }

  console.log(searchParams);

  const metadataGraph = searchParams.modelId
    ? metadata[searchParams.modelId]?.find((graph) => graph.slug === searchParams.slug)
    : undefined;

  console.log(searchParams.modelId, searchParams.slug);
  console.log(metadataGraph);

  return (
    <CircuitCLTProvider
      initialMetadata={metadata}
      initialModelToBaseUrl={modelToBaseUrl}
      initialClickedId={searchParams.clickedId}
      initialLogitDiff={searchParams.logitDiff}
      initialModel={searchParams.modelId}
      initialMetadataGraph={metadataGraph}
    >
      <CLTWrapper />
    </CircuitCLTProvider>
  );
}
