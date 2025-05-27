import { authOptions } from '@/app/api/auth/[...nextauth]/authOptions';
import { GraphProvider } from '@/components/provider/graph-provider';
import { prisma } from '@/lib/db';
import { Metadata } from 'next';
import { getServerSession } from 'next-auth/next';
import {
  getGraphMetadatasFromBucket,
  GRAPH_BASE_URL_TO_NAME,
  ModelToGraphMetadatasMap,
  supportedGraphModels,
} from './utils';
import GraphWrapper from './wrapper';

export async function generateMetadata({
  params,
  searchParams,
}: {
  params: { modelId: string };
  searchParams: { [key: string]: string | string[] | undefined };
}): Promise<Metadata> {
  const { modelId } = params;
  const slug = searchParams.slug as string | undefined;

  const title = `${slug ? `${slug} - ` : ''}${modelId.toUpperCase()} Attribution Graph`;
  const description = `Visualizing the biology of ${modelId.toUpperCase()} by generating attribution graphs.`;
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
      url,
    },
  };
}

// Helper function to ensure a modelId exists in the map
function ensureModelIdInMap(map: ModelToGraphMetadatasMap, modelId: string) {
  if (!map[modelId]) {
    // eslint-disable-next-line
    map[modelId] = [];
  }
}

// Helper function to add graph metadata to the map without duplicates
function addGraphMetadataToMap(map: ModelToGraphMetadatasMap, modelId: string, graphMetadata: any) {
  ensureModelIdInMap(map, modelId);

  // Ensure no existing graph with the same slug to avoid overriding featured graphs
  if (!map[modelId].find((graph) => graph.slug === graphMetadata.slug)) {
    map[modelId].push(graphMetadata);
  }
}

// Helper function to merge graph metadata arrays into the map
function mergeGraphMetadataArrays(map: ModelToGraphMetadatasMap, modelId: string, graphMetadataArray: any[]) {
  if (!map[modelId]) {
    // eslint-disable-next-line
    map[modelId] = graphMetadataArray;
  } else {
    // eslint-disable-next-line
    map[modelId] = [...map[modelId], ...graphMetadataArray];
  }
}

// Helper function to get graph metadata with user info
async function getGraphMetadataWithUser(where: any) {
  return prisma.graphMetadata.findMany({
    where,
    include: {
      user: {
        select: {
          name: true,
        },
      },
    },
  });
}

export default async function Page({
  params,
  searchParams,
}: {
  params: { modelId: string };
  searchParams: {
    clickedId?: string;
    logitDiff?: string;
    // model?: string;
    slug?: string;
    pinnedIds?: string;
    supernodes?: string;
    clerps?: string;
    pruningThreshold?: string;
    densityThreshold?: string;
  };
}) {
  const { modelId } = params;
  const session = await getServerSession(authOptions);

  // TODO: checks for model existence (current not used bc we have anthropic models)
  // const model = await getModelByIdWithSourceSets(modelId, await makeAuthedUserFromSessionOrReturnNull());
  // if (!model) {
  //   notFound();
  // }

  // iterate through all baseUrls/buckets, and merge all the metadata into a single map
  const modelIdToGraphMetadatasMap: ModelToGraphMetadatasMap = {};

  // first, get all the graphmetadatas from the buckets
  // eslint-disable-next-line
  for (const baseUrl of Object.keys(GRAPH_BASE_URL_TO_NAME)) {
    try {
      const modelIdToGraphMetadata = await getGraphMetadatasFromBucket(baseUrl);

      // eslint-disable-next-line
      Object.keys(modelIdToGraphMetadata)
        .filter((m) => supportedGraphModels.has(m))
        .forEach((m) => {
          mergeGraphMetadataArrays(modelIdToGraphMetadatasMap, m, modelIdToGraphMetadata[m]);
        });
    } catch (error) {
      console.error(`Failed to fetch metadata from ${baseUrl}:`, error);
    }
  }

  // now look up logged in user's graphmetadatas in our database
  const graphMetadatas =
    session && session.user && session.user.id ? await getGraphMetadataWithUser({ userId: session.user.id }) : [];

  // add those graphmetadatas to the modelIdToGraphMetadatasMap too
  graphMetadatas.forEach((graphMetadata) => {
    addGraphMetadataToMap(modelIdToGraphMetadatasMap, graphMetadata.modelId, graphMetadata);
  });

  // set the metadata graph to show, if specified in the url. if we don't have it, look it up in the database
  let metadataGraph;
  if (searchParams.slug) {
    metadataGraph = modelIdToGraphMetadatasMap[modelId]?.find((graph) => graph.slug === searchParams.slug);
    // if it's not in our map, look it up in the database
    if (!metadataGraph) {
      metadataGraph = await prisma.graphMetadata.findUnique({
        where: {
          modelId_slug: {
            modelId,
            slug: searchParams.slug,
          },
        },
        include: {
          user: {
            select: {
              name: true,
            },
          },
        },
      });
      // add this to our map
      if (metadataGraph) {
        addGraphMetadataToMap(modelIdToGraphMetadatasMap, modelId, metadataGraph);
      } else {
        console.error(`Graph with slug ${searchParams.slug} not found in database`);
        metadataGraph = undefined;
      }
    }
  }

  let parsedSupernodes: string[][] | undefined;
  try {
    parsedSupernodes = searchParams.supernodes ? JSON.parse(searchParams.supernodes) : undefined;
  } catch (error) {
    console.error('Error parsing supernodes:', error);
  }

  let parsedClerps: string[][] | undefined;
  try {
    parsedClerps = searchParams.clerps ? JSON.parse(searchParams.clerps) : undefined;
  } catch (error) {
    console.error('Error parsing clerps:', error);
  }

  return (
    <GraphProvider
      initialModelIdToMetadataGraphsMap={modelIdToGraphMetadatasMap}
      initialModel={modelId}
      initialMetadataGraph={metadataGraph}
      initialPinnedIds={searchParams.pinnedIds}
      initialClickedId={searchParams.clickedId}
      initialSupernodes={parsedSupernodes}
      initialClerps={parsedClerps}
      initialPruningThreshold={searchParams.pruningThreshold ? Number(searchParams.pruningThreshold) : undefined}
      initialDensityThreshold={searchParams.densityThreshold ? Number(searchParams.densityThreshold) : undefined}
    >
      <GraphWrapper />
    </GraphProvider>
  );
}
