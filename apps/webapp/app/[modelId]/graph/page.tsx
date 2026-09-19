import { authOptions } from '@/app/api/auth/[...nextauth]/authOptions';
import { GraphProvider } from '@/components/provider/graph-provider';
import { GraphStateProvider } from '@/components/provider/graph-state-provider';
import { prisma } from '@/lib/db';
import { getModelById } from '@/lib/db/model';
import { ASSET_BASE_URL } from '@/lib/env';
import { GraphMetadataWithPartialRelations, ModelWithPartialRelations } from '@/prisma/generated/zod';
import { Metadata } from 'next';
import { getServerSession } from 'next-auth/next';
import { notFound } from 'next/navigation';
import { ModelToGraphMetadatasMap } from './graph-types';
import {
  ADDITIONAL_MODELS_TO_LOAD,
  ANT_BUCKET_URL,
  ANTHROPIC_MODEL_TO_DISPLAY_NAME,
  ANTHROPIC_MODELS,
  getGraphMetadatasFromBucket,
  parseGraphClerps,
  parseGraphSupernodes,
} from './utils';
import GraphWrapper from './wrapper';

export async function generateMetadata(props: {
  params: Promise<{ modelId: string }>;
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}): Promise<Metadata> {
  const searchParams = await props.searchParams;
  const params = await props.params;
  const { modelId } = params;
  const slug = searchParams.slug as string | undefined;

  // use modelIdToModelDisplayName to get the model name if it's there. othewise use it directly
  const modelName = ANTHROPIC_MODEL_TO_DISPLAY_NAME.get(modelId) || modelId;

  const title = `${slug ? `${slug} - ` : ''}${modelName} Graph | Neuronpedia`;
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

// Helper function to add graph metadata to the map without duplicates
function addGraphMetadataToMap(map: ModelToGraphMetadatasMap, modelId: string, graphMetadata: any) {
  // https://github.com/safety-research/circuit-tracer/pull/34
  // replace ↵ with \n, → with \t, and \r with ↵ in the prompt tokens
  if (graphMetadata.promptTokens) {
    graphMetadata.promptTokens = graphMetadata.promptTokens.map((token: string) =>
      token.replace(/⏎/g, '\n').replace(/→/g, '\t').replace(/↵/g, '\r'),
    );
  }
  if (!map[modelId]) {
    map[modelId] = [];
  }
  // Ensure no existing graph with the same slug to avoid overriding featured graphs
  if (!map[modelId].find((graph) => graph.slug === graphMetadata.slug)) {
    map[modelId].push(graphMetadata);
  }
}

// Helper function to merge graph metadata arrays into the map
function mergeGraphMetadataArrays(map: ModelToGraphMetadatasMap, modelId: string, graphMetadataArray: any[]) {
  if (!map[modelId]) {
    map[modelId] = graphMetadataArray;
  } else {
    map[modelId] = [...map[modelId], ...graphMetadataArray];
  }
}

// Include the owner name and the newest featured solution. The graph provider
// applies that solution as the default subgraph when a graph is selected.
const GRAPH_METADATA_INCLUDE = {
  user: {
    select: {
      name: true,
    },
  },
  subgraphs: {
    where: { isFeaturedSolution: true },
    orderBy: { createdAt: 'desc' as const },
    take: 1,
  },
};

// Helper function to get graph metadata with user info
async function getGraphMetadataWithUser(where: any) {
  return prisma.graphMetadata.findMany({
    where,
    include: GRAPH_METADATA_INCLUDE,
  });
}

// Look up one graph by model and slug. Returns undefined when it does not exist.
async function findGraphMetadata(
  modelId: string,
  slug: string,
): Promise<GraphMetadataWithPartialRelations | undefined> {
  const graphMetadata = await prisma.graphMetadata.findUnique({
    where: { modelId_slug: { modelId, slug } },
    include: GRAPH_METADATA_INCLUDE,
  });
  return (graphMetadata as GraphMetadataWithPartialRelations | null) ?? undefined;
}

async function getFeaturedGraphs() {
  const featuredGraphs = await prisma.graphMetadata.findMany({
    where: {
      isFeatured: true,
    },
    include: GRAPH_METADATA_INCLUDE,
  });
  return featuredGraphs;
}

export default async function Page(props: {
  params: Promise<{ modelId: string }>;
  searchParams: Promise<{
    logitDiff?: string;
    slug?: string;
    pinnedIds?: string;
    supernodes?: string;
    clerps?: string;
    pruningThreshold?: string;
    densityThreshold?: string;
    embed?: string;
    subgraph?: string;
    generate?: string;
    sourceSet?: string;
  }>;
}) {
  const searchParams = await props.searchParams;
  const params = await props.params;
  const { modelId } = params;
  const session = await getServerSession(authOptions);

  const embedParam = searchParams.embed as string | undefined;
  const embed = embedParam === 'true';

  // iterate through all baseUrls/buckets, and merge all the metadata we care about into a single map
  const modelIdToGraphMetadatasMap: ModelToGraphMetadatasMap = {};

  // add any additional models that should be loaded but aren't in the map yet
  for (const additionalModel of ADDITIONAL_MODELS_TO_LOAD) {
    modelIdToGraphMetadatasMap[additionalModel] = [];
  }

  // we always load ant models so add it to the available models
  for (const antModel of ANTHROPIC_MODELS) {
    modelIdToGraphMetadatasMap[antModel] = [];
  }

  let model: ModelWithPartialRelations | undefined;

  // if this is an ant model, then fetch the graph metadatas from their bucket
  if (ANTHROPIC_MODELS.has(modelId)) {
    const modelIdToGraphMetadata = await getGraphMetadatasFromBucket(ANT_BUCKET_URL);

    Object.keys(modelIdToGraphMetadata)
      .filter((m) => ANTHROPIC_MODELS.has(m))
      .forEach((m) => {
        mergeGraphMetadataArrays(modelIdToGraphMetadatasMap, m, modelIdToGraphMetadata[m]);
      });
  } else {
    // check that the model exists in our database
    model = (await getModelById(modelId)) as ModelWithPartialRelations;
    if (!model) {
      console.error(`couldn't find model ${modelId} for graph page`);
      notFound();
    }
  }

  // default the source set to the model's default graph source set, then set it to the slug's source set later if specified
  let initialSourceSet =
    searchParams.sourceSet || (ANTHROPIC_MODELS.has(modelId) ? modelId : model?.defaultGraphSourceSetName || '');

  // always get the user's graphMetadatas from our database
  const graphMetadatas =
    session && session.user && session.user.id ? await getGraphMetadataWithUser({ userId: session.user.id }) : [];

  // add those graphmetadatas to the modelIdToGraphMetadatasMap too
  graphMetadatas.forEach((graphMetadata) => {
    addGraphMetadataToMap(modelIdToGraphMetadatasMap, graphMetadata.modelId, graphMetadata);
  });

  // get the featured graphs, add them to the map
  const featuredGraphs = await getFeaturedGraphs();
  featuredGraphs.forEach((graphMetadata) => {
    addGraphMetadataToMap(modelIdToGraphMetadatasMap, graphMetadata.modelId, graphMetadata);
  });

  // set the metadata graph to show, if specified in the url. if we don't have it, look it up in the database
  let metadataGraph: GraphMetadataWithPartialRelations | undefined;
  let parsedSupernodes: string[][] | undefined;
  let parsedClerps: string[][] | undefined;
  let pinnedIds: string | undefined;
  let pruningThreshold: number | undefined;
  let densityThreshold: number | undefined;
  let didFindMatchingSubgraph = false;
  if (searchParams.slug) {
    metadataGraph = modelIdToGraphMetadatasMap[modelId]?.find((graph) => graph.slug === searchParams.slug);
    // if it's not in our map, look it up in the database
    if (!metadataGraph) {
      metadataGraph = await findGraphMetadata(modelId, searchParams.slug);
      // add this to our map
      if (metadataGraph) {
        addGraphMetadataToMap(modelIdToGraphMetadatasMap, modelId, metadataGraph);
      } else {
        console.error(`Graph with slug ${searchParams.slug} not found in database`);
      }
    }

    if (metadataGraph) {
      initialSourceSet = metadataGraph.sourceSetName || '';
    }

    // now, if there is a subgraph, we use the subgraph's data instead of the searchparams
    if (searchParams.subgraph) {
      const foundSubgraph = await prisma.graphMetadataSubgraph.findUnique({
        where: {
          id: searchParams.subgraph,
          graphMetadata: {
            modelId,
            slug: searchParams.slug,
          },
        },
      });
      if (!foundSubgraph) {
        console.error(`Subgraph with id ${searchParams.subgraph} not found in database`);
      } else {
        didFindMatchingSubgraph = true;
        pinnedIds = foundSubgraph.pinnedIds.join(',');
        parsedSupernodes = parseGraphSupernodes(JSON.stringify(foundSubgraph.supernodes));
        parsedClerps = parseGraphClerps(JSON.stringify(foundSubgraph.clerps));
        pruningThreshold = foundSubgraph.pruningThreshold || undefined;
        densityThreshold = foundSubgraph.densityThreshold || undefined;
      }
    }

    if (!didFindMatchingSubgraph) {
      pinnedIds = searchParams.pinnedIds;

      try {
        parsedSupernodes = searchParams.supernodes ? JSON.parse(searchParams.supernodes) : undefined;
      } catch (error) {
        console.error('Error parsing params supernodes:', error);
      }

      try {
        parsedClerps = searchParams.clerps ? JSON.parse(searchParams.clerps) : undefined;
      } catch (error) {
        console.error('Error parsing params clerps:', error);
      }
    }
  } else {
    // No slug specified - pick a default graph
    if (modelId === 'gemma-2-2b' && initialSourceSet === 'clt-hp') {
      const cltSlug = 'factthecapitalof-1789506352829';
      metadataGraph = modelIdToGraphMetadatasMap['gemma-2-2b']?.find((graph) => graph.slug === cltSlug);
      if (!metadataGraph) {
        metadataGraph = await findGraphMetadata(modelId, cltSlug);
        if (metadataGraph) {
          addGraphMetadataToMap(modelIdToGraphMetadatasMap, modelId, metadataGraph);
        }
      }
    } else if (modelId === 'gemma-2-2b' && modelIdToGraphMetadatasMap['gemma-2-2b']) {
      // The default subgraph for this graph is its featured solution in the database.
      metadataGraph = modelIdToGraphMetadatasMap['gemma-2-2b'].find(
        (graph) => graph.slug === 'gemma-fact-dallas-austin',
      );
    }

    // If no graph selected yet, try the first featured graph for this model
    if (!metadataGraph) {
      const featuredForModel = featuredGraphs.find((g) => g.modelId === modelId);
      if (featuredForModel) {
        metadataGraph = modelIdToGraphMetadatasMap[modelId]?.find((graph) => graph.slug === featuredForModel.slug);
      }
    }

    // Fall back to the first graph in the map
    if (!metadataGraph && modelIdToGraphMetadatasMap[modelId]?.length > 0) {
      [metadataGraph] = modelIdToGraphMetadatasMap[modelId];
    }

    if (!metadataGraph) {
      console.error(`No default graph found for model ${modelId}`);
    }
  }

  const generateParam = searchParams.generate as string | undefined;
  const showGenerateModal = generateParam === 'true';

  return (
    <GraphStateProvider>
      <GraphProvider
        initialModelIdToMetadataGraphsMap={modelIdToGraphMetadatasMap}
        initialModel={modelId}
        initialMetadataGraph={metadataGraph}
        initialPinnedIds={pinnedIds}
        initialSupernodes={parsedSupernodes}
        initialClerps={parsedClerps}
        initialSourceSetName={initialSourceSet}
        initialPruningThreshold={
          pruningThreshold || (searchParams.pruningThreshold ? Number(searchParams.pruningThreshold) : undefined)
        }
        initialDensityThreshold={
          densityThreshold || (searchParams.densityThreshold ? Number(searchParams.densityThreshold) : undefined)
        }
      >
        {!embed && (
          <div className="flex w-full flex-col bg-red-200 px-3 py-1.5 text-center text-[11px] font-medium text-red-700 sm:hidden">
            Use a larger screen to view this page. UI is simplified for mobile.
          </div>
        )}
        <GraphWrapper hasSlug={!!searchParams.slug} showGenerateModal={showGenerateModal} />
      </GraphProvider>
    </GraphStateProvider>
  );
}
