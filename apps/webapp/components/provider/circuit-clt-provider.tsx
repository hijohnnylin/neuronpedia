'use client';

import {
  CLTFeature,
  CLTGraph,
  CLTMetadataGraph,
  CltVisState,
  ModelToCLTMetadataGraphsMap,
  formatCLTGraphData,
  isHideLayer,
  makeCltFetchUrl,
  metadataScanToModelDisplayName,
  nodeHasFeatureDetail,
} from '@/app/[modelId]/circuit/clt/clt-utils';
import { useRouter } from 'next-nprogress-bar';

import { usePathname, useSearchParams } from 'next/navigation';
import { ReactNode, createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE = 30;

// Define the context type
type CircuitCLTContextType = {
  metadata: ModelToCLTMetadataGraphsMap;
  selectedModelId: string;
  selectedMetadataGraph: CLTMetadataGraph | null;
  selectedGraph: CLTGraph | null;
  setSelectedModelId: (modelId: string) => void;
  setSelectedMetadataGraph: (graph: CLTMetadataGraph | null) => void;
  setMetadata: (metadata: ModelToCLTMetadataGraphsMap) => void;
  getGraph: (graphSlug: string) => Promise<CLTGraph>;
  metadataScanToModelDisplayName: Map<string, string>;
  modelToBaseUrl: Record<string, string>;

  // isLoadingGraphData
  isLoadingGraphData: boolean;
  setIsLoadingGraphData: (isLoading: boolean) => void;

  // visState
  visState: CltVisState;
  setVisState: (newState: Partial<CltVisState>) => void;
  updateVisStateField: <K extends keyof CltVisState>(key: K, value: CltVisState[K]) => void;

  // logitDiff
  logitDiff: string | null;
  setLogitDiff: (logitDiff: string | null) => void;
};

// Create the context with a default value
const CircuitCLTContext = createContext<CircuitCLTContextType | undefined>(undefined);

export function getGraphUrl(graphSlug: string, baseUrl: string): string {
  const url = makeCltFetchUrl(baseUrl, `graph_data/${graphSlug}.json`);
  return url;
}

async function fetchInBatches<T>(items: any[], fetchFn: (item: any) => Promise<T>, batchSize: number): Promise<T[]> {
  const results: T[] = [];
  for (let i = 0; i < items.length; i += batchSize) {
    const batchItems = items.slice(i, i + batchSize);
    const batchPromises = batchItems.map(fetchFn);
    const batchResults = await Promise.all(batchPromises);
    results.push(...batchResults);
  }
  return results;
}

// Provider component
export function CircuitCLTProvider({
  children,
  initialMetadata = {},
  initialModelToBaseUrl = {},
  initialClickedId,
  initialLogitDiff,
  initialModel,
  initialMetadataGraph,
}: {
  children: ReactNode;
  initialMetadata?: ModelToCLTMetadataGraphsMap;
  initialModelToBaseUrl?: Record<string, string>;
  initialClickedId?: string;
  initialLogitDiff?: string;
  initialModel?: string;
  initialMetadataGraph?: CLTMetadataGraph;
}) {
  const [metadata, setMetadata] = useState<ModelToCLTMetadataGraphsMap>(initialMetadata);
  const [selectedModelId, setSelectedModelId] = useState<string>(
    initialModel || (Object.keys(initialMetadata).length > 0 ? Object.keys(initialMetadata)[0] : ''),
  );
  const [selectedMetadataGraph, setSelectedMetadataGraph] = useState<CLTMetadataGraph | null>(
    initialMetadataGraph ||
      (selectedModelId && initialMetadata[selectedModelId]?.length > 0 ? initialMetadata[selectedModelId][0] : null),
  );
  const [selectedGraph, setSelectedGraph] = useState<CLTGraph | null>(null);
  const [isLoadingGraphData, setIsLoadingGraphData] = useState<boolean>(true);

  const [visState, setVisStateInternal] = useState<CltVisState>({
    pinnedIds: [],
    hiddenIds: [],
    hoveredId: null,
    hoveredNodeId: null,
    hoveredCtxIdx: null,
    clickedId: initialClickedId || null,
    clickedCtxIdx: null,
    linkType: 'both',
    isShowAllLinks: '',
    isSyncEnabled: '',
    subgraph: null,
    isEditMode: 1,
    isHideLayer: false,
    sg_pos: '',
    isModal: true,
    isGridsnap: false,
    supernodes: [],
  });

  const [logitDiff, setLogitDiff] = useState<string | null>(initialLogitDiff || null);

  const modelToBaseUrl = initialModelToBaseUrl;

  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const updateParams = (keysToValues: Record<string, string>) => {
    const params = new URLSearchParams(searchParams.toString());
    Object.entries(keysToValues).forEach(([key, value]) => {
      if (value) {
        params.set(key, value);
      } else {
        params.delete(key);
      }
    });
    router.replace(`${pathname}?${params.toString()}`);
  };

  useEffect(() => {
    console.log('selectedMetadataGraph', selectedMetadataGraph);
    if (selectedMetadataGraph) {
      console.log('updating params', selectedModelId, selectedMetadataGraph.slug);
      updateParams({ model: selectedModelId, slug: selectedMetadataGraph.slug });
    }
  }, [selectedMetadataGraph]);

  // When selectedgraph is set, set the correct isHideLayer value
  useEffect(() => {
    if (selectedGraph) {
      // if we have qParams (default queryparams/visstate), parse them as visState and set it
      if (selectedGraph.qParams) {
        const blankVisState: CltVisState = {
          pinnedIds: [],
          hiddenIds: [],
          hoveredId: null,
          hoveredNodeId: null,
          hoveredCtxIdx: null,
          clickedId: null,
          clickedCtxIdx: null,
          linkType: 'both',
          isShowAllLinks: '',
          isSyncEnabled: '',
          subgraph: null,
          isEditMode: 1,
          isHideLayer: isHideLayer(selectedGraph.metadata.scan),
          sg_pos: '',
          isModal: true,
          isGridsnap: false,
          supernodes: [],
        };
        // if the qparams has a clickedId, only set it in the visState if it exists in the nodes array
        setVisStateInternal(() => ({
          ...blankVisState,
          ...selectedGraph.qParams,
          clickedId:
            selectedGraph.qParams.clickedId &&
            selectedGraph.nodes.some((d) => d.nodeId === selectedGraph.qParams.clickedId)
              ? selectedGraph.qParams.clickedId
              : null,
        }));
      }
    }
  }, [selectedGraph]);

  // Function to update the entire visState
  const setVisState = (newState: Partial<CltVisState>) => {
    setVisStateInternal((prevState) => ({ ...prevState, ...newState }));
  };

  // Function to update a single field of visState
  const updateVisStateField = useCallback(<K extends keyof CltVisState>(key: K, value: CltVisState[K]) => {
    setVisStateInternal((prevState) => ({ ...prevState, [key]: value }));
  }, []);

  async function fetchFeatureDetail(modelId: string, feature: number, baseUrl: string): Promise<CLTFeature | null> {
    const response = await fetch(makeCltFetchUrl(baseUrl, `features/${modelId}/${feature}.json`));
    if (!response.ok) {
      console.error(`Failed to fetch feature detail for ${modelId}/${feature}`);
      return null;
    }
    const data = await response.json();
    return data as CLTFeature;
  }

  // Function to fetch graph data
  async function getGraph(graphSlug: string): Promise<CLTGraph> {
    const response = await fetch(getGraphUrl(graphSlug, modelToBaseUrl[selectedModelId]));

    if (!response.ok) {
      throw new Error(`Failed to fetch graph data for ${graphSlug}`);
    }

    const data = (await response.json()) as CLTGraph;
    const formattedData = formatCLTGraphData(data, logitDiff);

    const featureDetails = await fetchInBatches(
      formattedData.nodes,
      (d) => {
        if (nodeHasFeatureDetail(d)) {
          return fetchFeatureDetail(selectedModelId, d.feature, modelToBaseUrl[selectedModelId]);
        }
        return Promise.resolve(null);
      },
      FEATURE_DETAIL_DOWNLOAD_BATCH_SIZE,
    );

    formattedData.nodes.forEach((d, i) => {
      // eslint-disable-next-line no-param-reassign
      d.featureDetail = featureDetails[i] || undefined;
    });

    setIsLoadingGraphData(false);
    return formattedData;
  }

  // Fetch graph data when selected metadata graph changes
  useEffect(() => {
    if (selectedMetadataGraph) {
      setIsLoadingGraphData(true);
      getGraph(selectedMetadataGraph.slug).then((g) => {
        setSelectedGraph(g);
      });
    }
  }, [selectedMetadataGraph]);

  // Provide the context value
  const contextValue = useMemo(
    () => ({
      metadata,
      selectedModelId,
      selectedMetadataGraph,
      selectedGraph,
      modelToBaseUrl,
      setSelectedModelId,
      setSelectedMetadataGraph,
      setMetadata,
      getGraph,
      metadataScanToModelDisplayName,
      visState,
      setVisState,
      updateVisStateField,
      logitDiff,
      setLogitDiff,
      isLoadingGraphData,
      setIsLoadingGraphData,
    }),
    [
      metadata,
      selectedModelId,
      selectedMetadataGraph,
      selectedGraph,
      visState,
      logitDiff,
      updateVisStateField,
      isLoadingGraphData,
      setIsLoadingGraphData,
    ],
  );

  return <CircuitCLTContext.Provider value={contextValue}>{children}</CircuitCLTContext.Provider>;
}

// Custom hook to use the context
export function useCircuitCLT() {
  const context = useContext(CircuitCLTContext);
  if (context === undefined) {
    throw new Error('useCircuitCLT must be used within a CircuitCLTProvider');
  }
  return context;
}
