import FeatureDashboard from '@/app/[modelId]/[layer]/[index]/feature-dashboard';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/shadcn/hover-card';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import * as Tooltip from '@radix-ui/react-tooltip';
import { ExternalLinkIcon, HelpCircle } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { LoadingSquare } from '../svg/loading-square';

type ResidChannel = {
  id: string;
  index: number;
};

type TraceNode = {
  layer: number;
  neuron: number;
  read_weight: number;
  via_channel: number;
  write_weight: number;
  children?: TraceNode[] | null;
  parents?: TraceNode[] | null;
};

type ConnectedNeuron = {
  layer: number;
  index: string;
  current: boolean;
  resChannels: ResidChannel[];
  direction: 'forward' | 'backward' | 'current';
  readWeight?: number;
  writeWeight?: number;
};

type NeuronExplanation = {
  modelId: string;
  layer: string;
  index: string;
  explanations: { id: string; description: string }[];
};

type ResidChannelExplanation = {
  modelId: string;
  layer: string; // e.g., "0-resid", "2-resid"
  index: string; // channel index
  explanations: { id: string; description: string }[];
};

type SparsityData = {
  layer: number;
  neuron: number;
  trace_forward: TraceNode[];
  trace_backward: TraceNode[];
  neuronExplanations: NeuronExplanation[];
  residChannelExplanations: ResidChannelExplanation[];
};

// Helper function to extract all unique channels from trace data
// Orders channels: input channels (from backward trace) on left, output channels (from forward trace) on right
function extractChannels(traceForward: TraceNode[], traceBackward: TraceNode[]): ResidChannel[] {
  const inputChannels = new Set<number>();
  const outputChannels = new Set<number>();

  const collectInputChannels = (nodes: TraceNode[]) => {
    nodes.forEach((node) => {
      inputChannels.add(node.via_channel);
      if (node.parents) collectInputChannels(node.parents);
    });
  };

  const collectOutputChannels = (nodes: TraceNode[]) => {
    nodes.forEach((node) => {
      outputChannels.add(node.via_channel);
      if (node.children) collectOutputChannels(node.children);
    });
  };

  collectInputChannels(traceBackward);
  collectOutputChannels(traceForward);

  // Order: input-only channels first, then channels that are both, then output-only channels
  const inputOnlyChannels = Array.from(inputChannels)
    .filter((c) => !outputChannels.has(c))
    .sort((a, b) => a - b);
  const bothChannels = Array.from(inputChannels)
    .filter((c) => outputChannels.has(c))
    .sort((a, b) => a - b);
  const outputOnlyChannels = Array.from(outputChannels)
    .filter((c) => !inputChannels.has(c))
    .sort((a, b) => a - b);

  const orderedChannels = [...inputOnlyChannels, ...bothChannels, ...outputOnlyChannels];

  return orderedChannels.map((channelIndex) => ({
    id: `channel.${channelIndex}`,
    index: channelIndex,
  }));
}

// Helper function to build connected neurons from trace data
function buildConnectedNeurons(
  currentLayer: number,
  currentNeuronIndex: number,
  traceForward: TraceNode[],
  traceBackward: TraceNode[],
  resChannels: ResidChannel[],
): ConnectedNeuron[] {
  const neurons: ConnectedNeuron[] = [];
  const channelMap = new Map(resChannels.map((c) => [c.index, c]));

  // Add backward trace neurons (upstream)
  traceBackward.forEach((node) => {
    const channel = channelMap.get(node.via_channel);
    if (channel) {
      neurons.push({
        layer: node.layer,
        index: String(node.neuron),
        current: false,
        resChannels: [channel],
        direction: 'backward',
        readWeight: node.read_weight,
        writeWeight: node.write_weight,
      });
    }
  });

  // Add current neuron
  const currentChannels = [
    ...new Set([...traceForward.map((n) => n.via_channel), ...traceBackward.map((n) => n.via_channel)]),
  ]
    .map((idx) => channelMap.get(idx))
    .filter((c): c is ResidChannel => c !== undefined);

  neurons.push({
    layer: currentLayer,
    index: String(currentNeuronIndex),
    current: true,
    resChannels: currentChannels,
    direction: 'current',
  });

  // Add forward trace neurons (downstream)
  traceForward.forEach((node) => {
    const channel = channelMap.get(node.via_channel);
    if (channel) {
      neurons.push({
        layer: node.layer,
        index: String(node.neuron),
        current: false,
        resChannels: [channel],
        direction: 'forward',
        readWeight: node.read_weight,
        writeWeight: node.write_weight,
      });
    }
  });

  // Sort by layer, then by neuron index for consistent ordering
  return neurons.sort((a, b) => a.layer - b.layer || a.index.localeCompare(b.index));
}

// Helper function to compute neuron positions grouped by layer
// Backward neurons go on the left, forward/current neurons go on the right
// Layer heights are based on the number of neurons in each layer
function computeNeuronPositions(
  neurons: ConnectedNeuron[],
  leftBaseLeft: number, // Base position for backward neurons (left side)
  rightBaseLeft: number, // Base position for forward/current neurons (right side)
): Map<string, { left: number; top: number; layerIndex: number; indexInLayer: number }> {
  const positions = new Map<string, { left: number; top: number; layerIndex: number; indexInLayer: number }>();

  // Separate neurons by direction
  const backwardNeurons = neurons.filter((n) => n.direction === 'backward');
  const forwardAndCurrentNeurons = neurons.filter((n) => n.direction !== 'backward');

  // Group backward neurons by layer
  const backwardLayerGroups = new Map<number, ConnectedNeuron[]>();
  backwardNeurons.forEach((neuron) => {
    const group = backwardLayerGroups.get(neuron.layer) || [];
    group.push(neuron);
    backwardLayerGroups.set(neuron.layer, group);
  });

  // Group forward/current neurons by layer
  const forwardLayerGroups = new Map<number, ConnectedNeuron[]>();
  forwardAndCurrentNeurons.forEach((neuron) => {
    const group = forwardLayerGroups.get(neuron.layer) || [];
    group.push(neuron);
    forwardLayerGroups.set(neuron.layer, group);
  });

  // Get all unique layers for consistent vertical positioning
  const allLayers = new Set([...backwardLayerGroups.keys(), ...forwardLayerGroups.keys()]);
  const sortedLayers = Array.from(allLayers).sort((a, b) => a - b);

  // Calculate neuron count per layer (max of backward and forward sides)
  const neuronCountPerLayer = sortedLayers.map((layer) => {
    const backwardCount = (backwardLayerGroups.get(layer) || []).length;
    const forwardCount = (forwardLayerGroups.get(layer) || []).length;
    return Math.max(backwardCount, forwardCount, 1); // At least 1
  });

  // Calculate cumulative top positions based on neuron counts
  // Each layer gets height based on its neuron count
  const neuronSpacing = 24; // Vertical space per neuron
  const minLayerHeight = 40; // Minimum height for a layer
  const layerGap = 20; // Gap between layers

  const layerTops: number[] = [];
  let currentTop = 52; // Starting position
  sortedLayers.forEach((layer, idx) => {
    layerTops.push(currentTop);
    const layerHeight = Math.max(minLayerHeight, neuronCountPerLayer[idx] * neuronSpacing);
    currentTop += layerHeight + layerGap;
  });

  // Compute positions for backward neurons (left side)
  sortedLayers.forEach((layer, layerIndex) => {
    const neuronsInLayer = backwardLayerGroups.get(layer) || [];
    const layerBaseTop = layerTops[layerIndex];
    const layerHeight = Math.max(minLayerHeight, neuronCountPerLayer[layerIndex] * neuronSpacing);
    const layerCenterY = layerBaseTop + layerHeight / 2;

    neuronsInLayer.forEach((neuron, indexInLayer) => {
      // Same X position for all neurons in layer (at the left base)
      const neuronLeft = leftBaseLeft;

      // Spread neurons vertically within the layer, centered
      const verticalOffset = (indexInLayer - (neuronsInLayer.length - 1) / 2) * neuronSpacing;
      const neuronTop = layerCenterY + verticalOffset - 6; // -6 to center the circle

      positions.set(neuron.index, {
        left: neuronLeft,
        top: neuronTop,
        layerIndex,
        indexInLayer,
      });
    });
  });

  // Compute positions for forward/current neurons (right side)
  sortedLayers.forEach((layer, layerIndex) => {
    const neuronsInLayer = forwardLayerGroups.get(layer) || [];
    const layerBaseTop = layerTops[layerIndex];
    const layerHeight = Math.max(minLayerHeight, neuronCountPerLayer[layerIndex] * neuronSpacing);
    const layerCenterY = layerBaseTop + layerHeight / 2;

    neuronsInLayer.forEach((neuron, indexInLayer) => {
      // Same X position for all neurons in layer (at the right base)
      const neuronLeft = rightBaseLeft;

      // Spread neurons vertically within the layer, centered
      const verticalOffset = (indexInLayer - (neuronsInLayer.length - 1) / 2) * neuronSpacing;
      const neuronTop = layerCenterY + verticalOffset - 6; // -6 to center the circle

      positions.set(neuron.index, {
        left: neuronLeft,
        top: neuronTop,
        layerIndex,
        indexInLayer,
      });
    });
  });

  return positions;
}

// Example data for testing - kept for reference and as fallback
// const exampleData: SparsityData = {
//   layer: 2,
//   neuron: 1717,
//   trace_forward: [
//     {
//       layer: 6,
//       neuron: 2723,
//       read_weight: -0.3709927499294281,
//       via_channel: 506,
//       write_weight: -0.21734599769115448,
//       children: null,
//     },
//     {
//       layer: 5,
//       neuron: 5078,
//       read_weight: -0.3670603036880493,
//       via_channel: 506,
//       write_weight: -0.21734599769115448,
//       children: null,
//     },
//     {
//       layer: 3,
//       neuron: 1775,
//       read_weight: -0.2945079207420349,
//       via_channel: 506,
//       write_weight: -0.21734599769115448,
//       children: null,
//     },
//     {
//       layer: 3,
//       neuron: 3801,
//       read_weight: -0.28046295046806335,
//       via_channel: 506,
//       write_weight: -0.21734599769115448,
//       children: null,
//     },
//     {
//       layer: 4,
//       neuron: 5333,
//       read_weight: 0.3004470467567444,
//       via_channel: 249,
//       write_weight: 0.19179122149944305,
//       children: null,
//     },
//     {
//       layer: 6,
//       neuron: 5877,
//       read_weight: -0.2590516209602356,
//       via_channel: 506,
//       write_weight: -0.21734599769115448,
//       children: null,
//     },
//     {
//       layer: 4,
//       neuron: 4129,
//       read_weight: 0.29327937960624695,
//       via_channel: 249,
//       write_weight: 0.19179122149944305,
//       children: null,
//     },
//     {
//       layer: 3,
//       neuron: 46,
//       read_weight: 0.29263511300086975,
//       via_channel: 249,
//       write_weight: 0.19179122149944305,
//       children: null,
//     },
//     {
//       layer: 6,
//       neuron: 7171,
//       read_weight: -0.2514074146747589,
//       via_channel: 506,
//       write_weight: -0.21734599769115448,
//       children: null,
//     },
//     {
//       layer: 3,
//       neuron: 3530,
//       read_weight: -0.2436772584915161,
//       via_channel: 506,
//       write_weight: -0.21734599769115448,
//       children: null,
//     },
//   ],
//   trace_backward: [
//     {
//       layer: 0,
//       neuron: 3244,
//       write_weight: -0.32191646099090576,
//       via_channel: 1451,
//       read_weight: 0.16233937442302704,
//       parents: null,
//     },
//     {
//       layer: 0,
//       neuron: 6260,
//       write_weight: 0.1729516088962555,
//       via_channel: 1451,
//       read_weight: 0.16233937442302704,
//       parents: null,
//     },
//     {
//       layer: 0,
//       neuron: 2493,
//       write_weight: -0.16061851382255554,
//       via_channel: 1451,
//       read_weight: 0.16233937442302704,
//       parents: null,
//     },
//     {
//       layer: 0,
//       neuron: 2818,
//       write_weight: 0.1580868810415268,
//       via_channel: 1451,
//       read_weight: 0.16233937442302704,
//       parents: null,
//     },
//     {
//       layer: 0,
//       neuron: 7374,
//       write_weight: -0.15642257034778595,
//       via_channel: 1451,
//       read_weight: 0.16233937442302704,
//       parents: null,
//     },
//   ],
//   neuronExplanations: [],
//   residChannelExplanations: [],
// };

export default function ConnectedNeuronsPane({
  currentNeuron,
}: {
  currentNeuron: NeuronWithPartialRelations | undefined;
}) {
  const [hoveredNeuronIndex, setHoveredNeuronIndex] = useState<string | null>(null);
  const [hoveredChannelId, setHoveredChannelId] = useState<string | null>(null);
  const [hoveredFeature, setHoveredFeature] = useState<NeuronWithPartialRelations | undefined>();

  // State for API data
  const [sparsityData, setSparsityData] = useState<SparsityData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // const [debugData, setDebugData] = useState<string | null>(null);

  // const { setFeatureModalFeature, setFeatureModalOpen, getSource } = useGlobalContext();

  // Fetch connected neurons data from API when currentNeuron changes
  useEffect(() => {
    const fetchConnectedNeurons = async () => {
      if (!currentNeuron?.modelId || !currentNeuron?.layer || !currentNeuron?.index) {
        setSparsityData(null);
        return;
      }

      // Extract numeric layer from layer string (e.g., "2-mlp" -> 2)
      const layerMatch = currentNeuron.layer.match(/^(\d+)/);
      if (!layerMatch) {
        setError('Invalid layer format');
        return;
      }
      const layerNum = parseInt(layerMatch[1], 10);

      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(
          `/api/sparsity/connected-neurons?modelId=${encodeURIComponent(currentNeuron.modelId)}&layer=${layerNum}&index=${encodeURIComponent(currentNeuron.index)}`,
        );

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || `API error: ${response.status}`);
        }

        const apiData = await response.json();
        // setDebugData(JSON.stringify(apiData, null, 2));
        setSparsityData({
          layer: apiData.layer,
          neuron: apiData.neuron,
          trace_forward: apiData.trace_forward,
          trace_backward: apiData.trace_backward,
          neuronExplanations: apiData.neuronExplanations || [],
          residChannelExplanations: apiData.residChannelExplanations || [],
        });
      } catch (err) {
        console.error('Failed to fetch connected neurons:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch connected neurons');
        setSparsityData(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchConnectedNeurons();
  }, [currentNeuron?.modelId, currentNeuron?.layer, currentNeuron?.index]);

  // Derive channels and connected neurons from trace data (only if we have data)
  const resChannels = sparsityData ? extractChannels(sparsityData.trace_forward, sparsityData.trace_backward) : [];
  const connectedNeurons = sparsityData
    ? buildConnectedNeurons(
        sparsityData.layer,
        sparsityData.neuron,
        sparsityData.trace_forward,
        sparsityData.trace_backward,
        resChannels,
      )
    : [];

  const getAllNeuronExplanations = (layer: number, neuronIndex: string): { id: string; description: string }[] => {
    if (!sparsityData?.neuronExplanations) return [];
    const neuronLayer = `${layer}-mlp`;
    return sparsityData.neuronExplanations
      .filter((e) => e.layer === neuronLayer && e.index === neuronIndex)
      .flatMap((e) => e.explanations);
  };

  // Helper function to get the first explanation for a neuron
  const getNeuronExplanation = (
    layer: number,
    neuronIndex: string,
    bottomOrTop: 'bottom' | 'top' | undefined,
  ): string | null => {
    if (!sparsityData?.neuronExplanations) return null;
    const neuronLayer = `${layer}-mlp`;
    const explanation = sparsityData.neuronExplanations.find((e) => e.layer === neuronLayer && e.index === neuronIndex);
    if (!explanation?.explanations || explanation.explanations.length === 0) return null;

    const BOTTOM_EXPLANATION_SUFFIX = ' (negative activations)';
    if (bottomOrTop === 'bottom') {
      const bottomExplanation = explanation.explanations.find((e) => e.description.includes(BOTTOM_EXPLANATION_SUFFIX));
      return bottomExplanation ? bottomExplanation.description.replace(BOTTOM_EXPLANATION_SUFFIX, '') : null;
    }
    if (bottomOrTop === 'top') {
      return (
        explanation.explanations.find((e) => !e.description.includes(BOTTOM_EXPLANATION_SUFFIX))?.description || null
      );
    }
    // When bottomOrTop is undefined, return the first available explanation (prefer top, fallback to bottom)
    const topExplanation = explanation.explanations.find((e) => !e.description.includes(BOTTOM_EXPLANATION_SUFFIX));
    if (topExplanation) {
      return topExplanation.description;
    }
    // Fallback to bottom explanation if no top explanation exists
    const bottomExplanation = explanation.explanations.find((e) => e.description.includes(BOTTOM_EXPLANATION_SUFFIX));
    return bottomExplanation ? bottomExplanation.description.replace(BOTTOM_EXPLANATION_SUFFIX, '') : null;
  };

  // Helper function to get all explanations for a resid channel across ALL layers
  const getAllResidChannelExplanations = (
    channelIndex: number,
  ): { layer: string; explanations: { id: string; description: string }[] }[] => {
    if (!sparsityData?.residChannelExplanations) return [];
    return sparsityData.residChannelExplanations
      .filter((e) => e.index === channelIndex.toString())
      .map((e) => ({
        layer: e.layer,
        explanations: e.explanations,
      }))
      .sort((a, b) => {
        // Sort by layer number
        const layerA = parseInt(a.layer.split('-')[0], 10);
        const layerB = parseInt(b.layer.split('-')[0], 10);
        return layerA - layerB;
      });
  };

  // Layout constants - channels are centered, neurons positioned relative to channels
  const gapBetweenNeuronsAndChannels = 60;
  // Visual channel width (from leftmost to rightmost channel line edge)
  // Each channel is 5px wide, spaced 36px apart
  const visualChannelWidth = (resChannels.length - 1) * 36 + 5;
  const horizontalSpacing = 30;

  // Calculate neuron extents to determine inner container width
  const backwardNeurons = connectedNeurons.filter((n) => n.direction === 'backward');
  const forwardNeurons = connectedNeurons.filter((n) => n.direction !== 'backward');

  // Count max neurons per layer for width calculation
  const getMaxInLayer = (neurons: ConnectedNeuron[]) => {
    const counts = new Map<number, number>();
    neurons.forEach((n) => counts.set(n.layer, (counts.get(n.layer) || 0) + 1));
    return Math.max(0, ...counts.values());
  };
  const maxBackwardInLayer = getMaxInLayer(backwardNeurons);
  const maxForwardInLayer = getMaxInLayer(forwardNeurons);

  // Calculate widths needed on each side of channels
  const backwardWidth = maxBackwardInLayer * horizontalSpacing + gapBetweenNeuronsAndChannels + 20;
  const forwardWidth = maxForwardInLayer * horizontalSpacing + gapBetweenNeuronsAndChannels + 20;

  // Inner container width: enough to hold backward neurons + channels + forward neurons
  const innerContentWidth = backwardWidth + visualChannelWidth + forwardWidth;
  const innerContentCenter = innerContentWidth / 2;

  // Calculate absolute positions within the inner container
  const channelStartX = innerContentCenter - visualChannelWidth / 2;
  // Left neurons: same distance from leftmost channel as right neurons from rightmost channel
  const backwardNeuronsBaseLeft = channelStartX - gapBetweenNeuronsAndChannels - 8;
  const forwardNeuronsBaseLeft = channelStartX + visualChannelWidth + gapBetweenNeuronsAndChannels + 5;

  // Compute neuron positions grouped by layer
  const neuronPositions = computeNeuronPositions(
    connectedNeurons,
    backwardNeuronsBaseLeft, // Right edge of backward neurons area
    forwardNeuronsBaseLeft, // Left edge of forward neurons area
  );

  // Get number of unique layers for height calculation
  const uniqueLayers = new Set(connectedNeurons.map((n) => n.layer)).size;

  // Fixed height for current neuron rectangle (used for both rectangle and layer segment)
  const currentNeuronRectHeight = 68;
  // Height of the arrow/fade area above the current neuron rectangle
  const currentNeuronArrowAreaHeight = 28;

  // Compute layer label positions (vertically centered for each layer) with min/max for segment heights
  const layerLabelPositions = (() => {
    const layerGroups = new Map<number, number[]>();
    connectedNeurons.forEach((neuron) => {
      const position = neuronPositions.get(neuron.index);
      if (position) {
        const tops = layerGroups.get(neuron.layer) || [];
        tops.push(position.top);
        layerGroups.set(neuron.layer, tops);
      }
    });

    const positions: { layer: number; centerY: number; minY: number; maxY: number }[] = [];
    layerGroups.forEach((tops, layer) => {
      const minTop = Math.min(...tops);
      const maxTop = Math.max(...tops);
      const centerY = (minTop + maxTop) / 2 + 6; // +6 for circle center
      // Add padding around neurons (6 for circle radius + some margin)
      const padding = 12;
      positions.push({
        layer,
        centerY,
        minY: minTop - padding,
        maxY: maxTop + 12 + padding, // +12 for circle height
      });
    });

    return positions.sort((a, b) => a.layer - b.layer);
  })();

  // Calculate where channels start vertically (used for label positioning)
  const channelsStartY = layerLabelPositions.length > 0 ? layerLabelPositions[0].minY - 16 : 0;

  return (
    // TODO: hide if not relevant model
    <div
      className={`mb-2 hidden flex-col gap-x-2 overflow-hidden rounded-lg border bg-white px-3 pb-4 pt-2 text-xs shadow transition-all sm:mb-3 ${
        true ? 'sm:flex' : 'sm:hidden'
      }`}
    >
      <div className="mb-1.5 flex w-full flex-row items-center justify-center gap-x-1 text-[10px] font-normal uppercase text-slate-400">
        Explorer
        <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
          <Tooltip.Root>
            <Tooltip.Trigger asChild>
              <button type="button">
                <HelpCircle className="h-2.5 w-2.5" />
              </button>
            </Tooltip.Trigger>
            <Tooltip.Portal>
              <Tooltip.Content
                className="rounded bg-slate-500 px-3 py-2 text-xs text-white"
                sideOffset={5}
                side="right"
              >
                Explores the neural network connections between neurons and channels. Hover over the vertical lines for
                details on residual stream channels, or hover over the neurons for details on the neurons.
                <Tooltip.Arrow className="fill-slate-500" />
              </Tooltip.Content>
            </Tooltip.Portal>
          </Tooltip.Root>
        </Tooltip.Provider>
      </div>
      {error && <div className="mb-2 rounded bg-red-50 px-2 py-1 text-[10px] text-red-600">Error: {error}</div>}
      <div
        className="relative mb-3 mt-0 flex-col"
        style={{
          minHeight: `${Math.max(320, layerLabelPositions.length > 0 ? layerLabelPositions[layerLabelPositions.length - 1].maxY + 30 : 320)}px`,
        }}
      >
        {/* Loading state - centered */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <LoadingSquare size={24} />
          </div>
        )}
        {/* No data state - centered */}
        {!sparsityData && !isLoading && !error && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-400">
            No connected neurons data available
          </div>
        )}
        {/* Content - only render when we have data */}
        {sparsityData && (
          <>
            {/* <div>{debugData}</div> */}

            {/* Layer labels - absolutely positioned on far left of pane, aligned to top of layer */}
            {layerLabelPositions.map(({ layer, minY }) => (
              <div
                key={`layer-label-${layer}`}
                className="absolute text-[9px] font-medium text-slate-400"
                style={{
                  left: '-4px',
                  top: `${minY - 2}px`,
                  whiteSpace: 'nowrap',
                }}
              >
                L{layer}
              </div>
            ))}

            {/* Inner container - centered in space right of layer labels */}
            <div
              className="relative"
              style={{
                width: `${innerContentWidth}px`,
                height: '100%',
                marginLeft: `calc(50% - ${innerContentWidth / 2}px - 6px)`, // Offset left to account for layer labels on left
              }}
            >
              {/* Residual Stream label above channels */}
              <div
                className="absolute text-[10px] font-medium text-slate-400"
                style={{
                  // Center over the actual channel lines: from first channel to last channel + channel width
                  left: `${channelStartX + ((resChannels.length - 1) * 36 + 5) / 2}px`,
                  top: `${channelsStartY - 34}px`,
                  transform: 'translateX(-50%)',
                  whiteSpace: 'nowrap',
                }}
              >
                Residual Stream
              </div>

              {/* Connection lines between neurons and channels (rendered first so they appear below channels) */}
              {(() => {
                // Calculate max absolute weight across all non-current neurons for stroke width normalization
                const nonCurrentNeurons = connectedNeurons.filter((n) => !n.current);
                const maxAbsWeight = Math.max(
                  ...nonCurrentNeurons.map((n) => {
                    const weight = n.direction === 'backward' ? n.writeWeight : n.readWeight;
                    return Math.abs(weight ?? 0);
                  }),
                  0.001, // Prevent division by zero
                );

                return nonCurrentNeurons.map((neuron) => {
                  const position = neuronPositions.get(neuron.index);
                  if (!position) return null;

                  return (
                    <div key={`lines-${neuron.index}`}>
                      {neuron.resChannels.map((channel) => {
                        const channelIndex = resChannels.findIndex((c) => c.id === channel.id);
                        const channelLeft = channelStartX + channelIndex * 36;
                        const neuronLeft = position.left;
                        const neuronTop = position.top;

                        // Check if this path should be highlighted
                        const isNeuronHovered = hoveredNeuronIndex === neuron.index;
                        const isChannelHovered = hoveredChannelId === channel.id;

                        const isHighlighted = isNeuronHovered || isChannelHovered;

                        // Non-current neurons:
                        // - Backward (upstream) neurons WRITE to the channel → arrow to channel
                        // - Forward (downstream) neurons READ from the channel → arrow to neuron
                        const isBackward = neuron.direction === 'backward';
                        const isOutputLine = isBackward;

                        const neuronConnectionY = neuronTop + 6;
                        const channelConnectionY = neuronTop + 6;

                        // Determine which side of the neuron to connect to based on direction
                        // Backward neurons (left side): connect to right edge of neuron
                        // Forward neurons (right side): connect to left edge of neuron
                        const neuronCircleWidth = 12;
                        const neuronEdgeX = isBackward ? neuronLeft + neuronCircleWidth : neuronLeft;

                        const midX = channelLeft + 2.5 + (neuronEdgeX - channelLeft - 2.5) * 0.6;
                        const midY = (channelConnectionY + neuronConnectionY) / 2;

                        // Determine stroke color
                        const strokeColor = isHighlighted
                          ? '#64748b' // slate-500 - darker gray when highlighted
                          : '#cbd5e1'; // slate-300 - gray by default

                        // Calculate stroke width proportional to absolute weight (range: 1-4px)
                        const weight = isBackward ? neuron.writeWeight : neuron.readWeight;
                        const absWeight = Math.abs(weight ?? 0);
                        const normalizedWeight = absWeight / maxAbsWeight;
                        const strokeWidth = 1 + normalizedWeight * 3; // Scale from 1px to 4px

                        // Create unique marker IDs
                        const markerEndId = `arrow-end-${neuron.index}-${channel.id}`;
                        const markerStartId = `arrow-start-${neuron.index}-${channel.id}`;

                        const channelArrowOffset = isOutputLine ? -6 : 0;
                        const adjustedNeuronX = isBackward ? neuronEdgeX - 3 : neuronEdgeX - 2;
                        const adjustedChannelX = channelLeft + 2 + channelArrowOffset;

                        return (
                          <svg
                            key={`line-${neuron.index}-${channel.id}`}
                            className="pointer-events-none absolute"
                            style={{
                              left: 2.5,
                              top: 0,
                              width: '100%',
                              height: '100%',
                              overflow: 'visible',
                            }}
                          >
                            <defs>
                              {/* Arrow pointing forward (toward end of path) */}
                              <marker
                                id={markerEndId}
                                markerWidth="3"
                                markerHeight="3"
                                refX="2.7"
                                refY="1.5"
                                orient="auto"
                                markerUnits="strokeWidth"
                              >
                                <path d="M0,0 L0,3 L3,1.5 z" fill={strokeColor} />
                              </marker>
                              {/* Arrow pointing backward (toward start of path, for outputs) */}
                              <marker
                                id={markerStartId}
                                markerWidth="3"
                                markerHeight="3"
                                refX="0.3"
                                refY="1.5"
                                orient="auto"
                                markerUnits="strokeWidth"
                              >
                                <path d="M3,0 L3,3 L0,1.5 z" fill={strokeColor} />
                              </marker>
                            </defs>
                            <path
                              d={`M ${adjustedChannelX} ${channelConnectionY} 
                            Q ${midX} ${midY},
                              ${adjustedNeuronX} ${neuronConnectionY}`}
                              stroke={strokeColor}
                              strokeWidth={strokeWidth}
                              fill="none"
                              markerStart={isOutputLine ? `url(#${markerStartId})` : undefined}
                              markerEnd={isOutputLine ? undefined : `url(#${markerEndId})`}
                              style={{ transition: 'stroke 0.3s ease, stroke-width 0.3s ease' }}
                            />
                          </svg>
                        );
                      })}
                    </div>
                  );
                });
              })()}

              {resChannels.map((channel, arrayIndex) => {
                // Check if this channel is connected to the hovered neuron
                const isConnectedToHoveredNeuron = hoveredNeuronIndex
                  ? connectedNeurons
                      .find((n) => n.index === hoveredNeuronIndex)
                      ?.resChannels.some((c) => c.id === channel.id)
                  : false;

                // Check if this channel is in trace_backward (input) or trace_forward (output)
                const isInputCh = sparsityData?.trace_backward.some((n) => n.via_channel === channel.index) ?? false;
                const isOutputCh = sparsityData?.trace_forward.some((n) => n.via_channel === channel.index) ?? false;
                const isConnectedCh = isInputCh || isOutputCh;

                // Check if nothing is hovered
                const nothingHoveredCh = hoveredChannelId === null && hoveredNeuronIndex === null;

                // Check if this channel is being hovered or connected to hovered neuron, OR nothing hovered and connected
                const isHighlighted =
                  hoveredChannelId === channel.id || isConnectedToHoveredNeuron || (nothingHoveredCh && isConnectedCh);

                // Text colors - light blue by default, solid blue when highlighted
                const defaultTextColor = '#94a3b8'; // slate-400
                const hoverTextColor = '#0369a1'; // sky-700

                const channelHeight = Math.max(400, 64 + uniqueLayers * 80 + 40);

                // Compute segment boundaries based on layer positions
                const sortedLayerPositions = [...layerLabelPositions].sort((a, b) => a.layer - b.layer);

                // Create segments for each layer - segments connect at midpoints between layer neuron ranges
                const segments: { layer: number; top: number; height: number }[] = [];
                const currentNeuronLayer = sparsityData?.layer ?? 0;

                // Calculate boundaries between layers based on neuron positions
                const firstLayer = sortedLayerPositions[0];
                const boundaries: number[] = [firstLayer.minY - 16]; // Start a bit above first layer's neurons
                for (let i = 0; i < sortedLayerPositions.length - 1; i += 1) {
                  const currentLayerPos = sortedLayerPositions[i];
                  const nextLayerPos = sortedLayerPositions[i + 1];

                  // Boundary is midpoint between current layer's maxY and next layer's minY
                  const boundary = (currentLayerPos.maxY + nextLayerPos.minY) / 2;
                  boundaries.push(boundary);
                }
                // Add final boundary
                const lastLayer = sortedLayerPositions[sortedLayerPositions.length - 1];
                boundaries.push(lastLayer.maxY + 16);

                // Create segments using the boundaries
                sortedLayerPositions.forEach((layerPos, idx) => {
                  let segmentTop = boundaries[idx];
                  let segmentBottom = boundaries[idx + 1];

                  // For current neuron's layer, use fixed height centered on centerY
                  if (layerPos.layer === currentNeuronLayer) {
                    segmentTop = layerPos.centerY - currentNeuronRectHeight / 2;
                    segmentBottom = layerPos.centerY + currentNeuronRectHeight / 2;
                  } else {
                    // Adjust boundaries if adjacent to current neuron layer
                    if (idx > 0) {
                      const prevLayerPos = sortedLayerPositions[idx - 1];
                      if (prevLayerPos.layer === currentNeuronLayer) {
                        segmentTop = prevLayerPos.centerY + currentNeuronRectHeight / 2 + currentNeuronArrowAreaHeight;
                      }
                    }
                    if (idx < sortedLayerPositions.length - 1) {
                      const nextLayerPos = sortedLayerPositions[idx + 1];
                      if (nextLayerPos.layer === currentNeuronLayer) {
                        segmentBottom =
                          nextLayerPos.centerY - currentNeuronRectHeight / 2 - currentNeuronArrowAreaHeight;
                      }
                    }
                  }

                  segments.push({
                    layer: layerPos.layer,
                    top: segmentTop,
                    height: segmentBottom - segmentTop,
                  });
                });

                // Calculate the actual channel height based on segments
                const actualChannelHeight =
                  segments.length > 0
                    ? segments[segments.length - 1].top + segments[segments.length - 1].height
                    : channelHeight;

                return (
                  <div key={channel.id} className="absolute" style={{ left: `${channelStartX + arrayIndex * 36}px` }}>
                    {/* Channel number label */}
                    <div
                      className="absolute text-[9px] font-bold transition-colors"
                      style={{
                        top: `${channelsStartY - 16}px`,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        whiteSpace: 'nowrap',
                        color: isHighlighted ? hoverTextColor : defaultTextColor,
                      }}
                    >
                      {channel.index}
                    </div>
                    {/* Channel line - single hoverable element with one tooltip */}
                    {(() => {
                      // Check if this channel is in trace_backward (input) or trace_forward (output)
                      const isInputChannel =
                        sparsityData?.trace_backward.some((n) => n.via_channel === channel.index) ?? false;
                      const isOutputChannel =
                        sparsityData?.trace_forward.some((n) => n.via_channel === channel.index) ?? false;

                      // Check if nothing is hovered
                      const nothingHovered = hoveredChannelId === null && hoveredNeuronIndex === null;

                      // Channel is highlighted if: hovered, connected to hovered neuron, OR nothing hovered and it's a connected channel
                      const isChannelHovered = hoveredChannelId === channel.id;
                      const isFullHighlight =
                        isChannelHovered ||
                        isConnectedToHoveredNeuron ||
                        (nothingHovered && (isInputChannel || isOutputChannel));

                      // Calculate the channel line dimensions (excluding current neuron's layer)
                      // We need to render segments above and below the current neuron's rectangle separately
                      const aboveSegments = segments.filter((s) => s.layer < currentNeuronLayer);
                      const belowSegments = segments.filter((s) => s.layer > currentNeuronLayer);

                      // Colors
                      const defaultColor = 'rgba(3, 105, 161, 0.3)';
                      const fullColor = '#0369a1';
                      const fillColor = isFullHighlight ? fullColor : defaultColor;

                      // Get all explanations for this channel
                      const allExplanations = getAllResidChannelExplanations(channel.index);

                      // Calculate trigger bounds based on all segments
                      const allSegments = [...aboveSegments, ...belowSegments];
                      const triggerTop = allSegments.length > 0 ? Math.min(...allSegments.map((s) => s.top)) : 0;
                      const triggerBottom =
                        allSegments.length > 0 ? Math.max(...allSegments.map((s) => s.top + s.height)) : 0;
                      const triggerHeight = triggerBottom - triggerTop;

                      return (
                        <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
                          <Tooltip.Root>
                            <Tooltip.Trigger asChild>
                              <div
                                className="absolute w-[5px] cursor-pointer"
                                style={{
                                  top: `${triggerTop}px`,
                                  height: `${triggerHeight}px`,
                                }}
                                onMouseEnter={() => setHoveredChannelId(channel.id)}
                                onMouseLeave={() => setHoveredChannelId(null)}
                              >
                                {/* Segments above current neuron */}
                                {aboveSegments.map((segment, idx) => (
                                  <div
                                    key={`above-${segment.layer}`}
                                    className="absolute w-[5px]"
                                    style={{
                                      top: `${segment.top - triggerTop}px`,
                                      height: `${segment.height}px`,
                                      background:
                                        idx === 0
                                          ? `linear-gradient(to bottom, transparent 0%, ${fillColor} 20%, ${fillColor} 100%)`
                                          : fillColor,
                                      transition: 'background 0.3s ease',
                                      borderRadius: idx === 0 ? '2px 2px 0 0' : '0',
                                    }}
                                  />
                                ))}
                                {/* Segments below current neuron */}
                                {belowSegments.map((segment) => (
                                  <div
                                    key={`below-${segment.layer}`}
                                    className="absolute w-[5px]"
                                    style={{
                                      top: `${segment.top - triggerTop}px`,
                                      height: `${segment.height}px`,
                                      background: fillColor,
                                      transition: 'background 0.3s ease',
                                    }}
                                  />
                                ))}
                              </div>
                            </Tooltip.Trigger>
                            <Tooltip.Portal>
                              <Tooltip.Content
                                className="min-w-[240px] max-w-[240px] rounded-md border border-slate-300 bg-white px-3 py-3 text-xs text-slate-600 shadow-md"
                                sideOffset={3}
                                side={isOutputChannel && !isInputChannel ? 'left' : 'right'}
                                align="center"
                                avoidCollisions={false}
                              >
                                <div className="text-xs font-bold">Residual Channel {channel.index}</div>

                                {/* Writers: backward neurons that write to this channel */}
                                {(sparsityData?.trace_backward ?? []).some((n) => n.via_channel === channel.index) && (
                                  <div className="mt-2 text-[10px] font-semibold">Neurons Writing</div>
                                )}
                                {(sparsityData?.trace_backward ?? [])
                                  .filter((n) => n.via_channel === channel.index)
                                  .map((n) => {
                                    const neuronExplanations = getAllNeuronExplanations(n.layer, String(n.neuron));
                                    return (
                                      <div key={`writer-${n.neuron}`} className="mb-1 ml-2">
                                        <div className="text-[10px] font-medium">
                                          <Link
                                            href={`/${currentNeuron?.modelId}/${n.layer}-mlp/${n.neuron}`}
                                            className="font-mono text-sky-700 hover:underline"
                                          >
                                            {n.layer}-MLP @ {n.neuron}
                                          </Link>{' '}
                                          <span className="text-slate-400">(w: {n.write_weight.toFixed(3)})</span>
                                        </div>
                                        {neuronExplanations.length > 0 && (
                                          <div className="ml-3">
                                            {neuronExplanations
                                              .sort((a, b) => {
                                                const aIsBottom = a.description.includes('(negative activations)');
                                                const bIsBottom = b.description.includes('(negative activations)');
                                                return aIsBottom === bIsBottom ? 0 : aIsBottom ? 1 : -1;
                                              })
                                              .map((exp) => {
                                                const isBottomActivation =
                                                  exp.description.includes('(negative activations)');
                                                const signColorClass = isBottomActivation
                                                  ? 'text-rose-500'
                                                  : 'text-emerald-600';
                                                const sign = isBottomActivation ? '-' : '+';
                                                return (
                                                  <div
                                                    key={exp.id}
                                                    className="flex items-center gap-x-1 text-[9px] font-medium leading-snug"
                                                  >
                                                    <span
                                                      className={`font-mono text-[11px] font-bold leading-none ${signColorClass}`}
                                                    >
                                                      {sign}
                                                    </span>
                                                    <span className="text-slate-600">
                                                      {exp.description.replace(' (negative activations)', '')}
                                                    </span>
                                                  </div>
                                                );
                                              })}
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                {/* Readers: forward neurons that read from this channel */}
                                {(sparsityData?.trace_forward ?? []).some((n) => n.via_channel === channel.index) && (
                                  <div className="mt-2 text-[10px] font-semibold">Neurons Reading</div>
                                )}
                                {(sparsityData?.trace_forward ?? [])
                                  .filter((n) => n.via_channel === channel.index)
                                  .map((n) => {
                                    const neuronExplanations = getAllNeuronExplanations(n.layer, String(n.neuron));
                                    return (
                                      <div key={`reader-${n.neuron}`} className="mb-1 ml-2">
                                        <div className="text-[10px] font-medium">
                                          <Link
                                            href={`/${currentNeuron?.modelId}/${n.layer}-mlp/${n.neuron}`}
                                            className="font-mono text-sky-700 hover:underline"
                                          >
                                            {n.layer}-MLP @ {n.neuron}
                                          </Link>{' '}
                                          <span className="text-slate-400">(r: {n.read_weight.toFixed(3)})</span>
                                        </div>
                                        {neuronExplanations.length > 0 && (
                                          <div className="ml-3">
                                            {neuronExplanations
                                              .sort((a, b) => {
                                                const aIsBottom = a.description.includes('(negative activations)');
                                                const bIsBottom = b.description.includes('(negative activations)');
                                                return aIsBottom === bIsBottom ? 0 : aIsBottom ? 1 : -1;
                                              })
                                              .map((exp) => {
                                                const isBottomActivation =
                                                  exp.description.includes('(negative activations)');
                                                const signColorClass = isBottomActivation
                                                  ? 'text-rose-500'
                                                  : 'text-emerald-600';
                                                const sign = isBottomActivation ? '-' : '+';
                                                return (
                                                  <div
                                                    key={exp.id}
                                                    className="flex items-center gap-x-1 text-[9px] font-medium leading-snug"
                                                  >
                                                    <span
                                                      className={`font-mono text-[11px] font-bold leading-none ${signColorClass}`}
                                                    >
                                                      {sign}
                                                    </span>
                                                    <span className="text-slate-600">
                                                      {exp.description.replace(' (negative activations)', '')}
                                                    </span>
                                                  </div>
                                                );
                                              })}
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                {/* All layer explanations */}
                                {allExplanations.length > 0 && (
                                  <div className="mt-2">
                                    <div className="mb-0.5 text-[10px] font-semibold">Resid Explanations</div>
                                    {allExplanations.map((layerData) => (
                                      <div key={layerData.layer} className="mb-1 ml-2">
                                        <div className="text-[10px] font-medium uppercase text-slate-400">
                                          {/* Layer {layerData.layer.split('-')[0]}
                                          {' - '} */}
                                          <Link
                                            href={`/${currentNeuron?.modelId}/${layerData.layer}/${channel.index}`}
                                            className="font-mono text-sky-700 hover:underline"
                                          >
                                            {layerData.layer} @ {channel.index}
                                          </Link>
                                        </div>
                                        {layerData.explanations
                                          .sort((a, b) => {
                                            const aIsBottom = a.description.includes('(negative activations)');
                                            const bIsBottom = b.description.includes('(negative activations)');
                                            return aIsBottom === bIsBottom ? 0 : aIsBottom ? 1 : -1;
                                          })
                                          .map((exp) => {
                                            const isBottomActivation =
                                              exp.description.includes('(negative activations)');
                                            const signColorClass = isBottomActivation
                                              ? 'text-rose-500'
                                              : 'text-emerald-600';
                                            const sign = isBottomActivation ? '-' : '+';
                                            return (
                                              <div
                                                key={exp.id}
                                                className="mb-0.5 ml-3 flex items-center gap-x-1 text-[10px] font-medium leading-snug"
                                              >
                                                <span
                                                  className={`font-mono text-[12px] font-bold leading-none ${signColorClass}`}
                                                >
                                                  {sign}
                                                </span>
                                                <span className="text-slate-600">
                                                  {exp.description.replace(' (negative activations)', '')}
                                                </span>
                                              </div>
                                            );
                                          })}
                                      </div>
                                    ))}
                                  </div>
                                )}
                                <Tooltip.Arrow className="fill-slate-300" />
                              </Tooltip.Content>
                            </Tooltip.Portal>
                          </Tooltip.Root>
                        </Tooltip.Provider>
                      );
                    })()}
                    {/* Fade out at bottom */}
                    {(() => {
                      const isChannelHovered = hoveredChannelId === channel.id;
                      // Check if this channel is an input or output channel
                      const isOutputChan =
                        sparsityData?.trace_forward.some((n) => n.via_channel === channel.index) ?? false;
                      const isInputChan =
                        sparsityData?.trace_backward.some((n) => n.via_channel === channel.index) ?? false;
                      // Check if nothing is hovered
                      const nothingHovered = hoveredChannelId === null && hoveredNeuronIndex === null;

                      // Channel color - light blue by default, turns solid blue when highlighted
                      const fullColor = '#0369a1';
                      const defaultColorSolid = '#b3d2e3';

                      const isFullHighlight =
                        isConnectedToHoveredNeuron ||
                        isChannelHovered ||
                        (nothingHovered && (isInputChan || isOutputChan));

                      const fadeHeight = 16;

                      return (
                        <div
                          className="absolute"
                          style={{
                            top: `${actualChannelHeight}px`,
                            left: '0px',
                            width: '5px',
                            height: `${fadeHeight}px`,
                            background: isFullHighlight
                              ? `linear-gradient(to bottom, ${fullColor} 0%, ${fullColor} 20%, #ffffff 100%)`
                              : `linear-gradient(to bottom, ${defaultColorSolid} 0%, ${defaultColorSolid} 20%, #ffffff 100%)`,
                            transition: 'background 0.3s ease',
                          }}
                        />
                      );
                    })()}
                  </div>
                );
              })}

              {/* Render current neuron as a rounded rectangle spanning channel width */}
              {(() => {
                const currentNeuronData = connectedNeurons.find((n) => n.current);
                if (!currentNeuronData) return null;

                // Get the layer's centerY position from layerLabelPositions
                const layerPosition = layerLabelPositions.find((lp) => lp.layer === currentNeuronData.layer);
                if (!layerPosition) return null;

                // Get input and output channel indices for display
                const inputChannelIndices = (sparsityData?.trace_backward ?? [])
                  .map((n) => n.via_channel)
                  .sort((a, b) => a - b);
                const outputChannelIndices = (sparsityData?.trace_forward ?? [])
                  .map((n) => n.via_channel)
                  .sort((a, b) => a - b);

                // Unique channels for each side
                const uniqueInputChannels = [...new Set(inputChannelIndices)];
                const uniqueOutputChannels = [...new Set(outputChannelIndices)];

                // Rectangle dimensions - span the visual width of channel lines plus extra margin
                // Channel lines are 5px wide, spaced 36px apart
                // Visual span: from first line start to last line end = (n-1)*36 + 5
                // Add extra width on each side (as if edge channels are wider)
                const edgeChannelExtraWidth = 20;
                const rectPadding = 30;
                const rectWidth = Math.max(visualChannelWidth + edgeChannelExtraWidth * 2 + rectPadding * 2, 250);
                // Center the rectangle over the visual channel span
                // Visual center of channels = channelStartX + visualChannelWidth / 2
                const visualCenter = channelStartX + visualChannelWidth / 2;
                const rectLeft = visualCenter - rectWidth / 2;
                // Position vertically centered at the layer's centerY (same as segment)
                const rectTop = layerPosition.centerY - currentNeuronRectHeight / 2;

                // Use the shared arrow area height constant
                const arrowAreaHeight = currentNeuronArrowAreaHeight;

                // Build a map of channel index to the backward trace layer (for hover behavior)
                const channelToBackwardLayer = new Map<number, number>();
                (sparsityData?.trace_backward ?? []).forEach((node) => {
                  // Use the first (or any) backward layer for this channel
                  if (!channelToBackwardLayer.has(node.via_channel)) {
                    channelToBackwardLayer.set(node.via_channel, node.layer);
                  }
                });

                return (
                  <>
                    {/* Arrows/fades above rectangle for each channel */}
                    {resChannels.map((channel, idx) => {
                      const channelX = channelStartX + idx * 36 + 2; // Center of 5px line
                      const isInputChannel = uniqueInputChannels.includes(channel.index);

                      // Check if nothing is hovered
                      const nothingHovered = hoveredChannelId === null && hoveredNeuronIndex === null;

                      // Hover detection - simplified (no layer-specific hover)
                      const isConnectedToHoveredNeuron = hoveredNeuronIndex
                        ? connectedNeurons
                            .find((n) => n.index === hoveredNeuronIndex)
                            ?.resChannels.some((c) => c.id === channel.id)
                        : false;
                      const isChannelHovered = hoveredChannelId === channel.id;
                      const isOutputChannel = uniqueOutputChannels.includes(channel.index);
                      // Full highlight if: channel hovered, connected neuron hovered, OR nothing hovered and this is an INPUT or OUTPUT channel
                      const isFullHighlight =
                        isChannelHovered ||
                        isConnectedToHoveredNeuron ||
                        (nothingHovered && (isInputChannel || isOutputChannel));

                      // Colors
                      const defaultColor = 'rgba(3, 105, 161, 0.3)';
                      const fullColor = '#0369a1';

                      const fillColor = isFullHighlight ? fullColor : defaultColor;

                      if (isInputChannel) {
                        // Draw arrow pointing down for input channels
                        // Arrow head is 8px tall, line fills the rest
                        const arrowHeadHeight = 8;
                        const lineHeight = arrowAreaHeight - arrowHeadHeight;

                        return (
                          <svg
                            key={`arrow-input-${channel.id}`}
                            className="absolute cursor-pointer"
                            style={{
                              left: `${channelX - 6}px`,
                              top: `${rectTop - arrowAreaHeight}px`,
                              width: '13px',
                              height: `${arrowAreaHeight}px`,
                            }}
                            viewBox={`0 0 13 ${arrowAreaHeight}`}
                            onMouseEnter={() => setHoveredChannelId(channel.id)}
                            onMouseLeave={() => setHoveredChannelId(null)}
                          >
                            {/* Vertical line */}
                            <rect
                              x="4"
                              y="0"
                              width="5"
                              height={lineHeight}
                              fill={fillColor}
                              style={{ transition: 'fill 0.3s ease' }}
                            />
                            {/* Arrow head pointing down */}
                            <path
                              d={`M6.5 ${arrowAreaHeight}L0 ${lineHeight}H13L6.5 ${arrowAreaHeight}Z`}
                              fill={fillColor}
                              style={{ transition: 'fill 0.3s ease' }}
                            />
                          </svg>
                        );
                      }

                      // Don't draw fades above rectangle if there are no backward trace neurons
                      if (uniqueInputChannels.length === 0) {
                        return null;
                      }

                      // Draw fade to white for non-input channels (covers the channel line)
                      // For default state, use solid color equivalent of rgba(3, 105, 161, 0.3) on white for smooth gradient
                      const defaultColorSolid = '#b3d2e3';
                      const fadeBackground = isFullHighlight
                        ? `linear-gradient(to bottom, ${fullColor} 0%, #ffffff 90%)`
                        : `linear-gradient(to bottom, ${defaultColorSolid} 0%, #ffffff 90%)`;

                      return (
                        <div
                          key={`fade-${channel.id}`}
                          className="absolute cursor-pointer"
                          style={{
                            left: `${channelX - 2}px`,
                            top: `${rectTop - arrowAreaHeight}px`,
                            width: '5px',
                            height: `${arrowAreaHeight}px`,
                            background: fadeBackground,
                            transition: 'background 0.3s ease',
                          }}
                          onMouseEnter={() => setHoveredChannelId(channel.id)}
                          onMouseLeave={() => setHoveredChannelId(null)}
                        />
                      );
                    })}

                    {/* Current neuron rectangle */}
                    <div
                      className="absolute flex w-full flex-col overflow-hidden rounded-md border border-sky-700 bg-slate-50"
                      style={{
                        left: `${rectLeft}px`,
                        top: `${rectTop}px`,
                        width: `${rectWidth}px`,
                        height: `${currentNeuronRectHeight}px`,
                      }}
                    >
                      {/* Header text: Current [layer]@[index] - full width for background */}
                      <div className="mb-1 flex w-full items-center justify-center bg-sky-700 py-2 pt-2 text-center font-mono text-[10px] font-bold uppercase leading-none text-white">
                        {currentNeuron?.layer} - Index {currentNeuronData.index}
                      </div>

                      <div className="flex flex-col gap-x-2 text-[10px] font-bold">
                        <div className="flex flex-row items-center gap-x-1 px-2">
                          <span className="font-mono text-[12px] font-bold leading-none text-emerald-600">+</span>
                          <span className="text-slate-600">
                            {getNeuronExplanation(currentNeuronData.layer, currentNeuronData.index, 'top')}
                          </span>
                        </div>
                        <div className="flex flex-row items-center gap-x-1 px-2">
                          <span className="font-mono text-[12px] font-bold leading-none text-rose-500">-</span>
                          <span className="text-slate-600">
                            {getNeuronExplanation(currentNeuronData.layer, currentNeuronData.index, 'bottom')}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Arrows/fades below rectangle for each channel */}
                    {resChannels.map((channel, idx) => {
                      const channelX = channelStartX + idx * 36 + 2.5; // Center of 5px line
                      const isOutputChannel = uniqueOutputChannels.includes(channel.index);

                      // Check if nothing is hovered
                      const nothingHovered = hoveredChannelId === null && hoveredNeuronIndex === null;

                      // Hover detection - simplified (no layer-specific hover)
                      const isConnectedToHoveredNeuron = hoveredNeuronIndex
                        ? connectedNeurons
                            .find((n) => n.index === hoveredNeuronIndex)
                            ?.resChannels.some((c) => c.id === channel.id)
                        : false;
                      const isChannelHovered = hoveredChannelId === channel.id;
                      const isInputChannel = uniqueInputChannels.includes(channel.index);
                      // Full highlight if: channel hovered, connected neuron hovered, OR nothing hovered and this is an INPUT or OUTPUT channel
                      const isFullHighlight =
                        isChannelHovered ||
                        isConnectedToHoveredNeuron ||
                        (nothingHovered && (isInputChannel || isOutputChannel));

                      // Colors
                      const defaultColor = 'rgba(3, 105, 161, 0.3)';
                      const fullColor = '#0369a1';
                      // Solid color equivalent of rgba(3, 105, 161, 0.3) on white for smooth gradient
                      const defaultColorSolid = '#b3d2e3';

                      const fillColor = isFullHighlight ? fullColor : defaultColor;

                      // Position below the rectangle
                      const belowRectTop = rectTop + currentNeuronRectHeight;

                      if (isOutputChannel) {
                        // Draw funnel shape pointing down for output channels
                        // Starts wide (13px) at top, narrows to channel width (5px) at bottom
                        const arrowHeadHeight = 5;

                        return (
                          <svg
                            key={`arrow-output-${channel.id}`}
                            className="absolute cursor-pointer"
                            style={{
                              left: `${channelX - 6.5}px`,
                              top: `${belowRectTop}px`,
                              width: '13px',
                              height: `${arrowAreaHeight}px`,
                            }}
                            viewBox={`0 0 13 ${arrowAreaHeight}`}
                            onMouseEnter={() => setHoveredChannelId(channel.id)}
                            onMouseLeave={() => setHoveredChannelId(null)}
                          >
                            {/* Funnel shape: wide at top, narrows to channel width */}
                            <path
                              d={`M0 0 L13 0 L9 ${arrowHeadHeight} L9 ${arrowAreaHeight} L4 ${arrowAreaHeight} L4 ${arrowHeadHeight} Z`}
                              fill={fillColor}
                              style={{ transition: 'fill 0.3s ease' }}
                            />
                          </svg>
                        );
                      }

                      // Don't draw fades below rectangle if there are no forward trace neurons
                      if (uniqueOutputChannels.length === 0) {
                        return null;
                      }

                      // Draw fade from white for non-output channels
                      const fadeBackground = isFullHighlight
                        ? `linear-gradient(to bottom, #ffffff 15%, ${fullColor} 100%)`
                        : `linear-gradient(to bottom, #ffffff 15%, ${defaultColorSolid} 100%)`;

                      return (
                        <div
                          key={`fade-below-${channel.id}`}
                          className="absolute cursor-pointer"
                          style={{
                            left: `${channelX - 2.5}px`,
                            top: `${belowRectTop}px`,
                            width: '5px',
                            height: `${arrowAreaHeight}px`,
                            background: fadeBackground,
                            transition: 'background 0.3s ease',
                          }}
                          onMouseEnter={() => setHoveredChannelId(channel.id)}
                          onMouseLeave={() => setHoveredChannelId(null)}
                        />
                      );
                    })}
                  </>
                );
              })()}

              {/* Render non-current neurons as circles with explanation labels */}
              {connectedNeurons
                .filter((neuron) => !neuron.current)
                .map((neuron) => {
                  const position = neuronPositions.get(neuron.index);
                  if (!position) return null;

                  // Check if this neuron is active (hovered or connected to hovered channel)
                  const isConnectedToHoveredChannel = hoveredChannelId
                    ? neuron.resChannels.some((c) => c.id === hoveredChannelId)
                    : false;
                  const isActive = hoveredNeuronIndex === neuron.index || isConnectedToHoveredChannel;

                  // Determine background and border colors
                  const bgColor = isActive ? 'bg-slate-500' : 'bg-slate-300';
                  const borderColor = isActive ? 'border-slate-600' : 'border-slate-400';

                  // Get explanation for this neuron
                  const allExplanations = getAllNeuronExplanations(neuron.layer, neuron.index);
                  // Sort explanations: top (non-negative) first, then bottom (negative activations)
                  const sortedExplanations = [...allExplanations].sort((a, b) => {
                    const aIsBottom = a.description.includes('(negative activations)');
                    const bIsBottom = b.description.includes('(negative activations)');
                    return aIsBottom === bIsBottom ? 0 : aIsBottom ? 1 : -1;
                  });
                  const isBackward = neuron.direction === 'backward';

                  return (
                    <div key={neuron.index}>
                      {/* Explanation labels - stacked vertically */}
                      {sortedExplanations.length > 0 && (
                        <div
                          className="absolute flex max-w-[120px] flex-col"
                          style={{
                            top: `${position.top - (sortedExplanations.length > 1 ? 3 : 0)}px`,
                            ...(isBackward
                              ? {
                                  right: `calc(100% - ${position.left}px + 4px)`,
                                  alignItems: 'flex-end',
                                }
                              : {
                                  left: `${position.left + 16}px`,
                                  alignItems: 'flex-start',
                                }),
                          }}
                        >
                          {sortedExplanations.map((exp) => {
                            const isBottomActivation = exp.description.includes('(negative activations)');
                            const displayText = exp.description.replace(' (negative activations)', '');
                            const signColorClass = isBottomActivation ? 'text-rose-500' : 'text-emerald-600';
                            const sign = isBottomActivation ? '-' : '+';
                            return (
                              <div
                                key={exp.id}
                                className="flex items-center gap-x-0.5 text-[8px] font-medium leading-[10px]"
                                style={{ maxWidth: '120px' }}
                                title={displayText}
                              >
                                <span
                                  className={`font-mono text-[10px] font-bold ${signColorClass}`}
                                  style={{ lineHeight: '10px' }}
                                >
                                  {sign}
                                </span>
                                <span className="truncate text-slate-600">{displayText}</span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                      {/* Neuron circle with hover card showing feature dashboard */}
                      <HoverCard openDelay={300} closeDelay={400}>
                        <HoverCardTrigger asChild>
                          <Link
                            href={`/${currentNeuron?.modelId}/${neuron.layer}-mlp/${neuron.index}`}
                            aria-label={`Go to neuron ${neuron.layer}-MLP @ ${neuron.index}`}
                            className={`absolute block h-3 w-3 rounded-full border transition-colors ${bgColor} ${borderColor}`}
                            style={{
                              left: `${position.left}px`,
                              top: `${position.top}px`,
                            }}
                            onMouseEnter={() => {
                              setHoveredNeuronIndex(neuron.index);
                              // Fetch feature data if not already loaded for this neuron
                              const neuronLayer = `${neuron.layer}-mlp`;
                              if (
                                hoveredFeature &&
                                hoveredFeature.layer === neuronLayer &&
                                hoveredFeature.index === neuron.index
                              ) {
                                return;
                              }
                              // Reset and fetch new data
                              setHoveredFeature(undefined);
                              fetch(`/api/feature/${currentNeuron?.modelId}/${neuronLayer}/${neuron.index}`, {
                                method: 'GET',
                                headers: { 'Content-Type': 'application/json' },
                              })
                                .then((response) => response.json())
                                .then((n: NeuronWithPartialRelations) => {
                                  setHoveredFeature(n);
                                })
                                .catch((err) => {
                                  console.error(`Error fetching neuron data: ${err}`);
                                });
                            }}
                            onMouseLeave={() => setHoveredNeuronIndex(null)}
                          />
                        </HoverCardTrigger>
                        <HoverCardContent
                          className="max-h-[640px] min-h-[640px] w-[640px] min-w-[640px] max-w-[640px] overflow-y-auto border bg-white p-0"
                          side={neuron.direction === 'backward' ? 'left' : 'right'}
                          sideOffset={5}
                        >
                          {/* Original tooltip content */}
                          <div className="px-3 py-3 text-xs text-slate-600">
                            <Link
                              href={`/${currentNeuron?.modelId}/${neuron.layer}-mlp/${neuron.index}`}
                              className="flex flex-row items-center gap-x-1 font-mono text-xs font-bold uppercase text-sky-700 hover:underline"
                            >
                              {neuron.layer}-MLP @ {neuron.index} <ExternalLinkIcon className="h-3 w-3" />
                            </Link>
                            {allExplanations.length > 0 && (
                              <div className="mb-0 mt-3 text-[9px] font-bold uppercase text-slate-500">
                                Explanations
                              </div>
                            )}
                            {allExplanations.length > 0 && (
                              <div className="mb-3 ml-2">
                                {allExplanations
                                  .sort((a, b) => {
                                    const aIsBottom = a.description.includes('(negative activations)');
                                    const bIsBottom = b.description.includes('(negative activations)');
                                    return aIsBottom === bIsBottom ? 0 : aIsBottom ? 1 : -1;
                                  })
                                  .map((exp) => {
                                    const isBottomActivation = exp.description.includes('(negative activations)');
                                    const signColorClass = isBottomActivation ? 'text-rose-500' : 'text-emerald-600';
                                    const sign = isBottomActivation ? '-' : '+';
                                    return (
                                      <div
                                        key={exp.id}
                                        className="mb-0.5 flex items-center gap-x-1.5 text-[12px] font-medium leading-snug"
                                      >
                                        <span
                                          className={`font-mono text-[14px] font-bold leading-none ${signColorClass}`}
                                        >
                                          {sign}
                                        </span>
                                        <span className="text-slate-600">
                                          {exp.description.replace(' (negative activations)', '')}
                                        </span>
                                      </div>
                                    );
                                  })}
                              </div>
                            )}
                            <div className="mb-0 mt-2">
                              <span className="text-[9px] font-bold uppercase text-slate-500">
                                {neuron.direction !== 'backward' ? 'Reads From' : 'Writes To'}
                              </span>
                            </div>
                            {neuron.resChannels.map((channel) => (
                              <div key={channel.id} className="mb-1 font-mono text-[10px] font-medium">
                                <Link
                                  href={`/${currentNeuron?.modelId}/${neuron.layer}-resid/${channel.index}`}
                                  className="ml-2 uppercase text-sky-700 hover:underline"
                                >
                                  {neuron.layer}-resid @ {channel.index}
                                </Link>
                              </div>
                            ))}
                          </div>
                          {/* Feature dashboard */}
                          {hoveredFeature?.activations &&
                          hoveredFeature?.activations?.length > 0 &&
                          hoveredFeature?.layer === `${neuron.layer}-mlp` &&
                          hoveredFeature?.index === neuron.index ? (
                            <div className="-mt-2 h-[386px] w-full">
                              <FeatureDashboard
                                key={`${hoveredFeature?.modelId}-${hoveredFeature?.layer}-${hoveredFeature?.index}`}
                                initialNeuron={hoveredFeature}
                                embed
                                forceMiniStats
                              />
                            </div>
                          ) : (
                            <div className="flex h-[386px] w-full items-center justify-center">
                              <LoadingSquare className="h-6 w-6" />
                            </div>
                          )}
                        </HoverCardContent>
                      </HoverCard>
                    </div>
                  );
                })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
