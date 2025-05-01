'use client';

import { NeuronIdentifier } from '@/lib/utils/neuron-identifier';
// import Plotly from 'plotly.js-dist-min';
import {
  ExplanationWithPartialRelations,
  ListWithPartialRelations,
  NeuronWithPartialRelations,
} from 'prisma/generated/zod';
import React, { Dispatch, RefObject, SetStateAction, useMemo, useRef, useState } from 'react';

import dynamic from 'next/dynamic';
import createContextWrapper from './provider-util';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

// const Plot = createPlotlyComponent(Plotly);

type GraphRange = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

export type UmapListItem = {
  isEditing: boolean;
  description: string;
  neuron: NeuronWithPartialRelations | null;
};

export function deleteFeatureFromUmapMap(feat: NeuronIdentifier, map: Map<NeuronIdentifier, UmapListItem>) {
  map.forEach((value, key) => {
    if (key.equals(feat)) {
      map.delete(key);
    }
  });
}

export const [UmapContext, useUmapContext] = createContextWrapper<{
  searchText: string;
  setSearchText: Dispatch<SetStateAction<string>>;
  isSearching: boolean;
  setIsSearching: Dispatch<SetStateAction<boolean>>;
  loadingFeature: NeuronIdentifier | null;
  setLoadingFeature: Dispatch<SetStateAction<NeuronIdentifier | null>>;
  selectedFeatures: Map<NeuronIdentifier, UmapListItem>;
  setSelectedFeatures: Dispatch<SetStateAction<Map<NeuronIdentifier, UmapListItem>>>;
  graphRanges: GraphRange;
  setGraphRanges: Dispatch<SetStateAction<GraphRange>>;
  openedList: ListWithPartialRelations | undefined | null;
  setOpenedList: Dispatch<SetStateAction<ListWithPartialRelations | undefined | null>>;
  // @ts-ignore
  plotInactiveRef: RefObject<Plot>;
  // @ts-ignore
  plotActiveRef: RefObject<Plot>;
  loadFeature: (feature: NeuronIdentifier) => void;
  visibleUmapExplanations: ExplanationWithPartialRelations[];
  setVisibleUmapExplanations: Dispatch<SetStateAction<ExplanationWithPartialRelations[]>>;
  highlightedUmapExplanations: ExplanationWithPartialRelations[];
  setHighlightedUmapExplanations: Dispatch<SetStateAction<ExplanationWithPartialRelations[]>>;
  showLogSparsity: boolean;
  setShowLogSparsity: Dispatch<SetStateAction<boolean>>;
  layerToInitialColor: {
    [layer: string]: string;
  };
  setLayerToInitialColor: Dispatch<
    SetStateAction<{
      [layer: string]: string;
    }>
  >;
  // addAnnotationForExp: (exp: ExplanationWithPartialRelations) => void;
}>('UmapContext');

export const UMAP_HEIGHT = 300;
export const ZOOM_PADDING_Y = 0.1;
export const ZOOM_PADDING_X = 0.04;
// const ANNOTATION_BG_COLOR = '#404040';
export const UMAP_INITIAL_COLORS = ['#40B0A6', '#FFBE6A', '#dc2626'];
export const SPARSITY_COLOR_MAX = -1.5;
export const SPARSITY_COLOR_MIN = -4;
export const SPARSITY_COLORS = [
  '#f43e5c',
  '#f0396c',
  '#ea387c',
  '#e13b8a',
  '#d64097',
  '#c947a2',
  '#ba4dab',
  '#aa54b3',
  '#9959b8',
  '#875eba',
  '#7462bb',
  '#6165b9',
  '#4d67b5',
  '#3969b0',
  '#2369a9',
  '#0369a0',
];

export default function UmapProvider({ children }: { children: React.ReactNode }) {
  const [searchText, setSearchText] = useState('');
  const [loadingFeature, setLoadingFeature] = useState<NeuronIdentifier | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedFeatures, setSelectedFeatures] = useState<Map<NeuronIdentifier, UmapListItem>>(new Map());
  const [layerToInitialColor, setLayerToInitialColor] = useState<{
    [layer: string]: string;
  }>({});
  const [openedList, setOpenedList] = useState<ListWithPartialRelations | undefined | null>(); // null means loading
  const [graphRanges, setGraphRanges] = useState<GraphRange>({
    minX: 0,
    minY: 0,
    maxX: 0,
    maxY: 0,
  });
  const plotActiveRef = useRef<typeof Plot>(null);
  const plotInactiveRef = useRef<typeof Plot>(null);
  const [visibleUmapExplanations, setVisibleUmapExplanations] = useState<ExplanationWithPartialRelations[]>([]);
  const [showLogSparsity, setShowLogSparsity] = useState<boolean>(true);
  const [highlightedUmapExplanations, setHighlightedUmapExplanations] = useState<ExplanationWithPartialRelations[]>([]);

  // const formatStringWithLineBreaks = (inputString: string): string => {
  //   const maxLineLength = 25;
  //   let resultString = '';
  //   let currentLineLength = 0;

  //   inputString.split(' ').forEach((word) => {
  //     if (currentLineLength + word.length > maxLineLength) {
  //       resultString += '<br>';
  //       currentLineLength = 0;
  //     }
  //     resultString += `${word} `;
  //     currentLineLength += word.length + 1; // +1 for the space
  //   });

  //   return resultString.trim();
  // };

  // async function addAnnotationForExp(exp: ExplanationWithPartialRelations) {
  //   if (plotInactiveRef.current) {
  //     await Plotly.relayout((plotInactiveRef.current as any)?.el, {
  //       annotations: [],
  //     });
  //     Plotly.relayout((plotInactiveRef.current as any)?.el, {
  //       annotations: [
  //         {
  //           x: exp?.umap_x || 0,
  //           y: exp?.umap_y || 0,
  //           text: `${exp?.layer?.toUpperCase()}:${exp?.index}<br>${formatStringWithLineBreaks(exp?.description)}${
  //             exp?.umap_log_feature_sparsity
  //               ? `<br>Feature Sparsity: ${exp?.umap_log_feature_sparsity?.toFixed(3)}`
  //               : ''
  //           }`,
  //           showarrow: true,
  //           arrowcolor: ANNOTATION_BG_COLOR,
  //           arrowhead: 1,
  //           arrowsize: 1,
  //           yanchor: 'bottom',
  //           xanchor: 'center',
  //           align: 'left',
  //           ay: -5,
  //           ax: 0,
  //           bgcolor: ANNOTATION_BG_COLOR,
  //           font: {
  //             color: '#ffffff',
  //           },
  //         },
  //       ],
  //     });
  //   }
  // }

  function getFeatureFromSelectedFeatures(feature: NeuronIdentifier) {
    return [...selectedFeatures.keys()].find((selectedFeature) => selectedFeature.equals(feature));
  }

  function loadFeature(feature: NeuronIdentifier) {
    console.log(`loading feature at index: ${feature.index}`);
    const existingSelectedFeature = getFeatureFromSelectedFeatures(feature);
    if (existingSelectedFeature && selectedFeatures.get(existingSelectedFeature)?.neuron !== null) {
      console.log('already loaded');
      return;
    }
    setSelectedFeatures((prev) => {
      const newMap = new Map(prev);
      newMap.set(feature, {
        isEditing: false,
        description: '',
        neuron: null,
      });
      return newMap;
    });
    setLoadingFeature(feature);
    fetch(`/api/feature/${feature.modelId}/${feature.layer}/${feature.index}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .then((n: NeuronWithPartialRelations) => {
        // avoid race condition
        setSelectedFeatures((previousMap) => {
          const newMap = new Map(previousMap);
          // find the feature in the map and update it
          newMap.forEach((value, key) => {
            if (key.equals(feature)) {
              newMap.set(key, {
                isEditing: false,
                description: '',
                neuron: n,
              });
            }
          });
          return newMap;
        });
        setLoadingFeature(null);
      })
      .catch((error) => {
        console.error(`error submitting getting rest of neuron: ${error}`);
      });
  }

  const contextValue = useMemo(
    () => ({
      searchText,
      setSearchText,
      isSearching,
      setIsSearching,
      loadingFeature,
      setLoadingFeature,
      selectedFeatures,
      setSelectedFeatures,
      graphRanges,
      setGraphRanges,
      openedList,
      setOpenedList,
      plotInactiveRef,
      plotActiveRef,
      loadFeature,
      visibleUmapExplanations,
      setVisibleUmapExplanations,
      highlightedUmapExplanations,
      setHighlightedUmapExplanations,
      showLogSparsity,
      setShowLogSparsity,
      layerToInitialColor,
      setLayerToInitialColor,
      // addAnnotationForExp,
    }),
    [
      searchText,
      isSearching,
      loadingFeature,
      selectedFeatures,
      graphRanges,
      openedList,
      loadFeature,
      visibleUmapExplanations,
      highlightedUmapExplanations,
      showLogSparsity,
      layerToInitialColor,
      // addAnnotationForExp,
    ],
  );

  return <UmapContext.Provider value={contextValue}>{children}</UmapContext.Provider>;
}
