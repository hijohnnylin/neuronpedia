'use client';

import { NeuronWithPartialRelations } from 'prisma/generated/zod';
import { ReactNode, useEffect, useMemo, useState } from 'react';
import { useGlobalContext } from './global-provider';
import createContextWrapper from './provider-util';

export enum InferenceActivationAllState {
  DEFAULT,
  RUNNING,
  LOADED,
}

export type InferenceActivationAllResponse = {
  tokens: string[];
  result: InferenceActivationAllResult[];
  sortIndexes?: number[];
};

export type InferenceActivationAllResult = {
  modelId: string;
  layer: string;
  index: string;
  values: number[];
  maxValue: number;
  maxValueIndex: number;
  neuron: NeuronWithPartialRelations | undefined;
  dfaValues?: number[] | undefined;
  dfaTargetIndex?: number | undefined;
  dfaMaxValue?: number | undefined;
};

export const [InferenceActivationAllContext, useInferenceActivationAllContext] = createContextWrapper<{
  exploreState: InferenceActivationAllState;
  setExploreState: React.Dispatch<React.SetStateAction<InferenceActivationAllState>>;
  submitSearchAll: (
    modelId: string,
    text: string,
    selectedLayers: string[] | undefined,
    sourceSet: string,
    ignoreBos: boolean,
    sortIndex: number[],
  ) => void;
  tokens: string[];
  setTokens: React.Dispatch<React.SetStateAction<string[]>>;
  overallMaxValue: number;
  searchSortIndexes: number[];
  searchResults: InferenceActivationAllResult[];
  setSearchResults: React.Dispatch<React.SetStateAction<InferenceActivationAllResult[]>>;
  resultsGrid: (InferenceActivationAllResult | undefined)[][];
  setResultsGrid: React.Dispatch<React.SetStateAction<(InferenceActivationAllResult | undefined)[][]>>;
}>('InferenceActivationAllContext');

export default function InferenceActivationAllProvider({ children }: { children: ReactNode }) {
  const { showToastServerError, showToastMessage } = useGlobalContext();
  const [exploreState, setExploreState] = useState<InferenceActivationAllState>(InferenceActivationAllState.DEFAULT);
  const [searchResults, setSearchResults] = useState<InferenceActivationAllResult[]>([]);
  const [searchSortIndexes, setSearchSearchSortIndexes] = useState<number[]>([]);
  const [resultsGrid, setResultsGrid] = useState<(InferenceActivationAllResult | undefined)[][]>([]);
  const [overallMaxValue, setOverallMaxValue] = useState(-10);
  const [tokens, setTokens] = useState<string[]>([]);

  useEffect(() => {
    if (searchResults.length > 0) {
      let maxVal = -10;
      searchResults.forEach((a) => {
        if (a.maxValue > maxVal) {
          maxVal = a.maxValue;
        }
      });
      setOverallMaxValue(maxVal);
    }
  }, [searchResults]);

  async function submitSearchAll(
    modelId: string,
    text: string,
    selectedLayers: string[] | undefined,
    sourceSet: string,
    ignoreBos: boolean,
    sortIndexes: number[] = [],
  ) {
    if (!selectedLayers) {
      alert('Please select at least one layer to search.');
      return;
    }
    setExploreState(InferenceActivationAllState.RUNNING);
    try {
      const response = await fetch(`/api/search-all`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelId,
          text,
          selectedLayers,
          sortIndexes: sortIndexes as number[],
          sourceSet,
          ignoreBos,
        }),
      });
      const json = await response.json().catch(() => undefined);

      // A failed search still answers with JSON, just `{ error, message }` instead of a result. So
      // the status and the shape both have to be checked before this counts as LOADED: storing an
      // error body leaves `tokens` undefined while the searcher renders as though it had results,
      // and it crashes reading `tokens.length` rather than reporting the failure.
      if (!response.ok || !Array.isArray(json?.tokens) || !Array.isArray(json?.result)) {
        console.error('search-all failed', response.status, json);
        // 4xx messages are written for the client by `ApiError` (an unknown model, a query that is
        // too long), so they say something useful. Anything else is ours and stays generic.
        const clientMessage =
          response.status >= 400 && response.status < 500 ? json?.message || json?.error : undefined;
        if (clientMessage) {
          showToastMessage(clientMessage);
        } else {
          showToastServerError();
        }
        setExploreState(InferenceActivationAllState.DEFAULT);
        return;
      }

      const resp = json as InferenceActivationAllResponse;
      setSearchResults(resp.result);
      setSearchSearchSortIndexes(resp.sortIndexes || []);
      setTokens(resp.tokens);
      setExploreState(InferenceActivationAllState.LOADED);
    } catch (error) {
      showToastServerError();
      setExploreState(InferenceActivationAllState.DEFAULT);
      console.error(error);
    }
  }

  return (
    <InferenceActivationAllContext.Provider
      value={useMemo(
        () => ({
          exploreState,
          setExploreState,
          submitSearchAll,
          tokens,
          setTokens,
          searchResults,
          overallMaxValue,
          setSearchResults,
          searchSortIndexes,
          resultsGrid,
          setResultsGrid,
        }),
        [exploreState, submitSearchAll, tokens, searchResults, overallMaxValue, searchSortIndexes, resultsGrid],
      )}
    >
      {children}
    </InferenceActivationAllContext.Provider>
  );
}
