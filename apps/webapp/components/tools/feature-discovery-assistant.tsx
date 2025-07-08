/* eslint-disable jsx-a11y/label-has-associated-control -- customComponent doc examples don't work :( */

'use client';

import ModelSelector from '@/components/feature-selector/model-selector';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import { SearchExplanationsResponse } from '@/lib/utils/general';
import {
  SteerFeature,
  STEER_TEMPERATURE,
  STEER_FREQUENCY_PENALTY,
  STEER_SEED,
  STEER_STRENGTH_MULTIPLIER,
  STEER_SPECIAL_TOKENS,
  ChatMessage as SteerChatMessage, // Renaming to avoid conflict with local ChatMessage
} from '@/lib/utils/steer';
import { callSteerChatApi } from '@/lib/utils/steer-api'; // Import the new utility
// import { ExplanationWithPartialRelations } from '@/prisma/generated/zod'; // No longer directly used in this component's props/state
import { useState, useEffect, useRef, useCallback } from 'react';

// Define a type for our structured feature suggestion
type SuggestedSteerFeature = SteerFeature & {
  originalExplanation: string; // To show the source explanation
  score?: number; // Placeholder for potential future sorting/relevance score
  isSelected: boolean; // To manage selection state
};

// Define FeatureSteeringResult type
// This should extend SuggestedSteerFeature to include steering-specific results
type FeatureSteeringResult = SuggestedSteerFeature & {
  steeredText?: string;
  steeringError?: string;
  isSteeringLoading?: boolean;
};

export default function FeatureDiscoveryAssistant() {
  const { getDefaultModel, getInferenceEnabledModels, showToastServerError } = useGlobalContext();
  const [modelId, setModelId] = useState<string>(getDefaultModel()?.id || '');
  const [testPrompt, setTestPrompt] = useState<string>('');
  const [featureQuery, setFeatureQuery] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false); // For the main search
  const [suggestedFeatures, setSuggestedFeatures] = useState<SuggestedSteerFeature[]>([]);
  // steeringResults will store the outcome for each of the 5 suggestedFeatures
  const [steeringResults, setSteeringResults] = useState<FeatureSteeringResult[]>([]);
  const [hasSteered, setHasSteered] = useState<boolean>(false); // Tracks if "Steer Selected Features" has been clicked

  // State for auto-steer timer
  const [autoSteerTimerId, setAutoSteerTimerId] = useState<NodeJS.Timeout | null>(null);
  const activeTimerIdRef = useRef<NodeJS.Timeout | null>(null); // To robustly check active timer in setTimeout
  const [userInteracted, setUserInteracted] = useState<boolean>(false);
  const [triggerAutoSteer, setTriggerAutoSteer] = useState<boolean>(false);


  const handleSearch = async () => {
    // Reset timer and interaction states on new search
    if (autoSteerTimerId) {
      clearTimeout(autoSteerTimerId);
    }
    setAutoSteerTimerId(null);
    if (activeTimerIdRef.current) {
      clearTimeout(activeTimerIdRef.current);
      activeTimerIdRef.current = null;
    }
    setUserInteracted(false);
    setHasSteered(false); // Also reset hasSteered as per instructions

    if (!modelId || !testPrompt.trim() || !featureQuery.trim()) {
      console.warn('Please select a model, enter a test prompt, and enter a feature name to search.');
      return;
    }
    setIsLoading(true);
    setSuggestedFeatures([]);
    setSteeringResults([]);
    // setHasSteered(false); // Already reset above

    try {
      const response = await fetch('/api/explanation/search-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: featureQuery, modelId, offset: 0 }),
      });

      if (!response.ok) throw new Error(`API Error: ${response.status} ${response.statusText}`);
      const data = (await response.json()) as SearchExplanationsResponse; // Type assertion for data

      if (data.results && data.results.length > 0) {
        const topResults = data.results.slice(0, 5);
        const formattedFeatures: SuggestedSteerFeature[] = topResults.map((exp, index) => {
          const { neuron } = exp; // exp is an ExplanationWithPartialRelations here
          if (!neuron || !neuron.index) {
            console.warn('Explanation result missing neuron data:', exp); return null;
          }
          return {
            modelId: neuron.modelId,
            layer: neuron.layer,
            index: parseInt(neuron.index, 10),
            explanation: `Feature related to: "${exp.description}" (Original model: ${neuron.modelId}, Layer: ${neuron.layer}, Index: ${neuron.index})`,
            strength: neuron.maxActApprox || 1,
            originalExplanation: exp.description,
            isSelected: index < 2, // Default select top 2
          };
        }).filter(Boolean) as SuggestedSteerFeature[];
        setSuggestedFeatures(formattedFeatures);

        // Initialize steeringResults for all 5 discovered features
        const initialSteeringData: FeatureSteeringResult[] = formattedFeatures.map(feature => ({
          ...feature,
          isSteeringLoading: false, // Not loading initially
          steeredText: undefined,
          steeringError: undefined,
        }));
        setSteeringResults(initialSteeringData);

        // Start auto-steer timer if features were found
        if (formattedFeatures.length > 0) {
          const newTimerId = setTimeout(() => {
            if (!userInteracted && activeTimerIdRef.current === newTimerId) {
              console.log('Auto-steering: Timer elapsed without user interaction.');
              // Auto-select top 3 features
              setSuggestedFeatures(prevFeatures =>
                prevFeatures.map((feature, index) => ({
                  ...feature,
                  isSelected: index < 3,
                }))
              );
              setSteeringResults(prevResults =>
                prevResults.map((result, index) => ({
                  ...result,
                  isSelected: index < 3,
                }))
              );
              setTriggerAutoSteer(true); // Trigger effect to call handleSteerSelectedFeatures
            }
          }, 5000);
          activeTimerIdRef.current = newTimerId;
          setAutoSteerTimerId(newTimerId);
        }
      } else {
        alert('No features found for your query.');
      }
    } catch (error) {
      console.error('Failed to search features:', error);
      showToastServerError();
      alert(`Failed to search features. ${error instanceof Error ? error.message : 'Please check console for details.'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSteerSelectedFeatures = useCallback(async () => {
    const selectedForSteering = suggestedFeatures.filter(f => f.isSelected);
    if (selectedForSteering.length === 0) {
      alert('Please select at least one feature to steer.');
      return;
    }
    setHasSteered(true);

    // Set loading state only for selected features in the main steeringResults array
    setSteeringResults(prevGlobalResults =>
      prevGlobalResults.map(res => {
        const isSelectedForThisOperation = selectedForSteering.some(
          sf => sf.modelId === res.modelId && sf.layer === res.layer && sf.index === res.index
        );
        return isSelectedForThisOperation
          ? { ...res, isSteeringLoading: true, steeredText: undefined, steeringError: undefined }
          : res;
      })
    );

    const steeringPromises = selectedForSteering.map(async (featureToSteer): Promise<FeatureSteeringResult> => {
      try {
        const apiFeature = {
          modelId: featureToSteer.modelId,
          layer: featureToSteer.layer,
          index: featureToSteer.index,
          strength: (featureToSteer.strength || 1) * 1.5,
          explanation: featureToSteer.explanation,
        };
        const modelToSteer = (modelId !== 'gemma-2-2b') ? modelId : 'gemma-2-2b-it';
        const steeringData = await callSteerChatApi({
          modelId: modelToSteer,
          defaultChatMessages: [{ role: 'user', content: testPrompt }] as SteerChatMessage[],
          steeredChatMessages: [{ role: 'user', content: testPrompt }] as SteerChatMessage[],
          features: [apiFeature],
          temperature: STEER_TEMPERATURE, nTokens: 96, freqPenalty: STEER_FREQUENCY_PENALTY,
          seed: STEER_SEED, strengthMultiplier: STEER_STRENGTH_MULTIPLIER, steerSpecialTokens: STEER_SPECIAL_TOKENS,
        });
        const assistantMessage = steeringData?.STEERED?.chatTemplate?.findLast(msg => msg.role === 'model' || msg.role === 'assistant')?.content;
        return { ...featureToSteer, steeredText: assistantMessage || 'No response text.', steeringError: undefined, isSteeringLoading: false };
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Steering request failed';
        return { ...featureToSteer, steeredText: undefined, steeringError: errorMessage, isSteeringLoading: false };
      }
    });

    const settledResults = await Promise.allSettled(steeringPromises);

    const updatedResultsFromPromises: FeatureSteeringResult[] = settledResults.map((settledResult, index) => {
      if (settledResult.status === 'fulfilled') {
        return settledResult.value;
      }
      const { reason } = settledResult;
      console.error('Unexpected promise rejection post-settled:', reason);
      const errorMessage = reason instanceof Error ? reason.message : String(reason);
      const originalFeatureForRejection = selectedForSteering[index];
      return {
        ...originalFeatureForRejection,
        steeredText: undefined,
        steeringError: `Unexpected error: ${errorMessage}`,
        isSteeringLoading: false,
      };
    });

    setSteeringResults(prevGlobalResults =>
      prevGlobalResults.map(existingGlobalFeatureResult => {
        const newlySteeredData = updatedResultsFromPromises.find(
          nr => nr.modelId === existingGlobalFeatureResult.modelId &&
            nr.layer === existingGlobalFeatureResult.layer &&
            nr.index === existingGlobalFeatureResult.index
        );
        if (newlySteeredData) {
          return {
            ...existingGlobalFeatureResult,
            ...newlySteeredData,
            isSteeringLoading: false,
          };
        }
        return existingGlobalFeatureResult;
      })
    );
  }, [suggestedFeatures, modelId, testPrompt, showToastServerError]);

  useEffect(() => {
    if (triggerAutoSteer) {
      handleSteerSelectedFeatures();
      setTriggerAutoSteer(false); // Reset trigger
    }
  }, [triggerAutoSteer, handleSteerSelectedFeatures]);

  // Cleanup timer on component unmount
  useEffect(() => () => {
    if (activeTimerIdRef.current) {
      clearTimeout(activeTimerIdRef.current);
    }
    // Also clear state timer if it matches, though ref is primary for active timer
    if (autoSteerTimerId) {
      clearTimeout(autoSteerTimerId);
    }
  }, [autoSteerTimerId]); // Include autoSteerTimerId to re-run cleanup if it changes, though ref is main

  const handleCheckboxChange = (changedIndex: number) => {
    if (!userInteracted) {
      setUserInteracted(true);
    }
    if (autoSteerTimerId) {
      console.log('Auto-steering: User interacted, clearing timer.');
      clearTimeout(autoSteerTimerId);
      setAutoSteerTimerId(null);
    }
    if (activeTimerIdRef.current) {
      clearTimeout(activeTimerIdRef.current);
      activeTimerIdRef.current = null;
    }

    setSuggestedFeatures(prevFeatures =>
      prevFeatures.map((feature, idx) =>
        idx === changedIndex ? { ...feature, isSelected: !feature.isSelected } : feature
      )
    );
  };

  return (
    <div className="m-auto flex max-w-7xl flex-col gap-6 rounded-lg border py-6 px-4 sm:px-6 lg:px-8 shadow">
      <h1 className="text-xl font-semibold text-slate-700">Feature Discovery Assistant</h1>

      {/* Inputs Section */}
      <div className="flex flex-col md:flex-row md:space-x-4 md:items-end">
        <div className="flex-1 grid w-full items-center gap-1.5 mb-4 md:mb-0">
          <label htmlFor='featureDiscoveryModelSelector'>
            1. Select Target Model
            <ModelSelector
              id="featureDiscoveryModelSelector"
              modelId={modelId}
              modelIdChangedCallback={(newModelId) => {
                setModelId(newModelId);
                setSuggestedFeatures([]);
                setSteeringResults([]);
                setHasSteered(false);
                setTestPrompt('');
                // Reset timer and interaction states when model changes
                if (autoSteerTimerId) {
                  clearTimeout(autoSteerTimerId);
                  setAutoSteerTimerId(null);
                }
                if (activeTimerIdRef.current) {
                  clearTimeout(activeTimerIdRef.current);
                  activeTimerIdRef.current = null;
                }
                setUserInteracted(false);
              }}
              overrideModels={getInferenceEnabledModels()}
            />
          </label>
        </div>
        <div className="flex-1 grid w-full items-center gap-1.5 mb-4 md:mb-0">
          <label htmlFor="test-prompt-fda">
            2. Enter Test Prompt
            <input
              type="text"
              id="test-prompt-fda"
              placeholder="Enter your test prompt here..."
              value={testPrompt}
              onChange={(e) => setTestPrompt(e.target.value)}
              className="mt-1 flex h-10 w-full rounded-md border border-slate-300 bg-transparent px-3 py-2 text-sm placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            />
          </label>
        </div>
        <div className="flex-1 grid w-full items-center gap-1.5 mb-4 md:mb-0">
          <label htmlFor="feature-name-fda">
            3. Enter Desired Feature Name
            <input
              type="text"
              id="feature-name-fda"
              placeholder="Type a descriptive name..."
              value={featureQuery}
              onChange={(e) => setFeatureQuery(e.target.value)}
              className="mt-1 flex h-10 w-full rounded-md border border-slate-300 bg-transparent px-3 py-2 text-sm placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            />
          </label>
          <p className="text-xs text-slate-500">E.g., &quot;social isolation&quot;, &quot;positive sentiment&quot;, &quot;code generation&quot;</p>
        </div>
      </div>

      <Button onClick={handleSearch} disabled={isLoading || !modelId || !testPrompt.trim() || !featureQuery.trim()}>
        {isLoading ? 'Searching...' : 'Find Top 5 Matching Features'}
      </Button>

      {/* Discovered Features Section */}
      {suggestedFeatures.length > 0 && (
        <div className="mt-4">
          <h2 className="text-lg font-medium text-slate-600">Discovered Features:</h2>
          <div className="mt-2 flex flex-row flex-wrap -mx-2">
            {suggestedFeatures.map((feature, idx) => {
              const steeringResultForCard = steeringResults.find(
                sr => sr.modelId === feature.modelId && sr.layer === feature.layer && sr.index === feature.index
              );

              const cardClasses = [
                "w-full sm:w-1/2 lg:w-1/3 xl:w-1/4 p-2 mb-4 rounded-md bg-slate-50 shadow-sm transition-all duration-300 ease-in-out",
              ];
              if (hasSteered && !feature.isSelected) {
                cardClasses.push("filter grayscale opacity-50");
              }

              return (
                <div key={`feature-${feature.modelId}-${feature.layer}-${feature.index}`} className={cardClasses.join(' ')}>
                  <div className="flex items-center justify-between">
                    <h3 className="mb-1 font-semibold text-emerald-700">Feature #{idx + 1}</h3>
                    <input
                      type="checkbox"
                      id={`feature-checkbox-${idx}`}
                      checked={feature.isSelected}
                      onChange={() => handleCheckboxChange(idx)}
                      className="form-checkbox h-5 w-5 text-emerald-600 transition duration-150 ease-in-out"
                    />
                  </div>
                  <p className="mb-1 text-sm text-slate-600">
                    Derived from: &quot;{feature.originalExplanation}&quot;
                  </p>
                  <p className="mb-2 text-xs text-slate-500">
                    modelId: &apos;{feature.modelId}&apos;, layer: &apos;{feature.layer}&apos;, index: {feature.index}, strength: {feature.strength}
                  </p>

                  {/* Steering Result Display within Card */}
                  {steeringResultForCard && (steeringResultForCard.isSteeringLoading || (hasSteered && feature.isSelected)) && (
                    <div className="mt-3 border-t border-slate-200 pt-3">
                      {steeringResultForCard.isSteeringLoading ? (
                        <p className="text-sm text-slate-500 animate-pulse">Steering in progress...</p>
                      ) : steeringResultForCard.steeringError ? (
                        <div className="rounded-md bg-red-100 p-2">
                          <p className="text-sm font-medium text-red-700">Steering Error:</p>
                          <p className="text-xs text-red-600">{steeringResultForCard.steeringError}</p>
                        </div>
                      ) : steeringResultForCard.steeredText ? (
                        <div className="rounded-md bg-green-50 p-2">
                          <p className="text-sm font-medium text-green-700">Steered Output:</p>
                          <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-green-100 p-1.5 text-xs text-green-800">{steeringResultForCard.steeredText}</pre>
                        </div>
                      ) : (hasSteered && feature.isSelected) ? (
                        <p className="text-sm text-slate-500">Steering completed, but no text was returned.</p>
                      ) : null}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Select features and click &quot;Steer Selected Features&quot; to test their effect.
          </p>
        </div>
      )}

      {/* "Steer Selected Features" Button */}
      {suggestedFeatures.length > 0 && (
        <Button
          onClick={handleSteerSelectedFeatures}
          disabled={isLoading || suggestedFeatures.every(f => !f.isSelected) || steeringResults.some(sr => sr.isSteeringLoading)}
          className="mt-4"
        >
          {steeringResults.some(sr => sr.isSteeringLoading) ? 'Steering...' : 'Steer Selected Features'}
        </Button>
      )}
    </div>
  );
}
