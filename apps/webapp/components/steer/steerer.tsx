'use client';

import { SteerResultChat } from '@/app/api/steer-chat/route';
import ExplanationsSearcher from '@/components/explanations-searcher';
import FeatureSelector from '@/components/feature-selector/feature-selector';
import ModelSelector from '@/components/feature-selector/model-selector';
import NewVectorForm from '@/components/new-vector-form';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import { useIsMount } from '@/lib/hooks/use-is-mount';
import { SearchExplanationsType } from '@/lib/utils/general';
import { getFirstSourceSetForModel } from '@/lib/utils/source';
import {
  ChatMessage,
  convertOldSteerOutputToChatMessages,
  FeaturePreset,
  replaceSteerModelIdIfNeeded,
  STEER_FREQUENCY_PENALTY,
  STEER_METHOD,
  STEER_N_COMPLETION_TOKENS,
  STEER_N_COMPLETION_TOKENS_LARGE_LLM,
  STEER_N_COMPLETION_TOKENS_THINKING,
  STEER_SEED,
  STEER_SPECIAL_TOKENS,
  STEER_STRENGTH_MULTIPLIER,
  STEER_TEMPERATURE,
  SteerFeature,
  SteerPreset,
} from '@/lib/utils/steer';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { Model, Visibility } from '@prisma/client';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { NPLogprob, NPSteerMethod } from 'neuronpedia-inference-client';
import { useSearchParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import SteerAdvancedSettings from './advanced-settings';
import SteerCompletion from './completion';
import SteerCompletionChat from './completion-chat';
import SteerPresetSelector from './preset-selector';
import SteerSelectedFeature from './selected-feature';
import SteerTooltip from './tooltip';

const MODELS_TO_FILTER_OUT: string[] = []; // ['gpt-oss-20b']
const MODELS_TO_FILTER_OUT_PREFIX = ['gemma-3-'];
export const NNSIGHT_MODELS = ['llama3.3-70b-it', 'gpt-oss-20b'];

export default function Steerer({
  initialModelId,
  initialSource,
  initialIndex,
  initialStrength,
  initialSavedId,
  hideInitialSettingsOnMobile = false,
  initialPreset,
}: {
  initialModelId: string;
  initialSource?: string;
  initialIndex?: string;
  initialStrength?: string;
  initialSavedId?: string;
  hideInitialSettingsOnMobile?: boolean;
  initialPreset?: string;
}) {
  const {
    globalModels,
    setSignInModalOpen,
    showToastServerError,
    user,
    getInferenceEnabledForModel,
    getFirstInferenceEnabledModel,
    getInferenceEnabledModels,
  } = useGlobalContext();
  const searchParams = useSearchParams();
  const hideModelSelector = searchParams.get('hideModelSelector') === 'true';
  const hideAdvancedSettings = searchParams.get('hideAdvancedSettings') === 'true';
  const [inCompletionMode, setInCompletionMode] = useState(false);
  // this should never be blank
  const [modelId, setModelId] = useState(
    !getInferenceEnabledForModel(initialModelId) && getFirstInferenceEnabledModel()
      ? getFirstInferenceEnabledModel() || ''
      : initialModelId || '',
  );
  const [featurePresets, setFeaturePresets] = useState<FeaturePreset[]>([]);
  const [typedInText, setTypedInText] = useState('');
  const [defaultChatMessages, setDefaultChatMessages] = useState<ChatMessage[]>([]);
  const [steeredChatMessages, setSteeredChatMessages] = useState<ChatMessage[]>([]);
  const [defaultCompletionText, setDefaultCompletionText] = useState('');
  const [defaultCompletionLogProbs, setDefaultCompletionLogProbs] = useState<NPLogprob[] | null>(null);
  const [steeredCompletionText, setSteeredCompletionText] = useState('');
  const [steeredCompletionLogProbs, setSteeredCompletionLogProbs] = useState<NPLogprob[] | null>(null);

  // Default Steering Settings
  const [steerTokens, setSteerTokens] = useState(
    NNSIGHT_MODELS.includes(modelId) ? STEER_N_COMPLETION_TOKENS_LARGE_LLM : STEER_N_COMPLETION_TOKENS,
  );
  const [temperature, setTemperature] = useState(STEER_TEMPERATURE);
  const [freqPenalty, setFreqPenalty] = useState(STEER_FREQUENCY_PENALTY);
  const [strMultiple, setStrMultiple] = useState(STEER_STRENGTH_MULTIPLIER);
  const [steerSpecialTokens, setSteerSpecialTokens] = useState(STEER_SPECIAL_TOKENS);
  const [seed, setSeed] = useState(STEER_SEED);
  const [steerMethod, setSteerMethod] = useState<NPSteerMethod>(STEER_METHOD);
  const [randomSeed, setRandomSeed] = useState(false);

  const [selectedFeatures, setSelectedFeatures] = useState<SteerFeature[]>([]);
  const [isSteering, setIsSteering] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [showAddFeature, setShowAddFeature] = useState(false);
  const [showAddVector, setShowAddVector] = useState(false);
  const [showSettingsOnMobile, setShowSettingsOnMobile] = useState(
    initialSavedId === undefined && !hideInitialSettingsOnMobile,
  );
  const [model, setModel] = useState<Model | null>(null);
  const [loadingPresets, setLoadingPresets] = useState(false);
  const isMount = useIsMount();
  const [copying, setCopying] = useState(false);

  function setUrl(steerOutputId: string | null) {
    if (steerOutputId === null) {
      let newUrl = `/${modelId}/steer`;
      newUrl += searchParams.toString() ? `?${searchParams.toString()}` : '';
      window.history.replaceState({ ...window.history.state, as: newUrl, url: newUrl }, '', newUrl);
    } else {
      // check if searchparams has saved
      let newUrl = `/${modelId}/steer`;
      newUrl += `?saved=${steerOutputId}`;
      if (!searchParams.get('saved')) {
        newUrl += searchParams.toString() ? `&${searchParams.toString()}` : '';
      } else {
        // get all the params except saved
        newUrl += searchParams.toString()
          ? searchParams.toString().replace(`saved=${searchParams.get('saved')}`, '')
          : '';
      }
      window.history.replaceState({ ...window.history.state, as: newUrl, url: newUrl }, '', newUrl);
    }
  }

  function reset() {
    setDefaultChatMessages([]);
    setSteeredChatMessages([]);
    setDefaultCompletionText('');
    setSteeredCompletionText('');
    setDefaultCompletionLogProbs(null);
    setSteeredCompletionLogProbs(null);
    setTypedInText('');
    setUrl(null);
  }

  function resetFeatures() {
    setSelectedFeatures([]);
  }

  function loadPresets() {
    setLoadingPresets(true);
    fetch('/api/steer/presets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelId,
      }),
    })
      .then((response) => response.json())
      .then((data: SteerPreset) => {
        setModel(data.model);
        setFeaturePresets(data.featurePresets);
        if (initialPreset) {
          const preset = data.featurePresets.find((p) => p.alias === initialPreset);
          if (preset) {
            setSelectedFeatures(preset.features);
            if (preset.steerMethod) {
              setSteerMethod(preset.steerMethod);
            }
          }
        }
      })
      .catch((error) => {
        console.error(`error loading presets: ${error}`);
      })
      .finally(() => {
        setLoadingPresets(false);
      });
  }

  function findExplanationFromPresets(explanationModelId: string, layer: string, index: number) {
    // this exists because we prefer the explanations from the presets, rather than the actual top explanation from the feature
    // eslint-disable-next-line no-restricted-syntax
    for (const preset of featurePresets) {
      const feature = preset.features.find(
        (f) => f.modelId === explanationModelId && f.layer === layer && f.index === index,
      );
      if (feature) {
        return feature.explanation;
      }
    }
    return undefined;
  }

  async function loadSavedSteerOutput(steerOutputId: string) {
    setIsSteering(true);
    reset();
    await fetch(`/api/steer-load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        steerOutputId,
      }),
    })
      .then((response) => {
        if (response.status !== 200) {
          console.error(response);
          alert('Sorry, your message could not be sent at this time. Please try again later.');
          return null;
        }
        return response.json();
      })
      .then((resp: SteerResultChat | null) => {
        if (resp === null) {
          setIsSteering(false);
          setDefaultCompletionText('');
          setSteeredCompletionText('');
          setDefaultCompletionLogProbs(null);
          setSteeredCompletionLogProbs(null);
          return;
        }
        setIsSteering(false);
        if (resp.settings) {
          setTemperature(resp.settings.temperature);
          setSteerTokens(resp.settings.n_tokens);
          setFreqPenalty(resp.settings.freq_penalty);
          setSeed(resp.settings.seed);
          setStrMultiple(resp.settings.strength_multiplier);
          setSteerSpecialTokens(resp.settings.steer_special_tokens);
          setSteerMethod(resp.settings.steer_method);
        }
        setUrl(resp.id || '');
        // check if model is instruct
        if (!globalModels[modelId].instruct) {
          setDefaultCompletionText(resp.DEFAULT?.raw || '');
          setSteeredCompletionText(resp.STEERED?.raw || '');
          setDefaultCompletionLogProbs(resp.DEFAULT?.logprobs || null);
          setSteeredCompletionLogProbs(resp.STEERED?.logprobs || null);
          setTypedInText(resp.inputText || '');
        } else {
          // if chat template is null, we need to convert it (it's an older form of steering)
          // eslint-disable-next-line no-lonely-if
          if (resp.DEFAULT?.chatTemplate) {
            setDefaultChatMessages(resp.DEFAULT?.chatTemplate || []);
            setSteeredChatMessages(resp.STEERED?.chatTemplate || []);
            setDefaultCompletionLogProbs(resp.DEFAULT?.logprobs || null);
            setSteeredCompletionLogProbs(resp.STEERED?.logprobs || null);
          } else {
            const defaultRaw = resp.DEFAULT?.raw;
            const steeredRaw = resp.STEERED?.raw;
            if (defaultRaw && steeredRaw) {
              setDefaultChatMessages(convertOldSteerOutputToChatMessages(defaultRaw));
              setSteeredChatMessages(convertOldSteerOutputToChatMessages(steeredRaw));
            }
          }
        }
        const features = resp.features?.map((f) => ({
          modelId: f.modelId,
          layer: f.layer,
          index: parseInt(f.index, 10),
          explanation:
            findExplanationFromPresets(f.modelId, f.layer, parseInt(f.index, 10)) ||
            (f.neuron?.explanations && f.neuron?.explanations.length > 0
              ? f.neuron?.explanations[0].description
              : f.neuron?.vectorLabel || ''),
          strength: f.strength,
          hasVector: f.neuron?.vector && f.neuron?.vector?.length > 0,
        }));
        setSelectedFeatures(features || []);
      })
      .catch((error) => {
        showToastServerError();
        setIsSteering(false);
        console.error(error);
      });
  }

  async function setInitialSteerFeatureFromSourceAndIndex(source: string, index: string) {
    // use the searchparams "source" and "index" to set the initial steering configuration
    setIsSteering(true);
    reset();
    await fetch(`/api/feature/${modelId}/${source}/${index}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .then((resp: NeuronWithPartialRelations) => {
        setIsSteering(false);
        const feature = {
          modelId: resp.modelId,
          layer: resp.layer,
          index: parseInt(resp.index, 10),
          explanation:
            findExplanationFromPresets(resp.modelId, resp.layer, parseInt(resp.index, 10)) ||
            (resp.explanations && resp.explanations.length > 0
              ? resp.explanations[0].description
              : resp.vectorLabel || ''),
          strength: initialStrength ? parseFloat(initialStrength) : resp.vectorDefaultSteerStrength || 10,
          hasVector: resp.hasVector,
        };
        setSelectedFeatures([feature]);
      })
      .catch((error) => {
        showToastServerError();
        setIsSteering(false);
        console.error(error);
      });
  }

  useEffect(() => {
    if (modelId) {
      loadPresets();
      setSeed(STEER_SEED);
      const isCompletionMode = !globalModels[modelId].instruct;
      setInCompletionMode(isCompletionMode);
      setSteerSpecialTokens(!isCompletionMode);
      if (globalModels[modelId].thinking) {
        setSteerTokens(STEER_N_COMPLETION_TOKENS_THINKING);
      } else if (NNSIGHT_MODELS.includes(modelId)) {
        setSteerTokens(STEER_N_COMPLETION_TOKENS_LARGE_LLM);
      } else {
        setSteerTokens(STEER_N_COMPLETION_TOKENS);
      }
      setTemperature(STEER_TEMPERATURE);
      setStrMultiple(STEER_STRENGTH_MULTIPLIER);
      setFreqPenalty(STEER_FREQUENCY_PENALTY);
      setRandomSeed(false);
      setSteerMethod(STEER_METHOD);
      reset();
      resetFeatures();
    }
  }, [modelId]);

  useEffect(() => {
    if (isMount) {
      if (initialSavedId) {
        // load the default and steered from the steered id
        loadSavedSteerOutput(initialSavedId);
      } else if (initialSource && initialIndex) {
        // load the default and steered from the source and index
        setInitialSteerFeatureFromSourceAndIndex(initialSource, initialIndex);
      }
    }
  }, [initialSavedId, initialSource, initialIndex]);

  function selectedFeaturesHaveVector() {
    return selectedFeatures.find((f) => f.hasVector);
  }

  function addToSelectedFeatures(feature: SteerFeature) {
    // if selectedFeatures has a vector, don't allow this
    if (selectedFeaturesHaveVector() && !feature.hasVector) {
      alert(
        'You are currently only steering non-SAE vectors. You may steer SAE features, or non-SAE vectors, but not both at the same time. Please remove existing vectors to add this feature.',
      );
      return;
    }
    if (!selectedFeaturesHaveVector() && feature.hasVector) {
      alert(
        'You are currently only steering SAE features. You may steer SAE features, or non-SAE vectors, but not both at the same time. Please remove existing SAE features to add this vector.',
      );
      return;
    }
    setSelectedFeatures((prevSelectedFeatures) => [...prevSelectedFeatures, feature]);
  }

  function presetIsSelected(preset: FeaturePreset): boolean {
    const presetFeatures = preset.features;

    // check that ALL the features in the preset are selected
    const found = presetFeatures.every((f) =>
      selectedFeatures.find((sf) => {
        const match =
          sf.modelId === f.modelId && sf.layer === f.layer && sf.index === f.index && sf.strength === f.strength;
        return match;
      }),
    );
    return found;
  }

  function currentlySelectedPreset(): FeaturePreset | null {
    // eslint-disable-next-line no-restricted-syntax
    for (const preset of featurePresets) {
      if (presetIsSelected(preset)) {
        return preset;
      }
    }
    return null;
  }

  function setFeatureStrength(feature: SteerFeature, strength: number) {
    setSelectedFeatures(
      selectedFeatures.map((f) => {
        if (f.modelId === feature.modelId && f.layer === feature.layer && f.index === feature.index) {
          return {
            modelId: feature.modelId,
            layer: feature.layer,
            index: feature.index,
            explanation: f.explanation,
            strength,
            hasVector: feature.hasVector,
          };
        }
        return f;
      }),
    );
  }

  async function deleteUserVector(preset: FeaturePreset) {
    if (window.confirm('Are you sure you want to delete this vector?')) {
      await fetch(`/api/vector/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelId: preset.features[0].modelId,
          source: preset.features[0].layer,
          index: preset.features[0].index,
        }),
      });
      loadPresets();
    }
  }

  useEffect(() => {
    if (copying) {
      setTimeout(() => setCopying(false), 2000);
    }
  }, [copying]);

  return (
    <div className="relative flex h-full w-full flex-col items-start justify-start overflow-scroll sm:flex-row">
      {/* MOBILE: Button for toggling Settings */}
      <div className="absolute top-2.5 flex w-full flex-row justify-center px-4 sm:hidden">
        <div className="relative flex flex-1 flex-row justify-center gap-x-3">
          <Button
            variant="outline"
            onClick={() => setShowSettingsOnMobile(!showSettingsOnMobile)}
            className="z-10 text-xs shadow"
          >
            {currentlySelectedPreset()?.name
              ? `${currentlySelectedPreset()?.name} Mode Steering`
              : selectedFeatures.length > 0
                ? `Steering ${selectedFeatures.length} Feature${selectedFeatures.length === 1 ? '' : 's'}`
                : 'Steering Settings'}
            {showSettingsOnMobile ? (
              <ChevronUp className="ml-1.5 h-3.5 w-3.5" />
            ) : (
              <ChevronDown className="ml-1.5 h-3.5 w-3.5" />
            )}
          </Button>
        </div>
      </div>
      <div
        className={`w-full flex-col overflow-hidden px-5 pt-14 sm:h-full sm:basis-1/3 sm:pt-5 ${showSettingsOnMobile ? 'flex' : 'hidden sm:flex'
          }`}
      >
        <div
          className={`${hideModelSelector ? 'mb-0.5' : 'mb-4'
            } mt-2 flex w-full flex-row items-center justify-center gap-x-1 text-xl font-semibold leading-none tracking-tight text-slate-700 sm:mt-0 sm:justify-between`}
        >
          Steer Models
          <SteerTooltip />
        </div>
        {!hideModelSelector && (
          <>
            <div className="mb-0.5 flex w-full flex-row gap-x-1 text-left text-[10px] font-medium uppercase text-slate-500">
              Select Model to Steer
            </div>
            <div className="sm:ml-2">
              <ModelSelector
                modelId={modelId}
                modelIdChangedCallback={(newModelId) => {
                  setModelId(newModelId);
                }}
                overrideModels={getInferenceEnabledModels().filter(
                  (m) =>
                    !MODELS_TO_FILTER_OUT.includes(m) &&
                    !MODELS_TO_FILTER_OUT_PREFIX.some((prefix) => m.startsWith(prefix)),
                )}
              />
            </div>
          </>
        )}

        {showSearch ? (
          <div className="mt-1 flex flex-1 flex-col gap-y-3">
            <div className="relative my-3 flex flex-col rounded-md border border-slate-200 p-3 shadow-md">
              <Button
                variant="outline"
                className="absolute left-2 top-2 bg-slate-100 text-[11px] font-normal uppercase text-slate-500 hover:bg-slate-200"
                onClick={() => setShowSearch(false)}
                size="sm"
              >
                Back
              </Button>
              <div className="mb-2 mt-0.5 text-center text-[13px] font-medium text-slate-500">Search for Features</div>

              <div className="flex flex-col">
                <ExplanationsSearcher
                  initialModelId={modelId}
                  defaultTab={SearchExplanationsType.BY_MODEL}
                  initialSourceSetName={undefined}
                  showTabs={false}
                  showModelSelector={false}
                  hideResultDetails
                  filterToInferenceEnabled
                  isSteerSearch
                  onClickResultCallback={(result) => {
                    addToSelectedFeatures({
                      modelId: result.neuron?.modelId || '',
                      layer: result.neuron?.layer || '',
                      index: parseInt(result.neuron?.index || '0', 10),
                      explanation: result.description || '',
                      strength: result.neuron?.maxActApprox || 40,
                    });
                    setShowSearch(false);
                  }}
                  neverChangePageOnSearch
                />
              </div>
            </div>
          </div>
        ) : showAddFeature ? (
          <div className="mt-1 flex flex-1 flex-col gap-y-3">
            <div className="relative my-3 flex flex-col rounded-md border border-slate-200 p-3 shadow-md">
              <Button
                variant="outline"
                className="absolute left-2 top-2 bg-slate-100 text-[11px] font-normal uppercase text-slate-500 hover:bg-slate-200"
                onClick={() => setShowAddFeature(false)}
                size="sm"
              >
                Back
              </Button>
              <div className="mb-2 mt-0.5 text-center text-[13px] font-medium text-slate-500">Add a Feature</div>

              <div className="mt-3 flex flex-col">
                <FeatureSelector
                  showModel={false}
                  defaultModelId={replaceSteerModelIdIfNeeded(modelId)}
                  filterToInferenceEnabled
                  defaultSourceSet={
                    getFirstSourceSetForModel(
                      globalModels[replaceSteerModelIdIfNeeded(modelId)],
                      Visibility.PUBLIC,
                      true,
                      false,
                    )?.name || ''
                  }
                  exclusiveCallback
                  callback={async (feature) => {
                    await fetch(`/api/feature/${feature.modelId}/${feature.layer}/${feature.index}`, {
                      method: 'GET',
                      headers: { 'Content-Type': 'application/json' },
                    })
                      .then((response) => response.json())
                      .then((n: NeuronWithPartialRelations) => {
                        addToSelectedFeatures({
                          modelId: feature.modelId,
                          layer: feature.layer,
                          index: parseInt(feature.index, 10),
                          explanation:
                            n.explanations && n.explanations.length > 0 ? n.explanations[0].description || '' : '',
                          strength: 20,
                        });
                        setShowAddFeature(false);
                      })
                      .catch((error) => {
                        alert('Error getting that feature.');
                        console.error(error);
                      });
                  }}
                />
              </div>
            </div>
          </div>
        ) : showAddVector ? (
          <div className="mt-1 flex flex-1 flex-col gap-y-3">
            <div className="relative my-3 flex flex-col rounded-md border border-slate-200 p-3 shadow-md">
              <Button
                variant="outline"
                className="absolute left-2 top-2 bg-slate-100 text-[11px] font-normal uppercase text-slate-500 hover:bg-slate-200"
                onClick={() => setShowAddVector(false)}
                size="sm"
              >
                Back
              </Button>
              <div className="mb-2 mt-0.5 text-center text-[13px] font-medium text-slate-500">Add Vector</div>
              <div className="mt-1.5 text-[10px] leading-snug text-slate-500">
                {`This steering vector will appear in your presets. You'll be able to share steers that use it, but it
                will not appear in other users' presets.`}
              </div>

              <div className="mt-2 flex flex-col">
                {model && (
                  <NewVectorForm
                    defaultModel={model}
                    callback={() => {
                      loadPresets();
                      setShowAddVector(false);
                    }}
                  />
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-1 flex-col">
            <div className="mb-0.5 mt-4 text-left text-[10px] font-medium uppercase text-slate-500">
              Select a Preset
            </div>
            <SteerPresetSelector
              loadingPresets={loadingPresets}
              featurePresets={featurePresets}
              // eslint-disable-next-line react/jsx-no-bind
              presetIsSelected={presetIsSelected}
              selectedFeatures={selectedFeatures}
              setSelectedFeatures={setSelectedFeatures}
              setSteerMethod={setSteerMethod}
              setShowSettingsOnMobile={setShowSettingsOnMobile}
              // eslint-disable-next-line react/jsx-no-bind
              deleteUserVector={deleteUserVector}
              // eslint-disable-next-line react/jsx-no-bind
              loadSavedSteerOutput={loadSavedSteerOutput}
            />

            <div className="mb-0.5 mt-5 text-left text-[10px] font-medium uppercase text-slate-500">What to Steer</div>
            <div className="mt-2 flex w-full flex-row items-center justify-center gap-x-2">
              <Button
                onClick={() => setShowSearch(true)}
                size="sm"
                className="flex-1 border-slate-300 text-slate-500"
                variant="outline"
              >
                Search Features
              </Button>
              <Button
                onClick={() => setShowAddFeature(true)}
                size="sm"
                className="flex-1 border-slate-300 text-slate-500"
                variant="outline"
              >
                + Add Feature
              </Button>
              <Button
                onClick={() => {
                  if (!user) {
                    alert('Adding Custom Vectors requires a Neuronpedia account.');
                    setSignInModalOpen(true);
                  } else {
                    setShowAddVector(true);
                  }
                }}
                size="sm"
                className="hidden flex-1 border-slate-300 text-slate-500 sm:block"
                variant="outline"
              >
                + Add Vector
              </Button>
            </div>
            <div className="mt-1 flex max-h-[280px] min-h-0 w-full flex-col divide-y divide-slate-200 overflow-y-scroll py-0.5">
              {selectedFeatures.length === 0 && (
                <div className="flex w-full flex-row items-center justify-center py-2 pt-2 text-center text-sm font-medium leading-snug text-slate-600">
                  <div className="text-slate-300">No Features Selected</div>
                </div>
              )}
              {selectedFeatures.map((feature) => (
                <SteerSelectedFeature
                  key={feature.modelId + feature.layer + feature.index}
                  feature={feature}
                  // eslint-disable-next-line react/jsx-no-bind
                  setFeatureStrength={setFeatureStrength}
                  selectedFeatures={selectedFeatures}
                  // eslint-disable-next-line react/jsx-no-bind
                  setSelectedFeatures={setSelectedFeatures}
                  // eslint-disable-next-line react/jsx-no-bind
                  findExplanationFromPresets={findExplanationFromPresets}
                />
              ))}
            </div>
          </div>
        )}

        {!hideAdvancedSettings && (
          <SteerAdvancedSettings
            thinking={globalModels[modelId].thinking}
            inCompletionMode={inCompletionMode}
            steerTokens={steerTokens}
            setSteerTokens={setSteerTokens}
            temperature={temperature}
            setTemperature={setTemperature}
            freqPenalty={freqPenalty}
            setFreqPenalty={setFreqPenalty}
            strMultiple={strMultiple}
            setStrMultiple={setStrMultiple}
            seed={seed}
            setSeed={setSeed}
            randomSeed={randomSeed}
            setRandomSeed={setRandomSeed}
            steerSpecialTokens={steerSpecialTokens}
            setSteerSpecialTokens={setSteerSpecialTokens}
            steerMethod={steerMethod}
            setSteerMethod={setSteerMethod}
            isNnSightModel={NNSIGHT_MODELS.includes(modelId)}
          />
        )}

        <div className="w-full flex-row justify-center pb-20 sm:hidden">
          <Button
            variant={showSettingsOnMobile ? 'default' : 'outline'}
            onClick={() => {
              if (selectedFeatures.length === 0) {
                alert('Please select a preset mode to get started.');
              } else {
                setShowSettingsOnMobile(!showSettingsOnMobile);
              }
            }}
            className="fixed bottom-10 left-1/2 z-20 flex h-12 w-[160px] -translate-x-1/2 transform flex-row justify-center rounded-xl bg-emerald-600 px-5 py-3 text-sm shadow-md"
          >
            Start Chatting
          </Button>
        </div>
      </div>

      {inCompletionMode ? (
        <SteerCompletion
          showSettingsOnMobile={showSettingsOnMobile}
          isSteering={isSteering}
          defaultCompletionText={defaultCompletionText}
          defaultCompletionLogProbs={defaultCompletionLogProbs}
          setDefaultCompletionLogProbs={setDefaultCompletionLogProbs}
          steeredCompletionText={steeredCompletionText}
          steeredCompletionLogProbs={steeredCompletionLogProbs}
          setSteeredCompletionLogProbs={setSteeredCompletionLogProbs}
          modelId={modelId}
          selectedFeatures={selectedFeatures}
          typedInText={typedInText}
          setTypedInText={setTypedInText}
          // eslint-disable-next-line react/jsx-no-bind
          reset={reset}
          copying={copying}
          setCopying={setCopying}
          // eslint-disable-next-line react/jsx-no-bind
          setUrl={setUrl}
          temperature={temperature}
          steerTokens={steerTokens}
          freqPenalty={freqPenalty}
          randomSeed={randomSeed}
          seed={seed}
          strMultiple={strMultiple}
          setIsSteering={setIsSteering}
          setDefaultCompletionText={setDefaultCompletionText}
          setSteeredCompletionText={setSteeredCompletionText}
          steerMethod={steerMethod}
        />
      ) : (
        <SteerCompletionChat
          showSettingsOnMobile={showSettingsOnMobile}
          isSteering={isSteering}
          setIsSteering={setIsSteering}
          defaultChatMessages={defaultChatMessages}
          setDefaultChatMessages={setDefaultChatMessages}
          steeredChatMessages={steeredChatMessages}
          setSteeredChatMessages={setSteeredChatMessages}
          modelId={modelId}
          selectedFeatures={selectedFeatures}
          typedInText={typedInText}
          setTypedInText={setTypedInText}
          // eslint-disable-next-line react/jsx-no-bind
          reset={reset}
          copying={copying}
          setCopying={setCopying}
          // eslint-disable-next-line react/jsx-no-bind
          setUrl={setUrl}
          temperature={temperature}
          steerTokens={steerTokens}
          freqPenalty={freqPenalty}
          randomSeed={randomSeed}
          seed={seed}
          strMultiple={strMultiple}
          steerSpecialTokens={steerSpecialTokens}
          steerMethod={steerMethod}
        />
      )}
    </div>
  );
}
