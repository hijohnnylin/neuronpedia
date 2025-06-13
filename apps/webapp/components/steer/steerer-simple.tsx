/* eslint-disable no-console -- disabling this to match existing pattern */
// this is a radically simplified version of the steerer component
// it's for the gemma-scope demo and the explorables demo
// it has a lot of duplicate code with the steer component
// it should be refactored to share code with the steer component

'use client';

// React
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'; // useEffect, useRef added

// External Libraries
import * as Select from '@radix-ui/react-select';
import * as Slider from '@radix-ui/react-slider';
import { ChevronDown, ChevronDownIcon, ChevronUpIcon } from 'lucide-react';

// Application-specific
import { SteerResultChat } from '@/app/api/steer-chat/route';
import CustomTooltip from '@/components/custom-tooltip';
import { useGlobalContext } from '@/components/provider/global-provider'; // Still needed for loadSavedSteerOutput's error
import SimpleSteererLayout from '@/components/steer/shared/SimpleSteererLayout';
import SteererMoreOptionsContent from '@/components/steer/shared/SteererMoreOptionsContent';
import { useSimpleSteererLogic } from '@/hooks/useSimpleSteererLogic'; // Import the hook
import { ChatMessage, FeaturePreset, SteerFeature } from '@/lib/utils/steer';

const MAX_MESSAGE_LENGTH_CHARS = 128;

// Define a stable empty array for the default for excludedPresetNames
const DEFAULT_EXCLUDED_PRESET_NAMES: readonly string[] = Object.freeze([]);

export default function SteererSimple({
  initialModelId: initialModelIdFromProps, // Renamed to avoid conflict if we had local modelId state
  cappedHeight,
  showOptionsButton = true,
  excludedPresetNames = DEFAULT_EXCLUDED_PRESET_NAMES, // USE THE STABLE CONSTANT
}: {
  initialModelId: string;
  cappedHeight?: boolean;
  showOptionsButton?: boolean;
  excludedPresetNames?: readonly string[]; // Ensure type matches constant
}) {
  const [strengthLevel, setStrengthLevel] = useState(1);
  const { showToastServerError } = useGlobalContext(); // For loadSavedSteerOutput

  // transformFeaturesForApi callback specific to this component
  const transformFeaturesForApiCallback = useCallback(
    (featuresToTransform: SteerFeature[] /* , strengthConfigIgnored */) =>
      featuresToTransform.map((f) => ({
        modelId: f.modelId,
        layer: f.layer,
        index: f.index,
        explanation: f.explanation,
        strength: (f.strength || 1) * strengthLevel, // Uses strengthLevel from component's state
      })),
    [strengthLevel],
  );

  // Memoize initialStrengthConfig
  const stableInitialStrengthConfig = useMemo(() => ({ strengthLevel }), [strengthLevel]);

  // Ref to hold the actual logic for onPresetFeaturesSelected
  const onPresetFeaturesSelectedLogicRef = useRef<
    ((currentSelectedFeatures: SteerFeature[], preset?: FeaturePreset) => Promise<{ defaultChatMessages?: ChatMessage[]; steeredChatMessages?: ChatMessage[] } | void>) | null
  >(null);

  const {
    // modelId, // Use if model can be changed by this component, otherwise initialModelIdFromProps is enough for the hook
    // setModelId, // If model can be changed
    featurePresets,
    selectedFeatures,
    // setSelectedFeatures, // Managed by handlePresetChange
    defaultChatMessages,
    // setDefaultChatMessages, // Managed by hook's functions
    steeredChatMessages,
    // setSteeredChatMessages, // Managed by hook's functions
    typedInText,
    setTypedInText,
    isTuning,
    steerTokens,
    setSteerTokens,
    temperature,
    setTemperature,
    freqPenalty,
    setFreqPenalty,
    strMultiple,
    setStrMultiple,
    seed,
    setSeed,
    randomSeed,
    // setRandomSeed, // No longer used in this component
    steerSpecialTokens,
    setSteerSpecialTokens,
    showMoreOptions,
    normalEndRef,
    steeredEndRef,
    isLoadingPresets,
    // loadPresets, // Called internally by the hook
    sendChat,
    resetChatAndMessages,
    handlePresetChange,
    toggleMoreOptions,
    // Destructure setters needed by loadSavedSteerOutput
    setIsTuning, // Correctly destructure setIsTuning
    setDefaultChatMessages: hookSetDefaultChatMessages, // Keep aliases for clarity in loadSavedSteerOutputRef if preferred
    setSteeredChatMessages: hookSetSteeredChatMessages,
    setTemperature: hookSetTemperature,
    setSteerTokens: hookSetSteerTokens,
    setFreqPenalty: hookSetFreqPenalty,
    setSeed: hookSetSeed,
    setStrMultiple: hookSetStrMultiple,
    setSteerSpecialTokens: hookSetSteerSpecialTokens,
  } = useSimpleSteererLogic<{ strengthLevel: number }>({
    initialModelId: initialModelIdFromProps,
    presetsApiEndpoint: '/api/steer/presets',
    excludedPresetNames, // This will now be stable if defaulted
    initialStrengthConfig: stableInitialStrengthConfig, // USE THE MEMOIZED OBJECT
    transformFeaturesForApi: transformFeaturesForApiCallback,
    // Pass a stable callback that invokes the logic from the ref
    onPresetFeaturesSelected: useCallback(async (_currentSelectedFeatures: SteerFeature[], preset?: FeaturePreset) => {
      if (onPresetFeaturesSelectedLogicRef.current) {
        return onPresetFeaturesSelectedLogicRef.current(_currentSelectedFeatures, preset);
      }
      // Default return if ref is not set, though it should be by the time this is called
      return { defaultChatMessages: [], steeredChatMessages: [] };
    }, []), // This callback is stable
  });

  // Define loadSavedSteerOutput using setters from the hook
  const loadSavedSteerOutput = useCallback(
    async (steerOutputId: string) => {
      if (!steerOutputId) {
        hookSetDefaultChatMessages([]);
        hookSetSteeredChatMessages([]);
        return { defaultChatMessages: [], steeredChatMessages: [] };
      }
      setIsTuning(true);

      try {
        const response = await fetch(`/api/steer-load`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ steerOutputId }),
        });

        if (!response.ok) {
          // eslint-disable-next-line no-console
          console.error(`Error loading saved steer output! Status: ${response.status}`);
          setIsTuning(false);
          hookSetDefaultChatMessages([]);
          hookSetSteeredChatMessages([]);
          return { defaultChatMessages: [], steeredChatMessages: [] };
        }

        const resp = (await response.json()) as SteerResultChat | null;

        if (resp === null) {
          setIsTuning(false);
          hookSetDefaultChatMessages([]);
          hookSetSteeredChatMessages([]);
          return { defaultChatMessages: [], steeredChatMessages: [] };
        }

        setIsTuning(false);
        if (resp.settings) {
          hookSetTemperature(resp.settings.temperature);
          hookSetSteerTokens(resp.settings.n_tokens);
          hookSetFreqPenalty(resp.settings.freq_penalty);
          hookSetSeed(resp.settings.seed);
          hookSetStrMultiple(resp.settings.strength_multiplier);
          hookSetSteerSpecialTokens(resp.settings.steer_special_tokens);
        }
        const newDefaultMessages = resp.DEFAULT?.chatTemplate || [];
        const newSteeredMessages = resp.STEERED?.chatTemplate || [];
        hookSetDefaultChatMessages(newDefaultMessages);
        hookSetSteeredChatMessages(newSteeredMessages);
        return { defaultChatMessages: newDefaultMessages, steeredChatMessages: newSteeredMessages };
      } catch (error) {
        showToastServerError();
        setIsTuning(false);
        // eslint-disable-next-line no-console
        console.error(error);
        hookSetDefaultChatMessages([]);
        hookSetSteeredChatMessages([]);
        return { defaultChatMessages: [], steeredChatMessages: [] };
      }
    },
    [
      setIsTuning,
      hookSetDefaultChatMessages,
      hookSetSteeredChatMessages,
      hookSetTemperature,
      hookSetSteerTokens,
      hookSetFreqPenalty,
      hookSetSeed,
      hookSetStrMultiple,
      hookSetSteerSpecialTokens,
      showToastServerError,
    ],
  );

  // useEffect to update the ref with the latest logic
  useEffect(() => {
    onPresetFeaturesSelectedLogicRef.current = async (_currentSelectedFeatures: SteerFeature[], preset?: FeaturePreset) => {
      if (preset?.exampleSteerOutputId && loadSavedSteerOutput) {
        return loadSavedSteerOutput(preset.exampleSteerOutputId);
      }
      // If no example output, the hook's handlePresetChange already clears messages.
      // Or, we can explicitly return empty messages.
      return { defaultChatMessages: [], steeredChatMessages: [] };
    };
  }, [loadSavedSteerOutput]); // This effect depends on loadSavedSteerOutput
 
  // const examplePrompts = [ // Removed as no longer used
  //   { label: 'Tell me about yourself.', text: 'Tell me about yourself.' },
  //   { label: 'Tell me a one line story.', text: 'Tell me a one line story.' },
  //   { label: 'Write a haiku.', text: 'Write a haiku.' },
  //   { label: 'I wish...', text: 'Complete this: I wish...', fullLabel: 'Complete this: I wish...' },
  // ];
 
  const steeringControlsJsx = (
    <>
      <div className="mb-2 w-full flex-row justify-center pt-1.5 uppercase text-green-600">
        <div className="px-5 py-1 text-center text-[11px] font-bold">Steering Controls</div>
      </div>
      <div className="flex flex-col gap-y-1 rounded-xl bg-green-200 px-2 py-5">
        <div className="mt-0 hidden text-center text-[10px] font-bold uppercase leading-none text-green-800 sm:block">
          Feature
        </div>
        <div className="flex w-full flex-row items-center justify-center gap-x-1.5">
          <Select.Root
            disabled={isLoadingPresets || isTuning}
            value={
              featurePresets.find((p) =>
                p.features.length === selectedFeatures.length && // Basic check for same number of features
                p.features.every(pf => selectedFeatures.some(sf => sf.modelId === pf.modelId && sf.layer === pf.layer && sf.index === pf.index))
              )?.name || (featurePresets.length > 0 && selectedFeatures.length === 0 ? featurePresets[0].name : '') // Default to first if nothing selected
            }
            onValueChange={(presetName) => {
              handlePresetChange(presetName); // Use hook's handler
            }}
          >
            <Select.Trigger className="relative flex h-10 flex-1 select-none flex-row items-center justify-center gap-x-1.5 whitespace-pre rounded-xl border-2 border-green-700 bg-green-600 py-2 pl-4 pr-3 text-sm font-bold text-white hover:bg-green-700/50 focus:outline-none disabled:cursor-not-allowed disabled:opacity-70 sm:min-w-[100px] sm:pl-6 sm:pr-4 sm:text-sm">
              <Select.Value placeholder="Select feature..." />
              <Select.Icon>
                <ChevronDown className="absolute right-2 top-0 ml-0 h-full w-5 leading-none text-green-800" />
              </Select.Icon>
            </Select.Trigger>
            <Select.Portal>
              <Select.Content
                position="popper"
                align="center"
                sideOffset={-3}
                className="max-h-[320px] min-w-[100px] cursor-pointer overflow-hidden rounded-md border border-slate-300 bg-white text-xs font-medium text-slate-700 shadow"
              >
                <Select.ScrollUpButton className="flex justify-center border-b bg-slate-100 text-slate-300">
                  <ChevronUpIcon className="h-4 w-4" />
                </Select.ScrollUpButton>
                <Select.Viewport className="text-left">
                  {featurePresets.length > 0 ? (
                    featurePresets.map((preset) => (
                      <Select.Item
                        key={preset.name}
                        value={preset.name}
                        className="flex flex-col gap-y-0.5 overflow-hidden border-b px-3 py-2.5 font-sans text-xs text-slate-700 hover:bg-slate-100 focus:outline-none"
                      >
                        <Select.ItemText>
                          <span className="">{preset.name}</span>
                        </Select.ItemText>
                      </Select.Item>
                    ))
                  ) : (
                    <Select.Item
                      value="Loading..."
                      disabled
                      className="flex flex-col gap-y-0.5 overflow-hidden border-b px-3 py-2.5 font-sans text-xs text-slate-700"
                    >
                      <Select.ItemText>
                        <span className="">{isLoadingPresets ? 'Loading...' : 'No presets found'}</span>
                      </Select.ItemText>
                    </Select.Item>
                  )}
                </Select.Viewport>
                <Select.ScrollDownButton className="flex justify-center border-b bg-slate-100 text-slate-300">
                  <ChevronDownIcon className="h-4 w-4" />
                </Select.ScrollDownButton>
              </Select.Content>
            </Select.Portal>
          </Select.Root>
        </div>

        <div className="mt-4 hidden text-center text-[10px] font-bold uppercase leading-none text-green-800 sm:block">
          Strength
        </div>
        <div className="flex w-full flex-row items-center gap-x-0.5 px-2 pb-6 text-green-700">
          <Slider.Root
            defaultValue={[1]}
            value={[strengthLevel]}
            min={0}
            max={2}
            step={0.5}
            onValueChange={(value) => {
              setStrengthLevel(value[0]);
            }}
            className="relative mt-1 flex h-5 w-full flex-1 cursor-pointer items-center"
          >
            <Slider.Track className="relative h-[8px] grow rounded-full border border-green-600 bg-white group-hover:bg-green-50">
              <div className="absolute left-0 top-0 flex h-[8px] w-full flex-row">
                <div className="flex-1">
                  <div className="mt-[-7px] h-[20px] w-[1px] bg-green-700/0" />
                </div>
                <div className="flex-1">
                  <div className="mt-[-6px] h-[20px] w-[1px] bg-green-500" />
                </div>
                <div className="flex-1">
                  <div className="mt-[-6px] h-[20px] w-[1px] bg-green-500" />
                </div>
                <div className="flex-1">
                  <div className="mt-[-6px] h-[20px] w-[1px] bg-green-500" />
                </div>
              </div>
              <div className="absolute left-0 top-0 flex h-[8px] w-full flex-row">
                <div className="flex-1">
                  <div className="-ml-[8px] h-[16px] w-[16px] pt-[17px] text-[14px]">
                    <CustomTooltip trigger={<div>0️⃣</div>}>0️⃣ No Steering</CustomTooltip>
                  </div>
                </div>
                <div className="flex-1">
                  <div className="-ml-[8px] h-[16px] w-[16px] pt-[17px] text-[14px]">
                    <CustomTooltip trigger={<div>🤏</div>}>🤏 A Little</CustomTooltip>
                  </div>
                </div>{' '}
                <div className="flex-1">
                  <div className="-ml-[8px] h-[16px] w-[16px] pt-[17px] text-[14px]">
                    <CustomTooltip trigger={<div>⚖️</div>}>⚖️ Medium Strength</CustomTooltip>
                  </div>
                </div>{' '}
                <div className="flex-1">
                  <div className="-ml-[8px] h-[16px] w-[16px] pt-[17px] text-[14px]">
                    <CustomTooltip trigger={<div>💪</div>}>💪 Strong</CustomTooltip>
                  </div>
                  <div className="absolute right-0 top-1 -ml-[6.5px] mt-[18px] flex h-[16px] items-end justify-end text-[14px]">
                    <CustomTooltip trigger={<div>🤯</div>}>🤯 Quite A Bit</CustomTooltip>
                  </div>
                </div>{' '}
              </div>
              <Slider.Range className="absolute h-full rounded-full bg-green-400 group-hover:bg-green-700" />
            </Slider.Track>
            <Slider.Thumb className="flex h-[18px] w-9 items-center justify-center rounded-full border border-green-700 bg-white text-[9px] font-medium text-green-700 shadow group-hover:bg-green-100">
              {`+${strengthLevel.toFixed(1)}x`}
            </Slider.Thumb>
          </Slider.Root>
        </div>
      </div>
    </>
  );

  // const handleToggleMoreOptions = useCallback(() => { // Now from hook
  //   setShowMoreOptions((prev) => !prev);
  // }, [setShowMoreOptions]);

  return (
    <SimpleSteererLayout
      cappedHeight={cappedHeight}
      normalEndRef={normalEndRef} // From hook
      steeredEndRef={steeredEndRef} // From hook
      normalPanelTitle="Normal Gemma"
      steeredPanelTitle="Steered Gemma"
      initialNormalText="Hey, I'm normal Gemma!"
      initialSteeredText="Hey, I'm steered Gemma!"
      defaultChatMessages={defaultChatMessages} // From hook
      steeredChatMessages={steeredChatMessages} // From hook
      isTuning={isTuning} // From hook
      typedInText={typedInText} // From hook
      onTypedInTextChange={setTypedInText} // From hook
      onSendChat={sendChat} // From hook
      onResetChat={resetChatAndMessages} // From hook
      inputPlaceholder="Ask Gemma something..."
      isSendDisabled={isTuning || selectedFeatures.length === 0} // From hook
      isResetDisabled={defaultChatMessages.length === 0 || isTuning} // From hook
      // examplePrompts={examplePrompts} // Removed as SimpleSteererLayout no longer accepts this
      maxMessageLength={MAX_MESSAGE_LENGTH_CHARS} // Local const
      showMoreOptions={showMoreOptions} // From hook
      onToggleMoreOptions={toggleMoreOptions} // From hook
      moreOptionsContent={
        <SteererMoreOptionsContent
          strMultiple={strMultiple} // From hook
          setStrMultiple={setStrMultiple} // From hook
          steerSpecialTokens={steerSpecialTokens} // From hook
          setSteerSpecialTokens={setSteerSpecialTokens} // From hook
          steerTokens={steerTokens} // From hook
          setSteerTokens={setSteerTokens} // From hook
          temperature={temperature} // From hook
          setTemperature={setTemperature} // From hook
          freqPenalty={freqPenalty} // From hook
          setFreqPenalty={setFreqPenalty} // From hook
          seed={seed} // From hook
          setSeed={setSeed} // From hook
          randomSeed={randomSeed} // From hook
        // setRandomSeed={setRandomSeed} // SteererMoreOptionsContent does not accept this prop
        />
      }
      showOptionsButton={showOptionsButton} // Prop
    >
      {steeringControlsJsx}
    </SimpleSteererLayout>
  );
}
