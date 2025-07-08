/* eslint-disable no-console */
import { useState, useEffect, useRef, useCallback } from 'react';
import {
  ChatMessage,
  FeaturePreset,
  STEER_FREQUENCY_PENALTY,
  STEER_MAX_PROMPT_CHARS,
  STEER_SEED,
  STEER_SPECIAL_TOKENS,
  STEER_STRENGTH_MULTIPLIER,
  STEER_TEMPERATURE,
  SteerFeature,
  SteerPreset,
} from '@/lib/utils/steer';
// import { SteerResultChat } from '@/app/api/steer-chat/route'; // Will be replaced by SteerChatApiResponse
import { callSteerChatApi, SteerChatApiResponse } from '@/lib/utils/steer-api'; // Import the new utility
import { useGlobalContext } from '@/components/provider/global-provider';

// Define a stable empty array reference
const EMPTY_ARRAY: readonly string[] = Object.freeze([]);

interface UseSimpleSteererLogicProps<TStrengthConfig> {
  initialModelId: string;
  presetsApiEndpoint: string;
  excludedPresetNames?: readonly string[];
  initialStrengthConfig: TStrengthConfig;
  transformFeaturesForApi: (features: SteerFeature[], strengthConfig: TStrengthConfig) => SteerFeature[];
  onChatSuccess?: (response: SteerChatApiResponse | null) => void; // Updated to use SteerChatApiResponse
  onPresetFeaturesSelected?: (
    features: SteerFeature[],
    preset?: FeaturePreset,
  ) => { defaultChatMessages?: ChatMessage[]; steeredChatMessages?: ChatMessage[] } | void | Promise<{ defaultChatMessages?: ChatMessage[]; steeredChatMessages?: ChatMessage[] } | void>;
  initialShowMoreOptions?: boolean;
  starterPrompts?: Array<{ label: string; text: string; fullLabel?: string }>; // Added starterPrompts
}

export function useSimpleSteererLogic<TStrengthConfig>({
  initialModelId, // Assuming this is used to initialize modelId state
  presetsApiEndpoint,
  excludedPresetNames = EMPTY_ARRAY,
  initialStrengthConfig,
  transformFeaturesForApi,
  onChatSuccess,
  onPresetFeaturesSelected,
  initialShowMoreOptions = false,
  starterPrompts, // Added starterPrompts
}: UseSimpleSteererLogicProps<TStrengthConfig>) {
  const [modelId, setModelId] = useState(initialModelId); // Assuming initialModelId prop is used here
  const [featurePresets, setFeaturePresets] = useState<FeaturePreset[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<SteerFeature[]>([]);
  const [defaultChatMessages, setDefaultChatMessages] = useState<ChatMessage[]>([]);
  const [steeredChatMessages, setSteeredChatMessages] = useState<ChatMessage[]>([]);
  const [typedInText, setTypedInText] = useState('');
  const [isTuning, setIsTuning] = useState(false);
  const [steerTokens, setSteerTokens] = useState(48);
  const [temperature, setTemperature] = useState(STEER_TEMPERATURE);
  const [freqPenalty, setFreqPenalty] = useState(STEER_FREQUENCY_PENALTY);
  const [strMultiple, setStrMultiple] = useState(STEER_STRENGTH_MULTIPLIER);
  const [seed, setSeed] = useState(STEER_SEED);
  const [randomSeed, setRandomSeed] = useState(false); // SteererMoreOptionsContent will manage its own checkbox state for this
  const [steerSpecialTokens, setSteerSpecialTokens] = useState(STEER_SPECIAL_TOKENS);
  const [showMoreOptions, setShowMoreOptions] = useState(initialShowMoreOptions);
  const [isLoadingPresets, setIsLoadingPresets] = useState(false);
  const [hasInitialPresetBeenApplied, setHasInitialPresetBeenApplied] = useState(false);

  const normalEndRef = useRef<HTMLDivElement | null>(null);
  const steeredEndRef = useRef<HTMLDivElement | null>(null);

  const { showToastServerError } = useGlobalContext();

  const loadPresets = useCallback(async () => {
    console.log('[useSimpleSteererLogic] loadPresets function instance created/called. modelId:', modelId, 'presetsApiEndpoint:', presetsApiEndpoint, 'excludedPresetNames:', excludedPresetNames);
    setIsLoadingPresets(true);
    try {
      console.log(`[useSimpleSteererLogic] Preparing to fetch presets. modelId is "${modelId}", presetsApiEndpoint is "${presetsApiEndpoint}"`);
      if (!modelId) {
        console.error("[useSimpleSteererLogic] loadPresets: modelId is undefined or empty. Aborting fetch.");
        setFeaturePresets([]); // Clear presets on error or invalid modelId
        setIsLoadingPresets(false); // Ensure loading state is reset
        return; // Prevent fetch with undefined modelId
      }
      // Presets should always be fetched from the local endpoint
      const fetchUrl = presetsApiEndpoint;
      const headers: HeadersInit = { 'Content-Type': 'application/json' };

      const response = await fetch(fetchUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify({ modelId }),
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch presets: ${response.statusText}`);
      }
      const data: SteerPreset = await response.json();
      const filteredPresets = data.featurePresets.filter(
        (preset) => !preset.isUserVector && !excludedPresetNames.includes(preset.name),
      );
      setFeaturePresets(filteredPresets);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error(`Error loading presets: ${error}`);
      showToastServerError();
      setFeaturePresets([]); // Clear presets on error
    } finally {
      setIsLoadingPresets(false);
    }
  }, [modelId, presetsApiEndpoint, excludedPresetNames, showToastServerError]);

  const resetChatAndMessages = useCallback(() => {
    setDefaultChatMessages([]);
    setSteeredChatMessages([]);
    setTypedInText('');
    // Note: Component-specific reset logic (like setShowNormalResponse) should be handled in the component
  }, []);

  useEffect(() => {
    console.log('[useSimpleSteererLogic] useEffect to load presets triggered. modelId:', modelId);
    // Reset states when modelId changes to allow re-initialization for the new model
    resetChatAndMessages();
    setSelectedFeatures([]);
    setHasInitialPresetBeenApplied(false); // Allow initial preset to be applied for the new model
    loadPresets();
    setSeed(STEER_SEED); // Reset seed when model changes, similar to original
  }, [modelId, loadPresets, resetChatAndMessages]); // resetChatAndMessages is stable

  useEffect(() => {
    // Auto-select the first preset's features if none are selected yet
    // and then call the onPresetFeaturesSelected callback.
    // Also, auto-fill the first prompt.
    // Only run if initial auto-selection hasn't occurred AND conditions are met
    if (!hasInitialPresetBeenApplied && featurePresets.length > 0 && selectedFeatures.length === 0 && !isLoadingPresets) {
      const firstPreset = featurePresets[0];
      setSelectedFeatures(firstPreset.features);

      // Auto-fill chat input for the first preset
      if (starterPrompts && starterPrompts.length > 0) {
        setTypedInText(starterPrompts[0].text);
      } else {
        setTypedInText(''); // Clear if no corresponding prompt
      }

      if (onPresetFeaturesSelected) {
        Promise.resolve(onPresetFeaturesSelected(firstPreset.features, firstPreset)).then((messages) => {
          if (messages) {
            setDefaultChatMessages(messages.defaultChatMessages || []);
            setSteeredChatMessages(messages.steeredChatMessages || []);
          }
        });
      }
      setHasInitialPresetBeenApplied(true); // Mark that initial auto-selection has now happened
    }
  }, [
    hasInitialPresetBeenApplied, // Add as a dependency
    featurePresets,
    selectedFeatures, // Still need to check its length
    isLoadingPresets,
    onPresetFeaturesSelected,
    starterPrompts,
    // Stable setters like setTypedInText, setSelectedFeatures, etc., are often omitted
    // from deps if not strictly necessary by lint rules, assuming they are stable.
    // Adding them explicitly for clarity if preferred by project style:
    // setTypedInText, setSelectedFeatures, setDefaultChatMessages, setSteeredChatMessages
  ]);

  const sendChat = useCallback(
    async (overrideTypedInText?: string) => {
      setIsTuning(true);
      const originalInput = overrideTypedInText || typedInText;
      const currentText = `${originalInput} Tell me concisely.`;

      if (!currentText.trim()) {
        // eslint-disable-next-line no-alert
        alert('Please enter a message.');
        setIsTuning(false);
        return;
      }

      const newDefaultChatMessages: ChatMessage[] = [...defaultChatMessages, { content: currentText, role: 'user' }];
      const newSteeredChatMessages: ChatMessage[] = [...steeredChatMessages, { content: currentText, role: 'user' }];

      setDefaultChatMessages(newDefaultChatMessages);
      setSteeredChatMessages(newSteeredChatMessages);

      const defaultPromptToSendChars = newDefaultChatMessages.map((m) => m.content).join('').length;
      const steeredPromptToSendChars = newSteeredChatMessages.map((m) => m.content).join('').length;

      if (defaultPromptToSendChars >= STEER_MAX_PROMPT_CHARS || steeredPromptToSendChars >= STEER_MAX_PROMPT_CHARS) {
        // eslint-disable-next-line no-alert
        alert('Sorry, we limit the length of each chat conversation.\nPlease click Reset to start a new conversation.');
        setDefaultChatMessages(newDefaultChatMessages.slice(0, -1));
        setSteeredChatMessages(newSteeredChatMessages.slice(0, -1));
        setIsTuning(false);
        return;
      }

      const featuresForApi = transformFeaturesForApi(selectedFeatures, initialStrengthConfig);
      // eslint-disable-next-line no-console
      console.log(`Steering with: ${JSON.stringify(featuresForApi)}`);

      try {
        // The apiKey and apiBaseUrl are now handled by callSteerChatApi
        // The NEXT_PUBLIC_NEURONPEDIA_APIKEY environment variable will be used by callSteerChatApi

        const resp = await callSteerChatApi({
          modelId,
          features: featuresForApi,
          defaultChatMessages: newDefaultChatMessages,
          steeredChatMessages: newSteeredChatMessages,
          temperature,
          nTokens: steerTokens,
          freqPenalty,
          seed: randomSeed ? Math.floor(Math.random() * 200000000 - 100000000) : seed,
          strengthMultiplier: strMultiple,
          steerSpecialTokens,
        });

        // Error handling (including 429, 405, and other !response.ok cases)
        // is now managed within callSteerChatApi.
        // If callSteerChatApi throws an error, it will be caught by the catch block below.

        // const resp: SteerChatApiResponse | null = await response.json(); // This line is replaced by the callSteerChatApi call
        if (resp === null) { // Should not happen if response.ok
          throw new Error('Empty response from API');
        }
        setDefaultChatMessages(resp.DEFAULT?.chatTemplate || []);
        setSteeredChatMessages(resp.STEERED?.chatTemplate || []);
        setTypedInText('');
        if (onChatSuccess) {
          onChatSuccess(resp);
        }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error(error);
        showToastServerError();
        setDefaultChatMessages(newDefaultChatMessages.slice(0, -1));
        setSteeredChatMessages(newSteeredChatMessages.slice(0, -1));
      } finally {
        setIsTuning(false);
      }
    },
    [
      typedInText,
      defaultChatMessages,
      steeredChatMessages,
      modelId,
      selectedFeatures,
      initialStrengthConfig, // Hook prop, if it changes, sendChat should update
      transformFeaturesForApi, // Hook prop
      temperature,
      steerTokens,
      freqPenalty,
      randomSeed,
      seed,
      strMultiple,
      steerSpecialTokens,
      onChatSuccess, // Hook prop
      showToastServerError,
    ],
  );

  const handlePresetChange = useCallback(
    async (presetName: string) => {
      const presetIndex = featurePresets.findIndex((p) => p.name === presetName);
      const preset = presetIndex !== -1 ? featurePresets[presetIndex] : undefined;

      if (preset) {
        setSelectedFeatures(preset.features);
        setHasInitialPresetBeenApplied(true); // A user has made a selection, initial auto-select is no longer primary concern

        // Auto-fill chat input based on the index of the selected preset
        if (starterPrompts && starterPrompts.length > presetIndex && presetIndex !== -1) {
          setTypedInText(starterPrompts[presetIndex].text);
        } else {
          // Optionally clear or set a default if no corresponding prompt
          setTypedInText('');
        }

        if (onPresetFeaturesSelected) {
          // Reset messages before loading new ones from preset, or let callback decide
          setDefaultChatMessages([]);
          setSteeredChatMessages([]);
          const messages = await Promise.resolve(onPresetFeaturesSelected(preset.features, preset));
          if (messages) {
            setDefaultChatMessages(messages.defaultChatMessages || []);
            setSteeredChatMessages(messages.steeredChatMessages || []);
          }
        }
      }
    },
    [
      featurePresets,
      onPresetFeaturesSelected,
      starterPrompts,
      setTypedInText,
      setSelectedFeatures,
      setDefaultChatMessages,
      setSteeredChatMessages,
      setHasInitialPresetBeenApplied // Add the new setter to the dependency array
    ],
  );

  const toggleMoreOptions = useCallback(() => {
    setShowMoreOptions((prev) => !prev);
  }, []);

  const scrollToBottom = useCallback(() => {
    normalEndRef.current?.scrollTo({
      top: normalEndRef.current.scrollHeight,
      behavior: 'smooth',
    });
    steeredEndRef.current?.scrollTo({
      top: steeredEndRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, []);

  useEffect(() => {
    // Scroll logic depends on whether the normal panel is shown,
    // which is managed by the component using the hook.
    // For now, always try to scroll if messages exist.
    // Components can refine this by only calling scrollToBottom when appropriate.
    if (steeredChatMessages.length > 0 || defaultChatMessages.length > 0) {
      scrollToBottom();
    }
  }, [steeredChatMessages, defaultChatMessages, scrollToBottom]);

  return {
    modelId,
    setModelId,
    featurePresets,
    selectedFeatures,
    setSelectedFeatures,
    defaultChatMessages,
    setDefaultChatMessages,
    steeredChatMessages,
    setSteeredChatMessages,
    typedInText,
    setTypedInText,
    isTuning,
    setIsTuning, // Exposed setIsTuning
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
    setRandomSeed, // Allow component to control this if needed, though MoreOptionsContent might handle its own
    steerSpecialTokens,
    setSteerSpecialTokens,
    showMoreOptions,
    normalEndRef,
    steeredEndRef,
    isLoadingPresets,
    loadPresets, // Exposing for potential manual refresh, though it runs on modelId change
    sendChat,
    resetChatAndMessages,
    handlePresetChange,
    toggleMoreOptions,
    // scrollToBottom, // Exposing this if components need more fine-grained control
  };
}