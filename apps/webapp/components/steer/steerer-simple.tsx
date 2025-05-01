// this is a radically simplified version of the steerer component
// it's for the gemma-scope demo and the explorables demo
// it has a lot of duplicate code with the steer component
// it should be refactored to share code with the steer component

'use client';

import { SteerResultChat } from '@/app/api/steer-chat/route';
import CustomTooltip from '@/components/custom-tooltip';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import SteerChatMessage from '@/components/steer/chat-message';
import { LoadingSpinner } from '@/components/svg/loading-spinner';
import {
  ChatMessage,
  FeaturePreset,
  STEER_FREQUENCY_PENALTY,
  STEER_MAX_PROMPT_CHARS,
  STEER_N_COMPLETION_TOKENS_MAX,
  STEER_SEED,
  STEER_SPECIAL_TOKENS,
  STEER_STRENGTH_MULTIPLIER,
  STEER_STRENGTH_MULTIPLIER_MAX,
  STEER_TEMPERATURE,
  STEER_TEMPERATURE_MAX,
  SteerFeature,
  SteerPreset,
} from '@/lib/utils/steer';
import * as Select from '@radix-ui/react-select';
import * as Slider from '@radix-ui/react-slider';
import { ArrowUp, ChevronDown, ChevronDownIcon, ChevronUpIcon, RotateCcw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

const MAX_MESSAGE_LENGTH_CHARS = 128;

export default function SteererSimple({
  initialModelId,
  cappedHeight,
  showOptionsButton = true,
  excludedPresetNames = [],
}: {
  initialModelId: string;
  cappedHeight?: boolean;
  showOptionsButton?: boolean;
  excludedPresetNames?: string[];
}) {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [modelId, setModelId] = useState(initialModelId);
  const [featurePresets, setFeaturePresets] = useState<FeaturePreset[]>([]);

  function loadPresets() {
    fetch('/api/steer/presets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelId,
      }),
    })
      .then((response) => response.json())
      .then((data: SteerPreset) => {
        // Filter out any feature presets that have isUserVector set to true
        const filteredPresets = data.featurePresets.filter((preset) => {
          if (preset.isUserVector) {
            return false;
          }
          if (excludedPresetNames.includes(preset.name)) {
            return false;
          }
          return true;
        });
        setFeaturePresets(filteredPresets);
      })
      .catch((error) => {
        console.error(`error loading presets: ${error}`);
      });
  }

  const [defaultChatMessages, setDefaultChatMessages] = useState<ChatMessage[]>([]);
  const [steeredChatMessages, setSteeredChatMessages] = useState<ChatMessage[]>([]);
  const [typedInText, setTypedInText] = useState('');
  const [steerTokens, setSteerTokens] = useState(48);
  const [temperature, setTemperature] = useState(STEER_TEMPERATURE);
  const [freqPenalty, setFreqPenalty] = useState(STEER_FREQUENCY_PENALTY);
  const [strMultiple, setStrMultiple] = useState(STEER_STRENGTH_MULTIPLIER);
  const [seed, setSeed] = useState(STEER_SEED);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [randomSeed, setRandomSeed] = useState(false);
  const [steerSpecialTokens, setSteerSpecialTokens] = useState(STEER_SPECIAL_TOKENS);
  const [selectedFeatures, setSelectedFeatures] = useState<SteerFeature[]>([]);
  const [isTuning, setIsTuning] = useState(false);
  const { showToastServerError } = useGlobalContext();
  const [strengthLevel, setStrengthLevel] = useState(1);
  const normalEndRef = useRef<HTMLDivElement | null>(null);
  const steeredEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (featurePresets.length > 0) {
      setSelectedFeatures(featurePresets[0].features);
    }
  }, [featurePresets]);

  function reset() {
    setDefaultChatMessages([]);
    setSteeredChatMessages([]);
    setTypedInText('');
  }

  async function loadSavedSteerOutput(steerOutputId: string) {
    setIsTuning(true);
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
          alert('Sorry, your message could not be sent at this time. Please try again later.');
          console.log(response);
          return null;
        }
        return response.json();
      })
      .then((resp: SteerResultChat | null) => {
        if (resp === null) {
          setIsTuning(false);
          return;
        }
        setIsTuning(false);
        if (resp.settings) {
          setTemperature(resp.settings.temperature);
          setSteerTokens(resp.settings.n_tokens);
          setFreqPenalty(resp.settings.freq_penalty);
          setSeed(resp.settings.seed);
          setStrMultiple(resp.settings.strength_multiplier);
          setSteerSpecialTokens(resp.settings.steer_special_tokens);
        }
        // if chat template is null, we need to convert it (it's an old thing)
        if (resp.DEFAULT?.chatTemplate) {
          setDefaultChatMessages(resp.DEFAULT?.chatTemplate || []);
          setSteeredChatMessages(resp.STEERED?.chatTemplate || []);
        }
      })
      .catch((error) => {
        showToastServerError();
        setIsTuning(false);
        console.error(error);
      });
  }

  useEffect(() => {
    if (selectedFeatures.length > 0) {
      // find the featurepreset that matches the selected features
      const featPreset = featurePresets.find((p) =>
        p.features.find((f) =>
          selectedFeatures.find((sf) => sf.modelId === f.modelId && sf.layer === f.layer && sf.index === f.index),
        ),
      );
      loadSavedSteerOutput(featPreset?.exampleSteerOutputId || '');
    }
  }, [selectedFeatures]);

  function sendChat(overrideTypedInText?: string) {
    setIsTuning(true);

    const newDefaultChatMessages: ChatMessage[] = [
      ...defaultChatMessages,
      { content: overrideTypedInText || typedInText, role: 'user' },
    ];
    const newSteeredChatMessages: ChatMessage[] = [
      ...steeredChatMessages,
      { content: overrideTypedInText || typedInText, role: 'user' },
    ];
    // add to the chat messages (it will show up on UI as we load it)
    setDefaultChatMessages(newDefaultChatMessages);
    setSteeredChatMessages(newSteeredChatMessages);

    // calculate the number of characters in all the chat messages
    const defaultPromptToSendChars = newDefaultChatMessages.map((m) => m.content).join('').length;
    const steeredPromptToSendChars = newSteeredChatMessages.map((m) => m.content).join('').length;

    // check for character limit
    if (defaultPromptToSendChars >= STEER_MAX_PROMPT_CHARS || steeredPromptToSendChars >= STEER_MAX_PROMPT_CHARS) {
      alert('Sorry, we limit the length of each chat conversation.\nPlease click Reset to start a new conversation.');
      setIsTuning(false);
      return;
    }

    const selectedFeaturesStrengthOverridden = selectedFeatures.map((f) => ({
      modelId: f.modelId,
      layer: f.layer,
      index: f.index,
      explanation: f.explanation,
      strength: f.strength * strengthLevel,
    }));
    console.log(`steering with: ${JSON.stringify(selectedFeaturesStrengthOverridden)}`);

    // send the chat messages to the backend
    fetch(`/api/steer-chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        defaultChatMessages: newDefaultChatMessages,
        steeredChatMessages: newSteeredChatMessages,
        modelId,
        features: selectedFeaturesStrengthOverridden,
        temperature,
        n_tokens: steerTokens,
        freq_penalty: freqPenalty,
        seed: randomSeed ? Math.floor(Math.random() * 200000000 - 100000000) : seed,
        strength_multiplier: strMultiple,
        steer_special_tokens: steerSpecialTokens,
      }),
    })
      .then((response) => {
        if (response.status === 429 || response.status === 405) {
          alert('Sorry, we are limiting each user to 60 messages per hour. Please try again later.');
          console.log(response);
          return null;
        }
        if (response.status !== 200) {
          alert('Sorry, your message could not be sent at this time. Please try again later.');
          console.log(response);
          return null;
        }
        return response.json();
      })
      // check the response code
      .then((resp: SteerResultChat | null) => {
        if (resp === null) {
          // remove last message from chat messages UI
          setDefaultChatMessages(newDefaultChatMessages.slice(0, -1));
          setSteeredChatMessages(newSteeredChatMessages.slice(0, -1));
          setIsTuning(false);
        } else {
          setDefaultChatMessages(resp.DEFAULT?.chatTemplate || []);
          setSteeredChatMessages(resp.STEERED?.chatTemplate || []);
          setIsTuning(false);
          setTypedInText('');
        }
      })
      .catch((error) => {
        showToastServerError();
        setIsTuning(false);
        setDefaultChatMessages(newDefaultChatMessages.slice(0, -1));
        setSteeredChatMessages(newSteeredChatMessages.slice(0, -1));
        console.error(error);
      });
  }

  useEffect(() => {
    loadPresets();
    setSeed(STEER_SEED);
  }, [modelId]);

  const scrollToBottom = () => {
    if (normalEndRef.current) {
      normalEndRef.current?.scrollTo({
        top: normalEndRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
    if (steeredEndRef.current) {
      steeredEndRef.current?.scrollTo({
        top: steeredEndRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  };

  useEffect(() => {
    if (steeredChatMessages.length > 0 || defaultChatMessages.length > 0) {
      scrollToBottom();
    }
  }, [steeredChatMessages, defaultChatMessages]);

  const [showMoreOptions, setShowMoreOptions] = useState<boolean>(false);

  enum SteeringMethod {
    Activation = 'Activation',
    SAE = 'SAE',
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [steeringMethod, setSteeringMethod] = useState<SteeringMethod>(SteeringMethod.SAE);

  return (
    <div className="flex h-full w-full flex-col items-start justify-center gap-x-5 px-1.5 py-1 sm:max-h-screen sm:flex-row">
      <div className="order-2 flex h-full w-full flex-col items-center justify-center pb-0 sm:order-1">
        <div className="flex h-full w-full flex-1 flex-col">
          <div className="flex w-full flex-1 flex-col gap-x-0 px-0 pt-0 sm:flex-row sm:gap-x-2">
            <div
              ref={normalEndRef}
              className={`hidden flex-1 flex-col overflow-y-scroll rounded-xl bg-sky-100 px-5 py-2 text-left text-xs text-slate-400 shadow-md sm:flex ${
                cappedHeight ? `h-[400px] max-h-[400px] min-h-[400px]` : 'h-full max-h-[calc(100vh-111px)]'
              }`}
            >
              <div className="sticky top-1 mt-0 flex flex-row items-center justify-center uppercase text-sky-700 sm:top-0">
                <div className="rounded-full bg-sky-100 px-3 py-1 text-center text-[11px] font-bold shadow">
                  Normal Gemma
                </div>
              </div>
              <div className="pb-6 pt-3 text-[14px] font-medium leading-normal text-slate-600 sm:pb-6 sm:pt-3">
                {!isTuning && steeredChatMessages.length === 0 && (
                  <div className="w-full pt-8 text-center text-lg text-sky-500">
                    Hey, {`I'm normal Gemma!`}
                    <br />
                    Get started below.
                  </div>
                )}
                <SteerChatMessage
                  chatMessages={defaultChatMessages}
                  steered={false}
                  overrideDefaultColors="bg-sky-300 text-sky-700"
                />
                {isTuning && <LoadingSpinner className="text-sky-600" />}
              </div>
            </div>
            <div
              ref={steeredEndRef}
              className={`flex flex-1 flex-col overflow-y-scroll rounded-xl bg-green-100 px-3 py-2 text-left text-xs text-slate-400 shadow-md sm:px-5 ${
                cappedHeight ? `h-[400px] max-h-[400px] min-h-[400px]` : 'h-full max-h-[calc(100vh-111px)]'
              }`}
            >
              <div className="sticky top-1 mt-0 flex flex-row items-center justify-center uppercase text-green-600 sm:top-0">
                <div className="rounded-full bg-green-100 px-3 py-1 text-center text-[11px] font-bold shadow">
                  Steered Gemma
                </div>
              </div>
              <div className="pb-6 pt-3 text-[14px] font-medium leading-normal text-slate-600 sm:pb-6 sm:pt-3">
                {!isTuning && steeredChatMessages.length === 0 && (
                  <div className="w-full pt-8 text-center text-lg text-green-500">
                    Hey, {`I'm steered Gemma!`}
                    <br />
                    Get started below.
                  </div>
                )}
                <SteerChatMessage
                  chatMessages={steeredChatMessages}
                  steered
                  overrideDefaultColors="bg-green-300 text-green-700"
                />
                {isTuning && <LoadingSpinner className="text-green-600" />}
              </div>
            </div>
          </div>

          <div className="flex w-full flex-col items-center justify-center px-2">
            <div className="mt-3 flex w-full flex-row items-center justify-center gap-x-2">
              <div className="relative flex w-full flex-1 flex-row items-center justify-center">
                <input
                  name="searchQuery"
                  disabled={isTuning}
                  value={typedInText}
                  maxLength={MAX_MESSAGE_LENGTH_CHARS}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey && !isTuning) {
                      sendChat();
                      e.preventDefault();
                    }
                  }}
                  onChange={(e) => {
                    // if it's return, submit
                    if (e.target.value.indexOf('\n') === -1) {
                      setTypedInText(e.target.value);
                    }
                  }}
                  required
                  placeholder="Ask Gemma something..."
                  className="mt-0 w-full flex-1 resize-none rounded-full border border-slate-300 px-5 py-3.5 pr-10 text-left text-xs font-medium text-slate-800 placeholder-slate-400 shadow transition-all focus:border-slate-300 focus:shadow focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                />
                <div className="absolute right-2 flex h-full cursor-pointer items-center justify-center">
                  <ArrowUp
                    onClick={() => {
                      if (!isTuning) {
                        sendChat();
                      }
                    }}
                    className={`h-8 w-8 rounded-full ${
                      isTuning ? 'bg-slate-400' : 'bg-gBlue hover:bg-gBlue/80'
                    } p-1.5 text-white`}
                  />
                </div>
              </div>

              <button
                type="button"
                disabled={defaultChatMessages.length === 0 || isTuning}
                onClick={() => {
                  if (defaultChatMessages.length === 0) {
                    return;
                  }
                  reset();
                }}
                className="flex aspect-square h-10 cursor-pointer flex-row items-center justify-center gap-x-1.5 rounded-full bg-slate-200 px-5 text-sm text-slate-600 hover:bg-slate-300"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>
            </div>
            <div className="mb-2 mt-2.5 flex flex-row items-center justify-center gap-x-1.5 sm:mt-2.5 sm:gap-x-2">
              <button
                type="button"
                onClick={() => {
                  sendChat('Tell me about yourself.');
                }}
                className="mt-0 cursor-pointer rounded-full bg-slate-200 px-2 py-1 text-center text-[11px] font-medium leading-tight text-slate-600 transition-all hover:scale-105 hover:bg-gBlue/80 hover:text-white sm:px-3.5 sm:py-1.5 sm:text-[11px]"
              >
                Tell me about yourself.
              </button>
              <button
                type="button"
                onClick={() => {
                  sendChat('Tell me a one line story.');
                }}
                className="mt-0 cursor-pointer rounded-full bg-slate-200 px-2 py-1 text-center text-[11px] font-medium leading-tight text-slate-600 transition-all hover:scale-105 hover:bg-gBlue/80 hover:text-white sm:px-3.5 sm:py-1.5 sm:text-[11px]"
              >
                Tell me a one line story.
              </button>
              <button
                type="button"
                onClick={() => {
                  sendChat('Write a haiku.');
                }}
                className="mt-0 cursor-pointer rounded-full bg-slate-200 px-2 py-1 text-center text-[11px] font-medium leading-tight text-slate-600 transition-all hover:scale-105 hover:bg-gBlue/80 hover:text-white sm:px-3.5 sm:py-1.5 sm:text-[11px]"
              >
                Write a haiku.
              </button>
              <button
                type="button"
                onClick={() => {
                  sendChat('Complete this: I wish...');
                }}
                className="mt-0 cursor-pointer rounded-full bg-slate-200 px-2 py-1 text-center text-[11px] font-medium leading-tight text-slate-600 transition-all hover:scale-105 hover:bg-gBlue/80 hover:text-white sm:px-3.5 sm:py-1.5 sm:text-[11px]"
              >
                <span className="hidden sm:inline-block">Complete this:</span> I wish...
              </button>
            </div>
          </div>
        </div>
      </div>
      <div className="order-1 mt-3 flex max-h-full min-h-[200px] w-full flex-col justify-start overflow-y-scroll sm:order-2 sm:mt-0 sm:h-full sm:min-w-[25%] sm:max-w-[25%]">
        <div className="flex flex-col justify-start rounded-xl bg-green-100 px-2 pb-3">
          <div className="mb-2 w-full flex-row justify-center pt-1.5 uppercase text-green-600">
            <div className="px-5 py-1 text-center text-[11px] font-bold">Steering Controls</div>
          </div>
          <div className="flex flex-col gap-y-1 rounded-xl bg-green-200 px-2 py-5">
            <div className="mt-0 hidden text-center text-[10px] font-bold uppercase leading-none text-green-800 sm:block">
              Feature
            </div>
            <div className="flex w-full flex-row items-center justify-center gap-x-1.5">
              <Select.Root
                defaultValue={featurePresets.length > 0 ? featurePresets[0].name : 'Loading...'}
                value={
                  // find where selectedFeature matches a feature in the presets
                  featurePresets.find((p) =>
                    selectedFeatures.find((f) =>
                      p.features.find((fp) => fp.modelId === f.modelId && fp.layer === f.layer && fp.index === f.index),
                    ),
                  )?.name || 'Loading...'
                }
                onValueChange={(presetName) => {
                  const feat = featurePresets.find((p) => p.name === presetName);
                  setSelectedFeatures(feat?.features || []);
                }}
              >
                <Select.Trigger className="relative flex h-10 flex-1 select-none flex-row items-center justify-center gap-x-1.5 whitespace-pre rounded-xl border-2 border-green-700 bg-green-600 py-2 pl-4 pr-3 text-sm font-bold text-white hover:bg-green-700/50 focus:outline-none sm:min-w-[100px] sm:pl-6 sm:pr-4 sm:text-sm">
                  <Select.Value />
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
                          className="flex flex-col gap-y-0.5 overflow-hidden border-b px-3 py-2.5 font-sans text-xs text-slate-700 hover:bg-slate-100 focus:outline-none"
                        >
                          <Select.ItemText>
                            <span className="">Loading...</span>
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
              {/* <SignalHigh className="mt-1 h-4 w-4" /> */}
            </div>
          </div>
          {showMoreOptions && (
            <div className="mt-1 flex flex-col items-center rounded-xl bg-green-100 px-0 pb-0 pt-3.5">
              <div className="mb-1.5 hidden text-center text-[10px] font-bold uppercase leading-none text-green-800 sm:block">
                Steering Method
              </div>
              {/* <ToggleGroup.Root
                className="mx-2 inline-flex h-8 w-[90%] overflow-hidden rounded-full border-2 border-green-600 py-0"
                type="single"
                defaultValue={SteeringMethod.SAE}
                value={steeringMethod}
                onValueChange={(value) => {
                  alert('Activation not available yet.');
                  return;
                  setSteeringMethod(value as SteeringMethod);
                }}
                aria-label="steering method"
              >
                <ToggleGroup.Item
                  key={SteeringMethod.Activation}
                  className="flex-auto items-center rounded-r-full px-1 py-1 text-[10px] font-medium text-green-600 transition-all hover:bg-green-100 data-[state=on]:bg-green-600 data-[state=on]:text-white sm:px-4 sm:text-[11px]"
                  value={SteeringMethod.Activation}
                  aria-label={SteeringMethod.Activation}
                >
                  Activation
                </ToggleGroup.Item>
                <ToggleGroup.Item
                  key={SteeringMethod.SAE}
                  className="flex-auto items-center rounded-l-full px-1 py-1 text-[10px] font-medium text-green-600  transition-all hover:bg-green-100 data-[state=on]:bg-green-600 data-[state=on]:text-white sm:px-4  sm:text-[11px]"
                  value={SteeringMethod.SAE}
                  aria-label={SteeringMethod.SAE}
                >
                  SAE
                </ToggleGroup.Item>
              </ToggleGroup.Root> */}

              <div className="mb-1.5 mt-5 hidden text-center text-[10px] font-bold uppercase leading-none text-green-800 sm:block">
                Advanced
              </div>
              <div className="flex w-full flex-col gap-y-2 rounded-xl bg-green-200 px-2 py-3">
                <div className="flex w-full flex-row items-center justify-center gap-x-3">
                  <div className="w-[100px] text-center text-[12px] font-medium leading-tight text-green-800">
                    Strength Multiple
                  </div>
                  <input
                    type="number"
                    onChange={(e) => {
                      if (
                        parseFloat(e.target.value) < 0 ||
                        parseFloat(e.target.value) > STEER_STRENGTH_MULTIPLIER_MAX
                      ) {
                        alert(`Strength multiplier must be >= 0 and <= ${STEER_STRENGTH_MULTIPLIER_MAX}`);
                      } else {
                        setStrMultiple(parseFloat(e.target.value));
                      }
                    }}
                    className="max-w-[80px] flex-1 rounded-md border-green-400 py-1 text-center text-xs text-green-800"
                    value={strMultiple}
                  />
                </div>
                <div className="flex w-full flex-row items-center justify-center gap-x-1.5">
                  <div className="w-[150px] text-center text-[12px] font-medium leading-tight text-green-800">
                    Steer Special Tokens
                  </div>
                  <div className="flex w-[80px] flex-row items-center justify-start py-1">
                    <input
                      onChange={(e) => {
                        setSteerSpecialTokens(e.target.checked);
                      }}
                      type="checkbox"
                      checked={steerSpecialTokens}
                      className="h-5 w-5 cursor-pointer rounded border-green-400 bg-green-100 py-1 text-center text-xs text-green-800 checked:bg-green-600 checked:text-white"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        {showMoreOptions && (
          <div className="mt-3 flex h-full flex-col">
            <div className="flex flex-col rounded-xl bg-amber-100 px-2 pb-3 pt-3.5">
              <div className="hidden text-center text-[10px] font-bold uppercase leading-none text-amber-800 sm:block">
                Generation Settings
              </div>
              <div className="mt-1.5 flex flex-col gap-y-2 rounded-xl bg-amber-200 px-2 py-3">
                <div className="flex w-full flex-row items-center justify-center gap-x-3">
                  <div className="w-[100px] text-center text-[12px] font-medium leading-tight text-amber-800">
                    Num Tokens
                  </div>
                  <input
                    type="number"
                    onChange={(e) => {
                      if (parseInt(e.target.value, 10) > STEER_N_COMPLETION_TOKENS_MAX) {
                        alert(
                          `Due to compute constraints, the current allowed max tokens is: ${STEER_N_COMPLETION_TOKENS_MAX}`,
                        );
                      } else {
                        setSteerTokens(parseInt(e.target.value, 10));
                      }
                    }}
                    className="max-w-[80px] flex-1 rounded-md border-amber-400 py-1 text-center text-xs text-amber-800"
                    value={steerTokens}
                  />
                </div>
                <div className="flex w-full flex-row items-center justify-center gap-x-3">
                  <div className="w-[100px] text-center text-[12px] font-medium leading-tight text-amber-800">
                    Temperature
                  </div>
                  <input
                    type="number"
                    onChange={(e) => {
                      if (parseFloat(e.target.value) > STEER_TEMPERATURE_MAX || parseFloat(e.target.value) < 0) {
                        alert(`Temperature must be >= 0 and <= ${STEER_TEMPERATURE_MAX}`);
                      } else {
                        setTemperature(parseFloat(e.target.value));
                      }
                    }}
                    className="max-w-[80px] flex-1 rounded-md border-amber-400 py-1 text-center text-xs text-amber-800"
                    value={temperature}
                  />
                </div>
                <div className="flex w-full flex-row items-center justify-center gap-x-3">
                  <div className="w-[100px] text-center text-[12px] font-medium leading-tight text-amber-800">
                    Freq Penalty
                  </div>
                  <input
                    type="number"
                    onChange={(e) => {
                      if (parseFloat(e.target.value) > 2 || parseFloat(e.target.value) < -2) {
                        alert('Freq penalty must be >= -2 and <= 2');
                      } else {
                        setFreqPenalty(parseFloat(e.target.value));
                      }
                    }}
                    className="max-w-[80px] flex-1 rounded-md border-amber-400 py-1 text-center text-xs text-amber-800"
                    value={freqPenalty}
                  />
                </div>
                <div className="flex w-full flex-row items-center justify-center gap-x-3">
                  <div className="w-[100px] text-center text-[12px] font-medium leading-tight text-amber-800">Seed</div>
                  <input
                    type="number"
                    disabled={randomSeed}
                    onChange={(e) => {
                      if (parseInt(e.target.value, 10) > 100000000 || parseInt(e.target.value, 10) < -100000000) {
                        alert('Seed must be >= -100000000 and <= 100000000');
                      } else {
                        setSeed(parseInt(e.target.value, 10));
                      }
                    }}
                    className="max-w-[80px] flex-1 rounded-md border-amber-400 py-1 text-center text-xs text-amber-800 disabled:bg-amber-200 disabled:text-amber-400"
                    value={seed}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
        {showOptionsButton ? (
          <>
            <Button
              onClick={() => {
                setShowMoreOptions(!showMoreOptions);
              }}
              className="mb-0 mt-3 hidden h-11 rounded-lg border-amber-600 bg-amber-400 text-sm text-amber-700 shadow-md hover:bg-amber-500 sm:block"
            >
              {showMoreOptions ? 'Hide Options' : 'More Options'}
            </Button>
            {/* <div className="mb-1 mt-3 hidden h-full w-full flex-row items-end justify-end gap-x-0.5 text-[8px] text-slate-600 sm:flex">
              <div className="flex flex-row items-center gap-x-1 rounded bg-slate-100 px-[7px] py-[3px] text-slate-400 hover:bg-sky-200 hover:text-sky-700">
                <a
                  target="_blank"
                  href="https://neuronpedia.org/gemma-scope"
                  rel="noopener noreferrer"
                  className="select-none"
                >
                  Neuronpedia
                </a>
                <ExternalLinkIcon className="h-2 w-2" />
              </div>
            </div> */}
          </>
        ) : (
          <div className="h-full flex-1" />
        )}
      </div>
    </div>
  );
}
