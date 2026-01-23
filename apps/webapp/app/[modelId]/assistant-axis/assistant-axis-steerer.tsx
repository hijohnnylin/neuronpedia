'use client';

import { SteerResultChat } from '@/app/api/steer-chat/route';
import { useGlobalContext } from '@/components/provider/global-provider';
import { useIsMount } from '@/lib/hooks/use-is-mount';
import {
  ChatMessage,
  STEER_FREQUENCY_PENALTY,
  STEER_METHOD_ASSISTANT_CAP,
  STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS,
  STEER_SEED,
  STEER_SPECIAL_TOKENS,
  STEER_STRENGTH_MULTIPLIER,
  STEER_TEMPERATURE,
  SteerFeature,
} from '@/lib/utils/steer';
import { NPSteerMethod, SteerCompletionChatPost200ResponseAssistantAxisInner } from 'neuronpedia-inference-client';
import { useSearchParams } from 'next/navigation';
import { useCallback, useEffect, useRef, useState } from 'react';
import AssistantAxisChat from './assistant-axis-chat';
import { ChartData, buildChartData, combineChartData } from './persona-chart';

type PersonaCheckResult = SteerCompletionChatPost200ResponseAssistantAxisInner;

const PERSONA_MODELS = ['llama3.3-70b-it'];

export default function AssistantAxisSteerer({
  initialSavedId,
  initialSteerFeatures,
}: {
  initialSavedId?: string;
  initialSteerFeatures?: SteerFeature[];
}) {
  const { showToastServerError } = useGlobalContext();
  const searchParams = useSearchParams();
  // this should never be blank
  const [modelId] = useState(PERSONA_MODELS[0]);
  const [typedInText, setTypedInText] = useState('');
  const [defaultChatMessages, setDefaultChatMessages] = useState<ChatMessage[]>([]);
  const [steeredChatMessages, setSteeredChatMessages] = useState<ChatMessage[]>([]);

  // Default Steering Settings
  const [steerTokens, setSteerTokens] = useState(STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS);
  const [temperature, setTemperature] = useState(STEER_TEMPERATURE);
  const [freqPenalty, setFreqPenalty] = useState(STEER_FREQUENCY_PENALTY);
  const [strMultiple, setStrMultiple] = useState(STEER_STRENGTH_MULTIPLIER);
  const [steerSpecialTokens, setSteerSpecialTokens] = useState(STEER_SPECIAL_TOKENS);
  const [seed, setSeed] = useState(STEER_SEED);
  const [steerMethod, setSteerMethod] = useState<NPSteerMethod>(STEER_METHOD_ASSISTANT_CAP);
  const [randomSeed] = useState(false);

  const [selectedFeatures, setSelectedFeatures] = useState<SteerFeature[]>(initialSteerFeatures || []);
  const [currentSavedId, setCurrentSavedId] = useState<string | null>(initialSavedId || null);
  const [isSteering, setIsSteering] = useState(false);
  const isMount = useIsMount();

  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loadingChartData, setLoadingChartData] = useState(false);
  const skipChartAnimationRef = useRef(false);
  const [usePostCap, setUsePostCap] = useState(false);
  // Store raw persona check results so we can rebuild chart data when toggling pre/post cap
  const [rawSteeredAxis, setRawSteeredAxis] = useState<PersonaCheckResult | null>(null);
  const [rawDefaultAxis, setRawDefaultAxis] = useState<PersonaCheckResult | null>(null);
  const [scrollToTurnIndex, setScrollToTurnIndex] = useState<number | null>(null);

  // Callback for when a point on the persona chart is clicked
  const handleChartPointClick = useCallback((turn: number) => {
    setScrollToTurnIndex(turn);
  }, []);

  function setUrl(steerOutputId: string | null) {
    if (steerOutputId === null) {
      let newUrl = `/${modelId}/assistant-axis`;
      newUrl += searchParams.toString() ? `?${searchParams.toString()}` : '';
      window.history.replaceState({ ...window.history.state, as: newUrl, url: newUrl }, '', newUrl);
    } else {
      // check if searchparams has saved
      let newUrl = `/${modelId}/assistant-axis`;
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
    setTypedInText('');
    setLoadingChartData(false);
    setChartData(null);
    setCurrentSavedId(null);
    setRawSteeredAxis(null);
    setRawDefaultAxis(null);

    const newUrl = `/${modelId}/assistant-axis`;
    window.history.replaceState({ ...window.history.state, as: newUrl, url: newUrl }, '', newUrl);
  }

  // Handle assistant_axis data from the steer-chat response
  const handleAssistantAxisData = useCallback(
    (steeredData: PersonaCheckResult | null, defaultData: PersonaCheckResult | null) => {
      setLoadingChartData(true);
      // Store raw data for later rebuilding when toggling pre/post cap
      setRawSteeredAxis(steeredData);
      setRawDefaultAxis(defaultData);
      try {
        const steeredChartData = steeredData ? buildChartData(steeredData, 'steered', usePostCap) : null;
        const defaultChartData = defaultData ? buildChartData(defaultData, 'default', usePostCap) : null;
        const combinedData = combineChartData(steeredChartData, defaultChartData);
        setChartData(combinedData);
      } catch (error) {
        console.error(error);
      } finally {
        setLoadingChartData(false);
      }
    },
    [usePostCap],
  );

  // Rebuild chart data when usePostCap changes
  useEffect(() => {
    if (rawSteeredAxis || rawDefaultAxis) {
      const steeredChartData = rawSteeredAxis ? buildChartData(rawSteeredAxis, 'steered', usePostCap) : null;
      const defaultChartData = rawDefaultAxis ? buildChartData(rawDefaultAxis, 'default', usePostCap) : null;
      const combinedData = combineChartData(steeredChartData, defaultChartData);
      setChartData(combinedData);
    }
  }, [usePostCap, rawSteeredAxis, rawDefaultAxis]);

  async function loadSavedSteerOutput(steerOutputId: string) {
    setIsSteering(true);
    setCurrentSavedId(steerOutputId);
    skipChartAnimationRef.current = true;
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
          setDefaultChatMessages([]);
          setSteeredChatMessages([]);
          return;
        }
        setIsSteering(false);
        setCurrentSavedId(steerOutputId);
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

        setDefaultChatMessages(resp.DEFAULT?.chatTemplate || []);
        setSteeredChatMessages(resp.STEERED?.chatTemplate || []);

        const features = resp.features?.map((f) => ({
          modelId: f.modelId,
          layer: f.layer,
          index: parseInt(f.index, 10),
          explanation: '',
          strength: f.strength,
          hasVector: f.neuron?.vector && f.neuron?.vector?.length > 0,
        }));
        setSelectedFeatures(features || []);

        // Handle cached assistant_axis data for chart
        if (resp.assistant_axis && Array.isArray(resp.assistant_axis)) {
          let steeredAxis: PersonaCheckResult | null = null;
          let defaultAxis: PersonaCheckResult | null = null;
          for (const axisItem of resp.assistant_axis as PersonaCheckResult[]) {
            if (axisItem.type === 'STEERED') {
              steeredAxis = axisItem;
            } else if (axisItem.type === 'DEFAULT') {
              defaultAxis = axisItem;
            }
          }
          // Store raw data for toggle functionality
          setRawSteeredAxis(steeredAxis);
          setRawDefaultAxis(defaultAxis);
          handleAssistantAxisData(steeredAxis, defaultAxis);
        }
      })
      .catch((error) => {
        showToastServerError();
        setIsSteering(false);
        console.error(error);
      });
  }

  useEffect(() => {
    if (isMount) {
      if (initialSavedId) {
        // load the default and steered from the steered id
        loadSavedSteerOutput(initialSavedId);
      }
    }
  }, [initialSavedId]);

  return (
    <div className="relative flex h-[calc(100dvh)] w-full flex-col items-start justify-center overflow-hidden sm:h-full sm:flex-row">
      <AssistantAxisChat
        currentSavedId={currentSavedId}
        // eslint-disable-next-line react/jsx-no-bind
        loadSavedSteerOutput={loadSavedSteerOutput}
        chartData={chartData}
        loadingChartData={loadingChartData}
        skipChartAnimationRef={skipChartAnimationRef}
        onChartPointClick={handleChartPointClick}
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
        scrollToTurnIndex={scrollToTurnIndex}
        onAssistantAxisData={handleAssistantAxisData}
        initialSavedId={initialSavedId}
        setChartData={setChartData}
        usePostCap={usePostCap}
        setUsePostCap={setUsePostCap}
        rawSteeredAxis={rawSteeredAxis}
        rawDefaultAxis={rawDefaultAxis}
      />
    </div>
  );
}
