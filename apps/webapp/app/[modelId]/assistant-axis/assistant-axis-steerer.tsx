'use client';

import { SteerResultChat } from '@/app/api/steer-chat/route';
import { useAssistantAxisModalContext } from '@/components/provider/assistant-axis-modal-provider';
import { useGlobalContext } from '@/components/provider/global-provider';
import { useIsMount } from '@/lib/hooks/use-is-mount';
import AssistantAxisChat from './assistant-axis-chat';
import { Button } from '@/components/shadcn/button';
import {
  ChatMessage,
  STEER_FREQUENCY_PENALTY,
  STEER_METHOD_ASSISTANT_CAP,
  STEER_N_COMPLETION_TOKENS,
  STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS,
  STEER_SEED,
  STEER_SPECIAL_TOKENS,
  STEER_STRENGTH_MULTIPLIER,
  STEER_TEMPERATURE,
  SteerFeature,
} from '@/lib/utils/steer';
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import { ExternalLinkIcon } from 'lucide-react';
import { NPSteerMethod } from 'neuronpedia-inference-client';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useCallback, useEffect, useRef, useState } from 'react';
import AssistantAxisWelcomeModal from './assistant-axis-welcome-modal';
import { ChartData } from './persona-chart';
import { buildChartData } from './persona-chart';
import { combineChartData } from './persona-chart';
import PersonaChart from './persona-chart';

const PERSONA_MODELS = ['llama3.3-70b-it'];

export const CAP_GITHUB_URL = 'https://github.com/safety-research/assistant-axis';
export const CAP_PAPER_URL = 'https://arxiv.org/abs/2601.10387';
export const CAP_BLOG_URL = '#';
export const CAP_CONTACT_EMAIL = 'jacklindsey@anthropic.com,christina.lu@cs.ox.ac.uk';
export const CAP_VECTOR_URL = '/llama3.3-70b-it/40-neuronpedia-resid-post/101874252';

export const DEMO_BUTTONS = [
  { id: 'cmkjhhsu0000fgfu5pkv3zlmv', emoji: '😢', label: 'Isolation' },
  { id: 'cmkhii9hk0015ruw6zpzwan1z', emoji: '🌀', label: 'Sycophancy' },
  { id: 'cmkhj4zb5000vmj34bcicslcg', emoji: '💸', label: 'Tax Fraud' },
  { id: null, emoji: '✏️', label: 'Free Chat' },
] as const;

import { PersonaCheckResult } from './types';

export type { PersonaCheckTurn, PersonaCheckResult } from './types';

export default function AssistantAxisSteerer({
  initialSavedId,
  hideInitialSettingsOnMobile = false,
  initialSteerFeatures,
}: {
  initialSavedId?: string;
  hideInitialSettingsOnMobile?: boolean;
  initialSteerFeatures?: SteerFeature[];
}) {
  const { showToastServerError } = useGlobalContext();
  const searchParams = useSearchParams();
  // this should never be blank
  const [modelId, setModelId] = useState(PERSONA_MODELS[0]);
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
  const [randomSeed, setRandomSeed] = useState(false);

  const [selectedFeatures, setSelectedFeatures] = useState<SteerFeature[]>(initialSteerFeatures || []);
  const [currentSavedId, setCurrentSavedId] = useState<string | null>(initialSavedId || null);
  const [isSteering, setIsSteering] = useState(false);
  const [showSettingsOnMobile, setShowSettingsOnMobile] = useState(
    initialSavedId === undefined && !hideInitialSettingsOnMobile,
  );
  const isMount = useIsMount();

  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loadingChartData, setLoadingChartData] = useState(false);
  const skipChartAnimationRef = useRef(false);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [chartWidth, setChartWidth] = useState(340);
  const [chartHeight, setChartHeight] = useState(600);
  const [scrollToTurnIndex, setScrollToTurnIndex] = useState<number | null>(null);

  // Callback for when a point on the persona chart is clicked
  const handleChartPointClick = useCallback((turn: number) => {
    setScrollToTurnIndex(turn);
  }, []);

  // Track container width and height for responsive chart
  useEffect(() => {
    const container = chartContainerRef.current;
    if (!container) return;

    const updateDimensions = (entries?: ResizeObserverEntry[]) => {
      if (entries && entries[0]) {
        const { width, height } = entries[0].contentRect;
        setChartWidth(width || container.offsetWidth);
        // Use a minimum height of 300px if container height is too small
        setChartHeight(Math.max(height || container.offsetHeight, 300));
      } else {
        setChartWidth(container.offsetWidth);
        setChartHeight(Math.max(container.offsetHeight, 300));
      }
    };

    updateDimensions();
    const resizeObserver = new ResizeObserver(updateDimensions);
    resizeObserver.observe(container);

    return () => resizeObserver.disconnect();
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

    let newUrl = `/${modelId}/assistant-axis`;
    window.history.replaceState({ ...window.history.state, as: newUrl, url: newUrl }, '', newUrl);
  }

  // Handle assistant_axis data from the steer-chat response
  const handleAssistantAxisData = useCallback(
    (steeredData: PersonaCheckResult | null, defaultData: PersonaCheckResult | null) => {
      setLoadingChartData(true);
      try {
        const steeredChartData = steeredData ? buildChartData(steeredData, 'steered') : null;
        const defaultChartData = defaultData ? buildChartData(defaultData, 'default') : null;
        const combinedData = combineChartData(steeredChartData, defaultChartData);
        setChartData(combinedData);
      } catch (error) {
        console.error(error);
      } finally {
        setLoadingChartData(false);
      }
    },
    [],
  );

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
          for (const axisItem of resp.assistant_axis) {
            const result: PersonaCheckResult = {
              pc_titles: axisItem.pc_titles,
              turns: axisItem.turns,
            };
            if (axisItem.type === 'STEERED') {
              steeredAxis = result;
            } else if (axisItem.type === 'DEFAULT') {
              defaultAxis = result;
            }
          }
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
    <div className="relative flex  h-[calc(100dvh)] sm:h-full w-full flex-col items-start justify-center overflow-hidden sm:flex-row">

      <AssistantAxisChat
        currentSavedId={currentSavedId}
        loadSavedSteerOutput={loadSavedSteerOutput}
        chartData={chartData}
        loadingChartData={loadingChartData}
        skipChartAnimationRef={skipChartAnimationRef}
        onChartPointClick={handleChartPointClick}
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
      />
    </div>
  );
}
