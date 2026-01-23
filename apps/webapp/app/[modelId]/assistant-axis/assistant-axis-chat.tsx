import { SteerResultChat } from '@/app/api/steer-chat/route';
import CustomTooltip from '@/components/custom-tooltip';
import { useAssistantAxisModalContext } from '@/components/provider/assistant-axis-modal-provider';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import SteerChatMessage from '@/components/steer/chat-message';
import { LoadingSquare } from '@/components/svg/loading-square';
import { ASSET_BASE_URL, IS_ACTUALLY_NEURONPEDIA_ORG } from '@/lib/env';
import { ChatMessage, ERROR_STEER_MAX_PROMPT_CHARS, SteerFeature } from '@/lib/utils/steer';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { ArrowRightIcon, InfoCircledIcon, QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import copy from 'copy-to-clipboard';
import { EventSourceParserStream } from 'eventsource-parser/stream';
import {
  ArrowUp,
  BookOpenIcon,
  Check,
  Copy,
  Download,
  GithubIcon,
  Mail,
  Scroll,
  Share,
  Trash2,
  Undo2,
  X,
} from 'lucide-react';
import { NPSteerMethod, SteerCompletionChatPost200ResponseAssistantAxisInner } from 'neuronpedia-inference-client';
import Link from 'next/link';
import { MutableRefObject, useEffect, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import AssistantAxisWelcomeModal from './assistant-axis-welcome-modal';
import PersonaChart, { ChartData } from './persona-chart';
import { CAP_BLOG_URL, CAP_CONTACT_EMAIL, CAP_GITHUB_URL, CAP_PAPER_URL, CAP_VECTOR_URL, DEMO_BUTTONS } from './shared';

type PersonaCheckResult = SteerCompletionChatPost200ResponseAssistantAxisInner;

export default function AssistantAxisChat({
  isSteering,
  setIsSteering,
  defaultChatMessages,
  setDefaultChatMessages,
  steeredChatMessages,
  setSteeredChatMessages,
  modelId,
  selectedFeatures,
  reset,
  typedInText,
  setTypedInText,
  setUrl,
  temperature,
  steerTokens,
  freqPenalty,
  randomSeed,
  seed,
  strMultiple,
  steerSpecialTokens,
  steerMethod,
  scrollToTurnIndex,
  onAssistantAxisData,
  currentSavedId,
  loadSavedSteerOutput,
  chartData,
  loadingChartData,
  skipChartAnimationRef,
  onChartPointClick,
  initialSavedId,
  setChartData,
  usePostCap,
  setUsePostCap,
  rawSteeredAxis,
  rawDefaultAxis,
}: {
  isSteering: boolean;
  setIsSteering: (isSteering: boolean) => void;
  defaultChatMessages: ChatMessage[];
  setDefaultChatMessages: (chatMessages: ChatMessage[]) => void;
  steeredChatMessages: ChatMessage[];
  setSteeredChatMessages: (chatMessages: ChatMessage[]) => void;
  modelId: string;
  selectedFeatures: SteerFeature[];
  reset: () => void;
  typedInText: string;
  setTypedInText: (text: string) => void;
  setUrl: (url: string) => void;
  temperature: number;
  steerTokens: number;
  freqPenalty: number;
  randomSeed: boolean;
  seed: number;
  strMultiple: number;
  steerSpecialTokens: boolean;
  steerMethod: NPSteerMethod;
  scrollToTurnIndex?: number | null;
  onAssistantAxisData?: (steeredData: PersonaCheckResult | null, defaultData: PersonaCheckResult | null) => void;
  currentSavedId: string | null;
  loadSavedSteerOutput: (steerOutputId: string) => void;
  chartData: ChartData | null;
  loadingChartData: boolean;
  skipChartAnimationRef: MutableRefObject<boolean>;
  onChartPointClick: (turn: number) => void;
  initialSavedId: string | undefined;
  setChartData: (chartData: ChartData | null) => void;
  usePostCap: boolean;
  setUsePostCap: (usePostCap: boolean) => void;
  rawSteeredAxis: PersonaCheckResult | null;
  rawDefaultAxis: PersonaCheckResult | null;
}) {
  const normalEndRef = useRef<HTMLDivElement | null>(null);
  const steeredEndRef = useRef<HTMLDivElement | null>(null);
  const defaultScrollContainerRef = useRef<HTMLDivElement | null>(null);
  const steeredScrollContainerRef = useRef<HTMLDivElement | null>(null);
  const defaultMessageRefs = useRef<(HTMLDivElement | null)[]>([]);
  const steeredMessageRefs = useRef<(HTMLDivElement | null)[]>([]);
  const { showToastMessage, showToastServerError } = useGlobalContext();
  const abortControllerRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<{
    event?: string;
    data: string;
    id?: string;
    retry?: number;
  }> | null>(null);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [chartWidth, setChartWidth] = useState(200);
  const [chartHeight, setChartHeight] = useState(400);
  const [limitRemaining, setLimitRemaining] = useState<number | null>(null);
  const [isShareModalOpen, setIsShareModalOpen] = useState(false);
  const [urlCopied, setUrlCopied] = useState(false);

  const { setIsWelcomeModalOpen } = useAssistantAxisModalContext();

  function removeLastFailedUserMessage(defaultMsgs: ChatMessage[], steeredMsgs: ChatMessage[]) {
    if (defaultMsgs.length > 0 && defaultMsgs[defaultMsgs.length - 1].role === 'user') {
      setDefaultChatMessages(defaultMsgs.slice(0, -1));
    }
    if (steeredMsgs.length > 0 && steeredMsgs[steeredMsgs.length - 1].role === 'user') {
      setSteeredChatMessages(steeredMsgs.slice(0, -1));
    }
  }

  // Track chart container dimensions
  useEffect(() => {
    const container = chartContainerRef.current;
    if (!container) return;

    const updateDimensions = (entries?: ResizeObserverEntry[]) => {
      const isSmallScreen = window.innerWidth < 640; // sm breakpoint is 640px
      const heightAdjustment = isSmallScreen ? 70 : 0;

      if (entries && entries[0]) {
        const { width, height } = entries[0].contentRect;
        setChartWidth(width || container.offsetWidth);
        setChartHeight(Math.max((height || container.offsetHeight) - heightAdjustment, 300));
      } else {
        setChartWidth(container.offsetWidth);
        setChartHeight(Math.max(container.offsetHeight - heightAdjustment, 300));
      }
    };

    updateDimensions();
    const resizeObserver = new ResizeObserver(updateDimensions);
    resizeObserver.observe(container);

    // eslint-disable-next-line consistent-return
    return () => resizeObserver.disconnect();
  }, []);

  // Scroll to specific turn when scrollToTurnIndex changes
  useEffect(() => {
    if (scrollToTurnIndex === null || scrollToTurnIndex === undefined) return;

    // Turn 0 means scroll to the top of both conversations
    if (scrollToTurnIndex === 0) {
      if (defaultScrollContainerRef.current) {
        defaultScrollContainerRef.current.scrollTo({ top: 0, behavior: 'smooth' });
      }
      if (steeredScrollContainerRef.current) {
        steeredScrollContainerRef.current.scrollTo({ top: 0, behavior: 'smooth' });
      }
      return;
    }

    // turn is the assistant message turn (1-indexed from chart)
    // multiply by 2 and subtract 1 to get the correct message index
    const messageIndex = scrollToTurnIndex * 2 - 1;

    const defaultEl = defaultMessageRefs.current[messageIndex];
    const steeredEl = steeredMessageRefs.current[messageIndex];

    // Scroll within container only (don't scroll parent elements)
    if (defaultEl && defaultScrollContainerRef.current) {
      const container = defaultScrollContainerRef.current;
      const elementTop = defaultEl.offsetTop - container.offsetTop;
      container.scrollTo({ top: Math.max(0, elementTop - 50), behavior: 'smooth' });
    }
    if (steeredEl && steeredScrollContainerRef.current) {
      const container = steeredScrollContainerRef.current;
      const elementTop = steeredEl.offsetTop - container.offsetTop;
      container.scrollTo({ top: Math.max(0, elementTop - 50), behavior: 'smooth' });
    }
  }, [scrollToTurnIndex]);

  const scrollToNewestChatMessage = () => {
    normalEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    });
    if (steeredEndRef.current && steeredEndRef.current?.scrollHeight > 400) {
      steeredEndRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'end',
      });
    }
  };

  useEffect(() => {
    if (steeredChatMessages.length > 0 || defaultChatMessages.length > 0) {
      scrollToNewestChatMessage();
    }
  }, [steeredChatMessages, defaultChatMessages]);

  async function stopSteering() {
    if (readerRef.current) {
      try {
        await readerRef.current.cancel();
      } catch {
        // Ignore errors from canceling the reader
      }
      readerRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsSteering(false);
  }

  async function sendChat() {
    if (typedInText.trim().length === 0) {
      alert('Please enter a message.');
      return;
    }
    setIsSteering(true);

    // If the last message is a user message, replace it; otherwise, add a new one
    const newDefaultChatMessages: ChatMessage[] =
      defaultChatMessages.length > 0 && defaultChatMessages[defaultChatMessages.length - 1].role === 'user'
        ? [...defaultChatMessages.slice(0, -1), { content: typedInText, role: 'user' }]
        : [...defaultChatMessages, { content: typedInText, role: 'user' }];

    const newSteeredChatMessages: ChatMessage[] =
      steeredChatMessages.length > 0 && steeredChatMessages[steeredChatMessages.length - 1].role === 'user'
        ? [...steeredChatMessages.slice(0, -1), { content: typedInText, role: 'user' }]
        : [...steeredChatMessages, { content: typedInText, role: 'user' }];

    // add to the chat messages (it will show up on UI as we load it)
    setDefaultChatMessages(newDefaultChatMessages);
    setSteeredChatMessages(newSteeredChatMessages);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    const { signal } = abortControllerRef.current;

    // send the chat messages to the backend
    try {
      const stream = true;
      const response = await fetch(`/api/steer-chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          defaultChatMessages: newDefaultChatMessages,
          steeredChatMessages: newSteeredChatMessages,
          modelId,
          features: selectedFeatures,
          temperature,
          n_tokens: steerTokens,
          freq_penalty: freqPenalty,
          seed: randomSeed ? Math.floor(Math.random() * 200000000 - 100000000) : seed,
          strength_multiplier: strMultiple,
          steer_method: steerMethod,
          steer_special_tokens: steerSpecialTokens,
          stream,
          isAssistantAxis: true,
        }),
        signal,
      });
      if (!response || !response.body) {
        alert('Sorry, your message could not be sent at this time. Please try again later.');

        showToastServerError();
        setIsSteering(false);
        removeLastFailedUserMessage(newDefaultChatMessages, newSteeredChatMessages);
        return;
      }
      if (response.status === 429) {
        alert('Sorry, you have reached the maximum number of messages per hour. Please try again later.');
        setIsSteering(false);
        removeLastFailedUserMessage(newDefaultChatMessages, newSteeredChatMessages);
        return;
      }
      if (response.status === 400) {
        const errorBody = await response.json();
        if (errorBody.message === ERROR_STEER_MAX_PROMPT_CHARS) {
          alert(
            'The conversation is too long. Please reset the chat using the trash icon and start a new conversation.',
          );
          setIsSteering(false);
          removeLastFailedUserMessage(newDefaultChatMessages, newSteeredChatMessages);
          return;
        }
      }
      if (response.status !== 200) {
        if (response.status === 404) {
          alert(
            !IS_ACTUALLY_NEURONPEDIA_ORG
              ? 'Unable to steer with the selected feature. Did you check if you downloaded/imported this SAE?'
              : 'Unable to steer with the selected feature - it was not found.',
          );
        } else {
          setIsSteering(false);
          removeLastFailedUserMessage(newDefaultChatMessages, newSteeredChatMessages);
          const errorBody = await response.text();
          console.error(`Error ${response.status}: ${response.statusText}\n\n${errorBody}`);
          showToastServerError();
        }
      }
      // eslint-disable-next-line @typescript-eslint/no-shadow
      const limitRemaining = response.headers.get('x-limit-remaining');
      if (limitRemaining) {
        setLimitRemaining(Number(limitRemaining));
      }

      // check if the response is a stream
      const contentType = response.headers.get('content-type');
      if (contentType === 'text/event-stream') {
        const reader = response.body
          .pipeThrough(new TextDecoderStream())
          .pipeThrough(new EventSourceParserStream())
          .getReader();
        readerRef.current = reader;

        // Track the latest assistant_axis data from streaming chunks (one for each steer type)
        let latestSteeredAxis: PersonaCheckResult | null = null;
        let latestDefaultAxis: PersonaCheckResult | null = null;

        while (true) {
          // eslint-disable-next-line
          const { done, value } = await reader.read();
          if (done) {
            readerRef.current = null;
            setIsSteering(false);
            // Pass the assistant_axis data to the parent after streaming completes
            if (onAssistantAxisData && (latestSteeredAxis || latestDefaultAxis)) {
              onAssistantAxisData(latestSteeredAxis, latestDefaultAxis);
            }
            break;
          }
          const data = JSON.parse(value.data) as SteerResultChat;
          if (data.DEFAULT?.chatTemplate) {
            setDefaultChatMessages(data.DEFAULT?.chatTemplate || []);
          }
          if (data.STEERED?.chatTemplate) {
            setSteeredChatMessages(data.STEERED?.chatTemplate || []);
          }
          if (data.id) {
            setUrl(data.id);
          }
          // Track assistant_axis data from the response (now an array with type field)
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const assistantAxisArray = (data as any).assistant_axis as PersonaCheckResult[] | undefined;
          if (assistantAxisArray && Array.isArray(assistantAxisArray)) {
            console.log('[DEBUG] Received assistant_axis (streaming):', JSON.stringify(assistantAxisArray, null, 2));
            for (const axisItem of assistantAxisArray) {
              if (axisItem.type === 'STEERED') {
                latestSteeredAxis = axisItem;
              } else if (axisItem.type === 'DEFAULT') {
                latestDefaultAxis = axisItem;
              }
            }
          }
          setTypedInText('');
        }
      } else {
        const data = await response.json();
        setDefaultChatMessages(data.DEFAULT?.chatTemplate || []);
        setSteeredChatMessages(data.STEERED?.chatTemplate || []);
        setUrl(data.id);
        setTypedInText('');
        setIsSteering(false);
        // Pass the assistant_axis data to the parent for non-streaming response
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const assistantAxisArray = data.assistant_axis as PersonaCheckResult[] | undefined;
        console.log('[DEBUG] Received assistant_axis (non-streaming):', JSON.stringify(assistantAxisArray, null, 2));
        if (onAssistantAxisData && assistantAxisArray && Array.isArray(assistantAxisArray)) {
          let steeredAxis: PersonaCheckResult | null = null;
          let defaultAxis: PersonaCheckResult | null = null;
          for (const axisItem of assistantAxisArray) {
            // Debug: check if pcValuesPostCap exists in turns
            const hasPostCap = axisItem.turns?.some((t) => t.pcValuesPostCap);
            console.log(`[DEBUG] ${axisItem.type} axis - has pcValuesPostCap:`, hasPostCap);
            if (axisItem.turns && axisItem.turns.length > 0) {
              console.log(`[DEBUG] ${axisItem.type} first turn keys:`, Object.keys(axisItem.turns[0]));
              console.log(`[DEBUG] ${axisItem.type} first turn pcValuesPostCap:`, axisItem.turns[0].pcValuesPostCap);
            }
            if (axisItem.type === 'STEERED') {
              steeredAxis = axisItem;
            } else if (axisItem.type === 'DEFAULT') {
              defaultAxis = axisItem;
            }
          }
          onAssistantAxisData(steeredAxis, defaultAxis);
        }
      }
    } catch (error) {
      readerRef.current = null;
      if (error instanceof DOMException && error.name === 'AbortError') {
        showToastMessage('Steering aborted.');
        setIsSteering(false);
        removeLastFailedUserMessage(newDefaultChatMessages, newSteeredChatMessages);
      } else {
        console.error(error);
        setIsSteering(false);
        removeLastFailedUserMessage(newDefaultChatMessages, newSteeredChatMessages);
        showToastServerError();
      }
    }
  }

  return (
    <div className="relative flex h-[calc(100dvh)] w-full min-w-0 flex-col items-center text-sm font-medium leading-normal text-slate-500 sm:h-full sm:max-h-[calc(100dvh-180px)] sm:min-h-[calc(100dvh-180px)]">
      <AssistantAxisWelcomeModal onLoadDemo={loadSavedSteerOutput} onFreeChat={reset} initialSavedId={initialSavedId} />

      {/* Share Modal */}
      <Dialog
        open={isShareModalOpen}
        onOpenChange={(open) => {
          setIsShareModalOpen(open);
          if (!open) setUrlCopied(false);
        }}
      >
        <DialogContent className="bg-white sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Share Conversation</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <input
                type="text"
                readOnly
                value={typeof window !== 'undefined' ? window.location.href : ''}
                className="flex-1 rounded-md border border-slate-300 bg-slate-100 px-3 py-2 text-sm text-slate-600 outline-none"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 px-3"
                onClick={() => {
                  copy(window.location.href);
                  setUrlCopied(true);
                  setTimeout(() => setUrlCopied(false), 2000);
                }}
              >
                {urlCopied ? <Check className="h-4 w-4 text-green-600" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
            <Button
              type="button"
              variant="default"
              className="w-full"
              onClick={() => {
                const conversationData = {
                  modelId,
                  timestamp: new Date().toISOString(),
                  defaultConversation: defaultChatMessages,
                  steeredConversation: steeredChatMessages,
                  settings: {
                    temperature,
                    steerTokens,
                    freqPenalty,
                    strMultiple,
                    steerMethod,
                  },
                  chartData,
                  // Include raw axis data with pcValues and pcValuesPostCap
                  assistantAxis: {
                    steered: rawSteeredAxis,
                    default: rawDefaultAxis,
                  },
                };
                const blob = new Blob([JSON.stringify(conversationData, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `assistant-axis-conversation-${new Date().toISOString().slice(0, 10)}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }}
            >
              <Download className="mr-2 h-4 w-4" />
              Download Conversation (JSON)
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Demo buttons */}
      <div className="relative z-10 mb-2 flex h-[100px] max-h-[100px] min-h-[100px] w-full flex-row items-center justify-center gap-2 bg-slate-50 px-3 py-2 sm:mb-5 sm:h-[80px] sm:max-h-[80px] sm:min-h-[80px] sm:px-6">
        <div className="flex w-full max-w-screen-2xl flex-1 flex-row items-center justify-between gap-y-1">
          <div className="hidden flex-1 flex-row items-center justify-between gap-y-1 px-0.5 sm:flex">
            <div className="flex flex-col items-start justify-center gap-y-1">
              <div className="whitespace-pre text-[18px] font-bold leading-none tracking-tight text-slate-700 sm:text-xl sm:font-semibold">
                Assistant Axis
              </div>
              <div className="hidden text-[10px] text-slate-400 sm:block sm:text-xs">
                Lu et al.<span className="hidden sm:inline">, January 2026</span>
              </div>
            </div>
          </div>
          <div className="flex flex-1 flex-col items-center justify-center gap-y-0.5 rounded-md p-0 px-0.5 sm:p-2">
            <div className="flex w-full flex-row items-center justify-center px-0 sm:hidden sm:px-3">
              <div className="flex flex-1 flex-col items-start justify-center gap-y-0.5">
                <div className="px-0.5 text-[17px] font-bold text-slate-700 sm:hidden">Assistant Axis</div>
                <div className="flex px-0.5 text-[10px] font-medium uppercase text-slate-400">
                  <span>Lu et al., January 2026</span>
                </div>
              </div>
              <div className="flex flex-row items-center justify-center gap-x-1 sm:gap-x-1">
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    setIsWelcomeModalOpen(true);
                  }}
                  className="relative flex h-8 min-h-8 flex-row items-center justify-center gap-x-1.5 rounded border border-emerald-600 bg-emerald-50 px-2.5 py-1.5 font-sans text-[11px] font-semibold leading-none text-emerald-600 hover:bg-emerald-100 sm:h-7 sm:min-h-7 sm:gap-x-1 sm:text-[11px] sm:uppercase"
                >
                  Guide
                </button>
                <DropdownMenu.Root>
                  <DropdownMenu.Trigger asChild>
                    <button
                      type="button"
                      className="flex h-8 min-h-8 flex-row items-center justify-center gap-x-1.5 rounded border border-slate-200 bg-white px-2 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200 sm:h-7 sm:min-h-7"
                    >
                      <InfoCircledIcon className="h-3.5 w-3.5" />
                    </button>
                  </DropdownMenu.Trigger>
                  <DropdownMenu.Content
                    align="end"
                    className="min-w-[120px] rounded-md border border-slate-200 bg-white p-1 shadow-lg"
                  >
                    <DropdownMenu.Item asChild>
                      <Link
                        href={CAP_BLOG_URL}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex cursor-pointer items-center rounded px-2 py-1.5 text-[11px] font-semibold uppercase text-slate-500 outline-none hover:bg-slate-100"
                      >
                        Post
                      </Link>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item asChild>
                      <Link
                        href={CAP_PAPER_URL}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex cursor-pointer items-center rounded px-2 py-1.5 text-[11px] font-semibold uppercase text-slate-500 outline-none hover:bg-slate-100"
                      >
                        Paper
                      </Link>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item asChild>
                      <Link
                        href={CAP_GITHUB_URL}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex cursor-pointer items-center rounded px-2 py-1.5 text-[11px] font-semibold uppercase text-slate-500 outline-none hover:bg-slate-100"
                      >
                        GitHub
                      </Link>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item asChild>
                      <Link
                        href={`mailto:${CAP_CONTACT_EMAIL}`}
                        className="flex cursor-pointer items-center rounded px-2 py-1.5 text-[11px] font-semibold uppercase text-slate-500 outline-none hover:bg-slate-100"
                      >
                        Contact
                      </Link>
                    </DropdownMenu.Item>
                  </DropdownMenu.Content>
                </DropdownMenu.Root>
              </div>
            </div>
            <div className="hidden px-0.5 text-[10px] font-medium uppercase text-slate-400 sm:flex">
              <span>
                Select a Demo<span className="hidden sm:inline"> with Llama 3.3 70B</span>
              </span>
            </div>
            <div className="flex w-full flex-row items-center justify-between gap-x-1.5 gap-y-1 sm:px-0">
              {DEMO_BUTTONS.map((demo) => {
                const isDemoSelected = demo.id
                  ? currentSavedId === demo.id
                  : !currentSavedId || !DEMO_BUTTONS.some((d) => d.id === currentSavedId);

                return (
                  <Button
                    key={demo.label}
                    onClick={() => {
                      if (demo.id) {
                        loadSavedSteerOutput(demo.id);
                      } else {
                        reset();
                      }
                    }}
                    variant="outline"
                    size="sm"
                    className={`flex h-10 w-[90px] flex-row items-center justify-center gap-x-1 gap-y-1 text-xs hover:border-sky-300 hover:bg-sky-100 sm:w-32 ${
                      isDemoSelected && !isSteering
                        ? 'border-sky-400 bg-sky-100 text-sky-700'
                        : 'text-sky-700 hover:text-sky-700'
                    }`}
                  >
                    <span>{demo.emoji}</span>
                    <span className="text-[9px] sm:text-xs">{demo.label}</span>
                  </Button>
                );
              })}
            </div>
          </div>

          <div className="hidden flex-1 flex-row items-stretch justify-end gap-x-1.5 gap-y-1 sm:flex">
            <div className="flex min-w-24 flex-col gap-y-1">
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  setIsWelcomeModalOpen(true);
                }}
                className="relative flex h-16 min-h-16 flex-1 flex-row items-center justify-center gap-x-1.5 rounded border border-emerald-600 bg-emerald-50 px-3 py-1.5 font-sans text-[13px] font-semibold leading-none text-emerald-600 hover:bg-emerald-100 sm:h-7 sm:min-h-7 sm:gap-x-1 sm:text-[11px] sm:uppercase"
              >
                <QuestionMarkCircledIcon className="h-5 w-5 sm:h-3.5 sm:w-3.5" />
                Guide
              </button>
              <Link
                href={CAP_VECTOR_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="relative flex min-h-7 flex-1 flex-row items-center justify-center gap-x-1 rounded border border-slate-200 bg-white px-3 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
              >
                <ArrowRightIcon className="h-3.5 w-3.5" />
                Vector
              </Link>
            </div>
            <div className="flex min-w-24 flex-col gap-y-1">
              <Link
                href={CAP_BLOG_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex min-h-7 flex-1 flex-row items-center justify-center gap-x-1 rounded border border-slate-200 bg-white px-3 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
              >
                <BookOpenIcon className="h-3.5 w-3.5" />
                Blog Post
              </Link>
              <Link
                href={CAP_PAPER_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex min-h-7 flex-1 flex-row items-center justify-center gap-x-1 rounded border border-slate-200 bg-white px-3 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
              >
                <Scroll className="h-3.5 w-3.5" />
                Paper
              </Link>
            </div>
            <div className="flex min-w-24 flex-col gap-y-1">
              <Link
                href={CAP_GITHUB_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex min-h-7 flex-1 flex-row items-center justify-center gap-x-1 rounded border border-slate-200 bg-white px-3 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
              >
                <GithubIcon className="h-3.5 w-3.5" />
                GitHub
              </Link>
              <Link
                href={`mailto:${CAP_CONTACT_EMAIL}`}
                className="flex min-h-7 flex-1 flex-row items-center justify-center gap-x-1 rounded border border-slate-200 bg-white px-3 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
              >
                <Mail className="h-3.5 w-3.5" />
                Contact
              </Link>
            </div>
          </div>
        </div>
      </div>
      <div className="relative flex h-full w-full max-w-screen-2xl flex-col px-5 sm:flex-row 2xl:px-0">
        <div
          ref={defaultScrollContainerRef}
          className="absolute right-2 top-0 order-2 h-[calc(50dvh-140px)] max-h-[calc(50dvh-140px)] w-[calc(100dvw-135px)] max-w-[calc(100dvw-135px)] flex-1 flex-col overflow-y-scroll rounded-xl bg-slate-100 px-2 text-left text-xs text-slate-400 sm:relative sm:right-0 sm:order-1 sm:flex sm:h-full sm:max-h-[calc(100dvh-320px)] sm:min-h-[calc(100dvh-320px)] sm:w-full sm:max-w-full sm:px-5"
        >
          <div className="sticky top-0.5 flex w-full flex-row justify-center from-slate-100 via-slate-100 to-transparent pb-2 pt-2 uppercase text-slate-700 sm:top-0 sm:bg-gradient-to-b sm:pb-5 sm:pt-5">
            <div className="-mt-1 select-none rounded-full bg-slate-100 px-3 pb-1 pt-1 text-[11px] font-bold normal-case text-slate-600 sm:-mt-4 sm:bg-transparent sm:px-5 sm:pb-5 sm:pt-2 sm:text-sm">
              Default
            </div>
          </div>
          <div
            className="pb-3 pt-0 text-[14px] font-medium leading-normal text-slate-600 sm:pb-8 sm:pt-3"
            ref={normalEndRef}
          >
            {!isSteering && steeredChatMessages.length === 0 && (
              <div className="w-full pl-3 pt-2 text-center text-xs text-slate-600 sm:pt-8 sm:text-lg">
                {`I'm default Llama 3.3 70B.`}
                <div className="mt-3 hidden text-xs text-slate-500 sm:block sm:text-sm">{`I'm the model that's publicly available, with no activation capping.`}</div>
                <div className="mt-3 text-xs text-slate-500 sm:text-sm">Start a chat with me below.</div>
                <div className="mt-4 text-[9px] text-slate-500 sm:text-xs">
                  This demo is for research purposes and contains examples of AI failure modes, including harmful or
                  distressing outputs.
                </div>
              </div>
            )}
            <SteerChatMessage
              overrideTextSize="text-[10px]"
              chatMessages={defaultChatMessages}
              steered={false}
              messageRefs={defaultMessageRefs}
            />
            {isSteering &&
              (defaultChatMessages.length === 0 ||
                defaultChatMessages[defaultChatMessages.length - 1].role === 'user') && (
                <LoadingSquare className="px-1.5 py-3" />
              )}
          </div>
        </div>
        {/* PersonaChart in the middle */}
        <div
          ref={chartContainerRef}
          className="absolute left-0 top-0 order-1 h-full max-w-[120px] flex-1 flex-col overflow-hidden bg-white px-0 pb-40 sm:relative sm:order-2 sm:max-w-[300px]"
        >
          <div className="relative flex flex-col items-center justify-center gap-y-2">
            <div className="left-0 top-0 hidden w-full items-center justify-center px-2 pt-0 sm:absolute sm:flex sm:pt-[48px]">
              {/* Left arrow */}
              <div
                className="h-0 w-0"
                style={{
                  borderTop: '20px solid transparent',
                  borderBottom: '20px solid transparent',
                  borderRight: '20px solid #94a3b820',
                }}
              />
              <div
                className="h-10 w-[40%]"
                style={{
                  background: 'linear-gradient(to right, #94a3b820 0%, #94a3b830 30%, #94a3b800 80%, #94a3b800 100%)',
                }}
              />
              <div
                className="h-10 w-[40%]"
                style={{
                  background: 'linear-gradient(to right, #94a3b800 0%, #94a3b800 20%, #94a3b830 70%, #94a3b820 100%)',
                }}
              />
              {/* Right arrow */}
              <div
                className="h-0 w-0"
                style={{
                  borderTop: '20px solid transparent',
                  borderBottom: '20px solid transparent',
                  borderLeft: '20px solid #94a3b820',
                }}
              />
            </div>
            <div className="absolute left-1 top-0 flex w-full flex-col items-center pt-1 sm:left-0 sm:px-2.5 sm:pt-[54px]">
              <div className="flex w-full flex-row items-center justify-center">
                <div className="flex flex-1 flex-row items-center justify-center gap-x-1 text-[8.5px] uppercase sm:text-[9px]">
                  <div className="hidden text-lg sm:block">🧙</div>
                  <div className="font-semibold text-slate-600 sm:font-bold">Role-Play</div>
                </div>
                <div className="flex flex-1 flex-row items-center justify-center gap-x-1 text-[8.5px] uppercase sm:text-[9px]">
                  <div className="font-semibold text-slate-600 sm:font-bold">Assistant</div>
                  <div className="hidden text-lg sm:block">🤵🏻</div>
                </div>
              </div>
            </div>
            <div className="-mt-7 min-h-0 w-full flex-1 pl-1 pt-2 sm:mt-0 sm:pl-0">
              <PersonaChart
                data={chartData}
                loading={loadingChartData}
                isSteering={isSteering}
                width={chartWidth}
                height={chartHeight}
                skipAnimationRef={skipChartAnimationRef}
                onPointClick={onChartPointClick}
              />
            </div>
            {/* Pre-cap / Post-cap toggle - Hiding for now */}
            {chartData && (
              <div className="absolute left-1/2 top-[24px] hidden -translate-x-1/2 flex-row items-center justify-center gap-x-0 rounded border border-slate-200 bg-white/90 p-0 sm:hidden">
                <button
                  type="button"
                  onClick={() => setUsePostCap(false)}
                  className={`w-[56px] rounded px-0 py-[2px] text-[9px] font-medium transition-colors sm:text-[8px] ${
                    !usePostCap ? 'bg-sky-700 text-white' : 'bg-transparent text-slate-500 hover:bg-sky-100'
                  }`}
                >
                  Text
                </button>
                {(() => {
                  const hasPostCapData =
                    rawSteeredAxis?.turns?.some((t) => t.pcValuesPostCap) ||
                    rawDefaultAxis?.turns?.some((t) => t.pcValuesPostCap);
                  return (
                    <button
                      type="button"
                      onClick={() => hasPostCapData && setUsePostCap(true)}
                      disabled={!hasPostCapData}
                      className={`w-[56px] rounded px-0 py-[2px] text-[9px] font-medium transition-colors sm:text-[8px] ${
                        usePostCap
                          ? 'bg-sky-700 text-white'
                          : hasPostCapData
                            ? 'bg-transparent text-slate-500 hover:bg-sky-100'
                            : 'cursor-not-allowed bg-transparent text-slate-300'
                      }`}
                    >
                      Activations
                    </button>
                  );
                })()}
              </div>
            )}
            <div className="absolute left-0 top-[14px] flex w-full items-center justify-center">
              <CustomTooltip
                wide
                trigger={
                  <div className="hidden h-5 w-5 cursor-pointer items-center justify-center rounded-full bg-slate-200 p-1 text-xs font-semibold leading-none hover:bg-sky-200 sm:flex">
                    ?
                  </div>
                }
                side="right"
              >
                <div className="flex max-w-[320px] flex-col gap-y-2.5 pl-1 pr-0 text-xs leading-normal text-slate-700 sm:max-w-md">
                  <p className="leading-snug">
                    <div className="-ml-2 mb-1 font-semibold">What is this?</div>
                    This plot shows the mean activation collected on all response tokens per turn{' '}
                    <a
                      href="/llama3.3-70b-it/40-neuronpedia-resid-post/101874252"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sky-700 underline"
                    >
                      projected on layer 40
                    </a>{' '}
                    of the Assistant Axis.
                  </p>
                  <p className="leading-snug">
                    <div className="-ml-2 mb-1 font-semibold">What are the lines?</div>
                    The default line (gray) shows the projections from responses produced by an unsteered model. The
                    capped line (blue) shows the projections from responses produced by a steered model that was{` `}
                    <a
                      href={`${ASSET_BASE_URL}/cap/assistant_cap_with_vectors.json`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sky-700 underline"
                    >
                      activation capped
                    </a>{' '}
                    on layers 32 to 55, with the cap set to where the default persona&apos;s response typically lies.
                    See{' '}
                    <a
                      href={CAP_PAPER_URL}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sky-700 underline"
                    >
                      the paper
                    </a>{' '}
                    for more details.
                  </p>
                  <p className="leading-snug">
                    <div className="-ml-2 mb-1 font-semibold">
                      How are the projections calculated, and why does the capped model (blue) sometimes show a more
                      role-playing projection compared to the default model (gray)?
                    </div>
                    The projections are based on activations from the unsteered model, for both the default and capped
                    responses. This shows how an unsteered model &quot;reads&quot; the response produced by the capped
                    model. We use the activations collected from the unsteered model rather than the capped model,
                    because showing the projections based on activations from the capped model often results in a
                    straight vertical line near where the cap is set.
                    <br />
                    As a result of basing the projections on the unsteered model, the capped model may sometimes show a
                    more role-playing projection compared to the default model, due to the stochasticity of text
                    generation.
                  </p>
                  <p className="leading-snug">
                    <div className="-ml-2 mb-1 font-semibold">Where can I download the relevant vectors?</div>
                    The Assistant Axis vector used in this demo can be viewed and downloaded from{' '}
                    <a
                      href={CAP_VECTOR_URL}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sky-700 underline"
                    >
                      this Neuronpedia dashboard
                    </a>
                    . The activation capping configuration is available{' '}
                    <a
                      href={`${ASSET_BASE_URL}/cap/assistant_cap_with_vectors.json`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sky-700 underline"
                    >
                      as a JSON file
                    </a>
                    .
                  </p>
                </div>
              </CustomTooltip>
            </div>
          </div>
        </div>
        <div
          ref={steeredScrollContainerRef}
          className="absolute bottom-[270px] right-2 order-3 h-[calc(50dvh-140px)] max-h-[calc(50dvh-140px)] w-[calc(100dvw-135px)] max-w-[calc(100dvw-135px)] flex-1 flex-col overflow-y-scroll rounded-xl bg-sky-100 px-2 text-left text-xs text-slate-400 sm:relative sm:bottom-0 sm:right-0 sm:flex sm:h-full sm:max-h-[calc(100dvh-320px)] sm:min-h-[calc(100dvh-320px)] sm:w-full sm:max-w-full sm:px-5"
        >
          <div className="sticky top-0.5 flex w-full flex-row justify-center from-sky-100 via-sky-100 to-transparent pb-2 pt-2 uppercase text-sky-700 sm:top-0 sm:bg-gradient-to-b sm:pb-5 sm:pt-5">
            <div className="-mt-1 select-none rounded-full bg-sky-100 px-3 pb-1 pt-1 text-[11px] font-bold normal-case sm:-mt-4 sm:bg-transparent sm:px-5 sm:pb-5 sm:pt-2 sm:text-sm">
              Capped
            </div>
          </div>
          <div
            className="pb-3 pt-0 text-[14px] font-medium leading-normal text-slate-600 sm:pb-8 sm:pt-3"
            ref={steeredEndRef}
          >
            {!isSteering && steeredChatMessages.length === 0 && (
              <div className="w-full pl-3 pr-3 pt-2 text-center text-xs text-sky-700 sm:pt-8 sm:text-lg">
                {`I'm activation-capped Llama 3.3 70B.`}
                <div className="mt-3 hidden text-xs text-sky-700 sm:block sm:text-sm">{`I'm better at maintaining "assistant-like" behavior during conversations.`}</div>
                <div className="mt-3 text-xs text-sky-700 sm:text-sm">Start a chat with me below.</div>
                <div className="mt-4 text-[9px] text-sky-700 sm:text-xs">
                  This demo is for research purposes and contains examples of AI failure modes, including harmful or
                  distressing outputs.
                </div>
              </div>
            )}
            <SteerChatMessage
              overrideTextSize="text-[10px]"
              chatMessages={steeredChatMessages}
              steered
              messageRefs={steeredMessageRefs}
            />
            {isSteering &&
              (steeredChatMessages.length === 0 ||
                steeredChatMessages[steeredChatMessages.length - 1].role === 'user') && (
                <LoadingSquare className="px-1.5 py-3" />
              )}
          </div>
        </div>
      </div>
      <div className="-mt-[262px] flex w-full flex-col items-center justify-center px-2 pb-8 sm:mt-[-124px] sm:px-0 sm:pb-4">
        <div className="relative flex w-full flex-row items-center justify-end sm:max-w-xl sm:justify-center">
          <div
            className={`absolute left-12 flex -translate-x-full flex-col gap-y-2 pr-3 sm:left-0 ${DEMO_BUTTONS.some((demo) => demo.id && currentSavedId === demo.id) ? 'hidden' : ''}`}
          >
            <button
              type="button"
              title="Share chat"
              disabled={defaultChatMessages.length === 0 || isSteering}
              onClick={() => {
                if (defaultChatMessages.length === 0) {
                  return;
                }
                setIsShareModalOpen(true);
              }}
              className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-slate-300 text-slate-600 shadow hover:bg-slate-200 disabled:cursor-default disabled:text-slate-400 disabled:hover:bg-slate-300"
            >
              <Share className="h-4 w-4" />
            </button>
            <button
              type="button"
              title="Undo last message"
              disabled={
                defaultChatMessages.length < 2 ||
                isSteering ||
                DEMO_BUTTONS.some((demo) => demo.id && currentSavedId === demo.id)
              }
              onClick={() => {
                if (defaultChatMessages.length < 2) {
                  return;
                }
                // Find the last user message
                const lastUserMessage = defaultChatMessages.filter((m) => m.role === 'user').pop();
                // Remove last two messages (user + assistant) from both arrays
                setDefaultChatMessages(defaultChatMessages.slice(0, -2));
                setSteeredChatMessages(steeredChatMessages.slice(0, -2));
                // Put the last user message into the text box
                if (lastUserMessage) {
                  setTypedInText(lastUserMessage.content);
                }
                // Clear the saved URL query param
                setUrl('');
                // Remove last turn from chart data
                if (chartData && chartData.nTurns > 2) {
                  // More than one real turn remaining, just remove the last
                  setChartData({
                    ...chartData,
                    nTurns: chartData.nTurns - 1,
                    series: chartData.series.map((s) => ({
                      ...s,
                      points: s.points.filter((p) => p.turnIndex < chartData.nTurns - 1),
                    })),
                  });
                } else {
                  // Only one turn or less would remain, clear the chart
                  setChartData(null);
                }
              }}
              className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-slate-300 text-slate-600 shadow hover:bg-slate-200 disabled:cursor-default disabled:text-slate-400 disabled:hover:bg-slate-300"
            >
              <Undo2 className="h-4 w-4" />
            </button>
            <button
              type="button"
              title="Clear chat"
              disabled={
                defaultChatMessages.length === 0 ||
                isSteering ||
                DEMO_BUTTONS.some((demo) => demo.id && currentSavedId === demo.id)
              }
              onClick={() => {
                // eslint-disable-next-line
                if (confirm('Are you sure you want to reset the chat?')) {
                  if (defaultChatMessages.length === 0) {
                    return;
                  }
                  reset();
                }
              }}
              className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-slate-300 text-slate-600 shadow hover:bg-slate-200 disabled:cursor-default disabled:text-slate-400 disabled:hover:bg-slate-300"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
          <ReactTextareaAutosize
            name="searchQuery"
            disabled={isSteering || DEMO_BUTTONS.some((demo) => demo.id && currentSavedId === demo.id)}
            value={typedInText}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !isSteering) {
                sendChat();
                e.preventDefault();
              }
            }}
            onChange={(e) => {
              setTypedInText(e.target.value);
            }}
            required
            placeholder={
              DEMO_BUTTONS.some((demo) => demo.id && currentSavedId === demo.id) ? '' : 'Ask or say something...'
            }
            className={`mt-0 h-[90px] max-h-[90px] min-h-[90px] w-[calc(100dvw-60px)] max-w-[calc(100dvw-60px)] flex-1 resize-none rounded-xl border border-sky-100 bg-sky-50 px-4 py-3.5 pr-10 text-left text-xs font-medium text-slate-800 placeholder-sky-600/40 shadow-md transition-all focus:border-sky-200 focus:shadow-lg focus:outline-none focus:ring-0 disabled:border-slate-200 disabled:bg-slate-200 sm:h-[113px] sm:max-h-[113px] sm:min-h-[113px] sm:w-full sm:max-w-full sm:text-[13px] ${DEMO_BUTTONS.some((demo) => demo.id && currentSavedId === demo.id) ? 'hidden' : ''}`}
          />
          <button
            type="button"
            onClick={() => {
              if (!isSteering) {
                sendChat();
              } else {
                stopSteering();
              }
            }}
            disabled={DEMO_BUTTONS.some((demo) => demo.id && currentSavedId === demo.id)}
            className="absolute right-2 flex h-full cursor-pointer items-center justify-center disabled:hidden sm:right-4"
          >
            {!isSteering ? (
              <ArrowUp className="h-8 w-8 rounded-full bg-gBlue p-1.5 text-white hover:bg-gBlue/80" />
            ) : (
              <X className="h-8 w-8 rounded-full bg-red-400 p-1.5 text-white hover:bg-red-600" />
            )}
          </button>
          {limitRemaining !== null &&
            (limitRemaining > 0 ? (
              <div
                className={`absolute bottom-2 right-2 text-[9px] text-slate-500 ${DEMO_BUTTONS.some((demo) => demo.id && currentSavedId === demo.id) ? 'hidden' : ''}`}
              >
                Hourly Limit Left: {limitRemaining}
              </div>
            ) : (
              <div
                className={`absolute bottom-2 right-2 text-[9px] text-slate-500 ${DEMO_BUTTONS.some((demo) => demo.id && currentSavedId === demo.id) ? 'hidden' : ''}`}
              >
                Out of messages. Wait a bit and try again later.
              </div>
            ))}
          {DEMO_BUTTONS.some((demo) => demo.id && currentSavedId === demo.id) && (
            <div className="z-100 absolute left-0 right-0 top-0 flex h-full w-full flex-col items-center justify-start gap-y-1 rounded-lg bg-slate-400">
              <div className="z-100 pt-3 text-xs font-medium text-slate-600 sm:text-sm">
                This is a demo chat. Try your own conversation.
              </div>
              <Button
                variant="outline"
                className="z-100 mt-1 h-12 border-sky-600 bg-sky-50 px-5 py-3 text-[13px] text-sky-600 hover:bg-sky-100 hover:text-sky-800"
                onClick={() => {
                  reset();
                }}
              >
                Start New Chat
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
