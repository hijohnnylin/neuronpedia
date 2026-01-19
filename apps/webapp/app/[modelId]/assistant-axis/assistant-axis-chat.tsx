import { SteerResultChat } from '@/app/api/steer-chat/route';
import { AssistantAxisItem, PersonaCheckResult } from '@/app/[modelId]/assistant-axis/types';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import SteerChatMessage from '@/components/steer/chat-message';
import { LoadingSquare } from '@/components/svg/loading-square';
import { IS_ACTUALLY_NEURONPEDIA_ORG } from '@/lib/env';
import { ChatMessage, SteerFeature, ERROR_STEER_MAX_PROMPT_CHARS } from '@/lib/utils/steer';
import { EventSourceParserStream } from 'eventsource-parser/stream';
import copy from 'copy-to-clipboard';
import { ArrowUp, BookOpenIcon, GithubIcon, Mail, RotateCcw, Scroll, Share, Trash, Trash2, Undo2, X } from 'lucide-react';
import { NPSteerMethod } from 'neuronpedia-inference-client';
import { MutableRefObject, useEffect, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import PersonaChart, { ChartData } from './persona-chart';
import Link from 'next/link';
import { ArrowRightIcon, InfoCircledIcon, QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import { CAP_CONTACT_EMAIL, CAP_PAPER_URL, CAP_VECTOR_URL } from './assistant-axis-steerer';
import { CAP_BLOG_URL } from './assistant-axis-steerer';
import { CAP_GITHUB_URL } from './assistant-axis-steerer';
import AssistantAxisWelcomeModal from './assistant-axis-welcome-modal';
import { useAssistantAxisModalContext } from '@/components/provider/assistant-axis-modal-provider';
import { DEMO_BUTTONS } from './assistant-axis-steerer';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';

export default function AssistantAxisChat({
    showSettingsOnMobile,
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
}: {
    showSettingsOnMobile: boolean;
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
}) {
    const normalEndRef = useRef<HTMLDivElement | null>(null);
    const steeredEndRef = useRef<HTMLDivElement | null>(null);
    const defaultScrollContainerRef = useRef<HTMLDivElement | null>(null);
    const steeredScrollContainerRef = useRef<HTMLDivElement | null>(null);
    const defaultMessageRefs = useRef<(HTMLDivElement | null)[]>([]);
    const steeredMessageRefs = useRef<(HTMLDivElement | null)[]>([]);
    const { showToastMessage, showToastServerError } = useGlobalContext();
    const abortControllerRef = useRef<AbortController | null>(null);
    const readerRef = useRef<ReadableStreamDefaultReader<{ event?: string; data: string; id?: string; retry?: number }> | null>(null);
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const [chartWidth, setChartWidth] = useState(200);
    const [chartHeight, setChartHeight] = useState(400);
    const [limitRemaining, setLimitRemaining] = useState<number | null>(null);
    const [copying, setCopying] = useState(false);

    const { setIsWelcomeModalOpen } = useAssistantAxisModalContext();
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
                    alert('The conversation is too long. Please reset the chat using the trash icon and start a new conversation.');
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
                    const assistantAxisArray = (data as any).assistant_axis as AssistantAxisItem[] | undefined;
                    if (assistantAxisArray && Array.isArray(assistantAxisArray)) {
                        for (const axisItem of assistantAxisArray) {
                            const result: PersonaCheckResult = {
                                pc_titles: axisItem.pc_titles,
                                turns: axisItem.turns,
                            };
                            if (axisItem.type === 'STEERED') {
                                latestSteeredAxis = result;
                            } else if (axisItem.type === 'DEFAULT') {
                                latestDefaultAxis = result;
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
                const assistantAxisArray = data.assistant_axis as AssistantAxisItem[] | undefined;
                if (onAssistantAxisData && assistantAxisArray && Array.isArray(assistantAxisArray)) {
                    let steeredAxis: PersonaCheckResult | null = null;
                    let defaultAxis: PersonaCheckResult | null = null;
                    for (const axisItem of assistantAxisArray) {
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

    function removeLastFailedUserMessage(defaultMsgs: ChatMessage[], steeredMsgs: ChatMessage[]) {
        if (defaultMsgs.length > 0 && defaultMsgs[defaultMsgs.length - 1].role === 'user') {
            setDefaultChatMessages(defaultMsgs.slice(0, -1));
        }
        if (steeredMsgs.length > 0 && steeredMsgs[steeredMsgs.length - 1].role === 'user') {
            setSteeredChatMessages(steeredMsgs.slice(0, -1));
        }
    }

    return (
        <div
            className={`flex relative h-[calc(100dvh)] items-center w-full min-w-0 flex-col text-sm font-medium leading-normal text-slate-500 sm:h-full sm:max-h-[calc(100dvh-180px)] sm:min-h-[calc(100dvh-180px)]`}
        >

            <AssistantAxisWelcomeModal onLoadDemo={loadSavedSteerOutput} initialSavedId={initialSavedId} />
            {/* Demo buttons */}
            <div className="mb-2 sm:mb-5 w-full relative h-[100px] sm:h-[80px] min-h-[100px] sm:min-h-[80px] max-h-[100px] sm:max-h-[80px] z-10 flex flex-row items-center justify-center gap-2 bg-slate-50 px-3 py-2 sm:px-6">
                <div className="flex flex-row items-center justify-between gap-y-1 flex-1 w-full max-w-screen-2xl">
                    <div className="hidden sm:flex flex-row items-center justify-between gap-y-1 flex-1 px-0.5">
                        <div className="flex flex-col items-start justify-center gap-y-1">
                            <div className="text-[18px] font-bold sm:text-xl sm:font-semibold leading-none tracking-tight text-slate-700 whitespace-pre">Assistant Axis</div>
                            <div className="text-[10px] sm:text-xs hidden sm:block text-slate-400">Lu et al.<span className="hidden sm:inline">, January 2026</span></div>
                        </div>
                    </div>
                    <div className="flex flex-col items-center justify-center gap-y-0.5 flex-1 px-0.5 rounded-md p-0 sm:p-2">
                        <div className="flex sm:hidden flex-row w-full justify-center items-center px-0 sm:px-3">
                            <div className="flex flex-col items-start justify-center gap-y-0.5 flex-1">
                                <div className="text-slate-700 text-[17px] px-0.5 sm:hidden font-bold">Assistant Axis</div>
                                <div className="flex px-0.5 text-[10px] font-medium uppercase text-slate-400">
                                    <span>Lu et al., January 2026</span>
                                </div>
                            </div>
                            <div className="flex flex-row items-center justify-center gap-x-1 sm:gap-x-1">
                                <Link
                                    href="#"
                                    onClick={(e) => {
                                        e.preventDefault();
                                        setIsWelcomeModalOpen(true);
                                    }}
                                    className="h-8 sm:h-7 min-h-8 sm:min-h-7 flex flex-row items-center justify-center gap-x-1.5 sm:gap-x-1 rounded bg-emerald-50 border border-emerald-600 px-2.5 py-1.5 font-sans text-[11px] sm:text-[11px] font-semibold sm:uppercase leading-none text-emerald-600 hover:bg-emerald-100 relative"
                                >
                                    Guide
                                </Link>
                                <DropdownMenu.Root>
                                    <DropdownMenu.Trigger asChild>
                                        <button className="h-8 sm:h-7 min-h-8 sm:min-h-7 flex flex-row items-center justify-center gap-x-1.5 rounded bg-white border border-slate-200 px-2 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200">
                                            <InfoCircledIcon className="h-3.5 w-3.5" />
                                        </button>
                                    </DropdownMenu.Trigger>
                                    <DropdownMenu.Content align="end" className="min-w-[120px] bg-white border border-slate-200 rounded-md shadow-lg p-1">
                                        <DropdownMenu.Item asChild>
                                            <Link
                                                href={CAP_BLOG_URL}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="flex items-center px-2 py-1.5 text-[11px] font-semibold uppercase text-slate-500 hover:bg-slate-100 rounded cursor-pointer outline-none"
                                            >
                                                Post
                                            </Link>
                                        </DropdownMenu.Item>
                                        <DropdownMenu.Item asChild>
                                            <Link
                                                href={CAP_PAPER_URL}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="flex items-center px-2 py-1.5 text-[11px] font-semibold uppercase text-slate-500 hover:bg-slate-100 rounded cursor-pointer outline-none"
                                            >
                                                Paper
                                            </Link>
                                        </DropdownMenu.Item>
                                        <DropdownMenu.Item asChild>
                                            <Link
                                                href={CAP_GITHUB_URL}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="flex items-center px-2 py-1.5 text-[11px] font-semibold uppercase text-slate-500 hover:bg-slate-100 rounded cursor-pointer outline-none"
                                            >
                                                GitHub
                                            </Link>
                                        </DropdownMenu.Item>
                                        <DropdownMenu.Item asChild>
                                            <Link
                                                href={`mailto:${CAP_CONTACT_EMAIL}`}
                                                className="flex items-center px-2 py-1.5 text-[11px] font-semibold uppercase text-slate-500 hover:bg-slate-100 rounded cursor-pointer outline-none"
                                            >
                                                Contact
                                            </Link>
                                        </DropdownMenu.Item>
                                    </DropdownMenu.Content>
                                </DropdownMenu.Root>
                            </div>
                        </div>
                        <div className="hidden sm:flex px-0.5 text-[10px] font-medium uppercase text-slate-400">
                            <span>Select a Demo<span className="hidden sm:inline"> with Llama 3.3 70B</span></span>
                        </div>
                        <div className="flex flex-row items-center justify-between gap-y-1 gap-x-1.5 w-full sm:px-0">
                            {DEMO_BUTTONS.map((demo) => (
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
                                    className={`flex h-10 w-[90px] sm:w-32 gap-y-1 flex-row items-center justify-center gap-x-1 text-xs hover:border-sky-300 hover:bg-sky-100 ${demo.id
                                        ? currentSavedId === demo.id
                                            ? 'border-sky-400 bg-sky-100 text-sky-700'
                                            : 'text-sky-700 hover:text-sky-700'
                                        : !currentSavedId && !isSteering
                                            ? 'border-sky-400 bg-sky-100 text-sky-700'
                                            : 'text-sky-700 hover:text-sky-700'
                                        }`}
                                >
                                    <span>{demo.emoji}</span>
                                    <span className="text-[9px] sm:text-xs">{demo.label}</span>
                                </Button>
                            ))}
                        </div>

                    </div>

                    <div className="hidden sm:flex flex-row flex-1 gap-x-1.5 gap-y-1 items-stretch justify-end">
                        <div className="flex flex-col gap-y-1 min-w-24">
                            <Link
                                href="#"
                                onClick={(e) => {
                                    e.preventDefault();
                                    setIsWelcomeModalOpen(true);
                                }}
                                className="h-16 sm:h-7 min-h-16 sm:min-h-7 flex flex-row items-center flex-1 justify-center gap-x-1.5 sm:gap-x-1 rounded bg-emerald-50 border border-emerald-600 px-3 py-1.5 font-sans text-[13px] sm:text-[11px] font-semibold sm:uppercase leading-none text-emerald-600 hover:bg-emerald-100 relative"
                            >
                                <QuestionMarkCircledIcon className="h-5 w-5 sm:h-3.5 sm:w-3.5" />
                                Guide
                            </Link>
                            <Link
                                href={CAP_VECTOR_URL}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="min-h-7 flex flex-row items-center flex-1 justify-center gap-x-1 rounded bg-white border border-slate-200 px-3 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200 relative"
                            >
                                <ArrowRightIcon className="h-3.5 w-3.5" />
                                Vector
                            </Link>

                        </div>
                        <div className=" flex-col gap-y-1 min-w-24 flex">
                            <Link
                                href={CAP_BLOG_URL}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="min-h-7 flex flex-1 flex-row items-center justify-center gap-x-1 rounded bg-white border border-slate-200 px-3 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
                            >
                                <BookOpenIcon className="h-3.5 w-3.5" />
                                Blog Post
                            </Link>
                            <Link
                                href={CAP_PAPER_URL}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="min-h-7 flex flex-row items-center flex-1 justify-center gap-x-1 rounded bg-white border border-slate-200 px-3 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
                            >
                                <Scroll className="h-3.5 w-3.5" />
                                Paper
                            </Link>
                        </div>
                        <div className=" flex flex-col gap-y-1 min-w-24">
                            <Link
                                href={CAP_GITHUB_URL}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="min-h-7 flex flex-row items-center flex-1 justify-center gap-x-1 rounded bg-white border border-slate-200 px-3 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
                            >
                                <GithubIcon className="h-3.5 w-3.5" />
                                GitHub
                            </Link>
                            <Link
                                href={`mailto:${CAP_CONTACT_EMAIL}`}
                                className="min-h-7 flex flex-row items-center flex-1 justify-center gap-x-1 rounded bg-white border border-slate-200 px-3 py-1.5 font-sans text-[11px] font-semibold uppercase leading-none text-slate-500 hover:bg-slate-200"
                            >
                                <Mail className="h-3.5 w-3.5" />
                                Contact
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
            <div className="relative flex h-full w-full flex-col sm:flex-row max-w-screen-2xl px-5 2xl:px-0">
                <div
                    ref={defaultScrollContainerRef}
                    className="order-2 sm:order-1 sm:relative absolute top-0 right-2 sm:right-0 w-[calc(100dvw-135px)] max-w-[calc(100dvw-135px)] sm:max-w-full sm:flex sm:h-full  rounded-xl sm:w-full flex-1 flex-col overflow-y-scroll bg-slate-100 px-2 text-left text-xs text-slate-400 max-h-[calc(50dvh-140px)] h-[calc(50dvh-140px)] sm:max-h-[calc(100dvh-320px)] sm:min-h-[calc(100dvh-320px)] sm:px-5"
                >
                    <div className="sticky top-0.5 flex flex-row justify-center uppercase text-slate-700 sm:top-0 w-full sm:bg-gradient-to-b from-slate-100 via-slate-100 to-transparent pt-2 pb-2 sm:pt-5 sm:pb-5">
                        <div className="select-none rounded-full px-3 sm:px-5 text-[11px] font-bold bg-slate-100 sm:bg-transparent -mt-1 sm:-mt-4 pt-1 sm:pt-2 pb-1 sm:pb-5 sm:text-sm normal-case text-slate-600">
                            Default
                        </div>
                    </div>
                    <div
                        className="pb-3 pt-0 text-[14px] font-medium leading-normal text-slate-600 sm:pb-8 sm:pt-3"
                        ref={normalEndRef}
                    >
                        {!isSteering && steeredChatMessages.length === 0 && (
                            <div className="w-full pl-3 pt-2 sm:pt-8 text-center text-xs sm:text-lg text-slate-600">
                                {`I'm default Llama 3.3 70B.`}
                                <div className="mt-3 hidden sm:block text-xs sm:text-sm text-slate-500">{`I'm the model that's publicly available, with no activation capping.`}</div>
                                <div className="mt-3 text-xs sm:text-sm text-slate-500">{`Start a chat with me below.`}</div>

                            </div>
                        )}
                        <SteerChatMessage overrideTextSize="text-[10px]" chatMessages={defaultChatMessages} steered={false} messageRefs={defaultMessageRefs} />
                        {isSteering && (defaultChatMessages.length === 0 || defaultChatMessages[defaultChatMessages.length - 1].role === 'user') && <LoadingSquare className="px-1.5 py-3" />}
                    </div>
                </div>
                {/* PersonaChart in the middle */}
                <div
                    ref={chartContainerRef}
                    className="absolute sm:relative top-0 left-0 order-1 sm:order-2 h-full flex-1 max-w-[120px] sm:max-w-[300px] flex-col overflow-hidden bg-white pb-40 px-0"
                >
                    <div className="flex relative flex-col items-center justify-center gap-y-2">
                        {/* <div className="absolute top-0 left-0 text-center text-[12px] uppercase font-bold leading-normal text-slate-500 pt-3 w-full">Assistant Axis</div> */}

                        <div className="hidden sm:absolute sm:flex left-0 top-0 w-full items-center justify-center px-2 pt-0 sm:pt-[40px]">
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
                        <div className="absolute left-1 sm:left-0 top-0 flex w-full flex-col items-center sm:px-2.5 pt-1 sm:pt-[46px]">
                            <div className="flex w-full flex-row items-center justify-center">
                                <div className="flex flex-row items-center flex-1 justify-center gap-x-1 text-[8.5px] sm:text-[9px] uppercase">
                                    <div className="text-lg hidden sm:block">🧙</div>
                                    <div className="font-semibold sm:font-bold text-slate-600">Role-Play</div>
                                </div>
                                <div className="flex flex-row items-center flex-1 justify-center gap-x-1 text-[8.5px] sm:text-[9px] uppercase">
                                    <div className="font-semibold sm:font-bold text-slate-600">Assistant</div>
                                    <div className="text-lg hidden sm:block">🤵🏻</div>
                                </div>
                            </div>
                        </div>
                        <div className="min-h-0 w-full pl-1 sm:pl-0 flex-1 pt-2 -mt-7 sm:mt-0">
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
                    </div>
                </div>
                <div
                    ref={steeredScrollContainerRef}
                    className="order-3 sm:relative absolute sm:bottom-0 bottom-[270px] sm:right-0 right-2 w-[calc(100dvw-135px)] max-w-[calc(100dvw-135px)] sm:max-w-full sm:flex sm:h-full  rounded-xl sm:w-full flex-1 flex-col overflow-y-scroll bg-sky-100 px-2 text-left text-xs text-slate-400 max-h-[calc(50dvh-140px)] h-[calc(50dvh-140px)] sm:max-h-[calc(100dvh-320px)] sm:min-h-[calc(100dvh-320px)] sm:px-5"
                >
                    <div className="sticky top-0.5 flex flex-row justify-center uppercase text-sky-700 sm:top-0 w-full sm:bg-gradient-to-b from-sky-100 via-sky-100 to-transparent pt-2 pb-2 sm:pt-5 sm:pb-5">
                        <div className="select-none rounded-full px-3 sm:px-5 text-[11px] font-bold bg-sky-100 sm:bg-transparent -mt-1 sm:-mt-4 pt-1 sm:pt-2 pb-1 sm:pb-5 sm:text-sm normal-case">
                            Capped
                        </div>
                    </div>
                    <div
                        className="pb-3 pt-0 text-[14px] font-medium leading-normal text-slate-600 sm:pb-8 sm:pt-3"
                        ref={steeredEndRef}
                    >
                        {!isSteering && steeredChatMessages.length === 0 && (
                            <div className="w-full pl-3 pr-3 pt-2 sm:pt-8 text-center text-xs sm:text-lg text-sky-700">
                                {`I'm activation-capped Llama 3.3 70B.`}
                                <div className="mt-3 hidden sm:block text-xs sm:text-sm text-sky-700">{`I'm better at maintaining "assistant-like" behavior during conversations.`}</div>
                                <div className="mt-3 text-xs sm:text-sm text-sky-700">{`Start a chat with me below.`}</div>
                            </div>
                        )}
                        <SteerChatMessage overrideTextSize="text-[10px]" chatMessages={steeredChatMessages} steered messageRefs={steeredMessageRefs} />
                        {isSteering && (steeredChatMessages.length === 0 || steeredChatMessages[steeredChatMessages.length - 1].role === 'user') && <LoadingSquare className="px-1.5 py-3" />}
                    </div>
                </div>
            </div>
            <div className="-mt-[262px] flex w-full flex-col items-center justify-center px-2 sm:px-0 pb-8 sm:mt-[-124px] sm:pb-4">

                <div className="relative flex flex-row items-center justify-end sm:justify-center w-full sm:max-w-xl">
                    {DEMO_BUTTONS.some(demo => demo.id && currentSavedId === demo.id) && (
                        <div className="absolute top-0 left-0 right-0 flex w-full rounded-lg h-full flex flex-col gap-y-1 bg-slate-400/30 justify-start items-center">
                            <div className="text-slate-800 text-sm pt-4 text-xs font-medium text-slate-700">This is a demo chat. Try your own conversation.</div>
                            <Button variant="outline" className=" mt-2 bg-white text-sky-700 bg-sky-50 border-sky-500 hover:bg-sky-100" onClick={() => {
                                reset();
                            }}>Start New Chat</Button>
                        </div>
                    )}
                    <div className="absolute left-12 sm:left-0 -translate-x-full pr-3 flex flex-col gap-y-2">
                        <button
                            type="button"
                            title={copying ? 'Copied!' : 'Share chat'}
                            disabled={defaultChatMessages.length === 0 || isSteering}
                            onClick={() => {
                                if (defaultChatMessages.length === 0) {
                                    return;
                                }
                                setCopying(true);
                                copy(window.location.href);
                                alert(
                                    'Copied share link to clipboard.\nPaste it somewhere to share your conversation.',
                                );
                                setTimeout(() => setCopying(false), 2000);
                            }}
                            className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-slate-300 text-slate-600 shadow hover:bg-slate-200 disabled:cursor-default disabled:text-slate-400 disabled:hover:bg-slate-300"
                        >
                            <Share className="h-4 w-4" />
                        </button>
                        <button
                            type="button"
                            title="Undo last message"
                            disabled={defaultChatMessages.length < 2 || isSteering || DEMO_BUTTONS.some(demo => demo.id && currentSavedId === demo.id)}
                            onClick={() => {
                                if (defaultChatMessages.length < 2) {
                                    return;
                                }
                                // Find the last user message
                                const lastUserMessage = defaultChatMessages.filter(m => m.role === 'user').pop();
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
                                        series: chartData.series.map(s => ({
                                            ...s,
                                            points: s.points.filter(p => p.turnIndex < chartData.nTurns - 1),
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
                            disabled={defaultChatMessages.length === 0 || isSteering || DEMO_BUTTONS.some(demo => demo.id && currentSavedId === demo.id)}
                            onClick={() => {
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
                        disabled={isSteering || DEMO_BUTTONS.some(demo => demo.id && currentSavedId === demo.id)}
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
                        placeholder={DEMO_BUTTONS.some(demo => demo.id && currentSavedId === demo.id) ? "" : "Ask or say something..."}
                        className="mt-0 h-[90px] min-h-[90px] max-h-[90px] sm:h-[113px] sm:min-h-[113px] sm:max-h-[113px] w-[calc(100dvw-60px)] max-w-[calc(100dvw-60px)] sm:max-w-full sm:w-full flex-1 resize-none rounded-xl border bg-sky-50 border-sky-100 disabled:border-slate-200 px-4 py-3.5 pr-10 text-left text-xs font-medium text-slate-800 placeholder-sky-600/40 shadow-md transition-all focus:border-sky-200 focus:shadow-lg focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
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
                        disabled={DEMO_BUTTONS.some(demo => demo.id && currentSavedId === demo.id)}
                        className="absolute right-2 sm:right-4 flex h-full cursor-pointer items-center justify-center disabled:opacity-50"
                    >
                        {!isSteering ? (
                            <ArrowUp className="h-8 w-8 rounded-full bg-gBlue p-1.5 text-white hover:bg-gBlue/80" />
                        ) : (
                            <X className="h-8 w-8 rounded-full bg-red-400 p-1.5 text-white hover:bg-red-600" />
                        )}
                    </button>
                    {limitRemaining !== null ? limitRemaining > 0 ? (
                        <div className="text-[9px] absolute right-2 bottom-2 text-slate-500">Hourly Limit Left: {limitRemaining}</div>
                    ) : (
                        <div className="text-[9px] absolute right-2 bottom-2 text-slate-500">Out of messages. Wait a bit and try again later.</div>
                    ) : <></>}
                </div>
            </div>
        </div >
    );
}

