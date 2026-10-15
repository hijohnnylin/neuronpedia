'use client';

import { SteerResultChat } from '@/app/api/steer-chat/route';
import { useGlobalContext } from '@/components/provider/global-provider';
import { NPSteerType } from '@/lib/api/inference-types';
import { IS_ACTUALLY_NEURONPEDIA_ORG } from '@/lib/env';
import type { PersonaAxisDefinition } from '@/lib/utils/persona-axis';
import {
  ChatMessage,
  ERROR_STEER_MAX_PROMPT_CHARS,
  STEER_METHOD,
  STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS,
  STEER_PRESENCE_PENALTY,
  STEER_SEED,
  STEER_SPECIAL_TOKENS,
  STEER_STRENGTH_MULTIPLIER,
  STEER_TEMPERATURE,
} from '@/lib/utils/steer';
import type { PublicAxisReadout } from '@/lib/utils/steer-axis-public';
import { EventSourceParserStream } from 'eventsource-parser/stream';
import { ArrowUp, BookOpen, CircleHelp, Mail, Newspaper, Trash2, Undo2, X } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import NeuralTransparencyMessages, {
  GUTTER,
  GUTTER_CLASS,
  GUTTER_WIDTH,
  INPUT_BLUR_LEAD,
  UNDERLAP,
  UNDERLAP_CLASS,
} from './neural-transparency-messages';
import { tourHeldTurns, useNtTour } from './nt-tour';
import {
  NT_CHAT_ID,
  NT_CHAT_TOOL_CLASS,
  NT_EXAMPLES_BAR_ID,
  NT_GUTTER_CLASS,
  NT_INPUT_ID,
  NT_RIGHT_BUTTONS_ID,
  NT_TOUR_GUIDE_BUTTON_ID,
  NT_TOUR_SEEN_KEY,
} from './nt-tour-constants';
import { GUTTER_TINT } from './persona-sunburst';
import {
  NT_BLOG_URL,
  NT_CONTACT_URL,
  NT_EXAMPLE_CHATS,
  NT_FREE_CHAT_SYSTEM_PROMPT,
  NT_MODEL_ID,
  NT_PAPER_URL,
  NT_TITLE,
} from './shared';

/**
 * The ease the page settles with when the first message is sent. The same curve the readouts arrive
 * on, which ends slowly and so settles rather than stopping.
 */
const GROW_SHIFT = 'transition-[flex-grow] duration-500 ease-[cubic-bezier(0.16,1,0.3,1)]';

const NT_PATH = '/drift-calibrator';

// Stable defaults, because the chat reloads when these props change identity.
const NO_MESSAGES: ChatMessage[] = [];
const NO_READOUTS: PublicAxisReadout[] = [];

const assistantTurnCount = (messages: { role: string }[]) =>
  messages.filter((message) => message.role === 'assistant' || message.role === 'model').length;

/** A chat with no question left is empty, even if its system prompt is still there. */
const emptyIfNoQuestion = (messages: ChatMessage[]) => (messages.some((m) => m.role === 'user') ? messages : []);

/** The template a reply came back in, with the chat's system prompt kept at its head. */
function keepSystem(template: ChatMessage[], system: ChatMessage | undefined): ChatMessage[] {
  return system && template[0]?.role !== 'system' ? [system, ...template] : template;
}

/** A saved chat's question and reply, and the readouts up to and with that reply measured. */
type HeldTurn = { messages: ChatMessage[]; readouts: PublicAxisReadout[] };

/**
 * What the chat opens with. The tour can hold back a saved chat's last turns, so that the user
 * sends each question and sees its reply arrive. The held turns are in order.
 */
function openingChat(savedId: string | undefined, messages: ChatMessage[], readouts: PublicAxisReadout[]) {
  const count = savedId ? tourHeldTurns(savedId) : 0;
  const questions = messages.flatMap((message, index) => (message.role === 'user' ? [index] : []));
  if (count < 1 || questions.length <= count || questions[questions.length - 1] !== messages.length - 2) {
    return { messages, readouts, held: [] as HeldTurn[] };
  }
  const cut = questions[questions.length - count];
  const shownTurns = assistantTurnCount(messages.slice(0, cut));
  const upTo = (turns: number) => readouts.map((r) => ({ ...r, turns: (r.turns ?? []).slice(0, turns) }));
  return {
    messages: messages.slice(0, cut),
    readouts: upTo(shownTurns),
    held: questions.slice(-count).map((question, index) => ({
      messages: messages.slice(question, question + 2),
      readouts: upTo(shownTurns + index + 1),
    })),
  };
}

// How a held turn plays back: a wait for the reply, the reply a few words at a time, then a wait
// for its reading.
const REPLAY_WAIT_MS = 1200;
const REPLAY_WORDS_PER_STEP = 3;
const REPLAY_STEP_MS = 30;
const REPLAY_MEASURE_MS = 800;

const sleep = (ms: number) =>
  new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });

/** The saved examples and Free Chat as one segmented bar: an emoji, then one word per line. */
function ExampleBar({ currentSavedId }: { currentSavedId?: string }) {
  const items = [
    ...NT_EXAMPLE_CHATS.map((example) => ({
      key: example.savedId,
      emoji: example.emoji,
      title: example.title,
      href: `${NT_PATH}?savedId=${example.savedId}`,
      active: example.savedId === currentSavedId,
    })),
    {
      key: 'free-chat',
      emoji: '✏️',
      title: 'Free Chat',
      href: NT_PATH,
      active: !NT_EXAMPLE_CHATS.some((example) => example.savedId === currentSavedId),
    },
  ];
  return (
    <div className="flex max-w-full flex-row divide-x divide-slate-200 overflow-hidden rounded-xl border border-slate-200 bg-white">
      {items.map((item) => (
        <Link
          key={item.key}
          href={item.href}
          scroll={false}
          className={`flex flex-1 flex-row items-center justify-center gap-x-1.5 px-5 py-2 transition-colors sm:min-w-[110px] sm:flex-none sm:gap-x-2 sm:px-4 sm:py-1.5 ${
            item.active ? 'bg-slate-700 text-white' : 'bg-white text-slate-600 hover:bg-slate-100'
          }`}
        >
          <span className="text-sm leading-none sm:text-xl">{item.emoji}</span>
          <span className="flex flex-col items-center text-center text-[9px] font-medium leading-tight sm:items-start sm:text-left sm:text-[10px]">
            {item.title.split(' ').map((word) => (
              <span key={word} className="whitespace-nowrap">
                {word}
              </span>
            ))}
          </span>
        </Link>
      ))}
    </div>
  );
}

/** The intro cards above the empty chat, top to bottom. Each class is a full literal for Tailwind. */
const INTRO_CARDS = [
  {
    title: 'Problem',
    cardClass: 'bg-amber-100/60',
    titleClass: 'text-amber-700',
    bodyClass: 'text-slate-600',
    body: 'Humans communicate both verbally and non-verbally, but chats with AI miss things like body language and tone, leaving out crucial context and nuance.',
  },
  {
    title: 'Interface-Based Solution',
    cardClass: 'bg-emerald-100/60',
    titleClass: 'text-emerald-700',
    bodyClass: 'text-slate-600',
    body: (
      <>
        {`MIT Media Lab's `}
        <a
          href={NT_PAPER_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium text-emerald-700 underline decoration-emerald-700/40 underline-offset-2 hover:decoration-emerald-700"
        >
          {NT_TITLE}
        </a>
        {` adds live, interp-powered visualizations during chats, which reveal the AI's internal behaviors like sycophancy and toxicity.`}
      </>
    ),
  },
];

export default function NeuralTransparencyChat({
  /**
   * The axes this page draws and measures, read from the database by the server component above.
   * One list for both, so the wheel cannot hold an axis that is never asked for, or omit one whose
   * numbers arrive.
   */
  axes,
  /** The saved chat this page opened with, from `?savedId=`, if any. */
  savedId,
  initialMessages = NO_MESSAGES,
  initialReadouts = NO_READOUTS,
}: {
  axes: PersonaAxisDefinition[];
  savedId?: string;
  initialMessages?: ChatMessage[];
  initialReadouts?: PublicAxisReadout[];
}) {
  const { showToastMessage, showToastServerError } = useGlobalContext();
  const [opening] = useState(() => openingChat(savedId, initialMessages, initialReadouts));
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(opening.messages);
  const [readouts, setReadouts] = useState<PublicAxisReadout[]>(opening.readouts);
  const heldTurnsRef = useRef<HeldTurn[]>(opening.held);
  const [typedInText, setTypedInText] = useState('');
  /** The tour types a question for the user, who can send it but not change it. */
  const [draftLocked, setDraftLocked] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [limitRemaining, setLimitRemaining] = useState<number | null>(null);
  const [chatFull, setChatFull] = useState(false);
  /** The saved row that the URL names. Changes after each reply, without a server round trip. */
  const [currentSavedId, setCurrentSavedId] = useState(savedId);
  /** Saved row per number of replies, so that an undo can name the shorter chat in the URL. */
  const savedIdByTurnsRef = useRef(new Map<number, string>());

  const router = useRouter();
  const { startTour } = useNtTour({
    navigate: (href) => router.push(href, { scroll: false }),
    prefillHeldTurn: () => {
      const held = heldTurnsRef.current[0];
      if (held) {
        setTypedInText(held.messages[0].content);
        setDraftLocked(true);
      }
    },
    releaseHeldTurns: () => {
      setDraftLocked(false);
      const held = heldTurnsRef.current;
      if (held.length > 0) {
        heldTurnsRef.current = [];
        setTypedInText('');
        setChatMessages((shown) => [...shown, ...held.flatMap((turn) => turn.messages)]);
        setReadouts(held[held.length - 1].readouts);
      }
    },
  });
  // `true` until localStorage is read, so the "new" dot does not flash on for a returning user.
  const [hasSeenTour, setHasSeenTour] = useState(true);
  const markTourSeen = useCallback(() => {
    try {
      localStorage.setItem(NT_TOUR_SEEN_KEY, 'true');
    } catch (error) {
      console.error('Error setting localStorage:', error);
    }
    setHasSeenTour(true);
  }, []);
  const handleStartTour = useCallback(() => {
    markTourSeen();
    startTour();
  }, [markTourSeen, startTour]);
  // Start the tour on a first visit, unless the link opens a saved chat (`?savedId=`).
  useEffect(() => {
    try {
      const seen = localStorage.getItem(NT_TOUR_SEEN_KEY) === 'true';
      setHasSeenTour(seen);
      if (seen || savedId) return;
      markTourSeen();
      startTour();
    } catch (error) {
      console.error('Error checking localStorage:', error);
    }
    // Only on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<{
    event?: string;
    data: string;
    id?: string;
    retry?: number;
  }> | null>(null);

  /**
   * Whether to keep the newest text in view. Set from where the reader leaves the scrollbar, so
   * scrolling up to reread an earlier turn is not undone by the next streamed chunk, and coming
   * back to the bottom resumes following. Jumping there outright rather than smoothly is what
   * keeps this honest: a smooth scroll emits its own events mid-animation, which read as the
   * reader having scrolled away. A saved chat opens at its end, on the newest turn.
   */
  const followBottomRef = useRef(true);
  const chatContentRef = useRef<HTMLDivElement | null>(null);

  const onChatScroll = () => {
    const container = scrollContainerRef.current;
    if (container) {
      followBottomRef.current = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
    }
  };

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (container && chatMessages.length > 0 && followBottomRef.current) {
      container.scrollTop = container.scrollHeight;
    }
  }, [chatMessages]);

  const hasChatRef = useRef(chatMessages.length > 0);
  hasChatRef.current = chatMessages.length > 0;

  // The transcript grows after it renders, as replies and readouts lay out, so stay at the end.
  useEffect(() => {
    const container = scrollContainerRef.current;
    const content = chatContentRef.current;
    if (!container || !content) {
      return undefined;
    }
    const observer = new ResizeObserver(() => {
      if (followBottomRef.current && hasChatRef.current) {
        container.scrollTop = container.scrollHeight;
      }
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, []);

  /** Replaces the URL in place: a router navigation would reload the page from the server. */
  function showSavedId(id: string | undefined) {
    window.history.replaceState(null, '', id ? `${NT_PATH}?savedId=${id}` : NT_PATH);
    setCurrentSavedId(id);
  }

  // A link to this page whose key matches keeps this component mounted, so new props from the
  // server load that chat here.
  useEffect(() => {
    const next = openingChat(savedId, initialMessages, initialReadouts);
    setChatMessages(next.messages);
    setReadouts(next.readouts);
    heldTurnsRef.current = next.held;
    setDraftLocked(false);
    setTypedInText('');
    setChatFull(false);
    setCurrentSavedId(savedId);
    savedIdByTurnsRef.current = new Map(savedId ? [[assistantTurnCount(initialMessages), savedId]] : []);
    followBottomRef.current = initialMessages.length === 0;
  }, [savedId, initialMessages, initialReadouts]);

  function reset() {
    setChatMessages([]);
    setReadouts([]);
    setTypedInText('');
    setChatFull(false);
    savedIdByTurnsRef.current.clear();
    showSavedId(undefined);
  }

  // Only the unsteered column is generated here, so a STEERED readout would be a leftover from a
  // cached row rather than something this page measured.
  const keepDefault = (measured: PublicAxisReadout[]) => measured.filter((r) => r.type !== NPSteerType.STEERED);

  async function stopGenerating() {
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
    setIsGenerating(false);
  }

  /** Play back a held turn as if it were generated now: the question, a wait, the reply, its reading. */
  async function replayHeldTurn(held: HeldTurn) {
    heldTurnsRef.current = heldTurnsRef.current.slice(1);
    setDraftLocked(false);
    setTypedInText('');
    setIsGenerating(true);
    followBottomRef.current = true;
    const [question, reply] = held.messages;
    const asked = [...chatMessages, question];
    setChatMessages(asked);
    await sleep(REPLAY_WAIT_MS);
    const words = reply.content.split(/(?<=\s)/);
    for (let shown = REPLAY_WORDS_PER_STEP; shown < words.length; shown += REPLAY_WORDS_PER_STEP) {
      setChatMessages([...asked, { ...reply, content: words.slice(0, shown).join('') }]);
      // eslint-disable-next-line no-await-in-loop
      await sleep(REPLAY_STEP_MS);
    }
    setChatMessages([...asked, reply]);
    await sleep(REPLAY_MEASURE_MS);
    setReadouts(held.readouts);
    setIsGenerating(false);
  }

  async function sendChat() {
    if (typedInText.trim().length === 0) {
      alert('Please enter a message.');
      return;
    }
    if (heldTurnsRef.current.length > 0) {
      replayHeldTurn(heldTurnsRef.current[0]);
      return;
    }
    setIsGenerating(true);
    followBottomRef.current = true;

    // A new chat starts with the Free Chat system prompt.
    const base: ChatMessage[] =
      chatMessages.length > 0 ? chatMessages : [{ role: 'system', content: NT_FREE_CHAT_SYSTEM_PROMPT }];
    const system = base[0].role === 'system' ? base[0] : undefined;
    // If the last message is a user message, replace it; otherwise, add a new one
    const sent: ChatMessage[] =
      base[base.length - 1].role === 'user'
        ? [...base.slice(0, -1), { content: typedInText, role: 'user' }]
        : [...base, { content: typedInText, role: 'user' }];
    setChatMessages(sent);

    // Dropping the user turn again is what lets the box keep the text, so a failed send is
    // retryable rather than lost.
    const failed = () => {
      setIsGenerating(false);
      setChatMessages(emptyIfNoQuestion(sent.slice(0, -1)));
    };

    // Only the last frame of a reply has the id, because the row is saved after the reply ends.
    const showSaved = (data: SteerResultChat) => {
      if (data.defaultId && data.DEFAULT?.chatTemplate) {
        savedIdByTurnsRef.current.set(assistantTurnCount(data.DEFAULT.chatTemplate), data.defaultId);
        showSavedId(data.defaultId);
      }
    };

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    const { signal } = abortControllerRef.current;

    try {
      const response = await fetch(`/api/steer-chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          defaultChatMessages: sent,
          // No features means nothing to steer with, so /api/steer-chat runs the default column
          // alone and this is never generated from. It is sent so that adding a steered column
          // later is a UI change rather than a change to what the request means.
          steeredChatMessages: sent,
          modelId: NT_MODEL_ID,
          features: [],
          axes: axes.map((axis) => axis.name),
          temperature: STEER_TEMPERATURE,
          n_tokens: STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS,
          presence_penalty: STEER_PRESENCE_PENALTY,
          seed: STEER_SEED,
          strength_multiplier: STEER_STRENGTH_MULTIPLIER,
          steer_method: STEER_METHOD,
          steer_special_tokens: STEER_SPECIAL_TOKENS,
          stream: true,
        }),
        signal,
      });
      if (!response || !response.body) {
        showToastServerError();
        failed();
        return;
      }
      if (response.status === 429) {
        alert('Sorry, you have reached the maximum number of messages per hour. Please try again later.');
        failed();
        return;
      }
      if (response.status === 400) {
        const errorBody = await response.json();
        if (errorBody.message === ERROR_STEER_MAX_PROMPT_CHARS) {
          setChatFull(true);
          failed();
          return;
        }
      }
      if (response.status !== 200) {
        if (response.status === 404) {
          alert(
            !IS_ACTUALLY_NEURONPEDIA_ORG
              ? `Unable to chat with ${NT_MODEL_ID}. Is it imported, and is an inference server running it?`
              : `Unable to chat with ${NT_MODEL_ID} - it was not found.`,
          );
          failed();
          return;
        }
        console.error(`Error ${response.status}: ${response.statusText}\n\n${await response.text()}`);
        showToastServerError();
        failed();
        return;
      }

      const limitHeader = response.headers.get('x-limit-remaining');
      if (limitHeader) {
        setLimitRemaining(Number(limitHeader));
      }

      if (response.headers.get('content-type') === 'text/event-stream') {
        const reader = response.body
          .pipeThrough(new TextDecoderStream())
          .pipeThrough(new EventSourceParserStream())
          .getReader();
        readerRef.current = reader;
        while (true) {
          // eslint-disable-next-line no-await-in-loop
          const { done, value } = await reader.read();
          if (done) {
            readerRef.current = null;
            setIsGenerating(false);
            break;
          }
          const data = JSON.parse(value.data) as SteerResultChat;
          if (data.DEFAULT?.chatTemplate) {
            setChatMessages(keepSystem(data.DEFAULT.chatTemplate, system));
          }
          // Readouts arrive once the turn is generated, so this lands in one step near the end of
          // the stream rather than growing with the text.
          if (data.axes && data.axes.length > 0) {
            setReadouts(keepDefault(data.axes));
          }
          showSaved(data);
          setTypedInText('');
        }
      } else {
        const data = (await response.json()) as SteerResultChat;
        setChatMessages(keepSystem(data.DEFAULT?.chatTemplate || [], system));
        setReadouts(keepDefault(data.axes ?? []));
        showSaved(data);
        setTypedInText('');
        setIsGenerating(false);
      }
    } catch (error) {
      readerRef.current = null;
      if (error instanceof DOMException && error.name === 'AbortError') {
        showToastMessage('Generation aborted.');
      } else {
        console.error(error);
        showToastServerError();
      }
      failed();
    }
  }

  const awaitingResponse =
    isGenerating && (chatMessages.length === 0 || chatMessages[chatMessages.length - 1].role === 'user');
  const hasMessages = chatMessages.length > 0;

  return (
    <div className="flex h-[calc(100dvh-48px)] w-full min-w-0 flex-col items-center overflow-hidden text-sm font-medium leading-normal text-slate-500 sm:h-[calc(100dvh-75px)]">
      {/* The page header: title on the left, tutorial and links on the right, examples centred.
          On a phone the examples drop below the title, since the two do not fit side by side. */}
      <div className="flex w-full shrink-0 items-center justify-center bg-slate-100 px-2 py-3 sm:px-6 sm:py-4">
        <div className="relative flex w-full max-w-screen-2xl flex-col items-center gap-y-3 sm:flex-row sm:items-center">
          <div className="flex w-full flex-row items-center justify-between gap-x-3">
            <div className="flex flex-col items-start">
              <div className="mb-1.5 flex flex-row items-center gap-x-1">
                <div className="whitespace-nowrap rounded-full bg-slate-400 px-2 py-[3px] text-[9px] font-semibold uppercase leading-none tracking-wide text-white sm:text-[10px]">
                  UI Research
                </div>
                <div className="whitespace-nowrap rounded-full bg-slate-400 px-2 py-[3px] text-[9px] font-semibold uppercase leading-none tracking-wide text-white sm:text-[10px]">
                  Community
                </div>
              </div>
              <div className="whitespace-nowrap text-[13px] font-semibold leading-none text-slate-800 sm:text-[15px]">
                {NT_TITLE}
              </div>
              <div className="mt-[3px] whitespace-nowrap text-[10px] text-slate-500 sm:text-[11.5px]">
                Karny, Baez, Pataranutaporn
              </div>
            </div>
            <div id={NT_RIGHT_BUTTONS_ID} className="grid shrink-0 grid-cols-2 gap-1">
              <button
                type="button"
                id={NT_TOUR_GUIDE_BUTTON_ID}
                onClick={handleStartTour}
                className="relative flex w-24 items-center justify-center gap-x-1.5 rounded-md border border-sky-600 bg-sky-50 px-3 py-1 text-[11px] font-semibold text-sky-600 transition-colors hover:bg-sky-100"
              >
                <CircleHelp className="h-3 w-3" />
                Tutorial
                {!hasSeenTour && (
                  <span
                    aria-hidden="true"
                    className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full bg-red-500 ring-2 ring-slate-100"
                  />
                )}
              </button>
              <Link
                href={NT_PAPER_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex w-24 items-center justify-center gap-x-1.5 rounded-md border border-slate-400 bg-white px-3 py-1 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-50"
              >
                <Newspaper className="h-3 w-3" />
                Paper
              </Link>
              <Link
                href={NT_BLOG_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex w-24 items-center justify-center gap-x-1.5 rounded-md border border-slate-400 bg-white px-3 py-1 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-50"
              >
                <BookOpen className="h-3 w-3" />
                Blog
              </Link>
              <a
                href={NT_CONTACT_URL}
                className="flex w-24 items-center justify-center gap-x-1.5 rounded-md border border-slate-400 bg-white px-3 py-1 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-50"
              >
                <Mail className="h-3 w-3" />
                Contact
              </a>
            </div>
          </div>
          <div id={NT_EXAMPLES_BAR_ID} className="sm:absolute sm:left-1/2 sm:-translate-x-1/2">
            <ExampleBar currentSavedId={currentSavedId} />
          </div>
        </div>
      </div>

      {/* Empty, the greeting and the box you type in sit a third of the way down the page rather
          than halfway, which reads as waiting to start rather than as adrift; sending the first
          message settles them at the top and bottom of a transcript. The move is a transition on
          `flex-grow` -- these two spacers give up their share of the page as the transcript takes
          it, the lower one twice the share of the upper -- rather than a swap between two layouts,
          so the input the reader just typed into travels to where it now lives instead of vanishing
          from one place and appearing in another. */}

      <div aria-hidden className={`${GROW_SHIFT} w-full`} style={{ flexGrow: hasMessages ? 0 : 1 }} />

      {/* Narrower than the header above it, and centred: a row is a wheel plus a bubble that
          wraps well short of the full width, so at screen-2xl the transcript was a column of
          content down the left with nothing to its right. This once held a second column for the
          readouts, which now sit beside the turns they describe.

          Grown rather than `flex-1`, so its basis stays `auto`: with nothing said yet it is as tall
          as the greeting inside it, and it opens out to the page from there. `flex-1` would zero
          the basis, and a basis of `auto` cannot be animated to one of `0`. */}
      <div
        className={`${GROW_SHIFT} flex min-h-0 w-full max-w-5xl flex-row overflow-hidden pb-1 pl-0 pr-1 pt-0 sm:px-5 sm:pb-3 sm:pt-3`}
        style={{ flexGrow: hasMessages ? 1 : 0 }}
      >
        {/* The panel: the transcript and the box you type in, one surface. The scroll lives on the
            transcript alone rather than on the panel, so the input stays put while the conversation
            moves under it -- inside a single scrolling box it would have scrolled away with the
            first reply.

            No horizontal padding of its own, which the transcript and the input row have taken over
            between them. That is what lets the gutter below reach the edge of the panel. */}
        <div
          className={`relative flex min-h-0 w-full flex-1 flex-col overflow-hidden rounded-r-xl bg-slate-50 pb-0 pt-0 text-left text-xs text-slate-400 sm:rounded-xl sm:pb-10 sm:pt-1 ${UNDERLAP_CLASS} ${GUTTER_CLASS}`}
        >
          {/* The gutter: the whole reason the page reads as two columns. Everything on the left --
              the monitor, the wheels, the trait pills -- is laid out in the same rows as the
              replies, so the only thing dividing the page is this stripe of colour behind their
              left end. Full height of the panel rather than of the transcript, so the division
              still holds level with the box you type in, which has nothing to its left.

              Behind the content by being the only unpositioned thing here: its siblings are
              `relative`, so they paint over it without a stack of z-indexes. It fades in with the
              first message on the curve the rest of the page settles with, since with nothing said
              yet it would be a bare stripe beside a greeting. */}
          <div
            aria-hidden
            className={`${NT_GUTTER_CLASS} pointer-events-none absolute inset-y-0 left-0 transition-opacity duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] sm:rounded-xl ${GUTTER_TINT}`}
            style={{ width: GUTTER, opacity: hasMessages ? 1 : 0 }}
          />

          {/* min-h-0 is what lets this be shorter than its own content, so the overflow becomes
              scrollable rather than squashing the messages.

              The negative margin is what puts the conversation behind the box you type in instead
              of ending at it. A flex item's negative bottom margin takes nothing off where its
              siblings land -- the input row sits exactly where it did -- but it does let this box
              claim that much more height, so it runs to the foot of the panel and the input row,
              which comes after it and is positioned, paints over the end of it.

              Not while the page is still empty: with nothing said the panel is only as tall as what
              is in it, so a negative margin there would shorten the panel rather than lengthen this,
              and the greeting would come out underneath the input. It grows in on the same curve as
              the rest of the page settling. */}
          <div
            ref={scrollContainerRef}
            id={NT_CHAT_ID}
            data-saved-id={currentSavedId}
            data-generating={isGenerating || undefined}
            onScroll={onChatScroll}
            className="relative min-h-0 flex-1 overflow-y-auto transition-[margin-bottom] duration-500 ease-[cubic-bezier(0.16,1,0.3,1)]"
            style={{ marginBottom: hasMessages ? `calc(-1 * ${UNDERLAP})` : 0 }}
          >
            {/* At least as tall as the scroller, and a column so the transcript inside it can take
                that height, which the pinned monitor needs in a short chat. The top padding must
                equal the monitor's `CHART_TOP`. */}
            <div
              ref={chatContentRef}
              className={`flex min-h-full flex-col pt-4 text-[14px] font-medium leading-normal text-slate-600 ${
                // No padding under the transcript, which carries its own tail instead. The greeting
                // is not the transcript and still wants room between it and the box below it.
                hasMessages ? '' : 'pb-6'
              }`}
            >
              {!isGenerating && chatMessages.length === 0 && (
                <div className="mx-auto flex w-full max-w-xl flex-col gap-y-3 px-4 text-left sm:px-0">
                  {INTRO_CARDS.map((card) => (
                    <div key={card.title} className={`flex flex-col gap-y-2 rounded-xl px-4 py-3 ${card.cardClass}`}>
                      <div
                        className={`text-center text-[11px] font-semibold uppercase leading-none tracking-wide ${card.titleClass}`}
                      >
                        {card.title}
                      </div>
                      <p className={`text-[13px] font-normal leading-snug ${card.bodyClass}`}>{card.body}</p>
                    </div>
                  ))}
                  <p className="mt-1 text-center text-[13px] font-normal text-slate-500">
                    Try {NT_TITLE} with an AI model, Llama 3.1 8B.
                  </p>
                </div>
              )}
              <NeuralTransparencyMessages
                scrollContainerRef={scrollContainerRef}
                chatMessages={chatMessages}
                axes={axes}
                readouts={readouts}
                awaitingResponse={awaitingResponse}
                isGenerating={isGenerating}
              />
            </div>
          </div>

          {/* The band the transcript passes behind, between the scroller and the input row so that
              it covers the one and is covered by the other.

              Blur alone, with no wash over it: the two backgrounds it spans are the whole of what
              makes this page look like two columns, and anything laid over both would pull them
              towards each other exactly where the division most needs to hold. What softens the
              text is the blur, and what removes it is the mask -- a ramp from nothing at the top to
              full over the lead-in, held there for the rest. Masking the element is what lets a
              backdrop filter fade at all; opacity on it would fade a transparent box and leave the
              blur underneath at full strength the whole way up.

              Only once there is a transcript to pass behind, and never catching the pointer: a band
              this tall over the foot of the scroller would otherwise swallow a scroll begun on it. */}
          {hasMessages && (
            <div
              aria-hidden
              className="pointer-events-none absolute inset-x-0 bottom-0 backdrop-blur-md"
              style={{
                height: `calc(${UNDERLAP} + ${INPUT_BLUR_LEAD}px)`,
                maskImage: `linear-gradient(to bottom, transparent 0, black ${INPUT_BLUR_LEAD}px)`,
                WebkitMaskImage: `linear-gradient(to bottom, transparent 0, black ${INPUT_BLUR_LEAD}px)`,
              }}
            />
          )}

          <div className="relative flex w-full shrink-0 flex-row pb-3 pt-1">
            {/* The gutter's share of this row, held empty. The box you type in belongs to the chat,
                so it is centred on the chat rather than on the panel -- centred on the panel it
                would sit half under the monitor and read as belonging to both. Grown rather than
                swapped in, on the same curve the page settles with, so the box travels to the right
                as the first message lands instead of jumping there. */}
            <div
              aria-hidden
              className="hidden shrink-0 transition-[width] duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] sm:block"
              style={{ width: hasMessages ? GUTTER_WIDTH : 0 }}
            />
            {/* Room on the left for the undo and clear buttons, which hang off this box rather than
                sitting in it, and the same room again on the right so that what ends up centred is
                the box and its buttons together. */}
            <div className="flex min-w-0 flex-1 flex-col items-center justify-center px-2 sm:px-14">
              <div
                id={NT_INPUT_ID}
                className="relative flex w-full flex-row items-center justify-end pl-11 sm:max-w-xl sm:justify-center sm:pl-0"
              >
                <div
                  className={`${NT_CHAT_TOOL_CLASS} absolute left-12 flex -translate-x-full flex-col gap-y-2 pr-3 sm:left-0`}
                >
                  <button
                    type="button"
                    title="Undo last message"
                    disabled={chatMessages.length < 2 || isGenerating}
                    onClick={() => {
                      const lastUserMessage = chatMessages.filter((m) => m.role === 'user').pop();
                      const remaining = emptyIfNoQuestion(chatMessages.slice(0, -2));
                      setChatMessages(remaining);
                      setChatFull(false);
                      // A row saved earlier in this visit, if any. Otherwise the URL names no chat,
                      // rather than one longer than what is on screen.
                      showSavedId(savedIdByTurnsRef.current.get(assistantTurnCount(remaining)));
                      // The turn being undone was measured, so drop its reading too rather than leave
                      // the wheel showing a reply that is no longer on screen.
                      setReadouts(readouts.map((r) => ({ ...r, turns: (r.turns ?? []).slice(0, -1) })));
                      if (lastUserMessage) {
                        setTypedInText(lastUserMessage.content);
                      }
                    }}
                    className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-slate-300 text-slate-600 shadow hover:bg-slate-200 disabled:cursor-default disabled:text-slate-400 disabled:hover:bg-slate-300"
                  >
                    <Undo2 className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    title="Clear chat"
                    disabled={chatMessages.length === 0 || isGenerating}
                    onClick={() => {
                      // eslint-disable-next-line no-alert
                      if (confirm('Are you sure you want to reset the chat?')) {
                        reset();
                      }
                    }}
                    className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-rose-200 text-rose-700 shadow hover:bg-rose-100 disabled:cursor-default disabled:text-rose-400 disabled:hover:bg-rose-200"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
                {chatFull ? (
                  <div className="flex h-[90px] w-full flex-1 flex-col items-center justify-center gap-y-2 rounded-xl border border-amber-200 bg-amber-50 px-4 text-center shadow-md sm:h-[100px]">
                    <div className="text-[13px] font-semibold text-amber-700">Maximum chat length reached</div>
                    <button
                      type="button"
                      onClick={reset}
                      className="rounded-full bg-amber-600 px-4 py-1.5 text-[11px] font-semibold text-white shadow-sm hover:bg-amber-500"
                    >
                      Start New Chat
                    </button>
                  </div>
                ) : (
                  <>
                    <ReactTextareaAutosize
                      name="chatInput"
                      disabled={isGenerating}
                      readOnly={draftLocked}
                      value={typedInText}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey && !isGenerating) {
                          sendChat();
                          e.preventDefault();
                        }
                      }}
                      onChange={(e) => {
                        setTypedInText(e.target.value);
                      }}
                      required
                      placeholder="Ask or say something..."
                      className={`mt-0 h-[90px] max-h-[90px] min-h-[90px] w-full flex-1 resize-none rounded-xl border border-slate-200 px-4 py-3.5 pr-10 text-left text-xs font-medium placeholder-slate-400 ${
                        draftLocked ? 'bg-slate-100 text-slate-500' : 'bg-white text-slate-800'
                      } shadow-md transition-all focus:border-slate-300 focus:shadow-lg focus:outline-none focus:ring-0 disabled:border-slate-200 disabled:bg-slate-200 sm:h-[100px] sm:max-h-[100px] sm:min-h-[100px] sm:text-[13px]`}
                    />
                    <button
                      type="button"
                      onClick={() => {
                        if (!isGenerating) {
                          sendChat();
                        } else {
                          stopGenerating();
                        }
                      }}
                      className="absolute right-2 flex h-full cursor-pointer items-center justify-center sm:right-4"
                    >
                      {!isGenerating ? (
                        <ArrowUp className="h-8 w-8 rounded-full bg-sky-700 p-1.5 text-white hover:bg-sky-800" />
                      ) : (
                        <X className="h-8 w-8 rounded-full bg-red-400 p-1.5 text-white hover:bg-red-600" />
                      )}
                    </button>
                    {limitRemaining !== null && (
                      <div className="absolute bottom-2 right-2 text-[9px] text-slate-300">
                        {limitRemaining > 0
                          ? `Hourly Limit Left: ${limitRemaining}`
                          : 'Out of messages. Wait a bit and try again later.'}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div aria-hidden className={`${GROW_SHIFT} w-full`} style={{ flexGrow: hasMessages ? 0 : 2 }} />
    </div>
  );
}
