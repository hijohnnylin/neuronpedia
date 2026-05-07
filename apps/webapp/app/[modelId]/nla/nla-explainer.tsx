'use client';

import CustomTooltip from '@/components/custom-tooltip';
import { useNlaContext } from '@/components/provider/nla-provider';
import { MAX_TEXT_LENGTH, NLA_FREE_CHAT_DEMO_CACHE_ID } from '@/lib/nla-constants';
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { BookOpenIcon, GithubIcon, MailIcon, ScrollIcon, YoutubeIcon } from 'lucide-react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';
import NLADetailsColumn from './nla-details-column';
import NLAInputChat from './nla-input-chat';
import NlaIntroVideoModal from './nla-intro-video-modal';
import { useNlaTour } from './nla-tour';
import { NLA_DETAILS_ELEMENT_ID, NLA_TOUR_GUIDE_BUTTON_ELEMENT_ID } from './nla-tour-constants';
import { NLA_BLOG_URL, NLA_CONTACT_EMAIL, NLA_GITHUB_URL, NLA_PAPER_URL } from './nla-urls';
import ShareModal from './share-modal';

const NLA_TOUR_SEEN_KEY = 'nla-tour-seen';

const NLA_MODEL_TOGGLE_OPTIONS = [
  {
    modelId: 'llama3.3-70b-it',
    displayName: 'Llama 70B',
    layer: 53,
    url: '/llama3.3-70b-it/nla',
    hoverInfo: (
      <div className="flex flex-col gap-y-1.5">
        <p>
          <strong>Model:</strong> <code>Llama 3.3 70B Instruct</code>
        </p>
        <p>
          <strong>Activation Verbalizer:</strong>{' '}
          <a href="https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-av" target="_blank" rel="noopener noreferrer ">
            <code className="text-sky-600 hover:underline">kitft/Llama-3.3-70B-NLA-L53-av</code>
          </a>
        </p>
        <p>
          <strong>Activation Reconstructor:</strong>{' '}
          <a href="https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-ar" target="_blank" rel="noopener noreferrer">
            <code className="text-sky-600 hover:underline">kitft/Llama-3.3-70B-NLA-L53-ar</code>
          </a>
        </p>
        <p className="flex flex-col gap-y-1">
          <strong>Deployment</strong>
          <ul className="ml-4 list-inside list-disc">
            <li>
              <strong>Source Model:</strong> <code>fp8-quantized</code>
            </li>
            <li>
              <strong>Verbalizer:</strong> <code>fp8-quantized</code>
            </li>
            <li>
              <strong>Reconstructor*:</strong> <code>fp8-quantized</code>
            </li>
          </ul>
        </p>
        <p className="max-w-[320px] text-[10px] text-slate-500">
          *Quantization may have non-insignificant effects on the reconstructor&apos;s RMSE scores. Use with caution.
        </p>
      </div>
    ),
  },
  {
    modelId: 'gemma-3-27b-it',
    displayName: 'Gemma 27B',
    layer: 41,
    url: '/gemma-3-27b-it/nla',
    hoverInfo: (
      <div className="flex flex-col gap-y-1.5">
        <p>
          <strong>Model:</strong> <code>Gemma 3 27B Instruct</code>
        </p>
        <p>
          <strong>Activation Verbalizer:</strong>{' '}
          <a href="https://huggingface.co/kitft/nla-gemma3-27b-L41-av" target="_blank" rel="noopener noreferrer ">
            <code className="text-sky-600 hover:underline">kitft/nla-gemma3-27b-L41-av</code>
          </a>
        </p>
        <p>
          <strong>Activation Reconstructor:</strong>{' '}
          <a href="https://huggingface.co/kitft/nla-gemma3-27b-L41-ar" target="_blank" rel="noopener noreferrer">
            <code className="text-sky-600 hover:underline">kitft/nla-gemma3-27b-L41-ar</code>
          </a>
        </p>
        <p className="flex flex-col gap-y-1">
          <strong>Deployment</strong>
          <ul className="ml-4 list-inside list-disc">
            <li>
              <strong>Source Model:</strong> <code>fp8-quantized</code>
            </li>
            <li>
              <strong>Verbalizer:</strong> <code>fp8-quantized</code>
            </li>
            <li>
              <strong>Reconstructor*:</strong> <code>int4-quantized</code>
            </li>
          </ul>
        </p>
        <p className="max-w-[320px] text-[10px] leading-tight text-slate-500">
          *Quantization may have non-insignificant effects on the reconstructor&apos;s RMSE scores. Use with caution.
        </p>
      </div>
    ),
  },
] as const;

export default function NLAExplainer() {
  const {
    isEmbed,
    selectedModelId,
    handleModelChange,
    isChatStreaming,
    isLoading,
    isHydratingDemo,
    chatMessages,
    tokenizerFormat,
    activeDemoCacheId,
    handleClear,
    loadCacheById,
    isShareModalOpen,
    setIsShareModalOpen,
    shareDraft,
    shareError,
    setHighlightComment,
    setHighlightedRange,
    featuredDemos,
  } = useNlaContext();

  const searchParams = useSearchParams();
  const initialCacheId = searchParams.get('id');

  // Default to `true` so the red dot doesn't flash for returning users
  // before the localStorage check runs in the mount effect below.
  const [hasSeenTour, setHasSeenTour] = useState(true);
  const [isIntroVideoOpen, setIsIntroVideoOpen] = useState(false);

  const demosForModel = featuredDemos.filter((d) => d.modelId === selectedModelId);

  // Mirrors the chat panel's `isBusy`: disable demo-load and model-switch
  // while a chat stream, explanation, or demo hydrate is in flight so the
  // user can't race new state into a partially-rendered run.
  const isBusy = isChatStreaming || isLoading || isHydratingDemo;

  const startTour = useNlaTour({
    selectedModelId,
    handleModelChange,
    loadCacheById,
    setHighlightedRange,
  });

  const markTourSeen = useCallback(() => {
    try {
      localStorage.setItem(NLA_TOUR_SEEN_KEY, 'true');
    } catch (error) {
      console.error('Error setting localStorage:', error);
    }
    setHasSeenTour(true);
  }, []);

  const handleStartTour = () => {
    markTourSeen();
    handleClear();
    startTour();
  };

  // Auto-start the tour for first-time visitors. Skipped when the user
  // is deep-linking to a saved cache (`?id=...`), inside an embed, or
  // on a small viewport where the spotlight popovers don't fit. On
  // subsequent visits we just sync `hasSeenTour` from localStorage so
  // the Guide button can hide its "new" indicator.
  useEffect(() => {
    try {
      const seen = localStorage.getItem(NLA_TOUR_SEEN_KEY) === 'true';
      setHasSeenTour(seen);
      if (seen) return;
      if (isEmbed) return;
      if (initialCacheId) return;
      if (typeof window !== 'undefined' && window.innerWidth < 640) return;
      markTourSeen();
      startTour();
    } catch (error) {
      console.error('Error checking localStorage:', error);
    }
    // Only run on mount — subsequent URL changes (caused by hydrate)
    // shouldn't toggle the tour.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex min-h-0 w-full flex-1 flex-col overflow-x-hidden bg-slate-50">
      <div className="flex w-full items-center justify-center bg-slate-100 px-4 pt-3 sm:px-6 sm:pb-6 sm:pt-6">
        <div className="relative flex min-h-20 w-full flex-row items-center justify-between gap-0 sm:min-h-0">
          <button
            type="button"
            id={NLA_TOUR_GUIDE_BUTTON_ELEMENT_ID}
            onClick={handleStartTour}
            className="absolute left-0 top-0 z-20 flex h-7 w-14 items-center justify-center gap-x-1 rounded-md border border-emerald-600 bg-emerald-50 px-3 py-1 text-[11px] font-medium text-emerald-600 transition-colors hover:bg-emerald-100 sm:hidden"
          >
            Tutorial
            {!hasSeenTour && (
              <span
                aria-hidden="true"
                className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full bg-red-500 ring-2 ring-slate-100"
              />
            )}
          </button>
          <div className="absolute left-1/2 top-1 -ml-6 flex w-64 -translate-x-1/2 flex-col items-center justify-center gap-y-0 sm:relative sm:left-0 sm:ml-0 sm:w-full sm:translate-x-0 sm:items-start">
            <div
              className="whitespace-nowrap text-[11px] font-semibold leading-none text-slate-800 sm:-mt-2 sm:mb-0.5 sm:text-base"
              id="chat"
            >
              Natural Language Autoencoders
            </div>
            <CustomTooltip
              wide
              trigger={
                <div className="mb-2 cursor-pointer whitespace-nowrap text-[8px] text-slate-500 sm:mt-0.5 sm:text-[11px]">
                  Fraser-Taliente, Kantamneni, Ong et al. <span className="hidden sm:inline">2026</span>
                </div>
              }
              side="right"
              delayDuration={100}
            >
              <div>
                Kit Fraser-Taliente*, Subhash Kantamneni*‡, Euan Ong*, Daniel Mossing, Christina Lu, Paul C. Bogdan,
                Emmanuel Ameisen, James Chen, Dzmitry Kishylau, Adam Pearce, Julius Tarng, Alex Wu, Jeff Wu, Yang Zhang,
                Daniel M. Ziegler, Evan Hubinger, Joshua Batson, Jack Lindsey, Samuel Zimmerman, Samuel Marks
              </div>
            </CustomTooltip>
            <div className="hidden flex-row items-center justify-center gap-x-1 sm:flex">
              <button
                type="button"
                id={NLA_TOUR_GUIDE_BUTTON_ELEMENT_ID}
                onClick={handleStartTour}
                className="flex w-24 items-center justify-center gap-x-1 rounded-md border border-emerald-600 bg-emerald-50 px-3 py-1 text-[11px] font-medium text-emerald-600 transition-colors hover:bg-emerald-100"
              >
                <QuestionMarkCircledIcon className="h-3 w-3" />
                Tutorial
                {!hasSeenTour && (
                  <span
                    aria-hidden="true"
                    className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full bg-red-500 ring-2 ring-slate-100"
                  />
                )}
              </button>
              <Link
                href={NLA_PAPER_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex w-24 items-center justify-center gap-x-1 rounded-md border border-slate-500 bg-white px-3 py-1 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-50"
              >
                <ScrollIcon className="h-3 w-3" />
                Paper
              </Link>
            </div>
          </div>

          <div
            className={`absolute left-0 top-0 flex w-full flex-col items-center justify-center gap-x-5 gap-y-2 sm:-top-2.5 sm:left-1/2 sm:w-auto sm:-translate-x-1/2`}
          >
            <div
              className="flex w-full flex-col items-end justify-center gap-y-1 sm:w-auto sm:items-center sm:gap-y-1.5"
              id="nla-header"
            >
              <ToggleGroup.Root
                className="z-10 inline-flex h-7 overflow-hidden border-slate-200 data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50"
                type="single"
                value={selectedModelId}
                disabled={isBusy}
                onValueChange={(value) => {
                  if (value) handleModelChange(value);
                }}
                aria-label="Model"
              >
                {NLA_MODEL_TOGGLE_OPTIONS.map((opt) => (
                  <ToggleGroup.Item
                    key={opt.modelId}
                    value={opt.modelId}
                    aria-label={`${opt.displayName} (Layer ${opt.layer})`}
                    className="relative flex cursor-pointer items-center justify-center border bg-white px-2 text-[11px] font-medium text-slate-500 transition-colors first:rounded-l-lg last:rounded-r-lg hover:border-sky-700 hover:bg-sky-100 data-[state=on]:border data-[state=on]:border-sky-700 data-[state=on]:bg-sky-700 data-[state=off]:text-slate-500 data-[state=on]:text-white sm:gap-x-2 sm:px-5"
                  >
                    {/* Split displayName by space */}
                    <span className="sm:hidden">{opt.displayName.split(' ')[0]}</span>
                    <span className="hidden sm:inline">{opt.displayName}</span>

                    <div className="hidden items-center justify-center gap-x-1 sm:flex">
                      <span className="text-[8px] font-medium uppercase transition-colors">Layer {opt.layer}</span>
                    </div>
                    <CustomTooltip
                      side="bottom"
                      delayDuration={0}
                      wide
                      trigger={
                        <span
                          role="button"
                          tabIndex={-1}
                          aria-label={`About ${opt.displayName}`}
                          onClick={(e) => e.stopPropagation()}
                          onPointerDown={(e) => e.stopPropagation()}
                          className="right-2.5 top-[6.5px] hidden items-center text-current opacity-70 transition-opacity hover:opacity-100 sm:absolute sm:block"
                        >
                          <QuestionMarkCircledIcon className="h-3 w-3" />
                        </span>
                      }
                    >
                      <div>{opt.hoverInfo}</div>
                    </CustomTooltip>
                  </ToggleGroup.Item>
                ))}
              </ToggleGroup.Root>
              <div className="flex w-full flex-row items-end justify-end gap-x-[0px] gap-y-[1px] rounded-lg sm:-mt-5 sm:justify-center sm:rounded-2xl sm:bg-slate-200 sm:px-2 sm:py-2 sm:pt-5">
                <div className="flex w-full max-w-[calc(100dvw-90px)] flex-row items-center justify-center gap-x-[0px] gap-y-[1px] divide-x divide-slate-200 overflow-x-scroll rounded-lg border border-slate-200 sm:w-auto sm:max-w-none sm:overflow-hidden sm:rounded-xl sm:border-none">
                  {demosForModel.map((demo, index) => {
                    const label = demo.featuredDisplayName ?? demo.shareId;
                    const isActive = activeDemoCacheId === demo.cacheId;
                    return (
                      <button
                        key={demo.shareId}
                        type="button"
                        onClick={() =>
                          loadCacheById(
                            demo.cacheId,
                            demo.position ?? undefined,
                            demo.paragraph ?? undefined,
                            demo.highlightStart ?? undefined,
                            demo.highlightEnd ?? undefined,
                            demo.comment?.trim() ? demo.comment : undefined,
                          )
                        }
                        disabled={isBusy}
                        className={`${index === demosForModel.length - 1 ? 'hidden' : ''} flex-1 py-1.5 text-[11px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 sm:w-[105px] sm:min-w-[105px] sm:max-w-[105px] sm:flex-auto ${
                          isActive
                            ? 'border-sky-500 bg-sky-700 text-white'
                            : 'border-slate-200 bg-white text-slate-600 hover:bg-sky-100'
                        }`}
                      >
                        {(() => {
                          const firstSpace = label.indexOf(' ');
                          if (firstSpace === -1) {
                            return <span className="text-sm">{label}</span>;
                          }
                          const first = label.slice(0, firstSpace);
                          const second = label.slice(firstSpace + 1);
                          return (
                            <span className="flex flex-row items-center justify-center gap-x-1 px-1.5 sm:gap-x-2 sm:px-2">
                              <span className="text-sm sm:text-xl">{first}</span>
                              {(() => {
                                const secondFirstSpace = second.indexOf(' ');
                                if (secondFirstSpace === -1) {
                                  return <span className="text-[9px] sm:text-[10px]">{second}</span>;
                                }
                                const secondFirst = second.slice(0, secondFirstSpace);
                                const secondRest = second.slice(secondFirstSpace + 1);
                                return (
                                  <span className="flex flex-col text-[9px] leading-tight sm:text-[10px]">
                                    <span className="whitespace-nowrap">{secondFirst}</span>
                                    <span>{secondRest}</span>
                                  </span>
                                );
                              })()}
                            </span>
                          );
                        })()}
                      </button>
                    );
                  })}
                </div>
                <button
                  key="nla-free-chat"
                  type="button"
                  onClick={() => handleClear({ pinFreeChatDemo: true })}
                  disabled={isBusy}
                  className={`ml-1 rounded-lg border border-slate-200 px-1 py-1.5 text-[11px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 sm:ml-2 sm:w-[90px] sm:min-w-[90px] sm:max-w-[90px] sm:rounded-xl sm:border-none sm:py-1.5 ${
                    activeDemoCacheId === NLA_FREE_CHAT_DEMO_CACHE_ID
                      ? 'border-sky-500 bg-sky-700 text-white'
                      : 'border-slate-200 bg-white text-slate-600 hover:bg-sky-100'
                  }`}
                >
                  <span className="flex flex-row items-center justify-center gap-x-1 px-1.5 sm:gap-x-2.5 sm:px-2">
                    <span className="text-xs sm:text-xl">✏️</span>
                    <span className="text-[9px] leading-tight sm:text-[11px]">
                      Free
                      <br />
                      Chat
                    </span>
                  </span>
                </button>
              </div>
            </div>
          </div>

          <div className="hidden flex-col items-center justify-center sm:flex">
            <div className="flex flex-row items-center justify-center gap-0.5" id="nla-right-buttons">
              <div className="flex flex-1 flex-col gap-y-0.5">
                <Link
                  href={NLA_BLOG_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex min-w-28 items-center justify-center gap-x-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
                >
                  <BookOpenIcon className="h-3.5 w-3.5" />
                  Blog
                </Link>

                <Link
                  href={NLA_GITHUB_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex min-w-28 items-center justify-center gap-x-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
                >
                  <GithubIcon className="h-3.5 w-3.5" />
                  Code
                </Link>
              </div>
              <div className="flex flex-1 flex-col gap-y-0.5">
                <button
                  type="button"
                  onClick={() => setIsIntroVideoOpen(true)}
                  className="flex min-w-28 items-center justify-center gap-x-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
                >
                  <YoutubeIcon className="h-3.5 w-3.5" />
                  Video
                </button>
                <Link
                  href={`mailto:${NLA_CONTACT_EMAIL}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex min-w-28 items-center justify-center gap-x-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
                >
                  <MailIcon className="h-3.5 w-3.5" />
                  Contact
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Input / Results section */}
      <div
        className={`mx-auto flex min-h-0 w-full max-w-screen-xl flex-1 flex-col items-stretch justify-start bg-slate-50 px-2 pb-2 sm:px-6 sm:pt-3`}
      >
        <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col gap-y-1">
          <div className={`${isEmbed ? 'mt-0' : 'mt-2'} flex min-h-0 min-w-0 flex-1 flex-col gap-y-1`}>
            <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col gap-y-1 sm:flex-row sm:gap-x-5">
              <div className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-x-hidden">
                <NLAInputChat />
              </div>
              <div id={NLA_DETAILS_ELEMENT_ID} className="flex min-h-0 min-w-0 max-w-sm flex-1 basis-0 flex-col">
                {/* Errors (including NLA server / rate-limit 429s) are now
                    rendered above the chat textarea on the left side via
                    `nla-input-chat.tsx` — see `pendingChatInputRestore`
                    handling. Keeping the error close to the input keeps
                    failure recovery in one focused location. */}
                <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                  <NLADetailsColumn />
                </div>
              </div>
            </div>
            {(() => {
              const len = tokenizerFormat.formatChat(chatMessages).length;
              return len > MAX_TEXT_LENGTH ? (
                <span className="text-xs text-red-500">
                  Text must be {MAX_TEXT_LENGTH} characters or less (currently {len})
                </span>
              ) : null;
            })()}
          </div>
        </div>
      </div>

      <ShareModal
        open={isShareModalOpen}
        onOpenChange={setIsShareModalOpen}
        shareDraft={shareDraft}
        shareError={shareError}
        onCommentCommit={setHighlightComment}
      />

      <NlaIntroVideoModal open={isIntroVideoOpen} onOpenChange={setIsIntroVideoOpen} />
    </div>
  );
}
