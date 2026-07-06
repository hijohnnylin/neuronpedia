'use client';

import JlensChat from '@/components/jlens/jlens-chat';
import JlensCompletion from '@/components/jlens/jlens-completion';
import { JlensExport, JlensExportChat, JlensExportCompletion, parseFixture } from '@/components/jlens/jlens-export';
import { LensMode, LensModeContext, LensModeSetContext } from '@/components/jlens/jlens-lens-mode';
import JlensPanel from '@/components/jlens/jlens-panel';
import { useGlobalContext } from '@/components/provider/global-provider';
import { LoadingSquare } from '@/components/svg/loading-square';
import { ChineseTranslationsProvider } from '@/lib/utils/chinese-translations';
import { JlensShareLockedToken, JlensShareUiState } from '@/lib/utils/jlens-share';
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import { BookOpenIcon, GithubIcon, MailIcon, Newspaper, Scale, YoutubeIcon } from 'lucide-react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { useCallback, useEffect, useMemo, useState, useTransition } from 'react';
import JlensIntroVideoModal from './jlens-intro-video-modal';
import JlensModelSelector, { getNextJlensDemo } from './jlens-model-selector';
import { useJlensTour } from './jlens-tour';
import { JLENS_TOUR_GUIDE_BUTTON_ELEMENT_ID, JLENS_TOUR_SEEN_KEY } from './jlens-tour-constants';
import { JlensTourStepContext } from './jlens-tour-context';
import { JLENS_BLOG_URL, JLENS_CONTACT_EMAIL, JLENS_GITHUB_URL, JLENS_HF_URL, JLENS_PAPER_URL } from './jlens-urls';

// Failsafe for the navigation loading gate: if a `router.push` transition never
// settles (App Router state desync) and the URL never lands on the requested
// target, drop the skeleton after this long so the UI can't stay stuck. Long
// enough not to trip on genuinely slow navigations.
const NAV_WATCHDOG_MS = 5000;

// A loaded share (resolved server-side from `?shareId=`). The heavy run data
// lives at `url` (gzipped S3 blob, fetched client-side); the rest is UI state.
export interface JlensShareData {
  url: string;
  description: string | null;
  descriptionAttribution: string | null;
  lockedTokens: JlensShareLockedToken[];
  selectedPositions: number[];
  activeLensModeTab: string;
  topN: number;
  hideNonWordTokens: boolean;
  temperature: number;
  numCompletionTokens: number;
  numPromptTokens: number | null;
}

export default function JlensPageClient({
  modelId,
  share = null,
  inferenceAvailable = true,
}: {
  modelId: string;
  share?: JlensShareData | null;
  inferenceAvailable?: boolean;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { getInferenceEnabledForModel } = useGlobalContext();

  // The shareId currently reflected in the UI (drives the active-demo highlight
  // in the model selector). Mirrored into state — rather than read straight from
  // the URL every render — so editing a shared run's sidebar selection can drop
  // it (below) in place, without a real navigation. Re-synced whenever a genuine
  // navigation changes the URL's `?shareId=` (loading a share/demo, back/forward).
  const urlShareId = searchParams.get('shareId');
  const [currentShareId, setCurrentShareId] = useState<string | null>(urlShareId);
  useEffect(() => {
    setCurrentShareId(urlShareId);
  }, [urlShareId]);

  // Editing the sidebar selection on a shared run diverges from the shared
  // snapshot, so drop the shareId from both the URL (in place, no reload — the
  // loaded data stays on screen) and the tracked state (clears the active-demo
  // highlight).
  const clearShareId = useCallback(() => {
    setCurrentShareId(null);
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    if (!params.has('shareId')) return;
    params.delete('shareId');
    const query = params.toString();
    window.history.replaceState(null, '', `${window.location.pathname}${query ? `?${query}` : ''}`);
  }, []);
  const [lensMode, setLensMode] = useState<LensMode>(
    share && (share.activeLensModeTab === LensMode.LOGIT_LENS || share.activeLensModeTab === LensMode.DIFF)
      ? (share.activeLensModeTab as LensMode)
      : LensMode.JACOBIAN_LENS,
  );

  // Leaving a shared run for free chat (e.g. the model selector's "Free Chat"
  // button, or advancing past the last demo) resets the lens mode back to
  // Jacobian. Without this, a demo that pinned Logit-Lens/Diff (this component
  // stays mounted across the same-route navigation, so `lensMode` persists)
  // would leave free chat stuck on that mode. Loading a share instead restores
  // its own mode via the child's loaded-data effect, so we only reset here.
  const isShareActive = share != null;
  useEffect(() => {
    if (!isShareActive) {
      setLensMode(LensMode.JACOBIAN_LENS);
    }
  }, [isShareActive]);

  // `modelId`/`inferenceAvailable` come from the server component. We mirror
  // them into state so a model swap can happen client-side (no server round
  // trip / page flash) while still keeping the URL in sync. The effect re-syncs
  // whenever a real navigation occurs (loading a share, back/forward, demos).
  const [activeModelId, setActiveModelId] = useState(modelId);
  const [activeInferenceAvailable, setActiveInferenceAvailable] = useState(inferenceAvailable);

  // Demo / free-chat / cross-model-share navigations are real server round trips
  // (they change `?shareId=` on the same pathname, so no route loading boundary
  // fires). Driving `router.push` through a transition lets us show the loading
  // skeleton the instant a demo/free-chat button is clicked, rather than only
  // once the server responds and `JlensShareView` mounts.
  //
  // We gate the skeleton on our own `pendingNavHref` rather than the
  // transition's `isPending`: the App Router's internal state can desync (e.g.
  // after the client-side `window.history` updates below), and an occasional
  // `router.push` then leaves the transition pending forever — which would
  // strand the skeleton on both panels until another navigation. Tracking the
  // requested href ourselves lets us clear the gate as soon as the URL reflects
  // the target (normal case), with a watchdog timeout as a last resort.
  const [, startNavTransition] = useTransition();
  const [pendingNavHref, setPendingNavHref] = useState<string | null>(null);
  const navigate = useCallback(
    (href: string) => {
      setPendingNavHref(href);
      startNavTransition(() => router.push(href));
    },
    [router],
  );

  // Clear the navigation gate once the URL actually reflects the requested
  // target. Falls back to a watchdog so a transition that never settles can't
  // leave the loading skeleton up indefinitely.
  useEffect(() => {
    if (!pendingNavHref) return undefined;
    let targetModelId: string | null = null;
    let targetShareId: string | null = null;
    try {
      const url = new URL(pendingNavHref, window.location.origin);
      targetModelId = url.pathname.split('/').filter(Boolean)[0] ?? null;
      targetShareId = url.searchParams.get('shareId');
    } catch {
      // Malformed href — drop the gate rather than risk stranding it.
      setPendingNavHref(null);
      return undefined;
    }
    const landed =
      (targetModelId == null || targetModelId === modelId) && (targetShareId ?? null) === (urlShareId ?? null);
    if (landed) {
      setPendingNavHref(null);
      return undefined;
    }
    const timeout = setTimeout(() => setPendingNavHref(null), NAV_WATCHDOG_MS);
    return () => clearTimeout(timeout);
  }, [pendingNavHref, modelId, urlShareId]);

  useEffect(() => {
    setActiveModelId(modelId);
    setActiveInferenceAvailable(inferenceAvailable);
  }, [modelId, inferenceAvailable]);

  const handleModelSwap = (newModelId: string) => {
    if (newModelId === activeModelId) {
      return;
    }
    // Leaving a shared run needs a real navigation so the server can drop the
    // resolved share; only plain (non-share) swaps take the fast client path.
    if (share) {
      navigate(`/${newModelId}/jlens`);
      return;
    }
    setActiveModelId(newModelId);
    setActiveInferenceAvailable(getInferenceEnabledForModel(newModelId));
    window.history.pushState(null, '', `/${newModelId}/jlens`);
  };

  const [isIntroVideoOpen, setIsIntroVideoOpen] = useState(false);

  const { startTour, activeStep: activeTourStep } = useJlensTour({ navigate });

  // Defaults to `true` so the "new" dot on the Tutorial button doesn't flash
  // before the localStorage check runs on mount.
  const [hasSeenTour, setHasSeenTour] = useState(true);

  const markTourSeen = useCallback(() => {
    try {
      localStorage.setItem(JLENS_TOUR_SEEN_KEY, 'true');
    } catch (error) {
      console.error('Error setting localStorage:', error);
    }
    setHasSeenTour(true);
  }, []);

  const handleStartTour = useCallback(() => {
    markTourSeen();
    startTour();
  }, [markTourSeen, startTour]);

  // Auto-start the tour for first-time visitors, including on mobile. Skipped
  // only when the user is deep-linking to a shared run (`?shareId=...`). On
  // subsequent visits we just sync `hasSeenTour` from localStorage so the
  // Tutorial button can hide its "new" indicator.
  useEffect(() => {
    try {
      const seen = localStorage.getItem(JLENS_TOUR_SEEN_KEY) === 'true';
      setHasSeenTour(seen);
      if (seen) return;
      if (share) return;
      markTourSeen();
      startTour();
    } catch (error) {
      console.error('Error checking localStorage:', error);
    }
    // Only run on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <JlensTourStepContext.Provider value={activeTourStep}>
      <LensModeContext.Provider value={lensMode}>
        <LensModeSetContext.Provider value={setLensMode}>
          <div className="mx-auto flex max-h-[calc(100dvh-48px)] min-h-[calc(100dvh-48px)] w-full flex-col sm:max-h-[calc(100dvh-75px)] sm:min-h-[calc(100dvh-75px)]">
            <div className="flex w-full items-center justify-center bg-slate-100 px-3 pt-3 sm:px-6 sm:pb-6 sm:pt-6">
              <div className="relative flex min-h-20 w-full max-w-screen-2xl flex-row items-center justify-between gap-0 sm:min-h-0">
                <div className="absolute -top-0.5 left-0 flex flex-col items-start justify-center gap-y-0 sm:relative sm:w-full sm:items-start">
                  <div
                    className="whitespace-nowrap text-[14px] font-semibold leading-none text-slate-800 sm:-mt-2 sm:mb-0.5 sm:text-base"
                    id="jlens-header"
                  >
                    Jacobian Lens
                  </div>
                  <div className="mb-1.5 mt-[3px] whitespace-nowrap text-[10px] text-slate-500 sm:mt-0 sm:text-[11.5px]">
                    Gurnee et al.
                  </div>
                  <div className="hidden flex-row items-center justify-center gap-x-1 sm:flex">
                    <button
                      type="button"
                      id={JLENS_TOUR_GUIDE_BUTTON_ELEMENT_ID}
                      onClick={handleStartTour}
                      className="relative flex w-24 items-center justify-center gap-x-1.5 rounded-md border border-emerald-600 bg-emerald-50 px-3 py-1 text-[11px] font-semibold text-emerald-600 transition-colors hover:bg-emerald-100"
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
                      href={JLENS_PAPER_URL}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex w-20 items-center justify-center gap-x-1.5 rounded-md border border-slate-400 bg-white px-3 py-1 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-50"
                    >
                      <Newspaper className="h-3 w-3" />
                      Paper
                    </Link>
                    <button
                      type="button"
                      onClick={() => setIsIntroVideoOpen(true)}
                      className="flex w-20 items-center justify-center gap-x-1.5 rounded-md border border-slate-400 bg-white px-3 py-1 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-50"
                    >
                      <YoutubeIcon className="h-3 w-3" />
                      Video
                    </button>
                  </div>
                </div>

                <JlensModelSelector
                  modelId={activeModelId}
                  currentShareId={currentShareId}
                  onModelChange={handleModelSwap}
                  onNavigate={navigate}
                  mobileLeftSlot={
                    <button
                      type="button"
                      onClick={handleStartTour}
                      className="flex h-7 items-center justify-center gap-x-1 rounded-md border border-emerald-600 bg-emerald-50 px-3 py-1 text-[11px] font-medium text-emerald-600 transition-colors hover:bg-emerald-100"
                    >
                      <QuestionMarkCircledIcon className="hidden h-3 w-3 sm:block" />
                      Tutorial
                    </button>
                  }
                />

                <div className="hidden flex-col items-center justify-center sm:flex">
                  <div className="flex flex-row items-center justify-center gap-0.5" id="jlens-right-buttons">
                    <div className="flex flex-1 flex-col gap-y-0.5">
                      <Link
                        href={JLENS_BLOG_URL}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex min-w-28 items-center justify-center gap-x-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
                      >
                        <BookOpenIcon className="h-3.5 w-3.5" />
                        Blog
                      </Link>

                      <Link
                        href={JLENS_GITHUB_URL}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex min-w-28 items-center justify-center gap-x-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
                      >
                        <GithubIcon className="h-3.5 w-3.5" />
                        Code
                      </Link>
                    </div>
                    <div className="flex flex-1 flex-col gap-y-0.5">
                      <Link
                        href={JLENS_HF_URL}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex min-w-28 items-center justify-center gap-x-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
                      >
                        <Scale className="h-3.5 w-3.5" />
                        Weights
                      </Link>
                      <Link
                        href={`mailto:${JLENS_CONTACT_EMAIL}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex min-w-28 items-center justify-center gap-x-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
                      >
                        <MailIcon className="h-3.5 w-3.5" />
                        Contact
                      </Link>
                    </div>
                    <div className="flex flex-1 flex-col gap-y-0.5"></div>
                  </div>
                </div>
              </div>
            </div>
            <div className="mx-auto flex h-full min-h-0 w-full max-w-screen-xl flex-1 flex-col sm:py-3">
              {pendingNavHref ? (
                <JlensLoadingSkeleton />
              ) : share ? (
                <JlensShareView
                  key={share.url}
                  modelId={modelId}
                  share={share}
                  inferenceAvailable={inferenceAvailable}
                  onSidebarSelectionChange={clearShareId}
                  currentShareId={currentShareId}
                  onNavigate={navigate}
                />
              ) : (
                <JlensPanel key={activeModelId} modelId={activeModelId} inferenceAvailable={activeInferenceAvailable} />
              )}
            </div>
          </div>
          <JlensIntroVideoModal open={isIntroVideoOpen} onOpenChange={setIsIntroVideoOpen} />
        </LensModeSetContext.Provider>
      </LensModeContext.Provider>
    </JlensTourStepContext.Provider>
  );
}

// Blank two-panel skeleton shown while a shared run / demo loads: it mirrors the
// chat/completion + analysis-sidebar split of the real interface (rather than a
// single centered spinner) so the layout doesn't jump, with a loading square
// centered in each panel to signal the load.
function JlensLoadingSkeleton() {
  return (
    <div className="relative flex h-full min-h-full flex-1 flex-col rounded-xl bg-slate-50 sm:flex-row sm:gap-x-3">
      {/* Chat / completion panel skeleton (left, wider). */}
      <div className="flex min-h-0 w-full max-w-screen-lg grow-[3] basis-0 flex-col items-center justify-center bg-slate-200/40 px-3 sm:flex-1 sm:rounded-2xl">
        <LoadingSquare className="h-6 w-6" />
      </div>
      {/* Analysis sidebar panel skeleton (right, 40% on desktop). */}
      <div className="relative flex min-h-0 w-full grow-[3] basis-0 flex-col sm:w-auto sm:min-w-[40%] sm:max-w-[40%] sm:flex-none sm:shrink-0 sm:basis-auto">
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center border border-slate-200 bg-white sm:rounded-xl sm:shadow-lg">
          <LoadingSquare className="h-6 w-6" />
        </div>
      </div>
    </div>
  );
}

// Loads a shared run's S3 blob client-side (with a loading state), then renders
// the matching interface (chat or completion) with the restored UI state +
// sharer commentary.
function JlensShareView({
  modelId,
  share,
  inferenceAvailable,
  onSidebarSelectionChange,
  currentShareId,
  onNavigate,
}: {
  modelId: string;
  share: JlensShareData;
  inferenceAvailable: boolean;
  // Drops the shareId (URL + state) once the user edits the sidebar selection.
  onSidebarSelectionChange?: () => void;
  // The shareId currently reflected in the UI; used to locate the current demo
  // in its model's curated list so the commentary can offer a "Next demo" jump.
  currentShareId: string | null;
  // Navigate (through the page's transition, so the skeleton shows immediately).
  onNavigate: (href: string) => void;
}) {
  const router = useRouter();
  // When on one of the curated demos, offer a button in the commentary that
  // advances through the list (labelled "Free Chat" on the last one).
  const nextDemo = getNextJlensDemo(modelId, currentShareId);
  const onNextDemo = nextDemo ? () => onNavigate(nextDemo.href) : undefined;
  // On non-last demos (where the primary button advances to the next demo),
  // also offer a shortcut straight to free chat. On the last demo the primary
  // button is already "Free Chat", so no secondary button is needed.
  const onFreeChatDemo = nextDemo && nextDemo.label === 'Next demo' ? () => onNavigate(`/${modelId}/jlens`) : undefined;
  const [loadedData, setLoadedData] = useState<JlensExport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const uiState = useMemo<JlensShareUiState>(
    () => ({
      lockedTokens: share.lockedTokens,
      selectedPositions: share.selectedPositions,
      activeLensModeTab: share.activeLensModeTab,
      topN: share.topN,
      hideNonWordTokens: share.hideNonWordTokens,
      temperature: share.temperature,
      numCompletionTokens: share.numCompletionTokens,
      numPromptTokens: share.numPromptTokens,
    }),
    [share],
  );

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    // The blob is stored gzipped with Content-Encoding: gzip, so the browser
    // transparently decompresses it on fetch — we just read JSON.
    fetch(share.url, { cache: 'force-cache' })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Failed to load shared run (${res.status}).`);
        }
        const json = await res.json();
        const parsed = parseFixture(json);
        if (!cancelled) {
          setLoadedData({ ...parsed, uiState });
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [share.url, uiState]);

  if (loading) {
    return <JlensLoadingSkeleton />;
  }
  if (error) {
    return (
      <div className="flex max-h-[calc(100dvh-80px)] min-h-0 min-h-[calc(100dvh-80px)] w-full flex-col items-center justify-center gap-y-2 rounded-xl bg-slate-50">
        <div className="text-sm font-medium text-red-600">Could not load this shared run.</div>
        <div className="max-w-md text-center text-xs text-slate-400">{error}</div>
      </div>
    );
  }
  return (
    <ChineseTranslationsProvider>
      {loadedData?.kind === 'completion' ? (
        <JlensCompletion
          key={share.url}
          modelId={modelId}
          loadedData={loadedData as JlensExportCompletion}
          sharedDescription={share.description}
          sharedAttribution={share.descriptionAttribution}
          inferenceAvailable={inferenceAvailable}
          onClear={() => router.push(`/${modelId}/jlens`)}
          onSidebarSelectionChange={onSidebarSelectionChange}
          onNextDemo={onNextDemo}
          nextDemoLabel={nextDemo?.label}
          onFreeChatDemo={onFreeChatDemo}
        />
      ) : (
        <JlensChat
          key={share.url}
          modelId={modelId}
          loadedData={loadedData as JlensExportChat | null}
          sharedDescription={share.description}
          sharedAttribution={share.descriptionAttribution}
          inferenceAvailable={inferenceAvailable}
          onClear={() => router.push(`/${modelId}/jlens`)}
          onSidebarSelectionChange={onSidebarSelectionChange}
          onNextDemo={onNextDemo}
          nextDemoLabel={nextDemo?.label}
          onFreeChatDemo={onFreeChatDemo}
        />
      )}
    </ChineseTranslationsProvider>
  );
}
