'use client';

// The shared "Commentary By Human Editor" banner shown when viewing a shared
// run: an amber card with the editor's markdown note + a dismiss button. It's
// rendered in two spots so it can sit in the right place at each breakpoint:
// above the chat on mobile, and above the analysis panel on desktop.

import { ArrowRight, X } from 'lucide-react';
import { useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Owns the dismissable state for the commentary banner. Dismissing also drops
// the `shareId` from the URL in place (no navigation/reload) so the link is no
// longer "a shared run" while keeping the loaded data on screen.
export function useSharedCommentary(sharedDescription: string | null) {
  const [dismissed, setDismissed] = useState(false);
  const dismiss = () => {
    setDismissed(true);
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      params.delete('shareId');
      const query = params.toString();
      window.history.replaceState(null, '', `${window.location.pathname}${query ? `?${query}` : ''}`);
    }
  };
  return { showCommentary: !!sharedDescription && !dismissed, dismiss };
}

// Mobile short forms for the commentary action buttons. On narrow screens we
// swap the desktop labels for these compact versions.
const MOBILE_BUTTON_LABEL: Record<string, string> = {
  'Next demo': 'Next',
  'Free Chat': 'Custom',
};

function commentaryButtonMobileLabel(label: string) {
  return MOBILE_BUTTON_LABEL[label] ?? label;
}

export function JlensCommentary({
  description,
  attribution = null,
  onDismiss,
  onNext,
  nextLabel = 'Next demo',
  onFreeChat,
  className = '',
}: {
  description: string;
  // Attribution for the shared description (e.g. "by @username"), shown to the
  // right of the "Commentary By Human Editor" label. Null for the default creator.
  attribution?: string | null;
  onDismiss: () => void;
  // When set (i.e. viewing a predefined demo), shows a button on the right that
  // advances to the next curated demo (or free chat on the last one).
  onNext?: () => void;
  // Label for the next button ("Next demo", or "Free Chat" on the last demo).
  nextLabel?: string;
  // When set (i.e. viewing a non-last demo), shows a secondary slate-themed
  // button below the "Next demo" button that jumps straight to free chat.
  onFreeChat?: () => void;
  // Extra classes for the outer amber card (e.g. corner rounding per breakpoint).
  className?: string;
}) {
  return (
    <div
      className={`flex w-full flex-row items-center justify-start gap-x-1 border-amber-400 bg-amber-50 px-2 py-1.5 sm:border sm:px-3 ${className}`}
    >
      <div className="relative flex flex-1 flex-col gap-y-0.5 py-1 pl-1 pr-1 sm:px-2.5 sm:py-2 sm:pb-2.5">
        {attribution && (
          <div className="absolute bottom-0 right-0 font-sans text-[9.5px] font-normal normal-case leading-snug text-slate-400">
            {attribution}
          </div>
        )}
        <div className="jlens-markdown break-words font-sans text-[13px] font-normal leading-snug text-slate-800">
          <Markdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children, ...props }) => (
                <a
                  {...props}
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-amber-700 underline decoration-amber-700 underline-offset-2 hover:text-amber-900"
                >
                  {children}
                </a>
              ),
            }}
          >
            {description}
          </Markdown>
        </div>
      </div>
      {(onNext || onFreeChat) && (
        <div className="flex shrink-0 flex-col items-stretch gap-y-1.5 self-center">
          {onNext && (
            <button
              type="button"
              onClick={onNext}
              title={`Go to ${nextLabel}`}
              className="flex items-center justify-center gap-x-1 whitespace-nowrap rounded-md border border-amber-500 bg-amber-100 px-1.5 py-1 text-[9px] font-semibold text-amber-800 transition-colors hover:bg-amber-200 sm:rounded-lg sm:px-3 sm:py-1.5 sm:text-[11px]"
            >
              <span className="sm:hidden">{commentaryButtonMobileLabel(nextLabel)}</span>
              <span className="hidden sm:inline">{nextLabel}</span>
              <ArrowRight className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
            </button>
          )}
          {onFreeChat && (
            <button
              type="button"
              onClick={onFreeChat}
              title="Go to Free Chat"
              className="flex items-center justify-center gap-x-1 whitespace-nowrap rounded-md border border-slate-300 bg-slate-100 px-1.5 py-1 text-[9px] font-semibold text-slate-600 transition-colors hover:bg-slate-200 sm:rounded-lg sm:px-3 sm:py-1.5 sm:text-[11px]"
            >
              <span className="sm:hidden">{commentaryButtonMobileLabel('Free Chat')}</span>
              <span className="hidden sm:inline">Free Chat</span>
              <ArrowRight className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
            </button>
          )}
        </div>
      )}
      <button
        type="button"
        onClick={onDismiss}
        title="Dismiss commentary"
        aria-label="Dismiss commentary"
        className="hidden h-6 w-6 shrink-0 items-center justify-center self-start rounded-md text-slate-300 transition-colors hover:bg-slate-200 hover:text-slate-900"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
