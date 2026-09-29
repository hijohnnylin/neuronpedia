'use client';

import { useTurnstileSiteKey } from '@/components/provider/turnstile-provider';
import { TURNSTILE_SIGN_IN_ACTION } from '@/lib/utils/turnstile';
import { ReactNode, RefObject, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

const TURNSTILE_SCRIPT_URL = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';

type TurnstileAppearance = 'always' | 'interaction-only';

type TurnstileApi = {
  render: (element: HTMLElement, options: Record<string, unknown>) => string | undefined;
  remove: (widgetId: string) => void;
};

declare global {
  interface Window {
    turnstile?: TurnstileApi;
  }
}

let turnstileScript: Promise<TurnstileApi> | null = null;

function loadTurnstile(): Promise<TurnstileApi> {
  if (window.turnstile) {
    return Promise.resolve(window.turnstile);
  }
  if (!turnstileScript) {
    turnstileScript = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = TURNSTILE_SCRIPT_URL;
      script.async = true;
      script.onload = () => (window.turnstile ? resolve(window.turnstile) : reject(new Error('Turnstile is missing')));
      script.onerror = () => {
        turnstileScript = null;
        script.remove();
        reject(new Error('Could not load Turnstile'));
      };
      document.head.appendChild(script);
    });
  }
  return turnstileScript;
}

function TurnstileWidget({
  siteKey,
  appearance,
  onToken,
  onVisibleChange,
}: {
  siteKey: string;
  appearance: TurnstileAppearance;
  onToken: (token: string | null) => void;
  onVisibleChange?: (visible: boolean) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const onTokenRef = useRef(onToken);
  onTokenRef.current = onToken;
  const onVisibleChangeRef = useRef(onVisibleChange);
  onVisibleChangeRef.current = onVisibleChange;

  useEffect(() => {
    let cancelled = false;
    let widgetId: string | undefined;
    loadTurnstile()
      .then((turnstile) => {
        if (cancelled || !containerRef.current) {
          return;
        }
        widgetId = turnstile.render(containerRef.current, {
          sitekey: siteKey,
          action: TURNSTILE_SIGN_IN_ACTION,
          appearance,
          callback: (token: string) => onTokenRef.current(token),
          'expired-callback': () => onTokenRef.current(null),
          'error-callback': () => {
            onTokenRef.current(null);
            onVisibleChangeRef.current?.(true);
          },
          'before-interactive-callback': () => onVisibleChangeRef.current?.(true),
          'after-interactive-callback': () => onVisibleChangeRef.current?.(false),
        });
      })
      .catch((error) => console.error(error));
    return () => {
      cancelled = true;
      if (widgetId) {
        window.turnstile?.remove(widgetId);
      }
    };
  }, [siteKey, appearance]);

  return <div ref={containerRef} className="flex justify-center" />;
}

const ANCHOR_GAP = 4;
const SCREEN_MARGIN = 8;

// For forms too small for the widget. The card shows below the anchor, centered, only when
// Cloudflare needs a click or has an error.
function FloatingTurnstileWidget({
  siteKey,
  onToken,
  anchorRef,
}: {
  siteKey: string;
  onToken: (token: string | null) => void;
  anchorRef: RefObject<HTMLElement | null>;
}) {
  const [visible, setVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const cardRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    function place() {
      const anchor = anchorRef.current?.getBoundingClientRect();
      if (!anchor) {
        return;
      }
      const width = cardRef.current?.offsetWidth ?? 0;
      const centered = anchor.left + anchor.width / 2 - width / 2;
      const maxLeft = window.innerWidth - width - SCREEN_MARGIN;
      setPosition({ top: anchor.bottom + ANCHOR_GAP, left: Math.max(SCREEN_MARGIN, Math.min(centered, maxLeft)) });
    }
    place();
    const cardSize = new ResizeObserver(place);
    if (cardRef.current) {
      cardSize.observe(cardRef.current);
    }
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => {
      cardSize.disconnect();
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
    };
  }, [anchorRef]);

  if (typeof document === 'undefined') {
    return null;
  }
  return createPortal(
    <div
      ref={cardRef}
      style={position}
      className={
        visible
          ? 'fixed z-[100] rounded-lg border border-slate-200 bg-white p-3 shadow-lg'
          : 'pointer-events-none fixed -z-10 opacity-0'
      }
    >
      {visible && <div className="mb-1 text-center text-xs text-slate-600">Quick Robot Check 🤖</div>}
      <TurnstileWidget siteKey={siteKey} appearance="interaction-only" onToken={onToken} onVisibleChange={setVisible} />
    </div>,
    document.body,
  );
}

// A token works one time only. Call `reset` after each submit to get a new one.
// With `anchorRef`, the widget shows in a card below that element, not in the form.
export function useTurnstile(anchorRef?: RefObject<HTMLElement | null>): {
  enabled: boolean;
  ready: boolean;
  signInParams: Record<string, string> | undefined;
  reset: () => void;
  widget: ReactNode;
} {
  const siteKey = useTurnstileSiteKey();
  const [token, setToken] = useState<string | null>(null);
  const [widgetKey, setWidgetKey] = useState(0);

  const reset = useCallback(() => {
    setToken(null);
    setWidgetKey((key) => key + 1);
  }, []);

  return {
    enabled: siteKey !== '',
    ready: siteKey === '' || token !== null,
    signInParams: token ? { turnstile: token } : undefined,
    reset,
    widget: !siteKey ? null : anchorRef ? (
      <FloatingTurnstileWidget key={widgetKey} siteKey={siteKey} onToken={setToken} anchorRef={anchorRef} />
    ) : (
      <TurnstileWidget key={widgetKey} siteKey={siteKey} appearance="always" onToken={setToken} />
    ),
  };
}
