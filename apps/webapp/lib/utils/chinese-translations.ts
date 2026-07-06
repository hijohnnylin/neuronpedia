'use client';

// Client-side Chinese-token -> English translation lookup for the jlens UI.
//
// The translations file is served same-origin from `/public` (so there are no
// CORS issues and Vercel auto-compresses + edge-caches it). It's keyed by the
// tokenizer's *raw* byte-level vocab strings (e.g. "å¸¦æľī"). The lens stream,
// however, gives us *decoded* tokens (the real characters, e.g. "带有"), so we
// decode the raw keys back into real text via the GPT-2/Qwen byte-level table
// to build a lookup the UI can use directly.
//
// The file is fetched once (lazily, only when a jlens component mounts) and
// cached in localStorage for subsequent loads.

import { createContext, createElement, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';

const TRANSLATIONS_URL = '/chinese-translations-qwen.json';
const STORAGE_KEY = 'np-chinese-translations-v3';

// --------------------------------------------------------------------------- //
// Byte-level (GPT-2 / Qwen) decoding
// --------------------------------------------------------------------------- //

// Reproduces HF's `bytes_to_unicode`: a reversible map from each of the 256
// byte values to a printable unicode codepoint. We build the inverse here
// (printable char -> byte) so we can turn a raw vocab string back into bytes.
function buildByteDecoder(): Map<string, number> {
  const bs: number[] = [];
  const addRange = (from: string, to: string) => {
    for (let i = from.codePointAt(0)!; i <= to.codePointAt(0)!; i += 1) {
      bs.push(i);
    }
  };
  addRange('!', '~');
  addRange('\u00a1', '\u00ac');
  addRange('\u00ae', '\u00ff');

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }

  const decoder = new Map<string, number>();
  for (let i = 0; i < bs.length; i += 1) {
    decoder.set(String.fromCodePoint(cs[i]), bs[i]);
  }
  return decoder;
}

const BYTE_DECODER = buildByteDecoder();
const UTF8_DECODER = typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { fatal: false }) : null;

// Decode a raw byte-level token string back into its real text, or null if the
// string isn't a byte-level token (e.g. the file is already keyed by real
// characters) or decodes to invalid UTF-8 (a partial multi-byte token).
function decodeRawToken(raw: string): string | null {
  if (!UTF8_DECODER) {
    return null;
  }
  const bytes: number[] = [];
  for (const ch of raw) {
    const b = BYTE_DECODER.get(ch);
    if (b === undefined) {
      return null;
    }
    bytes.push(b);
  }
  const text = UTF8_DECODER.decode(Uint8Array.from(bytes));
  return text.includes('\ufffd') ? null : text;
}

// Build the in-memory lookup. Works whether the file is keyed by raw byte-level
// strings (decoded here) or already by real characters (stored as-is). Both the
// exact and trimmed forms are indexed so leading/trailing whitespace tokens
// still match.
function buildLookup(dict: Record<string, unknown>): Map<string, string> {
  const map = new Map<string, string>();
  const put = (key: string, value: string) => {
    if (key && !map.has(key)) {
      map.set(key, value);
    }
  };
  for (const rawKey of Object.keys(dict)) {
    const value = dict[rawKey];
    if (typeof value !== 'string' || value === '' || value === '?') {
      continue;
    }
    const decoded = decodeRawToken(rawKey);
    const key = decoded ?? rawKey;
    put(key, value);
    put(key.trim(), value);
  }
  return map;
}

// --------------------------------------------------------------------------- //
// Loading (singleton + localStorage cache)
// --------------------------------------------------------------------------- //

let cachedMap: Map<string, string> | null = null;
let inflight: Promise<Map<string, string>> | null = null;

async function loadTranslations(): Promise<Map<string, string>> {
  if (cachedMap) {
    return cachedMap;
  }
  if (inflight) {
    return inflight;
  }
  inflight = (async () => {
    if (typeof window !== 'undefined') {
      try {
        const stored = window.localStorage.getItem(STORAGE_KEY);
        if (stored) {
          cachedMap = buildLookup(JSON.parse(stored));
          return cachedMap;
        }
      } catch {
        // corrupt/unavailable cache — fall through to fetch
      }
    }

    const res = await fetch(TRANSLATIONS_URL);
    if (!res.ok) {
      throw new Error(`Failed to load Chinese translations: ${res.status}`);
    }
    const text = await res.text();
    cachedMap = buildLookup(JSON.parse(text));

    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem(STORAGE_KEY, text);
      } catch {
        // localStorage full/unavailable — keep the in-memory map only
      }
    }
    return cachedMap;
  })();
  return inflight;
}

// --------------------------------------------------------------------------- //
// React context + hooks
// --------------------------------------------------------------------------- //

const ChineseTranslationsContext = createContext<Map<string, string> | null>(null);

// Provider that lazily loads (and caches) the translations the first time it
// mounts. Only render this on jlens pages so the file is never fetched
// elsewhere.
export function ChineseTranslationsProvider({ children }: { children: ReactNode }) {
  const [map, setMap] = useState<Map<string, string> | null>(cachedMap);

  useEffect(() => {
    if (cachedMap) {
      setMap(cachedMap);
      return undefined;
    }
    let active = true;
    loadTranslations()
      .then((m) => {
        if (active) {
          setMap(m);
        }
      })
      .catch(() => {
        // best-effort: translations are an enhancement, never block the UI
      });
    return () => {
      active = false;
    };
  }, []);

  return createElement(ChineseTranslationsContext.Provider, { value: map }, children);
}

export type TranslateFn = (token: string) => string | undefined;

// Returns a stable lookup function: token (decoded string) -> English gloss.
export function useChineseTranslation(): TranslateFn {
  const map = useContext(ChineseTranslationsContext);
  return useMemo<TranslateFn>(
    () => (token: string) => {
      if (!map || !token) {
        return undefined;
      }
      return map.get(token) ?? map.get(token.trim());
    },
    [map],
  );
}
