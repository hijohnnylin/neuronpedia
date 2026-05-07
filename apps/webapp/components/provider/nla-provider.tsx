'use client';

import {
  ChatMessage,
  ChatTokenizerFormat,
  ExplainMeta,
  ExplainResult,
  NlaSourceWithModel,
  PartialUpdate,
  TokenInfo,
} from '@/app/[modelId]/nla/nla-types';
import {
  computeAutoSelection,
  groupTokensIntoMessages,
  MAX_TOKENS_TO_EXPLAIN,
  messageAllTokens,
} from '@/app/[modelId]/nla/nla-utils';
import {
  DEFAULT_COMPLETION_TOKENS,
  MAX_COMMENT_LENGTH,
  MAX_TEXT_LENGTH,
  NLA_FREE_CHAT_DEMO_CACHE_ID,
} from '@/lib/nla-constants';
import { EventSourceParserStream } from 'eventsource-parser/stream';
import { useSearchParams } from 'next/navigation';
import {
  createContext,
  Dispatch,
  ReactNode,
  SetStateAction,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

const TOKENIZE_PREVIEW_DEBOUNCE_MS = 300;

// ─── Chat tokenizer format abstraction ──────────────────────────────────────
// To add a new model format, implement ChatTokenizerFormat and add an entry to
// TOKENIZER_FORMATS below.

const qwenFormat: ChatTokenizerFormat = {
  id: 'qwen',
  turnStartToken: '<|im_start|>',
  turnEndToken: '<|im_end|>',
  assistantRoleName: 'assistant',
  formatSingleTurn(message, messageIndex, totalMessages) {
    const isLast = messageIndex === totalMessages - 1;
    return `<|im_start|>${message.role}\n${message.content}${isLast ? '' : '<|im_end|>\n'}`;
  },
  formatChat(messages) {
    return messages.map((m, i) => this.formatSingleTurn(m, i, messages.length)).join('');
  },
  parseChat(raw) {
    if (!raw.includes('<|im_start|>')) return null;
    const parts = raw.split('<|im_start|>').filter(Boolean);
    const messages: ChatMessage[] = [];
    let valid = true;
    parts.forEach((part) => {
      if (!valid) return;
      const newlineIdx = part.indexOf('\n');
      if (newlineIdx === -1) {
        valid = false;
        return;
      }
      const role = part.slice(0, newlineIdx).trim();
      if (role !== 'user' && role !== 'assistant') {
        valid = false;
        return;
      }
      const content = part.slice(newlineIdx + 1).replace(/<\|im_end\|>\n?$/, '');
      messages.push({ role, content });
    });
    if (!valid) return null;
    return messages.length > 0 ? messages : null;
  },
  isSpecialToken: (token) => token === '<|im_start|>' || token === '<|im_end|>',
};

const gemma3Format: ChatTokenizerFormat = {
  id: 'gemma3',
  turnStartToken: '<start_of_turn>',
  turnEndToken: '<end_of_turn>',
  assistantRoleName: 'model',
  formatSingleTurn(message, messageIndex, totalMessages) {
    const isLast = messageIndex === totalMessages - 1;
    const roleName = message.role === 'assistant' ? 'model' : message.role;
    return `<start_of_turn>${roleName}\n${message.content}${isLast ? '' : '<end_of_turn>\n'}`;
  },
  formatChat(messages) {
    return messages.map((m, i) => this.formatSingleTurn(m, i, messages.length)).join('');
  },
  parseChat(raw) {
    if (!raw.includes('<start_of_turn>')) return null;
    const parts = raw.split('<start_of_turn>').filter(Boolean);
    const messages: ChatMessage[] = [];
    let valid = true;
    parts.forEach((part) => {
      if (!valid) return;
      const newlineIdx = part.indexOf('\n');
      if (newlineIdx === -1) {
        valid = false;
        return;
      }
      const rawRole = part.slice(0, newlineIdx).trim();
      let role: 'user' | 'assistant';
      if (rawRole === 'user') role = 'user';
      else if (rawRole === 'model') role = 'assistant';
      else {
        valid = false;
        return;
      }
      const content = part.slice(newlineIdx + 1).replace(/<end_of_turn>\n?$/, '');
      messages.push({ role, content });
    });
    if (!valid) return null;
    return messages.length > 0 ? messages : null;
  },
  isSpecialToken: (token) => token === '<start_of_turn>' || token === '<end_of_turn>',
};

// Llama 3.x chat format. Each turn is wrapped as
//   <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
// with the trailing <|eot_id|> omitted on the last turn so the model continues
// generating. The BOS token <|begin_of_text|> is added automatically by the
// tokenizer, so we don't include it here. Mirrors Llama 3.3-Instruct's chat
// template (sans system prompt and tool calling).
const llama3Format: ChatTokenizerFormat = {
  id: 'llama3',
  turnStartToken: '<|start_header_id|>',
  turnEndToken: '<|eot_id|>',
  assistantRoleName: 'assistant',
  formatSingleTurn(message, messageIndex, totalMessages) {
    const isLast = messageIndex === totalMessages - 1;
    return `<|start_header_id|>${message.role}<|end_header_id|>\n\n${message.content}${isLast ? '' : '<|eot_id|>'}`;
  },
  formatChat(messages) {
    return messages.map((m, i) => this.formatSingleTurn(m, i, messages.length)).join('');
  },
  parseChat(raw) {
    if (!raw.includes('<|start_header_id|>')) return null;
    // Strip an optional leading BOS so pasted snippets that include it
    // still round-trip cleanly.
    const stripped = raw.replace(/^<\|begin_of_text\|>/, '');
    const parts = stripped.split('<|start_header_id|>').filter(Boolean);
    const messages: ChatMessage[] = [];
    let valid = true;
    parts.forEach((part) => {
      if (!valid) return;
      const headerEndIdx = part.indexOf('<|end_header_id|>');
      if (headerEndIdx === -1) {
        valid = false;
        return;
      }
      const role = part.slice(0, headerEndIdx).trim();
      if (role !== 'user' && role !== 'assistant') {
        valid = false;
        return;
      }
      let content = part.slice(headerEndIdx + '<|end_header_id|>'.length);
      content = content.replace(/^\n+/, '');
      content = content.replace(/<\|eot_id\|>\s*$/, '');
      messages.push({ role, content });
    });
    if (!valid) return null;
    return messages.length > 0 ? messages : null;
  },
  isSpecialToken: (token) =>
    token === '<|start_header_id|>' ||
    token === '<|end_header_id|>' ||
    token === '<|eot_id|>' ||
    token === '<|begin_of_text|>',
};

const TOKENIZER_FORMATS: ChatTokenizerFormat[] = [qwenFormat, gemma3Format, llama3Format];

function detectTokenizerFormat(modelId: string): ChatTokenizerFormat {
  const lower = modelId.toLowerCase();
  if (lower.includes('gemma')) return gemma3Format;
  if (lower.includes('llama')) return llama3Format;
  return qwenFormat;
}

function parseAnyChat(raw: string): { messages: ChatMessage[]; format: ChatTokenizerFormat } | null {
  for (const fmt of TOKENIZER_FORMATS) {
    const parsed = fmt.parseChat(raw);
    if (parsed) return { messages: parsed, format: fmt };
  }
  return null;
}

type InputMode = 'chat' | 'manual' | 'paste';

/** Snapshot encoded into an `NlaExplainShare` row (short `/nla/[id]` link). */
export type NlaShareDraft = {
  cacheId: string;
  position: number | null;
  paragraph: number | null;
  highlightStart: number | null;
  highlightEnd: number | null;
  initialComment: string | null;
  /** Featured demo row id — same link as POST /explain-share would create for this snapshot. */
  existingShareId?: string;
};

function normalizeNlaShareComment(value: string | null | undefined): string {
  return (value ?? '').trim();
}

/**
 * When the user opened Share from an untouched featured demo (still `activeDemoCacheId`),
 * reuse that row's short id instead of creating a duplicate via POST /explain-share.
 */
function existingFeaturedShareIdForDraft(
  demos: NlaFeaturedDemo[],
  modelId: string,
  activeDemoCacheId: string | null,
  draft: {
    cacheId: string;
    position: number | null;
    paragraph: number | null;
    highlightStart: number | null;
    highlightEnd: number | null;
    initialComment: string | null;
  },
): string | undefined {
  if (
    activeDemoCacheId == null ||
    activeDemoCacheId === NLA_FREE_CHAT_DEMO_CACHE_ID ||
    activeDemoCacheId !== draft.cacheId
  ) {
    return undefined;
  }
  const targetComment = normalizeNlaShareComment(draft.initialComment);
  for (const d of demos) {
    if (d.modelId !== modelId || d.cacheId !== draft.cacheId) continue;
    if ((d.position ?? null) !== (draft.position ?? null)) continue;
    if ((d.paragraph ?? null) !== (draft.paragraph ?? null)) continue;
    if ((d.highlightStart ?? null) !== (draft.highlightStart ?? null)) continue;
    if ((d.highlightEnd ?? null) !== (draft.highlightEnd ?? null)) continue;
    if (normalizeNlaShareComment(d.comment) !== targetComment) continue;
    return d.shareId;
  }
  return undefined;
}

/** Featured `/nla/[shareId]` demos loaded on the server and cached in context (no extra GET). */
export type NlaFeaturedDemo = {
  shareId: string;
  cacheId: string;
  modelId: string;
  modelDisplayName: string;
  modelOwner: string;
  position: number | null;
  paragraph: number | null;
  highlightStart: number | null;
  highlightEnd: number | null;
  comment: string;
  featuredDisplayName: string | null;
};

type NLAContextType = {
  // configuration
  isEmbed: boolean;
  nlaSources: NlaSourceWithModel[];

  // model/source selection
  selectedModelId: string;
  selectedNlaSourceId: string;
  modelIds: string[];
  modelDisplayMap: Map<string, { displayName: string; owner: string }>;
  filteredSources: NlaSourceWithModel[];
  selectedNlaSource: NlaSourceWithModel | undefined;
  tokenizerFormat: ChatTokenizerFormat;
  handleModelChange: (newModelId: string) => void;
  handleSourceChange: (newSourceId: string) => void;

  // chat / input
  chatMessages: ChatMessage[];
  setChatMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  inputMode: InputMode;
  setInputMode: Dispatch<SetStateAction<InputMode>>;
  isChatStreaming: boolean;
  setIsChatStreaming: Dispatch<SetStateAction<boolean>>;

  // settings
  temperature: number;
  setTemperature: Dispatch<SetStateAction<number>>;
  maxNewTokens: number;
  setMaxNewTokens: Dispatch<SetStateAction<number>>;
  showAdvanced: boolean;
  setShowAdvanced: Dispatch<SetStateAction<boolean>>;

  // tokenize
  tokenList: TokenInfo[];
  setTokenList: Dispatch<SetStateAction<TokenInfo[]>>;
  setLastTokenizedText: Dispatch<SetStateAction<string | null>>;

  // selection
  selectedTokenPositions: Set<number>;
  topLevelMode: 'auto' | 'manual';
  handleApplySelection: (newSet: Set<number>) => void;
  handleTopLevelModeChange: (mode: 'auto' | 'manual') => void;

  // results
  resultMap: Map<number, ExplainResult>;
  partialMap: Map<number, string>;
  selectedPosition: number | null;
  setSelectedPosition: Dispatch<SetStateAction<number | null>>;
  lockedPosition: number | null;
  setLockedPosition: Dispatch<SetStateAction<number | null>>;
  // Index of the paragraph (0=Message-Level, 1=Phrase-Level, 2=Token-Level)
  // currently highlighted in the details column. `null` means no highlight.
  // Driven by the `?paragraph=` URL param + by per-paragraph share buttons;
  // cleared whenever the user clicks a different chip. Mutually exclusive
  // with `highlightedRange` — setting one clears the other.
  highlightedParagraph: number | null;
  setHighlightedParagraph: (value: number | null) => void;
  // Character-index range into the focused token's raw `description`
  // string. When set, the corresponding text is highlighted in yellow in
  // the details column. Driven by `?highlightStart=&highlightEnd=` URL
  // params + by the in-paragraph text-selection share popover. Mutually
  // exclusive with `highlightedParagraph`.
  highlightedRange: { start: number; end: number } | null;
  setHighlightedRange: (value: { start: number; end: number } | null) => void;
  // Optional user-authored comment that annotates the currently-focused
  // demo/highlight/paragraph. Persisted via `?comment=` and displayed in
  // a panel above the details column's "Explanation at Token" header.
  // Anchored to `lockedPosition` — cleared when the user clicks out of
  // the token, clears the chat, switches model/source, or sends a new
  // message. Capped at `MAX_COMMENT_LENGTH` characters.
  highlightComment: string | null;
  setHighlightComment: (value: string | null) => void;
  isLoading: boolean;
  // True from the moment a demo-button click (or `?id=...` deep-link)
  // calls into the cache hydrate path until the cache fetch resolves and
  // the new state has been applied. The chat and details columns swap
  // their bodies for a centered "Loading…" placeholder while this is
  // true, so the previous demo's content doesn't linger (and the
  // commentary panel doesn't disappear ahead of the rest of the swap).
  isHydratingDemo: boolean;
  error: string | null;
  // When set, the chat input should restore this string into its textarea
  // and clear the field (via `setPendingChatInputRestore(null)`). Used by
  // the explain-429 handler (and other server-busy paths) to roll the
  // chat back to before the last user turn so the user can edit/retry.
  pendingChatInputRestore: string | null;
  setPendingChatInputRestore: Dispatch<SetStateAction<string | null>>;

  // actions
  handleSubmit: () => Promise<void>;
  handleClear: (options?: { pinFreeChatDemo?: boolean }) => void;
  handleShare: () => void;
  // The `override` arg lets per-paragraph and per-selection share buttons
  // emit a link with `?paragraph=` or `?highlightStart=&highlightEnd=`
  // even before the corresponding state update has flushed. Each field
  // takes effect only when explicitly provided (use `null` to omit a
  // param even if state has it set, `undefined` to fall back to state).
  handleShareExplanation: (override?: {
    paragraph?: number | null;
    range?: { start: number; end: number } | null;
    comment?: string | null;
  }) => void;
  loadCacheById: (
    cacheId: string,
    position?: number,
    paragraph?: number,
    highlightStart?: number,
    highlightEnd?: number,
    comment?: string,
  ) => Promise<void>;
  /** All featured demos (every model); UI filters by `selectedModelId`. */
  featuredDemos: NlaFeaturedDemo[];
  activeDemoCacheId: string | null;
  setActiveDemoCacheId: Dispatch<SetStateAction<string | null>>;
  onUserEdit: () => void;
  explainDisabled: boolean;
  // Disarms the auto-explain-after-chat trigger. Used by the chat panel
  // when the user aborts a streaming completion: without this the
  // auto-explain effect would fire on the partial conversation as soon
  // as `isChatStreaming` flips back to false.
  cancelPendingAutoExplain: () => void;
  // Drop `chatMessages[idx..end]` and reset the explain/tokenize state.
  // Used by the per-message edit flow.
  truncateChatFrom: (idx: number) => void;

  // share modal
  isShareModalOpen: boolean;
  setIsShareModalOpen: Dispatch<SetStateAction<boolean>>;
  shareDraft: NlaShareDraft | null;
  shareError: string | null;

  // misc
  detailsColumnRef: React.RefObject<HTMLDivElement | null>;
  // Increments whenever a cache is hydrated so the chat panel can scroll
  // to the bottom after the conversation has been swapped in.
  chatScrollNonce: number;
  // Trimmed explanation search query (length ≥ 2) for highlighting matches in
  // the details column; empty when search is inactive.
  explanationSearchNeedle: string;
  setExplanationSearchNeedle: Dispatch<SetStateAction<string>>;
  /** Bumps when chat is reset, model/source switches, or a cache/demo is loaded — clear local explanation search UI. */
  explanationSearchResetNonce: number;
};

const NLAContext = createContext<NLAContextType | undefined>(undefined);

export function NLAProvider({
  children,
  modelId,
  nlaSources = [],
  featuredDemos = [],
  isEmbed = false,
}: {
  children: ReactNode;
  modelId: string;
  nlaSources?: NlaSourceWithModel[];
  featuredDemos?: NlaFeaturedDemo[];
  isEmbed?: boolean;
}) {
  const searchParams = useSearchParams();
  const [temperature, setTemperature] = useState(0.7);
  const [isLoading, setIsLoading] = useState(false);
  const [tokenList, setTokenList] = useState<TokenInfo[]>([]);
  const [resultMap, setResultMap] = useState<Map<number, ExplainResult>>(new Map());
  const [partialMap, setPartialMap] = useState<Map<number, string>>(new Map());
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [meta, setMeta] = useState<ExplainMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pendingChatInputRestore, setPendingChatInputRestore] = useState<string | null>(null);
  const [selectedPosition, setSelectedPosition] = useState<number | null>(null);
  // When set, the details column is "pinned" to this position regardless
  // of which chip the cursor is currently hovering. Set by clicking an
  // explained chip; cleared by clicking elsewhere, the Deselect Token
  // button, sending a new message, clearing the chat, or running a new
  // explain.
  const [lockedPosition, setLockedPosition] = useState<number | null>(null);
  // Highlights one paragraph in the details column with a yellow tint.
  // Persisted to the URL as `?paragraph=N` (0-2) when a position is also
  // locked, so a copy-pasted address bar reproduces the exact view.
  const [highlightedParagraph, setHighlightedParagraphRaw] = useState<number | null>(null);
  // Highlights an arbitrary text range within one paragraph (character
  // indices into the focused token's raw `description`). Persisted as
  // `?highlightStart=&highlightEnd=`. Mutually exclusive with
  // `highlightedParagraph` — see the wrapper setters below.
  const [highlightedRange, setHighlightedRangeRaw] = useState<{ start: number; end: number } | null>(null);
  // Free-form comment annotating the current focus. Persisted as
  // `?comment=`. Truncated to MAX_COMMENT_LENGTH whenever it's set so
  // over-length values from the URL or a demo definition degrade
  // gracefully instead of being rejected outright.
  const [highlightCommentRaw, setHighlightCommentRaw] = useState<string | null>(null);
  const setHighlightComment = useCallback((value: string | null) => {
    if (value === null || value.length === 0) {
      setHighlightCommentRaw(null);
      return;
    }
    setHighlightCommentRaw(value.length > MAX_COMMENT_LENGTH ? value.slice(0, MAX_COMMENT_LENGTH) : value);
  }, []);
  const highlightComment = highlightCommentRaw;
  const setHighlightedParagraph = useCallback((value: number | null) => {
    setHighlightedParagraphRaw(value);
    // Mutual exclusion: a paragraph highlight and a sub-paragraph range
    // would visually fight each other, so adopting one clears the other.
    if (value !== null) setHighlightedRangeRaw(null);
  }, []);
  const setHighlightedRange = useCallback((value: { start: number; end: number } | null) => {
    setHighlightedRangeRaw(value);
    if (value !== null) setHighlightedParagraphRaw(null);
  }, []);
  const detailsColumnRef = useRef<HTMLDivElement | null>(null);
  const [isShareModalOpen, setIsShareModalOpen] = useState(false);
  const [shareDraft, setShareDraft] = useState<NlaShareDraft | null>(null);
  const [shareError, setShareError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxNewTokens, setMaxNewTokens] = useState(DEFAULT_COMPLETION_TOKENS);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([{ role: 'user', content: '' }]);
  const [lastTokenizedText, setLastTokenizedText] = useState<string | null>(null);
  const [activeDemoCacheId, setActiveDemoCacheId] = useState<string | null>(null);
  const [isHydratingDemo, setIsHydratingDemo] = useState(false);
  const [chatScrollNonce, setChatScrollNonce] = useState(0);
  const [explanationSearchNeedle, setExplanationSearchNeedle] = useState('');
  const [explanationSearchResetNonce, setExplanationSearchResetNonce] = useState(0);

  // Left-pane mode: 'chat' uses a streaming chat interface that talks to the
  // model; 'manual' lets the user hand-edit any sequence of turns; 'paste'
  // accepts a ChatMessages JSON array.
  const [inputMode, setInputMode] = useState<InputMode>('chat');
  const [isChatStreaming, setIsChatStreaming] = useState(false);

  // Selection state (which tokens to explain). Reset whenever the tokenize
  // preview returns a new tokenList; user can toggle individual chips or pick
  // a preset mode.
  const [selectedTokenPositions, setSelectedTokenPositions] = useState<Set<number>>(new Set());

  // Tracks the in-flight tokenize-preview request so it can be aborted
  // when inputs change or an explain/chat run takes over.
  const [isTokenizingPreview, setIsTokenizingPreview] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const tokenizePreviewAbortRef = useRef<AbortController | null>(null);
  const initialCacheIdRef = useRef<string | null>(null);
  // Captures the `?position=N` param so the hydrate path can default-focus
  // that token (instead of the first explained token in the last assistant
  // turn). One-shot: consumed by the hydrate, then cleared.
  const initialPositionRef = useRef<number | null>(null);
  // Same one-shot pattern as `initialPositionRef`, but for `?paragraph=N`.
  // Only honored when `initialPositionRef` is also set (a paragraph
  // highlight without a locked position would have nothing to anchor to).
  const initialParagraphRef = useRef<number | null>(null);
  // One-shot snapshot of `?highlightStart=&highlightEnd=`. Same anchoring
  // rule as paragraph: only honored alongside a locked position.
  const initialRangeRef = useRef<{ start: number; end: number } | null>(null);
  // One-shot snapshot of `?comment=`. Same anchoring rule: only honored
  // alongside a locked position.
  const initialCommentRef = useRef<string | null>(null);
  // True while we're applying a cache-hydrate (or for a brief window
  // after) so that the cacheId-cleanup useEffect doesn't fire on the
  // very state changes the hydrate itself produces.
  const isHydratingRef = useRef(false);
  // Cache / demo / share loads restore `selectedTokenPositions` from the
  // row's `sortedPositions`. `isHydratingRef` clears ~50ms after fetch;
  // if anything updates `tokenList` later without re-hydrating, we still
  // must not clobber that selection with `computeAutoSelection`. Cleared on
  // `resetResults` and on `onUserEdit` (typing, paste, sending a chat, etc.).
  const suppressDerivedAutoSelectionRef = useRef(false);
  // Next.js 15's `useSearchParams` treats our own `window.history.replaceState`
  // calls (e.g. mirroring `lockedPosition` to `?position=`) as soft
  // navigations and re-fires the searchParams effect. We only want to
  // capture the URL params once on mount; after that, ignore re-fires so
  // we don't accidentally re-arm the hydrate path and freeze
  // `isHydratingRef` to `true` forever.
  const hasInitialParamsRef = useRef(false);

  // The id of the most recently-completed explain run (or hydrated demo
  // cache). Sent on the next /explain call as `priorCacheId` so the server
  // can reuse those explanations for any prefix that still matches —
  // critical for chat-extension flows where the new prompt's `text` field
  // differs from the prior row's, defeating the route's exact-text-match
  // prior-aggregate query. Cleared on chat reset.
  const priorCacheIdRef = useRef<string | null>(null);

  // NLA source selection
  const modelIds = useMemo(() => {
    const ids = [...new Set(nlaSources.map((s) => s.modelId))];
    ids.sort((a, b) => a.localeCompare(b));
    return ids;
  }, [nlaSources]);

  const modelDisplayMap = useMemo(() => {
    const map = new Map<string, { displayName: string; owner: string }>();
    nlaSources.forEach((s) => {
      if (!map.has(s.modelId)) {
        map.set(s.modelId, { displayName: s.model.displayName || s.modelId, owner: s.model.owner });
      }
    });
    return map;
  }, [nlaSources]);

  const [selectedModelId, setSelectedModelId] = useState<string>(modelId || modelIds[0] || '');
  const [selectedNlaSourceId, setSelectedNlaSourceId] = useState<string>('');

  const tokenizerFormat = useMemo(() => detectTokenizerFormat(selectedModelId), [selectedModelId]);

  const resetResults = useCallback(() => {
    abortRef.current?.abort();
    tokenizePreviewAbortRef.current?.abort();
    setTokenList([]);
    setResultMap(new Map());
    setPartialMap(new Map());
    setMeta(null);
    setError(null);
    setSelectedPosition(null);
    setLockedPosition(null);
    setIsLoading(false);
    setIsChatStreaming(false);
    setIsTokenizingPreview(false);
    setLastTokenizedText(null);
    setSelectedTokenPositions(new Set());
    setShareDraft(null);
    setShareError(null);
    setIsShareModalOpen(false);
    setActiveDemoCacheId(null);
    priorCacheIdRef.current = null;
    suppressDerivedAutoSelectionRef.current = false;
    setExplanationSearchNeedle('');
    setExplanationSearchResetNonce((n) => n + 1);
  }, []);

  // Returns true if the chat has any non-empty user/assistant content.
  // The default chat state is a single empty user message, which we don't
  // consider as "existing chat" worth warning about.
  const hasExistingChat = useCallback(() => chatMessages.some((m) => m.content.trim().length > 0), [chatMessages]);

  const handleSourceChange = useCallback(
    (newSourceId: string) => {
      if (newSourceId === selectedNlaSourceId) return;
      // if (hasExistingChat()) {
      //   // eslint-disable-next-line no-alert
      //   if (!window.confirm('Switching models will clear the chat. Continue?')) return;
      // }
      setSelectedNlaSourceId(newSourceId);
      resetResults();
      setChatMessages([{ role: 'user', content: '' }]);
      const url = new URL(window.location.href);
      url.searchParams.delete('id');
      url.searchParams.delete('position');
      window.history.replaceState({}, '', url.toString());
    },
    [selectedNlaSourceId, resetResults, hasExistingChat],
  );

  const handleModelChange = useCallback(
    (newModelId: string) => {
      if (newModelId === selectedModelId) return;
      // if (hasExistingChat()) {
      //   // eslint-disable-next-line no-alert
      //   if (!window.confirm('Switching models will clear the chat. Continue?')) return;
      // }
      setSelectedModelId(newModelId);
      resetResults();
      setChatMessages([{ role: 'user', content: '' }]);
      const url = new URL(window.location.href);
      url.pathname = `/${newModelId}/nla`;
      url.searchParams.delete('id');
      url.searchParams.delete('position');
      window.history.replaceState({}, '', url.toString());
      const displayName = modelDisplayMap.get(newModelId)?.displayName || newModelId;
      document.title = `NLA \u2013 ${displayName}`;
    },
    [selectedModelId, resetResults, modelDisplayMap, hasExistingChat],
  );

  const filteredSources = useMemo(
    () => nlaSources.filter((s) => s.modelId === selectedModelId),
    [nlaSources, selectedModelId],
  );

  useEffect(() => {
    if (filteredSources.length > 0 && !filteredSources.find((s) => s.id === selectedNlaSourceId)) {
      setSelectedNlaSourceId(filteredSources[0].id);
    }
  }, [filteredSources, selectedNlaSourceId]);

  // NlaSource.id is only unique within a model (composite PK on (modelId, id)),
  // so the lookup must also pin to selectedModelId. Without this, a source
  // whose id collides across models could resolve to the wrong row.
  const selectedNlaSource = useMemo(
    () => nlaSources.find((s) => s.modelId === selectedModelId && s.id === selectedNlaSourceId),
    [nlaSources, selectedModelId, selectedNlaSourceId],
  );

  // Top-level chat-mode toggle: 'auto' = auto-explain after the user sends
  // a message; 'manual' = let the user adjust the selection and click
  // Explain themselves. This is decoupled from `selectionMode` (which
  // tracks the *algorithm* used to pick the initial selection) so that the
  // user's UI preference doesn't get clobbered when we recompute the
  // selection after every send.
  const [topLevelMode, setTopLevelMode] = useState<'auto' | 'manual'>('manual');

  // ─── Selection of which tokens to explain ───────────────────────────────
  // When `tokenList` updates (tokenize preview, chat stream settling, cache
  // hydrate, …), derive the default chip selection via `computeAutoSelection`
  // for top-level auto mode. Skipped while streaming (would churn token-by-token),
  // during hydrate (never stomp the row we're loading), while
  // `suppressDerivedAutoSelectionRef` guards post-`?id=` views until user
  // edits, and in manual mode (start empty so the user picks deliberately).
  useEffect(() => {
    if (tokenList.length === 0) {
      setSelectedTokenPositions(new Set());
      return;
    }
    if (isChatStreaming) return;
    // Same commit as `setTokenList` from hydrate; never stomp with auto.
    if (isHydratingRef.current) return;
    if (inputMode === 'chat' && topLevelMode === 'manual') {
      setSelectedTokenPositions(new Set());
      return;
    }
    // Loaded from ?id=: keep row selection until user edits (send / paste /
    // manual input tweak). Applies only in auto-style flows (manual clears above).
    if (suppressDerivedAutoSelectionRef.current) return;
    setSelectedTokenPositions(computeAutoSelection(tokenList, tokenizerFormat, MAX_TOKENS_TO_EXPLAIN));
  }, [tokenList, tokenizerFormat, isChatStreaming, inputMode, topLevelMode]);

  // Replace the whole selection set. The column applies its own cap-aware
  // logic (so drag-fills can stop at the limit and pulse the counter), and
  // the parent just commits the result + recomputes the mode label.
  const handleApplySelection = useCallback(
    (newSet: Set<number>) => {
      setSelectedTokenPositions(newSet);
    },
    [tokenList, tokenizerFormat],
  );

  // ─── Tokenize preview: populates the results panel with chips for the
  // current chat. Debounced. Skipped in chat mode (the chat panel itself
  // streams tokens into tokenList) and while a chat / explain stream is
  // in progress.
  useEffect(() => {
    if (isLoading) return undefined;
    if (isChatStreaming) return undefined;
    if (inputMode === 'chat') return undefined;

    const anyMessageHasContent = chatMessages.some((m) => m.content.trim().length > 0);
    const formattedText = tokenizerFormat.formatChat(chatMessages);

    if (!anyMessageHasContent || formattedText.length > MAX_TEXT_LENGTH) {
      setIsTokenizingPreview(false);
      setTokenList([]);
      setResultMap(new Map());
      setPartialMap(new Map());
      setSelectedPosition(null);
      setLastTokenizedText(null);
      return undefined;
    }

    setIsTokenizingPreview(true);
    const controller = new AbortController();
    tokenizePreviewAbortRef.current = controller;

    const timer = setTimeout(async () => {
      try {
        const res = await fetch('/api/nla/completion', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: formattedText,
            completion_tokens: 0,
            modelId: selectedModelId || undefined,
            nlaSourceId: selectedNlaSource?.id || undefined,
          }),
          signal: controller.signal,
        });

        if (res.ok) {
          const data = (await res.json()) as { tokens?: TokenInfo[] };
          if (data.tokens) {
            setTokenList(data.tokens);
            setResultMap(new Map());
            setPartialMap(new Map());
            setSelectedPosition(null);
            setMeta({ layer_index: 0, total: data.tokens.length, prompt_length: data.tokens.length });
            setLastTokenizedText(formattedText);
          }
        }
      } catch {
        // AbortError or network error — ignore
      } finally {
        if (!controller.signal.aborted) {
          setIsTokenizingPreview(false);
        }
      }
    }, TOKENIZE_PREVIEW_DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
    // isLoading is intentionally read via closure so transitioning isLoading to
    // false (after a stream completes) does not retrigger and clobber results.
    // isChatStreaming IS in deps: when chat streaming ends we want this
    // effect to fire once and tokenize the just-finished chat.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatMessages, inputMode, isChatStreaming, selectedModelId, selectedNlaSource, tokenizerFormat]);

  // Initialize from query params. The only user-facing param is `id`,
  // which fully reproduces the explainer state by hydrating from the stored
  // cache row (text, tokens, results, explained positions). Everything
  // else lives only in the row. We intentionally consume the params
  // exactly once — Next.js 15 makes `useSearchParams` reactive to our own
  // `replaceState` updates, so without the one-shot guard this would
  // re-fire on every URL mutation and stomp the hydrate-suppression flag.
  useEffect(() => {
    if (hasInitialParamsRef.current) return;
    hasInitialParamsRef.current = true;
    const idParam = searchParams.get('id');
    if (idParam) {
      initialCacheIdRef.current = idParam;
      const positionParam = searchParams.get('position');
      if (positionParam !== null) {
        const parsed = Number.parseInt(positionParam, 10);
        if (Number.isFinite(parsed) && parsed >= 0) {
          initialPositionRef.current = parsed;
          // `?highlightStart=&highlightEnd=` takes precedence over
          // `?paragraph=` — they're mutually exclusive in state, and the
          // text range is the more specific signal. Both still require
          // a locked position to anchor to.
          const startParam = searchParams.get('highlightStart');
          const endParam = searchParams.get('highlightEnd');
          let consumedRange = false;
          if (startParam !== null && endParam !== null) {
            const s = Number.parseInt(startParam, 10);
            const e = Number.parseInt(endParam, 10);
            if (Number.isFinite(s) && Number.isFinite(e) && s >= 0 && e > s) {
              initialRangeRef.current = { start: s, end: e };
              consumedRange = true;
            }
          }
          if (!consumedRange) {
            const paragraphParam = searchParams.get('paragraph');
            if (paragraphParam !== null) {
              const p = Number.parseInt(paragraphParam, 10);
              if (Number.isFinite(p) && p >= 0 && p <= 2) {
                initialParagraphRef.current = p;
              }
            }
          }
          // Comment is orthogonal to paragraph/range — captured
          // alongside either of them. Over-length URL comments are
          // truncated rather than rejected so a malformed/too-long
          // share link still round-trips a reasonable prefix.
          const commentParam = searchParams.get('comment');
          if (commentParam !== null && commentParam.length > 0) {
            initialCommentRef.current =
              commentParam.length > MAX_COMMENT_LENGTH ? commentParam.slice(0, MAX_COMMENT_LENGTH) : commentParam;
          }
        }
      }
      // Block the id-cleanup useEffect until hydrate's setStates have
      // settled (otherwise it'd see the hydrate's own state changes and
      // wipe the param before we use it).
      isHydratingRef.current = true;
    }
  }, [searchParams]);

  // Whenever something user-driven changes (chat content, temperature,
  // model, source), the cached row this URL points to is no longer the
  // source of truth — drop the `id` so reloads don't snap back to the
  // stale state. Intentionally NOT keyed on `resultMap`, because the
  // explain stream itself produces a fresh id we want to keep on the
  // URL. Skipped while a hydrate is in flight so the hydrate's own
  // setStates don't trigger this.
  useEffect(() => {
    if (isHydratingRef.current) return;
    if (typeof window === 'undefined') return;
    setLockedPosition(null);
    setHighlightedParagraph(null);
    setHighlightedRange(null);
    setHighlightComment(null);
    const url = new URL(window.location.href);
    if (
      !url.searchParams.has('id') &&
      !url.searchParams.has('position') &&
      !url.searchParams.has('paragraph') &&
      !url.searchParams.has('highlightStart') &&
      !url.searchParams.has('highlightEnd') &&
      !url.searchParams.has('comment')
    ) {
      return;
    }
    url.searchParams.delete('id');
    url.searchParams.delete('position');
    url.searchParams.delete('paragraph');
    url.searchParams.delete('highlightStart');
    url.searchParams.delete('highlightEnd');
    url.searchParams.delete('comment');
    window.history.replaceState({}, '', url.toString());
    // We intentionally don't depend on the wrapper setters here — they're
    // stable refs created via useCallback and would just bloat the deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatMessages, temperature, selectedModelId, selectedNlaSourceId]);

  // Mirror the locked-token onto the URL as `?position=` so a reload (or
  // a copy-paste of the address bar) re-opens the same details panel.
  // Skipped while a hydrate is in flight — hydrate manages the param
  // itself (deep-link / demo-cache flows write `?id=` and optionally
  // `?position=` up-front, then call setLockedPosition once the cache
  // arrives). Cleared when the user unlocks (via the Unlock button or
  // by clicking elsewhere) and on chat clear / new explain (which both
  // reset lockedPosition to null upstream).
  useEffect(() => {
    if (isHydratingRef.current) return;
    if (typeof window === 'undefined') return;
    const url = new URL(window.location.href);
    if (lockedPosition === null) {
      if (!url.searchParams.has('position')) return;
      url.searchParams.delete('position');
      window.history.replaceState({}, '', url.toString());
      return;
    }
    if (url.searchParams.get('position') === String(lockedPosition)) return;
    url.searchParams.set('position', String(lockedPosition));
    window.history.replaceState({}, '', url.toString());
  }, [lockedPosition]);

  // Mirror the highlighted paragraph onto the URL as `?paragraph=`. Only
  // emitted when there's also a locked position (a paragraph highlight
  // without an anchor isn't actionable for a recipient). Cleared whenever
  // the position is unlocked or the highlight is reset.
  useEffect(() => {
    if (isHydratingRef.current) return;
    if (typeof window === 'undefined') return;
    const url = new URL(window.location.href);
    const shouldHave = lockedPosition !== null && highlightedParagraph !== null;
    if (!shouldHave) {
      if (!url.searchParams.has('paragraph')) return;
      url.searchParams.delete('paragraph');
      window.history.replaceState({}, '', url.toString());
      return;
    }
    if (url.searchParams.get('paragraph') === String(highlightedParagraph)) return;
    url.searchParams.set('paragraph', String(highlightedParagraph));
    window.history.replaceState({}, '', url.toString());
  }, [highlightedParagraph, lockedPosition]);

  // Mirror the highlighted text range onto the URL as
  // `?highlightStart=&highlightEnd=`. Same anchoring rule as paragraph:
  // both params are only present when a position is also locked. Cleared
  // together as a pair so the URL never has just one of the two.
  useEffect(() => {
    if (isHydratingRef.current) return;
    if (typeof window === 'undefined') return;
    const url = new URL(window.location.href);
    const shouldHave = lockedPosition !== null && highlightedRange !== null;
    if (!shouldHave) {
      if (!url.searchParams.has('highlightStart') && !url.searchParams.has('highlightEnd')) return;
      url.searchParams.delete('highlightStart');
      url.searchParams.delete('highlightEnd');
      window.history.replaceState({}, '', url.toString());
      return;
    }
    const sStr = String(highlightedRange.start);
    const eStr = String(highlightedRange.end);
    if (url.searchParams.get('highlightStart') === sStr && url.searchParams.get('highlightEnd') === eStr) {
      return;
    }
    url.searchParams.set('highlightStart', sStr);
    url.searchParams.set('highlightEnd', eStr);
    window.history.replaceState({}, '', url.toString());
  }, [highlightedRange, lockedPosition]);

  // Mirror the authored comment onto the URL as `?comment=`. Same
  // anchoring rule as paragraph/range: only emitted when a position is
  // also locked, so a recipient opening the link has something to pin
  // the comment to. Cleared whenever the comment is reset or the
  // position is unlocked.
  useEffect(() => {
    if (isHydratingRef.current) return;
    if (typeof window === 'undefined') return;
    const url = new URL(window.location.href);
    const shouldHave = lockedPosition !== null && highlightComment !== null && highlightComment.length > 0;
    if (!shouldHave) {
      if (!url.searchParams.has('comment')) return;
      url.searchParams.delete('comment');
      window.history.replaceState({}, '', url.toString());
      return;
    }
    if (url.searchParams.get('comment') === highlightComment) return;
    url.searchParams.set('comment', highlightComment);
    window.history.replaceState({}, '', url.toString());
  }, [highlightComment, lockedPosition]);

  // Truncate the chat to keep only `chatMessages[0..idx)`. Used by the
  // per-message "edit" flow: the input chat picks the message content
  // into its typed-text field, then calls this with that message's
  // index. Explanations for tokens that belong to the surviving turns
  // are PRESERVED — only the trailing tokens (and any focus / pending
  // selection that pointed at them) are dropped, so the chips for
  // earlier turns keep their underlines and details panel content.
  // The id-cleanup useEffect handles wiping `?id=` / `?position=` from
  // the URL once `chatMessages` changes.
  const truncateChatFrom = useCallback(
    (idx: number) => {
      abortRef.current?.abort();
      tokenizePreviewAbortRef.current?.abort();

      const clamped = Math.max(0, Math.min(idx, chatMessages.length));
      const truncated = chatMessages.slice(0, clamped);

      // Map the kept chat messages back to their token positions. The
      // tokenizer is fed only the with-content messages, so the
      // grouped-messages array lines up 1:1 with that filtered list.
      const grouped = groupTokensIntoMessages(tokenList, tokenizerFormat);
      const keptMessagesWithContent = truncated.filter((m) => m.content.length > 0).length;
      let keptPositions: Set<number> | null = null;
      if (grouped.hasChatFormat && keptMessagesWithContent > 0) {
        const positions = new Set<number>();
        const limit = Math.min(keptMessagesWithContent, grouped.messages.length);
        for (let i = 0; i < limit; i += 1) {
          const m = grouped.messages[i];
          m.headerTokens.forEach((t) => positions.add(t.position));
          m.contentTokens.forEach((t) => positions.add(t.position));
          m.footerTokens.forEach((t) => positions.add(t.position));
        }
        keptPositions = positions;
      }

      if (keptPositions !== null && keptPositions.size > 0) {
        const kept = keptPositions;
        setTokenList((prev) => prev.filter((t) => kept.has(t.position)));
        setResultMap((prev) => {
          const next = new Map<number, ExplainResult>();
          prev.forEach((v, k) => {
            if (kept.has(k)) next.set(k, v);
          });
          return next;
        });
        setPartialMap((prev) => {
          const next = new Map<number, string>();
          prev.forEach((v, k) => {
            if (kept.has(k)) next.set(k, v);
          });
          return next;
        });
        setSelectedTokenPositions((prev) => {
          const next = new Set<number>();
          prev.forEach((p) => {
            if (kept.has(p)) next.add(p);
          });
          return next;
        });
        setLockedPosition((prev) => (prev !== null && kept.has(prev) ? prev : null));
        setSelectedPosition((prev) => (prev !== null && kept.has(prev) ? prev : null));
      } else {
        setTokenList([]);
        setResultMap(new Map());
        setPartialMap(new Map());
        setSelectedTokenPositions(new Set());
        setLockedPosition(null);
        setSelectedPosition(null);
        setMeta(null);
      }

      setError(null);
      setIsLoading(false);
      setIsTokenizingPreview(false);
      // The cached tokenized text no longer matches the truncated chat,
      // so force the next explain run to retokenize before submitting.
      setLastTokenizedText(null);

      setChatMessages(truncated);
    },
    [chatMessages, tokenList, tokenizerFormat],
  );

  // Stable ref so callers (e.g. the explain-429 handler inside
  // `handleSubmit`) can invoke the latest `truncateChatFrom` without
  // listing it as a dep — keeps the handleSubmit closure simple.
  const truncateChatFromRef = useRef(truncateChatFrom);
  truncateChatFromRef.current = truncateChatFrom;

  const handleClear = useCallback(
    (options?: { pinFreeChatDemo?: boolean }) => {
      resetResults();
      setChatMessages([{ role: 'user', content: '' }]);
      setHighlightedParagraph(null);
      setHighlightedRange(null);
      setHighlightComment(null);
      const url = new URL(window.location.href);
      url.searchParams.delete('id');
      url.searchParams.delete('position');
      url.searchParams.delete('paragraph');
      url.searchParams.delete('highlightStart');
      url.searchParams.delete('highlightEnd');
      url.searchParams.delete('comment');
      window.history.replaceState({}, '', url.toString());
      if (options?.pinFreeChatDemo) {
        setActiveDemoCacheId(NLA_FREE_CHAT_DEMO_CACHE_ID);
      }
    },
    [resetResults, setHighlightedParagraph, setHighlightedRange, setHighlightComment],
  );

  const handleShare = useCallback(() => {
    setShareDraft(null);
    setShareError(null);
    setIsShareModalOpen(true);

    const url = new URL(window.location.href);
    const cacheId = url.searchParams.get('id');
    if (!cacheId) {
      setShareError('Nothing has been explained yet.');
      return;
    }

    const focusPosition = lockedPosition ?? selectedPosition;
    let paragraph: number | null = null;
    let highlightStart: number | null = null;
    let highlightEnd: number | null = null;
    if (focusPosition !== null) {
      if (highlightedRange !== null && highlightedRange.start >= 0 && highlightedRange.end > highlightedRange.start) {
        highlightStart = highlightedRange.start;
        highlightEnd = highlightedRange.end;
      } else if (highlightedParagraph !== null && highlightedParagraph >= 0 && highlightedParagraph <= 2) {
        paragraph = highlightedParagraph;
      }
    }

    const initial = focusPosition !== null && highlightComment && highlightComment.length > 0 ? highlightComment : null;
    const draft: NlaShareDraft = {
      cacheId,
      position: focusPosition,
      paragraph,
      highlightStart,
      highlightEnd,
      initialComment: initial,
    };
    const existingShareId = existingFeaturedShareIdForDraft(featuredDemos, selectedModelId, activeDemoCacheId, draft);
    setShareDraft(existingShareId !== undefined ? { ...draft, existingShareId } : draft);
  }, [
    lockedPosition,
    selectedPosition,
    highlightedParagraph,
    highlightedRange,
    highlightComment,
    featuredDemos,
    selectedModelId,
    activeDemoCacheId,
  ]);

  // Share the currently-focused explanation. Same as `handleShare` but
  // additionally pins `?position=` so the recipient lands on the same
  // token's details panel. The optional `override` lets per-paragraph
  // and per-selection share buttons emit a link with `?paragraph=` or
  // `?highlightStart=&highlightEnd=` without first having to round-trip
  // through a state update. For each field: `undefined` falls back to
  // the current state; `null` explicitly omits the param. Range wins
  // over paragraph (matches the in-state mutual-exclusion rule).
  const handleShareExplanation = useCallback(
    (override?: {
      paragraph?: number | null;
      range?: { start: number; end: number } | null;
      comment?: string | null;
    }) => {
      setShareDraft(null);
      setShareError(null);
      setIsShareModalOpen(true);

      const url = new URL(window.location.href);
      const cacheId = url.searchParams.get('id');
      if (!cacheId) {
        setShareError('Nothing has been explained yet.');
        return;
      }
      const focusPosition = lockedPosition ?? selectedPosition;
      if (focusPosition === null) {
        setShareError('No explanation is currently focused.');
        return;
      }

      const range = override?.range !== undefined ? override.range : highlightedRange;
      const paragraphOverride = override?.paragraph !== undefined ? override.paragraph : highlightedParagraph;
      const commentFromOverride = override?.comment !== undefined ? override.comment : highlightComment;

      let paragraph: number | null = null;
      let highlightStart: number | null = null;
      let highlightEnd: number | null = null;
      if (range !== null && range.start >= 0 && range.end > range.start) {
        highlightStart = range.start;
        highlightEnd = range.end;
      } else if (paragraphOverride !== null && paragraphOverride >= 0 && paragraphOverride <= 2) {
        paragraph = paragraphOverride;
      }

      const initial =
        commentFromOverride !== null && commentFromOverride.length > 0
          ? commentFromOverride.length > MAX_COMMENT_LENGTH
            ? commentFromOverride.slice(0, MAX_COMMENT_LENGTH)
            : commentFromOverride
          : null;

      const draft: NlaShareDraft = {
        cacheId,
        position: focusPosition,
        paragraph,
        highlightStart,
        highlightEnd,
        initialComment: initial,
      };
      const existingShareId = existingFeaturedShareIdForDraft(featuredDemos, selectedModelId, activeDemoCacheId, draft);
      setShareDraft(existingShareId !== undefined ? { ...draft, existingShareId } : draft);
    },
    [
      lockedPosition,
      selectedPosition,
      highlightedParagraph,
      highlightedRange,
      highlightComment,
      featuredDemos,
      selectedModelId,
      activeDemoCacheId,
    ],
  );

  const handleSubmit = useCallback(
    async (positionsOverride?: Set<number>) => {
      const fullText = tokenizerFormat.formatChat(chatMessages);
      // Allow callers (specifically the auto-explain-after-chat effect) to
      // pass a freshly-computed selection set rather than relying on the
      // closure's `selectedTokenPositions`. Without this, the auto-explain
      // effect fires synchronously in the same commit phase as the
      // auto-selection effect — its closure captures the previous run's
      // selection, which (after the first explain) is already fully cached
      // in `resultMap`, so handleSubmit early-returns without explaining
      // the new turn's tokens.
      const effectiveSelection = positionsOverride ?? selectedTokenPositions;
      if (!fullText.trim() || isLoading) return;
      if (fullText.length > MAX_TEXT_LENGTH) {
        setError(
          `Max chat character length of ${MAX_TEXT_LENGTH} exceeded. Please start a new chat by clicking "Free Chat".`,
        );
        return;
      }

      // Cancel any pending tokenize-preview so it can't clobber state mid-stream.
      tokenizePreviewAbortRef.current?.abort();
      setIsTokenizingPreview(false);

      const effectiveSourceId = selectedNlaSource?.id;
      const tokensReady = lastTokenizedText === fullText && tokenList.length > 0;

      setIsLoading(true);
      // Keep `resultMap`/`partialMap` intact across follow-up Explain runs
      // so previously-explained tokens retain their results when the user
      // selects more tokens to explain. Tokens reuse stable positions
      // (the chat is append-only) so the existing entries stay valid.
      setError(null);
      if (!tokensReady) {
        setTokenList([]);
        setMeta(null);
      }

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        // Step 1: Tokenize the full text if we don't already have valid tokens.
        let resolvedTokens: TokenInfo[] = tokenList;
        if (!tokensReady) {
          const tokenizeRes = await fetch('/api/nla/completion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: fullText,
              completion_tokens: 0,
              modelId: selectedModelId || undefined,
              nlaSourceId: effectiveSourceId || undefined,
            }),
            signal: controller.signal,
          });

          if (!tokenizeRes.ok) {
            const data = await tokenizeRes.json();
            setError(data.error || `Tokenize error: ${tokenizeRes.status}`);
            setIsLoading(false);
            return;
          }

          const tokenizeData = (await tokenizeRes.json()) as { tokens: TokenInfo[] };
          setTokenList(tokenizeData.tokens);
          setMeta({ layer_index: 0, total: tokenizeData.tokens.length, prompt_length: tokenizeData.tokens.length });
          setLastTokenizedText(fullText);
          resolvedTokens = tokenizeData.tokens;
        }

        // Source of truth for positions to explain. Precedence:
        //   1. The live `selectedTokenPositions` (closure at handleSubmit
        //      creation time, so still useful for "user clicked Explain"
        //      paths where state is in sync).
        //   2. Auto-algo selection over the freshly-tokenized prompt
        //      (catches initial loads where the auto-submit fires before
        //      the parent's tokenList useEffect has populated the
        //      selection state).
        let positionSource: number[];
        if (effectiveSelection.size > 0) {
          positionSource = Array.from(effectiveSelection);
        } else if (resolvedTokens.length > 0) {
          const autoSet = computeAutoSelection(resolvedTokens, tokenizerFormat, MAX_TOKENS_TO_EXPLAIN);
          positionSource = Array.from(autoSet);
          setSelectedTokenPositions(autoSet);
        } else {
          positionSource = [];
        }

        // Only the *new* positions need to hit the upstream NLA server —
        // already-explained ones are kept in `resultMap` and reused.
        const pendingSelected = new Set<number>();
        for (const p of positionSource) {
          if (!resultMap.has(p)) pendingSelected.add(p);
        }

        // Nothing new to explain and we already have results — skip the
        // network round trip rather than re-explaining the entire prompt.
        if (pendingSelected.size === 0 && resultMap.size > 0) {
          setIsLoading(false);
          return;
        }

        if (pendingSelected.size === 0) {
          setError('No tokens selected for explanation.');
          setIsLoading(false);
          return;
        }

        // Per-submission cap applies only to *new* positions; the union
        // sent to the server can grow across submissions and is bounded
        // implicitly by the prompt's token count.
        const pendingCapped = Array.from(pendingSelected)
          .sort((a, b) => a - b)
          .slice(0, MAX_TOKENS_TO_EXPLAIN);

        // Cumulative set: every position the UI is currently showing or
        // about to explain. The server keys the resulting cache row on
        // this exact set, so the id it returns reflects the full
        // visible state and a `?id=...` reload reproduces it.
        const cumulative = new Set<number>(resultMap.keys());
        pendingCapped.forEach((p) => cumulative.add(p));
        const positionsArg = Array.from(cumulative).sort((a, b) => a - b);

        const priorCacheId = priorCacheIdRef.current;

        console.log('[nla-explain] handleSubmit →', {
          textLength: fullText.length,
          textHead: fullText.slice(0, 80),
          textTail: fullText.slice(-80),
          tokenListLength: resolvedTokens.length,
          alreadyExplainedPositions: Array.from(resultMap.keys()).sort((a, b) => a - b),
          positionSource,
          pendingNewPositions: pendingCapped,
          positionsArgSentToServer: positionsArg,
          priorCacheIdSentToServer: priorCacheId,
        });

        // Default-focus the first assistant token among the positions we're
        // about to explain (header → content → footer, including specials).
        // Falls back to the earliest pending position (i.e. the first
        // previously-unexplained token in the submit set) if no assistant
        // token was queued. One-shot on explain start — once the user
        // clicks elsewhere, their selection wins and we never auto-revert.
        const explainSet = new Set(pendingCapped);
        const grouped = groupTokensIntoMessages(resolvedTokens, tokenizerFormat);
        let defaultFocusPosition: number | null = null;
        for (const msg of grouped.messages) {
          if (msg.role !== 'assistant') continue;
          const hit = messageAllTokens(msg).find((t) => explainSet.has(t.position));
          if (hit) {
            defaultFocusPosition = hit.position;
            break;
          }
        }
        if (defaultFocusPosition === null && pendingCapped.length > 0) {
          // `pendingCapped` is already sorted ascending (see above), so
          // [0] is the earliest position the user queued for this run.
          defaultFocusPosition = pendingCapped[0];
        }
        // A new explain run invalidates any previously-pinned token: the
        // user is broadening the explained set, so re-pin onto the freshly
        // computed default focus (or unlock if there's nothing to focus).
        // Locking — not just hovering — so the selection survives the
        // user moving their mouse off the chip while explanations stream.
        if (defaultFocusPosition !== null) {
          setSelectedPosition(defaultFocusPosition);
          setLockedPosition(defaultFocusPosition);
        } else {
          setLockedPosition(null);
        }

        // Step 2: Stream explanations
        const explainRes = await fetch('/api/nla/explain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: fullText,
            temperature,
            modelId: selectedModelId || undefined,
            nlaSourceId: effectiveSourceId || undefined,
            positions: positionsArg,
            // Persisted alongside the cache row so a `?id=...` deep-link
            // can hydrate state without re-tokenizing.
            tokens: resolvedTokens.map((t) => t.token),
            // Lets the server reuse explanations from the prior cache row
            // when the new prompt prefix-extends it (the typical chat
            // continuation case). The route validates model/source/temp
            // and that the new text starts with the prior row's text
            // before trusting it.
            priorCacheId: priorCacheId || undefined,
          }),
          signal: controller.signal,
        });

        if (!explainRes.ok) {
          const data: { detail?: string; error?: string; limitPerWindow?: number } = await explainRes
            .json()
            .catch(() => ({}));
          if (explainRes.status === 429) {
            // Two flavors of 429 hit this branch:
            //   - middleware rate-limit (body has `limitPerWindow`)
            //   - NLA inference server is busy (body is `{ error: "NLA server error: 429" }`)
            // Both warrant the same UX: surface a friendly "try again later"
            // and roll the chat back to before the last user turn so the
            // textarea is repopulated with their message for a clean retry.
            const isRateLimit = typeof data?.limitPerWindow === 'number';
            const message = isRateLimit
              ? 'Hourly message limit reached. Please wait a bit and try again later.'
              : 'Servers busy - please try again later.';
            let lastUserIdx = -1;
            for (let i = chatMessages.length - 1; i >= 0; i -= 1) {
              if (chatMessages[i].role === 'user' && chatMessages[i].content.length > 0) {
                lastUserIdx = i;
                break;
              }
            }
            if (lastUserIdx >= 0) {
              const restoreText = chatMessages[lastUserIdx].content;
              // Drop the failed user turn (and any trailing assistant turn
              // that was streamed before the explain step kicked off).
              // `truncateChatFrom` also clears tokens/results/partials so
              // the chat returns to a "pre-send" state. It calls
              // `setError(null)` and `setIsLoading(false)` internally;
              // we re-set the error message immediately after.
              truncateChatFromRef.current(lastUserIdx);
              setPendingChatInputRestore(restoreText);
            }
            setError(message);
            setIsLoading(false);
            return;
          }
          setError(data.detail || data.error || `Explain error: ${explainRes.status}`);
          setIsLoading(false);
          return;
        }

        if (!explainRes.body) {
          setError('No response body');
          setIsLoading(false);
          return;
        }

        const contentType = explainRes.headers.get('content-type');
        if (contentType?.includes('text/event-stream')) {
          const reader = explainRes.body
            .pipeThrough(new TextDecoderStream())
            .pipeThrough(new EventSourceParserStream())
            .getReader();

          while (true) {
            // eslint-disable-next-line no-await-in-loop
            const { done, value } = await reader.read();
            if (done) break;

            const eventData = value.data;
            if (eventData === '[DONE]') break;

            const parsed = JSON.parse(eventData);

            if ('error' in parsed) {
              setError(parsed.detail || parsed.error || 'NLA server error');
              setIsLoading(false);
              break;
            }

            if ('layer_index' in parsed && 'total' in parsed) {
              setMeta(parsed as ExplainMeta);
            } else if ('done' in parsed && parsed.done === false) {
              const update = parsed as PartialUpdate;
              setPartialMap((prev) => {
                const next = new Map(prev);
                next.set(update.position, update.text);
                return next;
              });
            } else if ('description' in parsed) {
              const result = parsed as ExplainResult;
              setResultMap((prev) => {
                const next = new Map(prev);
                const existing = prev.get(result.position);
                if (existing) {
                  console.warn('[nla-explain] OVERWRITING existing result at position', result.position, {
                    oldDescription: existing.description,
                    newDescription: result.description,
                    changed: existing.description !== result.description,
                  });
                } else {
                  console.log('[nla-explain] new result at position', result.position, {
                    description: result.description,
                  });
                }
                next.set(result.position, result);
                return next;
              });
              // Intentionally retain the partialMap entry after the final
              // result lands. The partial text is the model's raw output
              // (still wrapped in `<explanation>` tags etc.) and the
              // details column surfaces it as the "raw response" view.
              // Chip/details "generating" state already keys on
              // `!result && partialText !== undefined`, so keeping the
              // partial around once `result` exists doesn't regress UI.
            } else if ('cacheId' in parsed && typeof parsed.cacheId === 'string') {
              // Server emits this last (after all results) once the
              // cumulative cache row has been written. Mirror it onto
              // the URL so reloads hydrate the full visible state.
              priorCacheIdRef.current = parsed.cacheId;
              const next = new URL(window.location.href);
              next.searchParams.set('id', parsed.cacheId);
              next.searchParams.delete('position');
              window.history.replaceState({}, '', next.toString());
            }
          }
        } else {
          const data = await explainRes.json();
          if (data.error) {
            setError(data.detail || data.error || 'NLA server error');
            setIsLoading(false);
            return;
          }
          if (data.results) {
            setMeta({
              layer_index: data.layer_index,
              total: data.results.length,
              prompt_length: data.prompt_length ?? data.results.length,
            });
            const map = new Map<number, ExplainResult>(
              data.results.map((r: ExplainResult) => [r.position, r] as const),
            );
            setResultMap(map);
          }
          if (typeof data.cacheId === 'string') {
            priorCacheIdRef.current = data.cacheId;
            const next = new URL(window.location.href);
            next.searchParams.set('id', data.cacheId);
            next.searchParams.delete('position');
            window.history.replaceState({}, '', next.toString());
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          // User cancelled
        } else {
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      } finally {
        setIsLoading(false);
        abortRef.current = null;
      }
    },
    [
      temperature,
      isLoading,
      chatMessages,
      selectedModelId,
      selectedNlaSource,
      tokenizerFormat,
      lastTokenizedText,
      tokenList,
      selectedTokenPositions,
      resultMap,
    ],
  );

  // Auto-explain when a chat stream finishes in top-level 'auto' mode.
  // The ref captures "we just finished a stream that was in auto mode";
  // it's intentionally NOT keyed on the algorithm-level `selectionMode`,
  // since that's reset to 'auto' on every new tokenList.
  const autoExplainAfterChatRef = useRef(false);
  const handleSubmitRef = useRef(handleSubmit);
  handleSubmitRef.current = handleSubmit;

  useEffect(() => {
    if (isChatStreaming && topLevelMode === 'auto') {
      autoExplainAfterChatRef.current = true;
    }
  }, [isChatStreaming, topLevelMode]);

  useEffect(() => {
    if (isChatStreaming) return;
    if (!autoExplainAfterChatRef.current) return;
    if (topLevelMode !== 'auto') {
      autoExplainAfterChatRef.current = false;
      return;
    }
    if (tokenList.length === 0) return;
    // Compute the fresh auto-selection here rather than reading
    // `selectedTokenPositions` — the separate auto-selection effect
    // queues its setState in the same commit, so by the time this effect
    // runs `selectedTokenPositions` is still the previous run's set
    // (already fully cached in `resultMap`, which would make
    // handleSubmit early-return). Pass the fresh set explicitly so
    // handleSubmit doesn't depend on the stale closure.
    const freshAutoSet = computeAutoSelection(tokenList, tokenizerFormat, MAX_TOKENS_TO_EXPLAIN);
    if (freshAutoSet.size === 0) return;
    autoExplainAfterChatRef.current = false;
    setSelectedTokenPositions(freshAutoSet);
    handleSubmitRef.current(freshAutoSet);
  }, [isChatStreaming, topLevelMode, tokenList, tokenizerFormat]);

  // Switching the top-level toggle.
  const handleTopLevelModeChange = useCallback(
    (mode: 'auto' | 'manual') => {
      setTopLevelMode(mode);
      if (mode === 'auto' && tokenList.length > 0) {
        // Manual → Auto: reset selection to the auto algorithm and arm
        // the auto-explain trigger. The existing auto-explain effect
        // already watches `selectedTokenPositions` + `topLevelMode`, so
        // it'll fire handleSubmit on the next render. The pending-
        // selected filter in handleSubmit skips already-explained
        // positions, so this is a no-op when everything's already done.
        const autoSet = computeAutoSelection(tokenList, tokenizerFormat, MAX_TOKENS_TO_EXPLAIN);
        setSelectedTokenPositions(autoSet);
        autoExplainAfterChatRef.current = true;
      }
    },
    [tokenList, tokenizerFormat],
  );

  // Hydrate the explainer directly from a stored cache row — no tokenize
  // or explain network round trip. Used both by the demo buttons and by
  // the `?id=...` deep-link path. The full row (text, tokens, results,
  // sortedPositions) lives in `/api/nla/cache/{id}`.
  const hydrateFromCache = useCallback(
    async (cacheId: string) => {
      setExplanationSearchResetNonce((n) => n + 1);
      suppressDerivedAutoSelectionRef.current = false;
      // Block the id-cleanup useEffect while we apply the cached state.
      // We re-arm it after the next paint so it fires only on user-driven
      // changes from then on.
      isHydratingRef.current = true;
      // Drive the chat/details "Loading…" placeholder. Set synchronously
      // (before any await) so the swap is visible the instant the user
      // clicks a demo button — without this the previous demo's commentary
      // would linger (or disappear ahead of the rest of the swap).
      setIsHydratingDemo(true);
      // Seed the prior-cacheId ref so a follow-up explain can prefix-extend
      // from this row (e.g., user loads a demo, sends another message).
      priorCacheIdRef.current = cacheId;
      try {
        const res = await fetch(`/api/nla/cache/${cacheId}`);
        if (!res.ok) {
          setError('Failed to load cached result');
          return;
        }
        const cache = (await res.json()) as {
          text: string;
          temperature: number;
          modelId: string;
          nlaSourceId: string;
          sortedPositions: number[];
          tokens: string[];
          resultJson: string;
        };

        if (cache.modelId) setSelectedModelId(cache.modelId);
        if (cache.nlaSourceId) {
          const match = nlaSources.find((s) => s.modelId === cache.modelId && s.id === cache.nlaSourceId);
          if (match) setSelectedNlaSourceId(match.id);
        }
        setTemperature(cache.temperature);

        const parsedChat = parseAnyChat(cache.text);
        setChatMessages(parsedChat ? parsedChat.messages : [{ role: 'user', content: cache.text }]);

        // Build the TokenInfo list from the stored token strings. The
        // cache row doesn't preserve per-token IDs (the chips don't use
        // them), so 0 is a fine placeholder.
        const newTokenList: TokenInfo[] = cache.tokens.map((tok, i) => ({
          token: tok,
          token_id: 0,
          position: i,
        }));
        setTokenList(newTokenList);
        setLastTokenizedText(cache.text);

        // Parse the persisted explain payload and rebuild resultMap.
        let parsedPayload: { results: ExplainResult[]; layer_index?: number; prompt_length?: number } = {
          results: [],
        };
        try {
          parsedPayload = JSON.parse(cache.resultJson);
        } catch {
          // ignore — fall through with empty results
        }
        const newResultMap = new Map<number, ExplainResult>();
        (parsedPayload.results || []).forEach((r) => {
          if (typeof r.position === 'number') newResultMap.set(r.position, r);
        });
        setResultMap(newResultMap);
        setPartialMap(new Map());
        setMeta({
          layer_index: parsedPayload.layer_index ?? 0,
          total: newTokenList.length,
          prompt_length: parsedPayload.prompt_length ?? newTokenList.length,
        });

        // Selection mirrors what's been explained.
        const sortedSelected = [...cache.sortedPositions].sort((a, b) => a - b);
        setSelectedTokenPositions(new Set(sortedSelected));
        suppressDerivedAutoSelectionRef.current = true;

        // Default-focus precedence:
        //   1. `?position=N` — only if the cached row actually has a result
        //      at that position. Otherwise we drop the param so the URL
        //      doesn't appear to deep-link to something we can't show.
        //   2. The first explained assistant token in turn order (header
        //      through footer, including chat specials) in the last assistant
        //      message.
        //   3. The lowest explained position (fallback).
        const explainSet = new Set(sortedSelected);
        const initialPosition = initialPositionRef.current;
        initialPositionRef.current = null;
        const initialParagraph = initialParagraphRef.current;
        initialParagraphRef.current = null;
        const initialRange = initialRangeRef.current;
        initialRangeRef.current = null;
        const initialComment = initialCommentRef.current;
        initialCommentRef.current = null;
        let defaultFocus: number | null = null;
        let lockFocus = false;
        if (initialPosition !== null && newResultMap.has(initialPosition)) {
          defaultFocus = initialPosition;
          lockFocus = true;
        } else {
          if (initialPosition !== null && typeof window !== 'undefined') {
            const url = new URL(window.location.href);
            if (
              url.searchParams.has('position') ||
              url.searchParams.has('paragraph') ||
              url.searchParams.has('highlightStart') ||
              url.searchParams.has('highlightEnd') ||
              url.searchParams.has('comment')
            ) {
              url.searchParams.delete('position');
              url.searchParams.delete('paragraph');
              url.searchParams.delete('highlightStart');
              url.searchParams.delete('highlightEnd');
              url.searchParams.delete('comment');
              window.history.replaceState({}, '', url.toString());
            }
          }
          const grouped = groupTokensIntoMessages(newTokenList, tokenizerFormat);
          for (let i = grouped.messages.length - 1; i >= 0; i -= 1) {
            const msg = grouped.messages[i];
            if (msg.role !== 'assistant') continue;
            const hit = messageAllTokens(msg).find((t) => explainSet.has(t.position));
            if (hit) {
              defaultFocus = hit.position;
              break;
            }
          }
          if (defaultFocus === null && sortedSelected.length > 0) {
            defaultFocus = sortedSelected[0];
          }
        }
        setSelectedPosition(defaultFocus);
        setLockedPosition(lockFocus ? defaultFocus : null);
        // Highlights are anchored to the locked position — apply them only
        // if we successfully locked on the URL-requested token. Range
        // takes precedence over paragraph (mirrors the URL-parse
        // precedence and the mutual-exclusion rule in state).
        if (lockFocus && initialRange !== null) {
          setHighlightedRange(initialRange);
        } else if (lockFocus && initialParagraph !== null) {
          setHighlightedParagraph(initialParagraph);
        } else {
          setHighlightedParagraph(null);
          setHighlightedRange(null);
        }
        // Comment is independent of paragraph/range — apply whenever
        // we successfully locked on the URL-requested token.
        if (lockFocus && initialComment !== null && initialComment.length > 0) {
          setHighlightComment(initialComment);
        } else {
          setHighlightComment(null);
        }

        setChatScrollNonce((n) => n + 1);

        setIsLoading(false);
        setError(null);
      } catch {
        setError('Failed to load cached result');
        suppressDerivedAutoSelectionRef.current = false;
      } finally {
        // Co-batched with the success-path setters above (or the catch's
        // setError) so the loading placeholder and the new content swap
        // in a single render — no flash of empty state in between.
        setIsHydratingDemo(false);
        // Defer release past the next render + effect cycle so the
        // hydrate's own setStates don't trip the id-cleanup useEffect.
        // 50ms is well within a frame budget but ample for React to flush.
        setTimeout(() => {
          isHydratingRef.current = false;
        }, 50);
      }
    },
    [nlaSources, tokenizerFormat, setHighlightComment, setHighlightedParagraph, setHighlightedRange],
  );

  const loadCacheById = useCallback(
    async (
      cacheId: string,
      position?: number,
      paragraph?: number,
      highlightStart?: number,
      highlightEnd?: number,
      comment?: string,
    ) => {
      // Bail if another hydrate or explain run is already in flight, so a
      // double-click on a demo button can't race two cache fetches into
      // overlapping state updates.
      if (isLoading || isHydratingDemo) return;
      setActiveDemoCacheId(cacheId);
      const hasPosition = typeof position === 'number' && Number.isFinite(position) && position >= 0;
      const hasRange =
        hasPosition &&
        typeof highlightStart === 'number' &&
        typeof highlightEnd === 'number' &&
        Number.isFinite(highlightStart) &&
        Number.isFinite(highlightEnd) &&
        highlightStart >= 0 &&
        highlightEnd > highlightStart;
      // Range and paragraph are mutually exclusive — range wins when both
      // are passed, mirroring the URL-parse precedence.
      const hasParagraph =
        hasPosition &&
        !hasRange &&
        typeof paragraph === 'number' &&
        Number.isFinite(paragraph) &&
        paragraph >= 0 &&
        paragraph <= 2;
      const hasComment = hasPosition && typeof comment === 'string' && comment.length > 0;
      const cappedComment = hasComment
        ? comment!.length > MAX_COMMENT_LENGTH
          ? comment!.slice(0, MAX_COMMENT_LENGTH)
          : comment!
        : null;
      if (hasPosition) {
        initialPositionRef.current = position;
      }
      if (hasRange) {
        initialRangeRef.current = { start: highlightStart, end: highlightEnd };
      } else if (hasParagraph) {
        initialParagraphRef.current = paragraph;
      }
      if (cappedComment !== null) {
        initialCommentRef.current = cappedComment;
      }
      // Sync the URL up-front so the share flow (which reads `?id=` from
      // window.location) works after a demo click. `isHydratingRef` is
      // flipped on inside hydrateFromCache, but we set it here too so the
      // id-cleanup effect doesn't wipe what we just wrote in the gap
      // before hydrate runs. Mirrors the deep-link path.
      isHydratingRef.current = true;
      if (typeof window !== 'undefined') {
        const url = new URL(window.location.href);
        url.searchParams.set('id', cacheId);
        if (hasPosition) {
          url.searchParams.set('position', String(position));
        } else {
          url.searchParams.delete('position');
        }
        if (hasRange) {
          url.searchParams.set('highlightStart', String(highlightStart));
          url.searchParams.set('highlightEnd', String(highlightEnd));
          url.searchParams.delete('paragraph');
        } else if (hasParagraph) {
          url.searchParams.set('paragraph', String(paragraph));
          url.searchParams.delete('highlightStart');
          url.searchParams.delete('highlightEnd');
        } else {
          url.searchParams.delete('paragraph');
          url.searchParams.delete('highlightStart');
          url.searchParams.delete('highlightEnd');
        }
        if (cappedComment !== null) {
          url.searchParams.set('comment', cappedComment);
        } else {
          url.searchParams.delete('comment');
        }
        window.history.replaceState({}, '', url.toString());
      }
      await hydrateFromCache(cacheId);
    },
    [isLoading, isHydratingDemo, hydrateFromCache],
  );

  // Auto-submit on initial load: `?id=...` hydrates directly from the
  // stored cache row — no tokenize / explain network calls.
  useEffect(() => {
    if (initialCacheIdRef.current) {
      const cacheId = initialCacheIdRef.current;
      initialCacheIdRef.current = null;
      hydrateFromCache(cacheId);
    }
  }, [hydrateFromCache]);

  const onUserEdit = useCallback(() => {
    setActiveDemoCacheId((prev) => (prev === NLA_FREE_CHAT_DEMO_CACHE_ID ? prev : null));
    suppressDerivedAutoSelectionRef.current = false;
  }, []);

  const cancelPendingAutoExplain = useCallback(() => {
    autoExplainAfterChatRef.current = false;
  }, []);

  const explainDisabled = chatMessages.every((m) => !m.content.trim()) || isTokenizingPreview;

  const contextValue = useMemo<NLAContextType>(
    () => ({
      isEmbed,
      nlaSources,
      selectedModelId,
      selectedNlaSourceId,
      modelIds,
      modelDisplayMap,
      filteredSources,
      selectedNlaSource,
      tokenizerFormat,
      handleModelChange,
      handleSourceChange,
      chatMessages,
      setChatMessages,
      inputMode,
      setInputMode,
      isChatStreaming,
      setIsChatStreaming,
      temperature,
      setTemperature,
      maxNewTokens,
      setMaxNewTokens,
      showAdvanced,
      setShowAdvanced,
      tokenList,
      setTokenList,
      setLastTokenizedText,
      selectedTokenPositions,
      topLevelMode,
      handleApplySelection,
      handleTopLevelModeChange,
      resultMap,
      partialMap,
      selectedPosition,
      setSelectedPosition,
      lockedPosition,
      setLockedPosition,
      highlightedParagraph,
      setHighlightedParagraph,
      highlightedRange,
      setHighlightedRange,
      highlightComment,
      setHighlightComment,
      isLoading,
      isHydratingDemo,
      error,
      pendingChatInputRestore,
      setPendingChatInputRestore,
      handleSubmit,
      handleClear,
      handleShare,
      handleShareExplanation,
      loadCacheById,
      featuredDemos,
      activeDemoCacheId,
      setActiveDemoCacheId,
      onUserEdit,
      explainDisabled,
      cancelPendingAutoExplain,
      truncateChatFrom,
      isShareModalOpen,
      setIsShareModalOpen,
      shareDraft,
      shareError,
      detailsColumnRef,
      chatScrollNonce,
      explanationSearchNeedle,
      setExplanationSearchNeedle,
      explanationSearchResetNonce,
    }),
    [
      isEmbed,
      nlaSources,
      selectedModelId,
      selectedNlaSourceId,
      modelIds,
      modelDisplayMap,
      filteredSources,
      selectedNlaSource,
      tokenizerFormat,
      handleModelChange,
      handleSourceChange,
      chatMessages,
      inputMode,
      isChatStreaming,
      temperature,
      maxNewTokens,
      showAdvanced,
      tokenList,
      selectedTokenPositions,
      topLevelMode,
      handleApplySelection,
      handleTopLevelModeChange,
      resultMap,
      partialMap,
      selectedPosition,
      lockedPosition,
      highlightedParagraph,
      setHighlightedParagraph,
      highlightedRange,
      setHighlightedRange,
      highlightComment,
      setHighlightComment,
      isLoading,
      isHydratingDemo,
      error,
      pendingChatInputRestore,
      handleSubmit,
      handleClear,
      handleShare,
      handleShareExplanation,
      loadCacheById,
      featuredDemos,
      activeDemoCacheId,
      onUserEdit,
      explainDisabled,
      cancelPendingAutoExplain,
      truncateChatFrom,
      isShareModalOpen,
      shareDraft,
      shareError,
      chatScrollNonce,
      explanationSearchNeedle,
      explanationSearchResetNonce,
    ],
  );

  return <NLAContext.Provider value={contextValue}>{children}</NLAContext.Provider>;
}

export function useNlaContext() {
  const context = useContext(NLAContext);
  if (context === undefined) {
    throw new Error('useNlaContext must be used within an NLAProvider');
  }
  return context;
}
