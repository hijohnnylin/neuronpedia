import { CONFIDENCE_THRESHOLD, MAX_TOKENS_TO_EXPLAIN } from '@/lib/nla-constants';
import { ChatMessage, TokenInfo, TokenMessageGroup } from './nla-types';

// Re-exported so existing client modules can keep importing it from here.
export { MAX_TOKENS_TO_EXPLAIN };

// Stable key identifying "the current tokenList corresponds to these chat
// turns", replacing the old rendered-template-string comparison (the frontend
// no longer renders chat templates — the NLA server does).
export function chatMessagesKey(messages: ChatMessage[]): string {
  return JSON.stringify(messages.map((m) => [m.role, m.content]));
}

export function scoreColor(score: number | null): string {
  if (score === null) return 'bg-slate-100';
  if (score >= 0.7) return 'bg-emerald-100';
  if (score >= 0.5) return 'bg-yellow-100';
  if (score >= 0.3) return 'bg-orange-100';
  return 'bg-red-100';
}

export function scoreBorderColor(score: number | null): string {
  if (score === null) return 'border-slate-300';
  if (score >= 0.7) return 'border-emerald-400';
  if (score >= 0.5) return 'border-yellow-400';
  if (score >= 0.3) return 'border-orange-400';
  return 'border-red-400';
}

// Maps a Relative-MSE score to a coarse, human-readable confidence label and a
// matching text-color class. Buckets are anchored at CONFIDENCE_THRESHOLD
// (interpreted as "RMSE at-or-above this is low confidence"): "high" when
// comfortably below (CONFIDENCE_THRESHOLD - 0.2), "medium" when below the
// threshold, and "low" at-or-above — keeping the chat underline and details
// pill in sync. Lower RMSE = better reconstruction.
export function confidenceLabel(score: number | null): { label: string; color: string } {
  if (score === null) return { label: 'Unknown', color: 'text-slate-500' };
  if (score < CONFIDENCE_THRESHOLD - 0.2) return { label: 'High', color: 'text-sky-600' };
  if (score < CONFIDENCE_THRESHOLD) return { label: 'Medium', color: 'text-sky-600' };
  return { label: 'Low', color: 'text-orange-500' };
}

// Relative MSE: MSE(norm(pred), norm(target)) / Var(dataset), where the
// denominator (`norm`) is the source's mean MSE for predicting the dataset
// mean of normed vectors. 0 = perfect reconstruction, 1 = no better than
// predicting the mean, > 1 = worse than the mean predictor.
// `mse` is optional on the wire, not just nullable: the server omits it when no reconstructor is
// loaded, so callers reading it off an ExplainResult can hand us undefined.
export function computeRelativeMse(mse: number | null | undefined, norm: number): number | null {
  if (mse === null || mse === undefined || norm <= 0) return null;
  return mse / norm;
}

export function cleanPartialText(raw: string): string {
  return raw
    .replace(/<\/?explanation>/g, '')
    .replace(/<explanation\s*$/g, '')
    .replace(/<\/explanation\s*$/g, '')
    .trim();
}

// Which display bucket a token's server-computed `section` maps to. Structural
// scaffolding + headers render as header chips; turn-end markers as footer
// chips; everything else (message content / generated text) as content.
type SectionBucket = 'header' | 'content' | 'footer';
function sectionBucket(section: string | null | undefined): SectionBucket {
  if (section === 'footer') return 'footer';
  if (section === 'header' || section === 'scaffold') return 'header';
  return 'content';
}

// Whether any token carries server-computed span metadata. When false, no chat
// grouping is possible (raw-text) and the caller renders tokens plainly.
function hasSpans(tokens: TokenInfo[]): boolean {
  return tokens.some((t) => t.section != null || t.role != null || t.message_index != null);
}

// Group the flat, span-tagged token stream into per-message bubbles. Purely
// span-driven (role / section / message_index from the NLA server) — no
// per-model-family token knowledge. Mirrors the jlens `groupTokensBySpans`.
//
// A new group opens when the source message index changes, or (within the
// generated assistant response) when a new harmony block begins (a header token
// following a non-header token).
export function groupTokensIntoMessages(tokens: TokenInfo[]): {
  messages: TokenMessageGroup[];
  hasChatFormat: boolean;
} {
  if (tokens.length === 0 || !hasSpans(tokens)) {
    return { messages: [], hasChatFormat: false };
  }

  const groupIds: number[] = new Array(tokens.length).fill(0);
  let groupId = -1;
  let curKey: string | null = null;
  let prevSection: string | null = null;
  let genBlock = 0;
  for (let i = 0; i < tokens.length; i += 1) {
    const t = tokens[i];
    const mi = t.message_index ?? null;
    let key: string;
    if (mi != null) {
      key = `msg:${mi}`;
    } else {
      if (t.section === 'header' && prevSection != null && prevSection !== 'header') {
        genBlock += 1;
      }
      key = `gen:${genBlock}`;
    }
    if (key !== curKey) {
      groupId += 1;
      curKey = key;
    }
    groupIds[i] = groupId;
    prevSection = t.section ?? null;
  }

  const messages: TokenMessageGroup[] = [];
  let lastId = -1;
  for (let i = 0; i < tokens.length; i += 1) {
    const t = tokens[i];
    if (groupIds[i] !== lastId) {
      messages.push({ role: 'assistant', headerTokens: [], contentTokens: [], footerTokens: [] });
      lastId = groupIds[i];
    }
    const group = messages[messages.length - 1];
    const bucket = sectionBucket(t.section);
    if (bucket === 'header') group.headerTokens.push(t);
    else if (bucket === 'footer') group.footerTokens.push(t);
    else group.contentTokens.push(t);
    if (t.role != null) {
      group.role = t.role === 'user' ? 'user' : 'assistant';
    }
  }

  return { messages, hasChatFormat: messages.length > 0 };
}

export function messageAllTokens(msg: TokenMessageGroup): TokenInfo[] {
  return [...msg.headerTokens, ...msg.contentTokens, ...msg.footerTokens];
}

export function computeLastSelection(tokens: TokenInfo[], maxTokens: number): Set<number> {
  return new Set(tokens.slice(-maxTokens).map((t) => t.position));
}

export function computeLastUserSelection(tokens: TokenInfo[], maxTokens: number): Set<number> {
  const grouped = groupTokensIntoMessages(tokens);
  if (!grouped.hasChatFormat) {
    return computeLastSelection(tokens, maxTokens);
  }
  let lastUserIdx = -1;
  for (let i = grouped.messages.length - 1; i >= 0; i -= 1) {
    if (grouped.messages[i].role === 'user') {
      lastUserIdx = i;
      break;
    }
  }
  if (lastUserIdx < 0) {
    return computeLastSelection(tokens, maxTokens);
  }
  const userTokens = messageAllTokens(grouped.messages[lastUserIdx]);
  return new Set(userTokens.slice(-maxTokens).map((t) => t.position));
}

export function computeAutoSelection(tokens: TokenInfo[], maxTokens: number): Set<number> {
  if (tokens.length <= maxTokens) {
    return new Set(tokens.map((t) => t.position));
  }

  const grouped = groupTokensIntoMessages(tokens);
  if (!grouped.hasChatFormat || grouped.messages.length === 0) {
    return computeLastSelection(tokens, maxTokens);
  }

  const lastMsg = grouped.messages[grouped.messages.length - 1];
  if (lastMsg.role === 'user') {
    return computeLastSelection(tokens, maxTokens);
  }

  // Last message is assistant: 1/5 from end of last user, 4/5 from start of last assistant.
  const userPart = Math.floor(maxTokens / 5);
  const assistantPart = maxTokens - userPart;

  const positions = new Set<number>();

  let lastUserIdx = -1;
  for (let i = grouped.messages.length - 2; i >= 0; i -= 1) {
    if (grouped.messages[i].role === 'user') {
      lastUserIdx = i;
      break;
    }
  }
  if (lastUserIdx >= 0) {
    const userTokens = messageAllTokens(grouped.messages[lastUserIdx]);
    userTokens.slice(-userPart).forEach((t) => positions.add(t.position));
  }

  const assistantTokens = messageAllTokens(lastMsg);
  assistantTokens.slice(0, assistantPart).forEach((t) => positions.add(t.position));

  return positions;
}
