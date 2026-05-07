import { CONFIDENCE_THRESHOLD, MAX_TOKENS_TO_EXPLAIN } from '@/lib/nla-constants';
import { ChatMessage, ChatTokenizerFormat, TokenInfo, TokenMessageGroup } from './nla-types';

// Re-exported so existing client modules can keep importing it from here.
export { MAX_TOKENS_TO_EXPLAIN };

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
export function computeRelativeMse(mse: number | null, norm: number): number | null {
  if (mse === null || norm <= 0) return null;
  return mse / norm;
}

export function cleanPartialText(raw: string): string {
  return raw
    .replace(/<\/?explanation>/g, '')
    .replace(/<explanation\s*$/g, '')
    .replace(/<\/explanation\s*$/g, '')
    .trim();
}

export function groupTokensIntoMessages(
  tokens: TokenInfo[],
  fmt: ChatTokenizerFormat,
): {
  messages: TokenMessageGroup[];
  hasChatFormat: boolean;
} {
  const hasTurnStart = tokens.some((t) => fmt.isSpecialToken(t.token));
  if (!hasTurnStart) {
    return { messages: [], hasChatFormat: false };
  }

  const messages: TokenMessageGroup[] = [];
  let i = 0;

  while (i < tokens.length) {
    const tok = tokens[i];

    if (tok.token === fmt.turnStartToken) {
      const headerTokens: TokenInfo[] = [tok];
      i += 1;

      let role: 'user' | 'assistant' = 'user';
      while (i < tokens.length) {
        const t = tokens[i];
        headerTokens.push(t);
        if (t.token.includes('user')) role = 'user';
        if (t.token.includes(fmt.assistantRoleName)) role = 'assistant';
        i += 1;
        if (t.token.includes('\n')) break;
      }

      const contentTokens: TokenInfo[] = [];
      while (i < tokens.length && tokens[i].token !== fmt.turnEndToken && tokens[i].token !== fmt.turnStartToken) {
        contentTokens.push(tokens[i]);
        i += 1;
      }

      const footerTokens: TokenInfo[] = [];
      if (i < tokens.length && tokens[i].token === fmt.turnEndToken) {
        footerTokens.push(tokens[i]);
        i += 1;
        while (i < tokens.length && tokens[i].token !== fmt.turnStartToken) {
          footerTokens.push(tokens[i]);
          i += 1;
        }
      }

      messages.push({ role, headerTokens, contentTokens, footerTokens });
    } else {
      if (messages.length > 0) {
        messages[messages.length - 1].contentTokens.push(tok);
      }
      i += 1;
    }
  }

  return { messages, hasChatFormat: true };
}

export function messageAllTokens(msg: TokenMessageGroup): TokenInfo[] {
  return [...msg.headerTokens, ...msg.contentTokens, ...msg.footerTokens];
}

/**
 * Pre-tokenization chat bubble: header / body / footer segments match
 * `formatSingleTurn` so `renderTokenGroup` can reuse the same newline, ↵,
 * and blank-line rules as real chips. Pass `{ preview: true }`; header/footer
 * tokens then get the same flex wrapper classes as tokenized bubbles
 * (`buildLines` + `renderChip` preview branch).
 */
export function buildSyntheticUserTurnPreviewGroup(
  fmt: ChatTokenizerFormat,
  message: ChatMessage,
  messageIndex: number,
  totalMessages: number,
  positionBase: number,
): TokenMessageGroup {
  const isLast = messageIndex === totalMessages - 1;
  let header: string;
  let footer: string;
  if (fmt.id === 'qwen') {
    header = `<|im_start|>${message.role}\n`;
    footer = isLast ? '' : '<|im_end|>\n';
  } else if (fmt.id === 'gemma3') {
    const roleName = message.role === 'assistant' ? 'model' : message.role;
    header = `<start_of_turn>${roleName}\n`;
    footer = isLast ? '' : '<end_of_turn>\n';
  } else if (fmt.id === 'llama3') {
    // Real `formatSingleTurn` uses `\n\n` before content; preview uses one `\n` so the
    // header chip does not show a double line break before the body (inference unchanged).
    header = `<|start_header_id|>${message.role}<|end_header_id|>\n`;
    footer = isLast ? '' : '<|eot_id|>';
  } else {
    const full = fmt.formatSingleTurn(message, messageIndex, totalMessages);
    const at = full.indexOf(message.content);
    if (at < 0) {
      return {
        role: message.role,
        headerTokens: [],
        contentTokens: [{ token: full, token_id: 0, position: positionBase }],
        footerTokens: [],
      };
    }
    header = full.slice(0, at);
    footer = full.slice(at + message.content.length);
  }

  let pos = positionBase;
  const headerTokens: TokenInfo[] = header ? [{ token: header, token_id: 0, position: pos++ }] : [];
  const contentTokens: TokenInfo[] =
    message.content.length > 0 ? [{ token: message.content, token_id: 0, position: pos++ }] : [];
  const footerTokens: TokenInfo[] = footer ? [{ token: footer, token_id: 0, position: pos++ }] : [];

  return {
    role: message.role,
    headerTokens,
    contentTokens,
    footerTokens,
  };
}

export function computeLastSelection(tokens: TokenInfo[], maxTokens: number): Set<number> {
  return new Set(tokens.slice(-maxTokens).map((t) => t.position));
}

export function computeLastUserSelection(
  tokens: TokenInfo[],
  fmt: ChatTokenizerFormat,
  maxTokens: number,
): Set<number> {
  const grouped = groupTokensIntoMessages(tokens, fmt);
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

export function computeAutoSelection(tokens: TokenInfo[], fmt: ChatTokenizerFormat, maxTokens: number): Set<number> {
  if (tokens.length <= maxTokens) {
    return new Set(tokens.map((t) => t.position));
  }

  const grouped = groupTokensIntoMessages(tokens, fmt);
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
