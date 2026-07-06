// Chat formatting + token grouping for the jlens chat interface.
//
// The inference server applies the model's real chat template server-side
// (it receives a structured `chat: [{role, content}]` payload). This module
// owns the *client-side* knowledge needed to:
//   1. detect which family a model belongs to (by model id), and
//   2. group the flat per-position token stream the server returns back into
//      user / assistant message bubbles for display, using each family's
//      special turn tokens.
//
// To support a new model family, implement a `JlensChatFormat` and add it to
// `JLENS_CHAT_FORMATS` below (order matters only for `detectChatFormat`, which
// falls through to the qwen default).

import { LensChatMessage, LensTokenMessage } from '@/lib/utils/lens';

export type ChatRole = 'user' | 'assistant';

export interface JlensChatFormat {
  id: string;
  // Substrings of the model id that select this format (lower-cased match).
  matchModelId: (lowerModelId: string) => boolean;
  // The token string that opens a turn (e.g. `<|im_start|>`).
  turnStartToken: string;
  // The token string that closes a turn (e.g. `<|im_end|>`).
  turnEndToken: string;
  // The role label this family uses for the assistant in its header
  // (e.g. gemma uses `model`).
  assistantRoleName: string;
  // Whether a decoded token is one of this family's structural/special tokens.
  isSpecialToken: (token: string) => boolean;
}

// Qwen / ChatML family. Used by Qwen3 (incl. Qwen3.6) and the generic
// ChatML fallback.
const qwenFormat: JlensChatFormat = {
  id: 'qwen',
  matchModelId: (id) => id.includes('qwen') || id.includes('chatml'),
  turnStartToken: '<|im_start|>',
  turnEndToken: '<|im_end|>',
  assistantRoleName: 'assistant',
  isSpecialToken: (token) => token === '<|im_start|>' || token === '<|im_end|>' || token === '<|endoftext|>',
};

// Gemma 3 family.
const gemma3Format: JlensChatFormat = {
  id: 'gemma3',
  matchModelId: (id) => id.includes('gemma'),
  turnStartToken: '<start_of_turn>',
  turnEndToken: '<end_of_turn>',
  assistantRoleName: 'model',
  isSpecialToken: (token) =>
    token === '<start_of_turn>' || token === '<end_of_turn>' || token === '<bos>' || token === '<eos>',
};

// Ordered list of known formats. `detectChatFormat` returns the first match,
// falling back to the qwen/ChatML default.
export const JLENS_CHAT_FORMATS: JlensChatFormat[] = [gemma3Format, qwenFormat];

export function detectChatFormat(modelId: string): JlensChatFormat {
  const lower = (modelId || '').toLowerCase();
  for (const fmt of JLENS_CHAT_FORMATS) {
    if (fmt.matchModelId(lower)) {
      return fmt;
    }
  }
  return qwenFormat;
}

export function toChatPayload(messages: { role: ChatRole; content: string }[]): LensChatMessage[] {
  return messages.map((m) => ({ role: m.role, content: m.content }));
}

export interface JlensTokenGroup {
  role: ChatRole;
  headerTokens: LensTokenMessage[];
  contentTokens: LensTokenMessage[];
  footerTokens: LensTokenMessage[];
}

// Group the flat per-position token stream into message bubbles using the
// family's turn tokens. Mirrors the NLA chat grouping: a turn opens at
// `turnStartToken`, its header runs to the first newline (carrying the role
// label), content runs until the next turn-end / turn-start, and the footer
// captures the closing turn-end (plus any trailing tokens before the next
// turn). Tokens before the first turn (e.g. a leading BOS) are attached to a
// synthetic leading group so they still render and remain hoverable.
export function groupTokensIntoMessages(
  tokens: LensTokenMessage[],
  fmt: JlensChatFormat,
): { messages: JlensTokenGroup[]; hasChatFormat: boolean } {
  const hasTurnStart = tokens.some((t) => t.token === fmt.turnStartToken);
  if (!hasTurnStart) {
    return { messages: [], hasChatFormat: false };
  }

  const messages: JlensTokenGroup[] = [];
  let i = 0;

  // Leading tokens before the first turn (BOS etc.) → prepend to the first
  // header once we open it.
  const leading: LensTokenMessage[] = [];
  while (i < tokens.length && tokens[i].token !== fmt.turnStartToken) {
    leading.push(tokens[i]);
    i += 1;
  }

  let isFirstTurn = true;
  while (i < tokens.length) {
    const tok = tokens[i];

    if (tok.token === fmt.turnStartToken) {
      const headerTokens: LensTokenMessage[] = isFirstTurn && leading.length > 0 ? [...leading, tok] : [tok];
      isFirstTurn = false;
      i += 1;

      let role: ChatRole = 'user';
      while (i < tokens.length) {
        const t = tokens[i];
        headerTokens.push(t);
        if (t.token.includes('user')) role = 'user';
        if (t.token.includes(fmt.assistantRoleName)) role = 'assistant';
        i += 1;
        if (t.token.includes('\n')) break;
      }

      const contentTokens: LensTokenMessage[] = [];
      while (i < tokens.length && tokens[i].token !== fmt.turnEndToken && tokens[i].token !== fmt.turnStartToken) {
        contentTokens.push(tokens[i]);
        i += 1;
      }

      const footerTokens: LensTokenMessage[] = [];
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
