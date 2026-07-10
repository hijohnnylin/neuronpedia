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

export interface JlensTokenGroup {
  role: ChatRole;
  headerTokens: LensTokenMessage[];
  contentTokens: LensTokenMessage[];
  footerTokens: LensTokenMessage[];
  // Optional raw role label (e.g. `system`, `developer`) for formats that
  // surface more roles than the `user`/`assistant` display split. `role` above
  // is the display side (user = right, everything else = left); `roleLabel`
  // preserves the true role so callers can map bubbles back to chat messages.
  roleLabel?: string;
  // Optional harmony channel (`analysis` / `final` / `commentary`) for models
  // whose assistant turns are split into channels (gpt-oss). Undefined for
  // simple ChatML/Gemma turns.
  channel?: string;
}

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
  // Optional format-specific grouping. When present it fully replaces the
  // default turn-start/turn-end grouping — used by families whose structure the
  // default state machine can't express (gpt-oss harmony: channels + multiple
  // end markers). Groups produced this way may not map 1:1 to chat messages
  // (e.g. an injected system turn), so inline message editing is disabled for
  // such formats (see `jlens-chat.tsx`).
  groupTokens?: (tokens: LensTokenMessage[]) => { messages: JlensTokenGroup[]; hasChatFormat: boolean };
  // Optional extractor for the clean, human-readable assistant text from a raw
  // completion string (used to store the assistant turn for re-send / copy).
  // Defaults to stripping trailing `turnEndToken`s. Harmony overrides this to
  // pull out just the `final` channel content.
  parseCompletion?: (completion: string) => string;
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

// Llama 3.x family. Each turn is wrapped as
//   <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
// with the leading <|begin_of_text|> BOS added by the tokenizer. The header
// runs `<|start_header_id|>` `{role}` `<|end_header_id|>` `\n\n`, so the default
// turn-start/turn-end grouping (which reads the header up to the first newline)
// picks up the role and closes the header on the `\n\n` token.
const llama3Format: JlensChatFormat = {
  id: 'llama3',
  matchModelId: (id) => id.includes('llama'),
  turnStartToken: '<|start_header_id|>',
  turnEndToken: '<|eot_id|>',
  assistantRoleName: 'assistant',
  isSpecialToken: (token) =>
    token === '<|start_header_id|>' ||
    token === '<|end_header_id|>' ||
    token === '<|eot_id|>' ||
    token === '<|begin_of_text|>' ||
    token === '<|end_of_text|>',
};

// --------------------------------------------------------------------------
// gpt-oss (OpenAI Harmony) family.
//
// Harmony is structurally different from ChatML/Gemma: a message opens with
// `<|start|>`, carries a role then an optional `<|channel|>` + channel name,
// then `<|message|>`, its content, and closes with `<|end|>` (intermediate) or
// `<|return|>` (the final assistant reply) / `<|call|>` (a tool call). A single
// assistant response is emitted as SEPARATE messages per channel, e.g.:
//   <|start|>assistant<|channel|>analysis<|message|>…reasoning…<|end|>
//   <|start|>assistant<|channel|>final<|message|>…answer…<|return|>
// so the default single-turn state machine can't group it. `groupHarmonyTokens`
// walks this structure directly.
// --------------------------------------------------------------------------
const HARMONY_START = '<|start|>';
const HARMONY_END = '<|end|>';
const HARMONY_MESSAGE = '<|message|>';
const HARMONY_CHANNEL = '<|channel|>';
const HARMONY_RETURN = '<|return|>';
const HARMONY_CALL = '<|call|>';
// Tokens that close a harmony message (any of these ends the current turn's
// content). `<|start|>` also implicitly ends the previous turn.
const HARMONY_END_TOKENS = new Set([HARMONY_END, HARMONY_RETURN, HARMONY_CALL]);

const gptOssFormat: JlensChatFormat = {
  id: 'gpt-oss',
  matchModelId: (id) => id.includes('gpt-oss'),
  turnStartToken: HARMONY_START,
  // The generic "turn end"; harmony actually uses several (see END_TOKENS).
  turnEndToken: HARMONY_END,
  assistantRoleName: 'assistant',
  isSpecialToken: (token) => {
    const t = token.trim();
    return (
      t === HARMONY_START ||
      t === HARMONY_END ||
      t === HARMONY_MESSAGE ||
      t === HARMONY_CHANNEL ||
      t === HARMONY_RETURN ||
      t === HARMONY_CALL ||
      t === '<|endoftext|>'
    );
  },
  groupTokens: (tokens) => groupHarmonyTokens(tokens),
  parseCompletion: (completion) => parseHarmonyCompletion(completion),
};

// Group a harmony token stream into per-message bubbles. Everything from
// `<|start|>` through `<|message|>` (role + optional channel markers) becomes
// the dim header; the message body is content; the closing end marker (plus any
// stray trailing tokens before the next turn) is the footer.
function groupHarmonyTokens(tokens: LensTokenMessage[]): { messages: JlensTokenGroup[]; hasChatFormat: boolean } {
  const hasTurnStart = tokens.some((t) => t.token.trim() === HARMONY_START);
  if (!hasTurnStart) {
    return { messages: [], hasChatFormat: false };
  }

  const messages: JlensTokenGroup[] = [];
  let i = 0;

  // Tokens before the first `<|start|>` (rare for harmony) → prepend to the
  // first header so they still render and stay hoverable.
  const leading: LensTokenMessage[] = [];
  while (i < tokens.length && tokens[i].token.trim() !== HARMONY_START) {
    leading.push(tokens[i]);
    i += 1;
  }

  let isFirstTurn = true;
  while (i < tokens.length) {
    if (tokens[i].token.trim() !== HARMONY_START) {
      // Stray token outside a turn: attach to the current bubble's content.
      if (messages.length > 0) {
        messages[messages.length - 1].contentTokens.push(tokens[i]);
      }
      i += 1;
      continue;
    }

    const headerTokens: LensTokenMessage[] = isFirstTurn && leading.length > 0 ? [...leading, tokens[i]] : [tokens[i]];
    isFirstTurn = false;
    i += 1;

    // Header: role text, then (optionally) `<|channel|>` + channel name, up to
    // and including `<|message|>`.
    let roleText = '';
    let channelText = '';
    let sawChannel = false;
    while (i < tokens.length) {
      const t = tokens[i];
      const trimmed = t.token.trim();
      // A malformed turn (no `<|message|>`): bail before consuming the marker.
      if (trimmed === HARMONY_START || HARMONY_END_TOKENS.has(trimmed)) {
        break;
      }
      headerTokens.push(t);
      i += 1;
      if (trimmed === HARMONY_MESSAGE) {
        break;
      }
      if (trimmed === HARMONY_CHANNEL) {
        sawChannel = true;
      } else if (sawChannel) {
        channelText += t.token;
      } else {
        roleText += t.token;
      }
    }

    const roleLabel = roleText.trim();
    const role: ChatRole = roleLabel === 'user' ? 'user' : 'assistant';
    const channel = channelText.trim();

    const contentTokens: LensTokenMessage[] = [];
    while (
      i < tokens.length &&
      !HARMONY_END_TOKENS.has(tokens[i].token.trim()) &&
      tokens[i].token.trim() !== HARMONY_START
    ) {
      contentTokens.push(tokens[i]);
      i += 1;
    }

    const footerTokens: LensTokenMessage[] = [];
    if (i < tokens.length && HARMONY_END_TOKENS.has(tokens[i].token.trim())) {
      footerTokens.push(tokens[i]);
      i += 1;
    }

    messages.push({ role, roleLabel, channel, headerTokens, contentTokens, footerTokens });
  }

  return { messages, hasChatFormat: true };
}

// Pull the clean assistant answer out of a raw harmony completion: the `final`
// channel's message content (the analysis/reasoning channel is dropped, which
// matches harmony's own behavior of stripping prior reasoning from history).
// While streaming (before `final` arrives) fall back to the text after the last
// `<|message|>`, with any leftover harmony markers stripped.
function parseHarmonyCompletion(completion: string): string {
  const finalMatch = completion.match(
    /<\|channel\|>final<\|message\|>([\s\S]*?)(?:<\|return\|>|<\|end\|>|<\|call\|>|$)/,
  );
  if (finalMatch) {
    return finalMatch[1].trim();
  }
  const lastMsg = completion.lastIndexOf(HARMONY_MESSAGE);
  const tail = lastMsg >= 0 ? completion.slice(lastMsg + HARMONY_MESSAGE.length) : completion;
  return tail.replace(/<\|[^|]*\|>/g, '').trim();
}

// Ordered list of known formats. `detectChatFormat` returns the first match,
// falling back to the qwen/ChatML default.
export const JLENS_CHAT_FORMATS: JlensChatFormat[] = [gemma3Format, gptOssFormat, llama3Format, qwenFormat];

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

// Group the flat per-position token stream into message bubbles using the
// family's turn tokens. Mirrors the NLA chat grouping: a turn opens at
// `turnStartToken`, its header runs to the first newline (carrying the role
// label), content runs until the next turn-end / turn-start, and the footer
// captures the closing turn-end (plus any trailing tokens before the next
// turn). Tokens before the first turn (e.g. a leading BOS) are attached to a
// synthetic leading group so they still render and remain hoverable.
//
// Formats that need bespoke structure (e.g. gpt-oss harmony) supply their own
// `groupTokens`, which fully replaces the logic below.
export function groupTokensIntoMessages(
  tokens: LensTokenMessage[],
  fmt: JlensChatFormat,
): { messages: JlensTokenGroup[]; hasChatFormat: boolean } {
  if (fmt.groupTokens) {
    return fmt.groupTokens(tokens);
  }
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

      messages.push({ role, roleLabel: role, headerTokens, contentTokens, footerTokens });
    } else {
      if (messages.length > 0) {
        messages[messages.length - 1].contentTokens.push(tok);
      }
      i += 1;
    }
  }

  return { messages, hasChatFormat: true };
}
