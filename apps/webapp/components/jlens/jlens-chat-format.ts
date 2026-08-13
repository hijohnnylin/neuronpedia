// Chat token grouping for the jlens chat interface.
//
// The inference server is the single source of truth for chat structure: it
// applies the model's real chat template and returns per-token span metadata
// (role / channel / section / message index) on every streamed token (see
// `LensTokenSpan` in `@/lib/utils/lens`). This module just regroups that flat
// per-position token stream into user / assistant message bubbles for display.
//
// There is deliberately NO per-model-family knowledge here anymore (no special
// turn tokens, no `detectChatFormat`, no harmony state machine): all of that now
// lives server-side in the engine's tokenize layer, so a new model family needs
// no frontend change. When the tokens carry no spans (raw-text / reproduction
// requests, or legacy shared runs saved before spans existed) the caller falls
// back to a plain, ungrouped render.

import { LensCharContinuation, LensChatMessage, LensTokenMessage } from '@/lib/utils/lens';

export type ChatRole = 'user' | 'assistant';

export interface JlensTokenGroup {
  role: ChatRole;
  headerTokens: LensTokenMessage[];
  contentTokens: LensTokenMessage[];
  footerTokens: LensTokenMessage[];
  // The true role label (e.g. `system`, `developer`) for formats that surface
  // more roles than the `user`/`assistant` display split. `role` above is the
  // display side (user = right, everything else = left); `roleLabel` preserves
  // the real role so callers can map bubbles back to chat messages.
  roleLabel?: string;
  // The channel (`analysis` / `final` / `commentary`) for assistant turns split
  // into channels — harmony's named channels (gpt-oss) and reasoning-tag models'
  // `<think>` blocks both land here. Undefined for simple turns.
  channel?: string;
  // The source chat-message index for input turns (from the server spans), or
  // undefined for generated turns / turns without a 1:1 message mapping. Used to
  // decide whether inline message editing is safe (see `jlens-chat.tsx`).
  messageIndex?: number;
}

export function toChatPayload(messages: { role: ChatRole; content: string }[]): LensChatMessage[] {
  return messages.map((m) => ({ role: m.role, content: m.content }));
}

// The text a run of tokens spells out. This is NOT `tokens.map(t => t.token).join('')`: a
// character split across tokens (an emoji, usually) has its whole glyph repeated in every
// contributing position's `token`, deliberately, so each chip renders the emoji instead of a
// row of `. Joining those strings emits the emoji once per token — which is how an assistant
// turn stored for the next request came back with its emoji doubled, retokenizing the whole
// turn differently. `is_char_continuation` marks the repeats, so text drops them.
//
// Every path that turns tokens back into text goes through here: the stored assistant message,
// the copy buttons, and the prefill a steered re-run continues from.
export function tokensToText(tokens: readonly (LensCharContinuation & { token: string })[]): string {
  return tokens
    .filter((t) => !t.is_char_continuation)
    .map((t) => t.token)
    .join('');
}

// Which `messages` entry each bubble belongs to, or `null` when a bubble can't be
// mapped (template scaffolding, or a turn not yet in `messages`).
//
// Bubbles are NOT 1:1 with messages, so a bubble's own position is never a valid
// message index: a reasoning model splits one assistant message into `analysis`
// and `final` bubbles, and harmony can split it further. Callers that act on a
// message (editing truncates the conversation at it) must go through this.
//
// Input turns use the server's span message index, which is authoritative: it
// counts the request's messages rather than the template's, so a
// template-injected system preamble doesn't shift it. The trailing generated
// turn carries no index, so it maps to the final assistant message — and only
// once that message exists, which it doesn't mid-stream.
export function messageIndicesForGroups(
  groups: JlensTokenGroup[],
  messages: readonly { role: string }[],
): (number | null)[] {
  const lastIdx = messages.length - 1;
  const generatedIdx = messages[lastIdx]?.role === 'assistant' ? lastIdx : null;
  return groups.map((group) => {
    if (group.messageIndex != null) {
      return group.messageIndex <= lastIdx ? group.messageIndex : null;
    }
    return group.role === 'assistant' ? generatedIdx : null;
  });
}

// Whether any token carries server-computed span metadata. When false the caller
// should render the tokens plainly (no chat grouping is possible).
function hasSpans(tokens: LensTokenMessage[]): boolean {
  return tokens.some((t) => t.section != null || t.role != null || t.message_index != null);
}

// Which display bucket a token's `section` maps to. Structural scaffolding and
// headers render as dim header chips; turn-end markers as dim footer chips;
// everything else (message content, generated text) as hoverable lens chips.
function sectionBucket(section: string | null | undefined): 'header' | 'content' | 'footer' {
  if (section === 'footer') return 'footer';
  if (section === 'header' || section === 'scaffold') return 'header';
  return 'content';
}

// Group the flat, span-tagged token stream into per-message bubbles.
//
// A new group opens when the logical turn changes: the source message index
// changes, or (within the generated assistant response) a new block begins. A
// new block is signalled either structurally — a `header` token following a
// non-header token, as harmony's `<|end|>` -> `<|start|>` does — or by the
// channel switching between two named channels, which is how a reasoning-tag
// model moves from `analysis` to `final` (`</think>` closes the reasoning block
// and the answer follows with no header of its own). Leading scaffolding with no
// role (e.g. a BOS before the first turn) is folded into the first real group's
// header so it still renders and stays hoverable.
export function groupTokensBySpans(tokens: LensTokenMessage[]): {
  messages: JlensTokenGroup[];
  hasChatFormat: boolean;
} {
  if (tokens.length === 0 || !hasSpans(tokens)) {
    return { messages: [], hasChatFormat: false };
  }

  // Pass 1: assign a group id to each token.
  const groupIds: number[] = new Array(tokens.length).fill(0);
  let groupId = -1;
  let curKey: string | null = null;
  let prevSection: string | null = null;
  let prevChannel: string | null = null;
  let genBlock = 0;
  for (let i = 0; i < tokens.length; i += 1) {
    const t = tokens[i];
    const mi = t.message_index ?? null;
    const channel = t.channel ?? null;
    let key: string;
    if (mi != null) {
      key = `msg:${mi}`;
    } else {
      // Generated / scaffold token. A new harmony block starts when a header
      // token follows a non-header token (e.g. `<|end|>` then `<|start|>`).
      const startsHeaderBlock = t.section === 'header' && prevSection != null && prevSection !== 'header';
      // ...and a reasoning-tag block ends without one, so a switch between two
      // named channels (`analysis` -> `final`) opens a block too. Both sides must
      // be named: harmony's structural markers sit at `null` before their channel
      // name is parsed, and splitting on that would break a single harmony block.
      const switchesChannel = prevChannel != null && channel != null && channel !== prevChannel;
      if (startsHeaderBlock || switchesChannel) {
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
    prevChannel = channel;
  }

  // Pass 2: build the groups.
  const raw: JlensTokenGroup[] = [];
  let lastId = -1;
  for (let i = 0; i < tokens.length; i += 1) {
    const t = tokens[i];
    if (groupIds[i] !== lastId) {
      raw.push({ role: 'assistant', headerTokens: [], contentTokens: [], footerTokens: [] });
      lastId = groupIds[i];
    }
    const group = raw[raw.length - 1];
    const bucket = sectionBucket(t.section);
    if (bucket === 'header') group.headerTokens.push(t);
    else if (bucket === 'footer') group.footerTokens.push(t);
    else group.contentTokens.push(t);
    if (t.role != null && group.roleLabel === undefined) {
      group.roleLabel = t.role;
      group.role = t.role === 'user' ? 'user' : 'assistant';
    }
    if (t.channel && group.channel === undefined) {
      group.channel = t.channel;
    }
    if (t.message_index != null && group.messageIndex === undefined) {
      group.messageIndex = t.message_index;
    }
  }

  // Fold a leading role-less scaffold group (e.g. a lone BOS) into the next
  // group's header so it renders without producing a stray empty bubble.
  if (
    raw.length > 1 &&
    raw[0].roleLabel === undefined &&
    raw[0].contentTokens.length === 0 &&
    raw[0].footerTokens.length === 0
  ) {
    const leading = raw.shift() as JlensTokenGroup;
    raw[0].headerTokens = [...leading.headerTokens, ...raw[0].headerTokens];
  }

  return { messages: raw, hasChatFormat: raw.length > 0 };
}

// Extract the clean, human-readable assistant text from the generated tokens of
// a completed turn (for storing the assistant message for re-send / copy).
//
// When the response is split into harmony channels, only the `final` channel's
// content is kept (the analysis / reasoning channel is dropped, matching
// harmony's own behavior of stripping prior reasoning from history). While the
// `final` channel hasn't arrived yet (streaming), fall back to the content of
// whatever channel is present. For plain turns, all generated content is kept.
//
// `prefill` is the assistant prefill the turn continued from, and is joined on
// here rather than by the caller because the seam between the two must survive
// verbatim: the first generated token normally carries the leading space, so
// trimming the generated side turned a "hi" prefill continued by " how are
// you?" into "hihow are you?" — which then retokenized differently on the next
// turn, silently changing the analyzed prompt.
export function extractAssistantText(tokens: LensTokenMessage[], prefill = ''): string {
  const generated = tokens.filter((t) => t.is_generated);
  if (generated.length === 0) {
    return prefill;
  }
  const spanned = generated.some((t) => t.section != null || t.channel != null);
  if (!spanned) {
    // No spans (legacy / raw): best-effort strip of any residual markers.
    return joinPrefill(prefill, tokensToText(generated).replace(/<\|[^|]*\|>/g, ''));
  }
  const hasChannels = generated.some((t) => t.channel);
  let picked: LensTokenMessage[];
  if (hasChannels) {
    const finalContent = generated.filter((t) => t.channel === 'final' && sectionBucket(t.section) === 'content');
    picked = finalContent.length > 0 ? finalContent : generated.filter((t) => sectionBucket(t.section) === 'content');
  } else {
    picked = generated.filter((t) => sectionBucket(t.section) === 'content');
  }
  return joinPrefill(prefill, tokensToText(picked));
}

// Trailing whitespace is always dropped (the chat template re-adds its own
// newline before the turn-end token, so keeping it would double up). Leading
// whitespace is only dropped when the generated text starts the message —
// after a prefill it is part of the message.
function joinPrefill(prefill: string, generated: string): string {
  return prefill.length > 0 ? prefill + generated.trimEnd() : generated.trim();
}
