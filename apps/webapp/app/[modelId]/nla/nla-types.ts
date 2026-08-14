/**
 * Client-side NLA types.
 *
 * The response shapes are re-exported from `@/lib/api/nla-types`, which is generated from
 * `apps/nla/openapi.json`. They are re-exported rather than imported directly because the NLA
 * UI is a dozen files deep and they all already point here.
 */
import type { ExplainResult, TokenInfo } from '@/lib/api/nla-types';

export type { ExplainResult, TokenInfo };

// The server's `ChatMessageInput` accepts any role string and an optional harmony `channel`; the
// UI only ever builds the two-role, content-only subset, so it keeps a narrower type. It is
// assignable to the generated one, which is what matters at the fetch boundary.
export type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

// Per-token chat-span metadata computed server-side (the NLA server applies the model's real
// chat template). `section` buckets a token as structural header / message content / turn-end
// footer / template scaffold. All null for raw text. Present on both generated shapes above;
// named separately here for the UI helpers that take just the span.
export type TokenSpan = Pick<TokenInfo, 'role' | 'section' | 'channel' | 'message_index'>;

// SSE frames from /explain. Not in the spec — they are not response bodies — so these stay
// hand-written; `apps/nla/tests/test_frame_contract.py` pins the server side of both.
export type PartialUpdate = {
  position: number;
  text: string;
  done: false;
};

export type ExplainMeta = {
  layer_index: number;
  total: number;
  prompt_length: number;
};

export type NlaSourceWithModel = {
  id: string;
  modelId: string;
  displayName: string;
  description: string;
  url: string;
  author: string;
  av: string;
  ar: string;
  layerNum: number;
  servers: string[];
  norm: number;
  createdAt: Date | string;
  model: {
    id: string;
    displayName: string;
    owner: string;
  };
};

export type TokenMessageGroup = {
  role: 'user' | 'assistant';
  headerTokens: TokenInfo[];
  contentTokens: TokenInfo[];
  footerTokens: TokenInfo[];
};
