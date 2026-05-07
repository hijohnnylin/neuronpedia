export type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

export type ChatTokenizerFormat = {
  id: string;
  turnStartToken: string;
  turnEndToken: string;
  assistantRoleName: string;
  formatChat: (messages: ChatMessage[]) => string;
  /** One turn's substring exactly as concatenated in `formatChat` (for UI preview). */
  formatSingleTurn: (message: ChatMessage, messageIndex: number, totalMessages: number) => string;
  parseChat: (raw: string) => ChatMessage[] | null;
  isSpecialToken: (token: string) => boolean;
};

export type TokenInfo = {
  token: string;
  token_id: number;
  position: number;
  // Byte-fragment metadata for byte-level BPE tokenizers (Qwen, Llama, etc.).
  // When a multi-byte UTF-8 glyph (emoji, CJK char) is split across multiple
  // model tokens, the server emits the *merged* glyph string for every
  // token in the run, plus a 0-based `fragment_index` and the run's
  // `fragment_count`. Default 0/1 means "this token decodes cleanly on
  // its own."
  fragment_index?: number;
  fragment_count?: number;
};

export type ExplainResult = {
  token: string;
  token_id: number;
  position: number;
  l2_norm: number;
  description: string;
  mse: number | null;
  cosine_similarity: number | null;
  generated: boolean;
  fragment_index?: number;
  fragment_count?: number;
};

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
