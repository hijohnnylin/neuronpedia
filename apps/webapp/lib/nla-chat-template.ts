// Server-side chat templating for the NLA API routes.
//
// Mirrors the format definitions used by the chat UI in
// `components/provider/nla-provider.tsx`. The frontend renders templates
// client-side and passes the rendered `text` to /api/nla/* routes; the
// API routes need to render the same templates server-side for callers
// who supply `messages` instead of `text`.
//
// If you change a template here, change the corresponding definition in
// `nla-provider.tsx` too (and vice versa). The chat UI's
// `parseChat` / `isSpecialToken` helpers live only on the frontend
// because they're UI-only.

export type ChatMessage = { role: 'user' | 'assistant'; content: string };

function formatQwen(messages: ChatMessage[]): string {
  return messages
    .map((m, i) => {
      const isLast = i === messages.length - 1;
      return `<|im_start|>${m.role}\n${m.content}${isLast ? '' : '<|im_end|>\n'}`;
    })
    .join('');
}

function formatGemma3(messages: ChatMessage[]): string {
  return messages
    .map((m, i) => {
      const isLast = i === messages.length - 1;
      const roleName = m.role === 'assistant' ? 'model' : m.role;
      return `<start_of_turn>${roleName}\n${m.content}${isLast ? '' : '<end_of_turn>\n'}`;
    })
    .join('');
}

function formatLlama3(messages: ChatMessage[]): string {
  return messages
    .map((m, i) => {
      const isLast = i === messages.length - 1;
      return `<|start_header_id|>${m.role}<|end_header_id|>\n\n${m.content}${isLast ? '' : '<|eot_id|>'}`;
    })
    .join('');
}

/**
 * Render a chat-templated string for `modelId`. The detection is the same
 * loose substring match the frontend uses (gemma → gemma3, llama → llama3,
 * else qwen). Returns a string in the canonical chat template the model's
 * tokenizer expects, ready to be passed to NLA `/tokenize` or `/explain`.
 */
export function formatChatForModel(modelId: string, messages: ChatMessage[]): string {
  const lower = (modelId || '').toLowerCase();
  if (lower.includes('gemma')) return formatGemma3(messages);
  if (lower.includes('llama')) return formatLlama3(messages);
  return formatQwen(messages);
}

/**
 * Append an assistant turn to the chat-templated `prompt`. Used after a
 * non-streaming OpenRouter completion to assemble the canonical full
 * `prompt + assistant_completion` string for a final NLA tokenize pass.
 *
 * We re-render the full message list (prompt messages + new assistant turn)
 * rather than concatenating strings, so the chat-template's "last turn
 * omits the closing tag" behavior applies consistently.
 */
export function formatChatWithAssistantCompletion(
  modelId: string,
  promptMessages: ChatMessage[],
  assistantCompletion: string,
): string {
  return formatChatForModel(modelId, [...promptMessages, { role: 'assistant', content: assistantCompletion }]);
}

export function isChatMessageArray(value: unknown): value is ChatMessage[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (m) =>
        m !== null &&
        typeof m === 'object' &&
        'role' in m &&
        (m.role === 'user' || m.role === 'assistant') &&
        'content' in m &&
        typeof (m as { content: unknown }).content === 'string',
    )
  );
}
