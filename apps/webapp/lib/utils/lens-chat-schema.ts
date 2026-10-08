// Validation for chat payloads sent to the lens routes (`/api/lens/prompt`,
// `/api/lens/share`). Both routes take the same message and tool shapes.

import {
  MAX_LENS_CHAT_TOOL_ARGUMENT_CHARS,
  MAX_LENS_CHAT_TOOL_CALLS,
  MAX_LENS_CHAT_TOOL_NAME_CHARS,
  MAX_LENS_CHAT_TOOLS,
  MAX_LENS_CHAT_TOOLS_CHARS,
} from '@/lib/utils/lens';
import * as yup from 'yup';

export const LENS_CHAT_ROLES = ['user', 'assistant', 'system', 'tool'] as const;

const MAX_TOOL_CALL_ID_CHARS = 128;

const jsonLength = (value: unknown) => JSON.stringify(value ?? null).length;

const toolCallSchema = yup.object({
  name: yup.string().min(1).max(MAX_LENS_CHAT_TOOL_NAME_CHARS).required(),
  arguments: yup
    .object()
    .optional()
    .default(undefined)
    .test(
      'arguments-size',
      `Tool call arguments must be at most ${MAX_LENS_CHAT_TOOL_ARGUMENT_CHARS} characters of JSON`,
      (value) => value === undefined || jsonLength(value) <= MAX_LENS_CHAT_TOOL_ARGUMENT_CHARS,
    ),
  id: yup.string().max(MAX_TOOL_CALL_ID_CHARS).optional(),
});

// One chat message. `maxChars` caps `content` for every role.
export function lensChatMessageSchema(maxChars: number) {
  return yup
    .object({
      role: yup.string().oneOf(LENS_CHAT_ROLES).required(),
      // A tool call often has no text, so empty content is valid.
      content: yup.string().defined().strict().max(maxChars),
      toolCalls: yup.array().of(toolCallSchema).max(MAX_LENS_CHAT_TOOL_CALLS).optional(),
      toolCallId: yup.string().max(MAX_TOOL_CALL_ID_CHARS).optional(),
    })
    .test('tool-fields', 'Only assistant messages have toolCalls, and only tool messages have toolCallId', (m) => {
      if (m.toolCalls?.length && m.role !== 'assistant') return false;
      if (m.toolCallId && m.role !== 'tool') return false;
      return true;
    });
}

// Tool definitions the chat template writes into the prompt (OpenAI function schema).
export const lensChatToolsSchema = yup
  .array()
  .of(yup.object().required())
  .max(MAX_LENS_CHAT_TOOLS)
  .optional()
  .test(
    'tools-size',
    `Tool definitions must be at most ${MAX_LENS_CHAT_TOOLS_CHARS} characters of JSON`,
    (value) => value === undefined || jsonLength(value) <= MAX_LENS_CHAT_TOOLS_CHARS,
  );
