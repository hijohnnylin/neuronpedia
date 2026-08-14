// Shared types for the NLA API routes.
//
// Chat templating is NO LONGER done here (or anywhere on the webapp): the NLA
// inference server applies the model's real chat template and returns per-token
// span metadata, so there is no per-model-family template building on the
// frontend/API. This file now only carries the structured-message type + a
// validation guard used when forwarding `messages` to the NLA server.

export type ChatMessage = { role: 'user' | 'assistant'; content: string };

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
