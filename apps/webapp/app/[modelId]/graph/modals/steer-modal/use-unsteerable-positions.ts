import { useEffect, useState } from 'react';

// Stable identity so a caller can use the result as an effect dependency.
const NO_POSITIONS: ReadonlySet<number> = new Set<number>();

/**
 * Prompt positions the graph server refuses to steer at.
 *
 * The server owns the rule (`unsteerable_token_positions` in `chat_prompt.py`):
 * `/steer` silently drops any feature targeting one of these, and
 * `/parse-chat-prompt` reports the same set from the same function. Hiding
 * controls by position rather than by token string is what keeps the two from
 * drifting — and it means the webapp needs no list of BOS literals.
 *
 * Pass the same `prompt` / `modelId` / `sourceSetName` the steer request uses.
 * These are indices into the server's tokenization of that prompt, so a
 * different prompt is a different index space.
 *
 * Fetched lazily, on `enabled` rather than on graph selection: viewing a graph
 * touches the graph server nowhere else, and making it do so would wake a cold
 * GPU pod on every page view. Someone opening the steer modal is about to wake
 * it anyway. Resolved sets are kept per prompt, so reopening doesn't refetch.
 *
 * While it's in flight, and if it fails, the set is empty and every slider
 * renders. That is the pre-existing failure mode rather than a new one: steering
 * one of these positions has always been a no-op the server drops without
 * complaint, so the cost is an ineffective control, never a wrong result.
 */
export default function useUnsteerablePositions({
  enabled,
  prompt,
  modelId,
  sourceSetName,
}: {
  enabled: boolean;
  prompt: string | undefined;
  modelId: string | undefined;
  sourceSetName: string | undefined | null;
}): ReadonlySet<number> {
  const [resolved, setResolved] = useState<{ key: string; positions: ReadonlySet<number> } | null>(null);

  const key = enabled && prompt && modelId ? JSON.stringify([modelId, sourceSetName ?? '', prompt]) : null;

  useEffect(() => {
    if (!key || resolved?.key === key) {
      return undefined;
    }
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch('/api/graph/parse-chat-prompt', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, modelId, sourceSetName: sourceSetName || undefined }),
        });
        if (cancelled || !response.ok) {
          return;
        }
        const json = await response.json();
        if (cancelled) {
          return;
        }
        setResolved({
          key,
          positions: new Set<number>(Array.isArray(json.unsteerable_positions) ? json.unsteerable_positions : []),
        });
      } catch {
        // Leave it unresolved so the next open retries.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [key, resolved?.key, prompt, modelId, sourceSetName]);

  // Only answer for the prompt actually asked about — a stale set from the
  // previous graph would hide the wrong columns.
  return resolved?.key === key && resolved ? resolved.positions : NO_POSITIONS;
}
