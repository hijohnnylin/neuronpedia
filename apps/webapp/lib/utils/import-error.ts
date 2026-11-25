const CONNECTION_LOST_MESSAGE =
  'The connection to the import stream was lost before it completed. Check the webapp server logs for details.';

/**
 * Turns whatever an EventSource `onerror` hands us into a readable message.
 *
 * The browser fires `onerror` both for a dropped connection (a bare `Event` with no payload) and for a
 * server-sent `event: error` frame (a `MessageEvent` whose `data` is the server's JSON payload). Calling
 * `JSON.stringify` on either one yields `{"isTrusted":true}`, which tells the user nothing.
 */
export function describeImportError(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof e === 'string') return e;
  if (e && typeof e === 'object') {
    const { data } = e as { data?: unknown };
    if (typeof data === 'string' && data.length > 0) {
      try {
        const parsed = JSON.parse(data) as { message?: unknown };
        if (parsed && typeof parsed.message === 'string' && parsed.message.length > 0) return parsed.message;
      } catch {
        // not JSON, fall through to the raw payload
      }
      return data;
    }
  }
  return CONNECTION_LOST_MESSAGE;
}
