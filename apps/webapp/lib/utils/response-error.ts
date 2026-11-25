/**
 * The error message to show for a failed (non-ok) response.
 *
 * Uses the API's `message` or `error` field when the body is JSON, and falls back to the HTTP
 * status when it is not, e.g. an HTML 502/504 page from a proxy or a redirect to a sign-in page.
 */
export async function responseErrorMessage(res: Response): Promise<string> {
  const data = await res.json().catch(() => null);
  return data?.message || data?.error || `Error: ${res.status}`;
}
