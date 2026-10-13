// Pages with a complex UI. When embedded in an iframe, they show a bar that opens the full page.
const INTERACTIVE_EMBED_PAGES = new Set(['assistant-axis', 'graph', 'jlens', 'nla', 'steer']);

/**
 * Returns true when the path is an interactive page: the first path segment (/graph) or the one after
 * the model ID (/[model]/jlens).
 */
export function isInteractiveEmbedPath(pathname: string): boolean {
  return pathname
    .split('/')
    .filter(Boolean)
    .slice(0, 2)
    .some((segment) => INTERACTIVE_EMBED_PAGES.has(segment));
}

/** Returns the URL of the same page without its embed parameter. */
export function getFullPageHref(pathname: string, searchParams: URLSearchParams): string {
  const params = new URLSearchParams(searchParams);
  params.delete('embed');
  const query = params.toString();
  return `${pathname}${query ? `?${query}` : ''}`;
}
