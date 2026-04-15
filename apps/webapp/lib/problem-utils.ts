import { MAX_TITLE_LENGTH } from '@/app/explorer/explorer-shared';

export { detectTypeFromUrl } from './problem-url-types';

// ─── URL Metadata Fetching ─────────────────────────────────────────────────

function decodeEntities(str: string): string {
  return str
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/g, "'");
}

const BLOCKED_IP_RANGES = [
  /^127\./, // loopback
  /^10\./, // RFC 1918
  /^172\.(1[6-9]|2\d|3[01])\./, // RFC 1918
  /^192\.168\./, // RFC 1918
  /^169\.254\./, // link-local
  /^0\./, // current network
  /^100\.(6[4-9]|[7-9]\d|1[0-2]\d)\./, // shared address space
  /^::1$/, // IPv6 loopback
  /^f[cd]/, // IPv6 private
  /^fe80:/, // IPv6 link-local
];

const MAX_RESPONSE_BYTES = 512 * 1024; // 512KB

function isPrivateUrl(urlString: string): boolean {
  try {
    const parsed = new URL(urlString);
    const { hostname } = parsed;
    if (hostname === 'localhost' || hostname === '[::1]') return true;
    if (BLOCKED_IP_RANGES.some((re) => re.test(hostname))) return true;
    // Block cloud metadata endpoints
    if (hostname === '169.254.169.254' || hostname === 'metadata.google.internal') return true;
    return false;
  } catch {
    return true;
  }
}

export async function fetchUrlMetadata(url: string): Promise<{
  title: string | null;
  description: string | null;
  author: string | null;
}> {
  if (isPrivateUrl(url)) {
    throw new Error('URLs pointing to private/internal networks are not allowed');
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);

  const res = await fetch(url, {
    headers: {
      'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.5',
    },
    redirect: 'follow',
    signal: controller.signal,
  });

  clearTimeout(timeout);

  if (!res.ok) {
    throw new Error(`Upstream returned ${res.status}`);
  }

  const contentType = res.headers.get('content-type') || '';
  if (!contentType.includes('text/html') && !contentType.includes('application/xhtml')) {
    return { title: null, description: null, author: null };
  }

  // Read body with size limit to prevent OOM
  const reader = res.body?.getReader();
  if (!reader) {
    return { title: null, description: null, author: null };
  }
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > MAX_RESPONSE_BYTES) {
      reader.cancel();
      break;
    }
    chunks.push(value);
  }
  const html = new TextDecoder().decode(Buffer.concat(chunks));

  // Extract title
  let title: string | null = null;
  const ogTitleMatch =
    html.match(/<meta[^>]+property=["']og:title["'][^>]+content=["']([^"']+)["']/i) ||
    html.match(/<meta[^>]+content=["']([^"']+)["'][^>]+property=["']og:title["']/i);
  if (ogTitleMatch) {
    [, title] = ogTitleMatch;
  } else {
    const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
    if (titleMatch) {
      [, title] = titleMatch;
    }
  }

  // Extract description
  let description: string | null = null;
  const ogDescMatch =
    html.match(/<meta[^>]+property=["']og:description["'][^>]+content=["']([^"']+)["']/i) ||
    html.match(/<meta[^>]+content=["']([^"']+)["'][^>]+property=["']og:description["']/i);
  if (ogDescMatch) {
    [, description] = ogDescMatch;
  } else {
    const descMatch =
      html.match(/<meta[^>]+name=["']description["'][^>]+content=["']([^"']+)["']/i) ||
      html.match(/<meta[^>]+content=["']([^"']+)["'][^>]+name=["']description["']/i);
    if (descMatch) {
      [, description] = descMatch;
    }
  }

  // Extract author
  let author: string | null = null;
  const authorPatterns = [
    /<meta[^>]+name=["']author["'][^>]+content=["']([^"']+)["']/i,
    /<meta[^>]+content=["']([^"']+)["'][^>]+name=["']author["']/i,
    /<meta[^>]+name=["']citation_author["'][^>]+content=["']([^"']+)["']/i,
    /<meta[^>]+content=["']([^"']+)["'][^>]+name=["']citation_author["']/i,
    /<meta[^>]+property=["']article:author["'][^>]+content=["']([^"']+)["']/i,
    /<meta[^>]+content=["']([^"']+)["'][^>]+property=["']article:author["']/i,
    /<meta[^>]+property=["']og:article:author["'][^>]+content=["']([^"']+)["']/i,
    /<meta[^>]+content=["']([^"']+)["'][^>]+property=["']og:article:author["']/i,
  ];
  for (const pattern of authorPatterns) {
    const match = html.match(pattern);
    if (match) {
      [, author] = match;
      break;
    }
  }

  // Decode and clean
  if (title) {
    title = decodeEntities(title).trim();
    const parsedUrl = new URL(url);
    if (parsedUrl.hostname === 'github.com' || parsedUrl.hostname === 'www.github.com') {
      // Extract repo name from path (e.g. /user/repo or /user/repo/...)
      const pathParts = parsedUrl.pathname.split('/').filter(Boolean);
      if (pathParts.length >= 2) {
        title = `${pathParts[1]} - GitHub`;
      }
    }
    title = title.slice(0, MAX_TITLE_LENGTH);
  }
  if (description) description = decodeEntities(description).trim();
  if (author) author = decodeEntities(author).trim();

  return { title, description, author };
}
