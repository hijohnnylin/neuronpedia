import { lookup } from 'node:dns/promises';
import { isIPv4, isIPv6 } from 'node:net';
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

function ipv4IsBlocked(ip: string): boolean {
  const parts = ip.split('.').map(Number);
  if (parts.length !== 4 || parts.some((p) => Number.isNaN(p) || p < 0 || p > 255)) return true;
  const [a, b] = parts;
  if (a === 0) return true; // 0.0.0.0/8 (incl. unspecified)
  if (a === 127) return true; // loopback
  if (a === 10) return true; // RFC 1918
  if (a === 172 && b >= 16 && b <= 31) return true; // RFC 1918
  if (a === 192 && b === 168) return true; // RFC 1918
  if (a === 169 && b === 254) return true; // link-local (incl. cloud metadata 169.254.169.254)
  if (a === 100 && b >= 64 && b <= 127) return true; // CGNAT / shared address space
  if (a >= 224) return true; // 224/4 multicast, 240/4 reserved, 255.255.255.255 broadcast
  return false;
}

// Decode an IPv4-mapped / IPv4-compatible IPv6 address to its dotted v4 form.
// Handles both the dotted (`::ffff:127.0.0.1`) and hex (`::ffff:7f00:1`, which
// is how Node normalizes the former) representations. Returns null if the
// address doesn't embed an IPv4.
function extractEmbeddedV4(addr: string): string | null {
  const dotted = addr.match(/((?:\d{1,3}\.){3}\d{1,3})$/);
  if (dotted) return dotted[1];
  // `::[ffff:]HHHH:LLLL` — the trailing 32 bits carry the IPv4.
  const hex = addr.match(/^::(?:ffff:)?([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
  if (hex) {
    const hi = parseInt(hex[1], 16);
    const lo = parseInt(hex[2], 16);
    return `${(hi >> 8) & 0xff}.${hi & 0xff}.${(lo >> 8) & 0xff}.${lo & 0xff}`;
  }
  return null;
}

function ipv6IsBlocked(ip: string): boolean {
  const addr = ip.toLowerCase();
  if (addr === '::' || addr === '::1') return true; // unspecified, loopback
  if (addr.startsWith('fe80')) return true; // link-local
  if (addr.startsWith('fc') || addr.startsWith('fd')) return true; // unique local fc00::/7
  if (addr.startsWith('ff')) return true; // multicast
  // IPv4-mapped/compatible (e.g. ::ffff:169.254.169.254): validate the embedded v4.
  const embeddedV4 = extractEmbeddedV4(addr);
  if (embeddedV4) return ipv4IsBlocked(embeddedV4);
  return false;
}

function ipIsBlocked(ip: string): boolean {
  if (isIPv4(ip)) return ipv4IsBlocked(ip);
  if (isIPv6(ip)) return ipv6IsBlocked(ip);
  return true; // unparseable → block
}

// Resolve the hostname and reject if ANY resolved address is private/internal.
// Checking the resolved IPs (not the hostname string) is what defeats
// attacker-controlled DNS records that point at internal hosts, and numeric/
// alternate-encoding hosts (the WHATWG URL parser normalizes those to a dotted
// IP that dns.lookup returns verbatim). Note: a determined DNS-rebinding
// attacker could still flip the record between this check and the fetch's own
// resolution (TOCTOU) — acceptable residual risk for this metadata fetcher.
async function assertHostAllowed(hostname: string): Promise<void> {
  const host = hostname.replace(/^\[|\]$/g, ''); // strip IPv6 literal brackets
  let addresses: { address: string }[];
  try {
    addresses = await lookup(host, { all: true, verbatim: true });
  } catch {
    throw new Error('URLs pointing to private/internal networks are not allowed');
  }
  if (addresses.length === 0 || addresses.some(({ address }) => ipIsBlocked(address))) {
    throw new Error('URLs pointing to private/internal networks are not allowed');
  }
}

// fetch() wrapper that validates the target (scheme + resolved IPs) before each
// request and follows redirects manually, re-validating every hop — an open
// redirect to an internal host would otherwise bypass a one-time up-front check.
async function safeFetch(startUrl: string, init: RequestInit, maxRedirects = 5): Promise<Response> {
  let currentUrl = startUrl;
  for (let hop = 0; ; hop += 1) {
    const parsed = new URL(currentUrl);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error('Only http and https URLs are allowed');
    }
    // eslint-disable-next-line no-await-in-loop
    await assertHostAllowed(parsed.hostname);
    // eslint-disable-next-line no-await-in-loop
    const res = await fetch(currentUrl, { ...init, redirect: 'manual' });
    if (res.status >= 300 && res.status < 400) {
      const location = res.headers.get('location');
      if (!location) return res;
      if (hop >= maxRedirects) throw new Error('Too many redirects');
      res.body?.cancel().catch(() => {});
      currentUrl = new URL(location, currentUrl).toString();
      continue;
    }
    return res;
  }
}

const MAX_RESPONSE_BYTES = 2 * 1024 * 1024; // 2MB — some SSR apps (e.g. LessWrong) emit og: meta tags well past the first 512KB
// We stop reading early once we've seen og:title AND og:description — these are the main signals we need.
// Don't stop on </head>: React 19 / Next.js apps (e.g. LessWrong) hoist document metadata and emit
// og: tags later in the streamed body, well after </head>.
const OG_TITLE_RE = /<meta[^>]+property=["']og:title["']/i;
const OG_DESC_RE = /<meta[^>]+property=["']og:description["']/i;

// LessWrong and Alignment Forum share a codebase and database. Both sites are gated by aggressive
// bot protection (Cloudflare/Vercel) that rejects non-browser TLS fingerprints regardless of headers.
// Their public GraphQL endpoint at lesswrong.com/graphql is not gated and serves post data for both.
// URLs look like https://www.{lesswrong.com,alignmentforum.org}/posts/<id>/<slug>.
const LW_AF_HOSTS = new Set(['www.lesswrong.com', 'lesswrong.com', 'www.alignmentforum.org', 'alignmentforum.org']);
const LW_POST_ID_RE = /\/posts\/([A-Za-z0-9]{10,24})(?:\/|$)/;

async function fetchLessWrongMetadata(url: string): Promise<{
  title: string | null;
  description: string | null;
  author: string | null;
} | null> {
  let postId: string | null = null;
  try {
    const parsed = new URL(url);
    if (!LW_AF_HOSTS.has(parsed.hostname)) return null;
    const match = parsed.pathname.match(LW_POST_ID_RE);
    if (!match) return null;
    [, postId] = match;
  } catch {
    return null;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const res = await fetch('https://www.lesswrong.com/graphql', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        'User-Agent':
          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      },
      body: JSON.stringify({
        query:
          '{ post(input: {selector: {_id: "' +
          postId +
          '"}}) { result { title user { displayName } contents { plaintextDescription } } } }',
      }),
      signal: controller.signal,
    });
    if (!res.ok) {
      console.log('[url-metadata] LW GraphQL non-OK', { postId, status: res.status });
      return null;
    }
    const json = (await res.json()) as {
      data?: {
        post?: {
          result?: {
            title?: string | null;
            user?: { displayName?: string | null } | null;
            contents?: { plaintextDescription?: string | null } | null;
          } | null;
        };
      };
      errors?: unknown;
    };
    const result = json?.data?.post?.result;
    if (!result) {
      console.log('[url-metadata] LW GraphQL no result', { postId, errors: json?.errors });
      return null;
    }
    const title = result.title ? result.title.slice(0, MAX_TITLE_LENGTH) : null;
    const rawDescription = result.contents?.plaintextDescription ?? null;
    // plaintextDescription returns the full post body; trim to a meta-description-like length.
    const description = rawDescription
      ? (() => {
          const collapsed = rawDescription.replace(/\s+/g, ' ').trim();
          return collapsed.length > 300 ? `${collapsed.slice(0, 300).trimEnd()}…` : collapsed;
        })()
      : null;
    const author = result.user?.displayName ?? null;
    console.log('[url-metadata] ✓ LW GraphQL', { postId, title, author });
    return { title, description, author };
  } catch (err) {
    console.log('[url-metadata] LW GraphQL threw', {
      postId,
      error: err instanceof Error ? `${err.name}: ${err.message}` : String(err),
    });
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

export async function fetchUrlMetadata(url: string): Promise<{
  title: string | null;
  description: string | null;
  author: string | null;
}> {
  const t0 = Date.now();
  console.log('[url-metadata] ▶ start', url);

  // LessWrong / Alignment Forum: their bot protection blocks server-side HTML fetches regardless
  // of headers (TLS fingerprinting). Use their public GraphQL API instead.
  const lwMeta = await fetchLessWrongMetadata(url);
  if (lwMeta) {
    console.log('[url-metadata] ✓ via LW GraphQL', { url, elapsedMs: Date.now() - t0 });
    return lwMeta;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);

  let res: Response;
  try {
    res = await safeFetch(url, {
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"macOS"',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'no-cache',
        Pragma: 'no-cache',
      },
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeout);
    console.log('[url-metadata] ✗ fetch threw', {
      url,
      elapsedMs: Date.now() - t0,
      error: err instanceof Error ? `${err.name}: ${err.message}` : String(err),
    });
    throw err;
  }

  clearTimeout(timeout);

  console.log('[url-metadata] ← response', {
    url,
    finalUrl: res.url,
    status: res.status,
    contentType: res.headers.get('content-type'),
    contentLength: res.headers.get('content-length'),
    elapsedMs: Date.now() - t0,
  });

  if (!res.ok) {
    throw new Error(`Upstream returned ${res.status}`);
  }

  const contentType = res.headers.get('content-type') || '';
  if (!contentType.includes('text/html') && !contentType.includes('application/xhtml')) {
    console.log('[url-metadata] ✗ non-HTML content-type, skipping metadata extraction', contentType);
    return { title: null, description: null, author: null };
  }

  // Read body with size limit to prevent OOM
  const reader = res.body?.getReader();
  if (!reader) {
    console.log('[url-metadata] ✗ response body has no reader');
    return { title: null, description: null, author: null };
  }
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  let truncated = false;
  let stoppedEarly = false;
  const decoder = new TextDecoder();
  let decodedSoFar = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > MAX_RESPONSE_BYTES) {
      reader.cancel();
      truncated = true;
      break;
    }
    chunks.push(value);
    // Stream-decode to check if we've seen the signals we care about; if so, stop early.
    decodedSoFar += decoder.decode(value, { stream: true });
    if (OG_TITLE_RE.test(decodedSoFar) && OG_DESC_RE.test(decodedSoFar)) {
      reader.cancel();
      stoppedEarly = true;
      break;
    }
  }
  const html = new TextDecoder().decode(Buffer.concat(chunks));
  console.log('[url-metadata] read body', {
    bytes: totalBytes,
    truncated,
    stoppedEarly,
    htmlChars: html.length,
    maxBytes: MAX_RESPONSE_BYTES,
  });

  const headSnippetMatch = html.match(/<head[\s\S]*?<\/head>/i);
  console.log('[url-metadata] html preview', {
    hasHead: !!headSnippetMatch,
    headChars: headSnippetMatch ? headSnippetMatch[0].length : 0,
    ogTitleInHtml: /<meta[^>]+property=["']og:title["']/i.test(html),
    ogDescInHtml: /<meta[^>]+property=["']og:description["']/i.test(html),
    titleTagInHtml: /<title[^>]*>/i.test(html),
    descMetaInHtml: /<meta[^>]+name=["']description["']/i.test(html),
    authorMetaInHtml: /<meta[^>]+name=["']author["']/i.test(html),
    headStart: html.slice(0, 500),
  });

  // Extract title
  let title: string | null = null;
  let titleSource = 'none';
  const ogTitleMatch =
    html.match(/<meta[^>]+property=["']og:title["'][^>]+content=["']([^"']+)["']/i) ||
    html.match(/<meta[^>]+content=["']([^"']+)["'][^>]+property=["']og:title["']/i);
  if (ogTitleMatch) {
    [, title] = ogTitleMatch;
    titleSource = 'og:title';
  } else {
    const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
    if (titleMatch) {
      [, title] = titleMatch;
      titleSource = '<title>';
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

  console.log('[url-metadata] ✓ extracted', {
    url,
    titleSource,
    title,
    description: description ? `${description.slice(0, 80)}${description.length > 80 ? '…' : ''}` : null,
    author,
    totalElapsedMs: Date.now() - t0,
  });

  return { title, description, author };
}
