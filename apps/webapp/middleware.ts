import { Ratelimit } from '@upstash/ratelimit';
import { ipAddress } from '@vercel/functions';
import { kv } from '@vercel/kv';
import { NextRequest, NextResponse } from 'next/server';
import { API_KEY_HEADER_NAME, CONTACT_EMAIL_ADDRESS, ENABLE_RATE_LIMITER, HIGHER_LIMIT_API_TOKENS } from './lib/env';

const RATE_LIMIT_WINDOW = '60 m';

const NO_LIMIT_ENDPOINTS = ['/api/steer-load'];

// Rate-limit entry. `exact: true` requires path-component matching (pathname
// === endpoint OR pathname.startsWith(endpoint + '/')) so the bucket isn't
// shared with hyphen-suffixed siblings — e.g. `/api/nla/explain` with
// `exact: true` will NOT also count requests to `/api/nla/explain-saelens`
// or `/api/nla/explain-share`. Default (omitted/false) preserves the
// historical loose `startsWith` matching, which is intentional for
// catch-alls (`/`, `/api`) and the `/api/steer` family aggregation
// (`/api/steer-chat`, `/api/steer-logits`, …).
type RateLimitEntry = { endpoint: string; limit: number; exact?: boolean };

const NORMAL_RATE_LIMITS: RateLimitEntry[] = [
  { endpoint: '/', limit: 25000 },
  { endpoint: '/api', limit: 25000 },
  { endpoint: '/api/activation/new', limit: 1000 },
  { endpoint: '/api/explanation/search', limit: 200 },
  { endpoint: '/api/steer', limit: 120 },
  { endpoint: '/api/search-topk-by-token', limit: 500 },
  { endpoint: '/api/search-all', limit: 1600 },
  { endpoint: '/api/graph/generate', limit: 30 },
  { endpoint: '/api/features/upload-batch', limit: 1000 },
  { endpoint: '/api/model/new', limit: 5 },
  { endpoint: '/api/source-set/new', limit: 10 },
  { endpoint: '/api/graph/tokenize', limit: 300 },
  { endpoint: '/api/auth/signin', limit: 5 },
  { endpoint: '/api/explorer/url-metadata', limit: 120 },
  { endpoint: '/api/explorer/node/submit-urls', limit: 30 },
  { endpoint: '/api/explorer/node/new', limit: 60 },
  { endpoint: '/api/explorer/node/update', limit: 120 },
  { endpoint: '/api/explorer/node/delete', limit: 60 },
  { endpoint: '/api/explorer/node/approve', limit: 60 },
  { endpoint: '/api/explorer/comment/new', limit: 120 },
  { endpoint: '/api/explorer/comment/delete', limit: 60 },
  { endpoint: '/api/explorer/edge/new', limit: 60 },
  { endpoint: '/api/explorer/edge/delete', limit: 60 },
  { endpoint: '/api/saelens/latent-decoder', limit: 3600 },
  { endpoint: '/api/nla/explain-saelens', limit: 120 },
  // Each NLA chat send issues 2 requests to /api/nla/completion (streaming
  // chat + canonical re-tokenize), so 240/hour ≈ 120 user-perceived messages
  // per hour. The chat UI divides by 2 before displaying the counter.
  // External API callers issue 1 request per generation (no post-stream
  // re-tokenize), so they effectively get 2x the throughput vs the UI —
  // intentional, since we don't want researchers to burn through their
  // budget on the UI's double-fire pattern.
  { endpoint: '/api/nla/completion', limit: 240 },
  // `exact: true` so this bucket is independent of /api/nla/explain-saelens
  // and /api/nla/explain-share (without `exact`, prefix-matching would also
  // count those endpoints' requests against this bucket).
  { endpoint: '/api/nla/explain', limit: 120, exact: true },
  // Cheap read-only metadata endpoint; generous default so docs/SDK calls
  // never get rate-limited mid-flight.
  { endpoint: '/api/nla/sources', limit: 1200 },
];

const HIGHER_RATE_LIMITS: RateLimitEntry[] = [
  { endpoint: '/', limit: 25000 },
  { endpoint: '/api', limit: 25000 },
  { endpoint: '/api/activation/new', limit: 3000 }, // higher
  { endpoint: '/api/explanation/search', limit: 3000 }, // higher
  { endpoint: '/api/steer', limit: 1000 }, // higher
  { endpoint: '/api/search-topk-by-token', limit: 1200 }, // higher
  { endpoint: '/api/search-all', limit: 1600 },
  { endpoint: '/api/graph/generate', limit: 320 }, // higher
  { endpoint: '/api/features/upload-batch', limit: 1000 },
  { endpoint: '/api/model/new', limit: 5 },
  { endpoint: '/api/source-set/new', limit: 10 },
  { endpoint: '/api/graph/tokenize', limit: 300 },
  { endpoint: '/api/auth/signin', limit: 5 },
  { endpoint: '/api/saelens/latent-decoder', limit: 3600 },
  { endpoint: '/api/nla/explain-saelens', limit: 120 },
  { endpoint: '/api/nla/completion', limit: 240 },
  { endpoint: '/api/nla/explain', limit: 120, exact: true },
  { endpoint: '/api/nla/sources', limit: 1200 },
];

function pathMatchesEndpoint(pathname: string, endpoint: string, exact: boolean | undefined) {
  if (exact) {
    return pathname === endpoint || pathname.startsWith(`${endpoint}/`);
  }
  return pathname.startsWith(endpoint);
}

type CompiledLimiter = { endpoint: string; exact: boolean | undefined; limiter: Ratelimit };

const normalRateLimiters: CompiledLimiter[] = [];

for (const { endpoint, limit, exact } of NORMAL_RATE_LIMITS) {
  normalRateLimiters.push({
    endpoint,
    exact,
    limiter: new Ratelimit({
      redis: kv,
      prefix: endpoint,
      limiter: Ratelimit.slidingWindow(limit, RATE_LIMIT_WINDOW),
    }),
  });
}

const higherRateLimiters: CompiledLimiter[] = [];

for (const { endpoint, limit, exact } of HIGHER_RATE_LIMITS) {
  higherRateLimiters.push({
    endpoint,
    exact,
    limiter: new Ratelimit({
      redis: kv,
      prefix: `higher-${endpoint}`,
      limiter: Ratelimit.slidingWindow(limit, RATE_LIMIT_WINDOW),
    }),
  });
}

export default async function middleware(request: NextRequest) {
  const requestHeaders = new Headers(request.headers);
  const ip = ipAddress(request) ?? '127.0.0.1';
  const pathname = request.nextUrl.pathname.toLowerCase();

  const isEmbedSearchParam = request.nextUrl.searchParams.get('embed');
  const isEmbed = isEmbedSearchParam === 'true' || pathname.startsWith('/embed/');
  requestHeaders.set('x-is-embed', isEmbed ? 'true' : 'false');

  if (!ENABLE_RATE_LIMITER) {
    const res = NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    });
    if (pathname.startsWith('/api')) {
      res.headers.append('Access-Control-Allow-Origin', '*');
      res.headers.append('Access-Control-Allow-Methods', 'GET, POST');
      res.headers.append('Access-Control-Allow-Headers', 'Content-Type');
    }
    return res;
  }
  let wasRateLimited = false;
  let foundEndpoint = '';
  let foundEndpointLimit = 0;
  const remaining = 0;

  const apiKey = request.headers.get(API_KEY_HEADER_NAME);
  const rateLimitersToUse =
    apiKey && HIGHER_LIMIT_API_TOKENS.includes(apiKey) ? higherRateLimiters : normalRateLimiters;

  for (const { endpoint, exact, limiter } of rateLimitersToUse) {
    if (pathMatchesEndpoint(pathname, endpoint, exact) && !NO_LIMIT_ENDPOINTS.some((ep) => pathname.startsWith(ep))) {
      // eslint-disable-next-line
      const { success, pending, limit, reset, remaining } = await limiter.limit(ip);
      requestHeaders.set('x-limit-remaining', remaining.toString());
      wasRateLimited = !success;
      foundEndpoint = endpoint;
      foundEndpointLimit = limit;
      // if not rated limited by this endpoint, keep going to find endpoints that might be rate limited
      if (wasRateLimited) {
        break;
      }
    }
  }
  if (wasRateLimited) {
    return NextResponse.json(
      {
        endpoint: foundEndpoint,
        limitPerWindow: foundEndpointLimit,
        requestWindow: RATE_LIMIT_WINDOW,
        remainingRequests: remaining,
        error: `Rate limit exceeded for this endpoint. The limit for this endpoint (${foundEndpoint}) is ${foundEndpointLimit} requests per ${RATE_LIMIT_WINDOW}. Contact ${CONTACT_EMAIL_ADDRESS} to increase your rate limit.`,
      },
      { status: 429 },
    );
  }

  const res = NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });
  const limitRemaining = requestHeaders.get('x-limit-remaining');
  if (pathname.startsWith('/api')) {
    res.headers.append('Access-Control-Allow-Origin', '*');
    res.headers.append('Access-Control-Allow-Methods', 'GET, POST');
    res.headers.append('Access-Control-Allow-Headers', 'Content-Type');
    if (limitRemaining) {
      res.headers.set('x-limit-remaining', limitRemaining);
    }
  }
  return res;
}
