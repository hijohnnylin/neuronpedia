import { API_KEY_HEADER_NAME } from '@/lib/constants';
import { env } from '@/lib/env';
import { Ratelimit } from '@upstash/ratelimit';
import { kv } from '@vercel/kv';
import { NextRequest, NextResponse } from 'next/server';

const RATE_LIMIT_WINDOW = '60 m';

const NORMAL_RATE_LIMITS = [
  { endpoint: '/', limit: 25000 },
  { endpoint: '/api', limit: 25000 },
  { endpoint: '/api/activation/new', limit: 1000 },
  { endpoint: '/api/explanation/search', limit: 200 },
  { endpoint: '/api/steer', limit: 300 },
  { endpoint: '/api/search-topk-by-token', limit: 500 },
  { endpoint: '/api/search-all', limit: 1600 },
  { endpoint: '/api/graph/generate', limit: 30 },
  { endpoint: '/api/features/upload-batch', limit: 1000 },
  { endpoint: '/api/model/new', limit: 5 },
  { endpoint: '/api/source-set/new', limit: 10 },
  { endpoint: '/api/graph/tokenize', limit: 300 },
];

const HIGHER_RATE_LIMITS = [
  { endpoint: '/', limit: 25000 },
  { endpoint: '/api', limit: 25000 },
  { endpoint: '/api/activation/new', limit: 1200 }, // higher
  { endpoint: '/api/explanation/search', limit: 3000 }, // higher
  { endpoint: '/api/steer', limit: 1000 }, // higher
  { endpoint: '/api/search-topk-by-token', limit: 1200 }, // higher
  { endpoint: '/api/search-all', limit: 1600 },
  { endpoint: '/api/graph/generate', limit: 320 }, // higher
  { endpoint: '/api/features/upload-batch', limit: 1000 },
  { endpoint: '/api/model/new', limit: 5 },
  { endpoint: '/api/source-set/new', limit: 10 },
  { endpoint: '/api/graph/tokenize', limit: 300 },
];

const normalRateLimiters: { endpoint: string; limiter: Ratelimit }[] = [];
// eslint-disable-next-line no-restricted-syntax
for (const { endpoint, limit } of NORMAL_RATE_LIMITS) {
  normalRateLimiters.push({
    endpoint,
    limiter: new Ratelimit({
      redis: kv,
      prefix: endpoint,
      limiter: Ratelimit.slidingWindow(limit, RATE_LIMIT_WINDOW),
    }),
  });
}

const higherRateLimiters: { endpoint: string; limiter: Ratelimit }[] = [];
// eslint-disable-next-line no-restricted-syntax
for (const { endpoint, limit } of HIGHER_RATE_LIMITS) {
  higherRateLimiters.push({
    endpoint,
    limiter: new Ratelimit({
      redis: kv,
      prefix: `higher-${endpoint}`,
      limiter: Ratelimit.slidingWindow(limit, RATE_LIMIT_WINDOW),
    }),
  });
}

export default async function middleware(request: NextRequest) {
  const requestHeaders = new Headers(request.headers);
  const ip = request.ip ?? '127.0.0.1';
  const pathname = request.nextUrl.pathname.toLowerCase();

  const isEmbedSearchParam = request.nextUrl.searchParams.get('embed');
  const isEmbed = isEmbedSearchParam === 'true' || pathname.startsWith('/embed/');
  requestHeaders.set('x-is-embed', isEmbed ? 'true' : 'false');

  if (!env.ENABLE_RATE_LIMITER) {
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
    apiKey && env.HIGHER_LIMIT_API_TOKENS.includes(apiKey) ? higherRateLimiters : normalRateLimiters;

  // eslint-disable-next-line no-restricted-syntax
  for (const { endpoint, limiter } of rateLimitersToUse) {
    if (pathname.startsWith(endpoint)) {
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
        error: `Rate limit exceeded for this endpoint. The limit for this endpoint (${foundEndpoint}) is ${foundEndpointLimit} requests per ${RATE_LIMIT_WINDOW}. Contact ${env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS} to increase your rate limit.`,
      },
      { status: 429 },
    );
  }

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
