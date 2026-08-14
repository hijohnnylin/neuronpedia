import * as Sentry from '@sentry/nextjs';
import { NextRequest, NextResponse } from 'next/server';
import { ValidationError } from 'yup';
import { ZodError } from 'zod';

/**
 * The API's error contract, applied centrally by the wrappers in `with-user.ts`.
 *
 * Two rules it exists to enforce.
 *
 * A message only reaches a client if someone wrote it for a client. Everything else becomes a
 * fixed string, because raw exception text carries things we do not want to publish: a prisma
 * error names the source file and line it was thrown from, and an upstream python traceback
 * carries absolute server paths. Constructing an `ApiError` is the act of declaring a message
 * safe, and it is the only way to say anything specific.
 *
 * And every 5xx reaches Sentry. `instrumentation.ts` wires `onRequestError`, which Next.js only
 * calls for errors that *escape* a handler — so catching an error here also means taking over
 * responsibility for reporting it. Handling an error without reporting it is how a route becomes
 * invisible.
 */

const SERVER_FAULT_MESSAGE = 'Something went wrong on our end. The error has been reported.';

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** The request was wrong: malformed, missing a field, or out of range. */
export const badRequest = (message: string) => new ApiError(400, message);

/** Authenticated, but not allowed to touch this. */
export const forbidden = (message: string) => new ApiError(403, message);

export const notFound = (message: string) => new ApiError(404, message);

/**
 * A service we depend on failed. Ours, not the caller's, so the detail stays in `cause` where
 * only Sentry sees it.
 */
export const upstreamError = (serviceName: string, cause?: unknown) =>
  new ApiError(502, `The ${serviceName} service is unavailable. Please try again shortly.`, { cause });

function classify(error: unknown): { status: number; message: string } {
  if (error instanceof ApiError) {
    return { status: error.status, message: error.message };
  }
  // Validation messages are generated from a schema we wrote, so they describe the caller's
  // input rather than our internals, and they are the most useful thing we can say.
  if (error instanceof ValidationError || error instanceof ZodError) {
    return { status: 400, message: error.message };
  }
  // Everything unclassified — including every prisma error — is our fault and stays opaque.
  return { status: 500, message: SERVER_FAULT_MESSAGE };
}

export function toErrorResponse(error: unknown, request?: NextRequest): NextResponse {
  const { status, message } = classify(error);

  if (status >= 500) {
    const route = request ? `${request.method} ${request.nextUrl.pathname}` : 'api';
    console.error(`${route} -> ${status}`, error);
    Sentry.captureException(error, request ? { tags: { route: request.nextUrl.pathname } } : undefined);
  }

  // v1 answers with `error` on 49 routes and `message` on 35, and the frontend reads whichever
  // its author happened to pick. Emitting both keeps every existing reader working; v2 picks one.
  return NextResponse.json({ error: message, message }, { status });
}
