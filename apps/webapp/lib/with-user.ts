import { NextRequest, NextResponse } from 'next/server';

import { toErrorResponse } from './api-error';
import { getAuthenticatedUserFromApiKey, getUserById, makeAuthedUserFromSessionOrReturnNull } from './db/user';
import { API_KEY_HEADER_NAME } from './env';

export type AuthenticatedUser = {
  id: string;
  name: string;
};

type NextRouteHandler = (request: NextRequest, arg?: any) => Promise<NextResponse> | NextResponse;

/**
 * Every wrapper below routes its failures through here, which is what gives all 107 wrapped
 * routes one error contract (see `api-error.ts`) without editing them.
 *
 * This cannot change what a route that already catches does, because its own catch runs first —
 * the effect is confined to the errors that currently escape, which today become an unparseable
 * plain-text 500 from the framework. Auth resolution is inside the try on purpose: it hits the
 * database, so it can fail like anything else.
 */
async function handleErrors(request: NextRequest, run: () => Promise<NextResponse> | NextResponse) {
  try {
    return await run();
  } catch (error) {
    return toErrorResponse(error, request);
  }
}

// ================ MARK: Optionally Authenticated User ================

export interface RequestOptionalUser extends NextRequest {
  user: AuthenticatedUser | null;
}

type NextHandlerWithUser<T = any> = (request: RequestOptionalUser, arg?: T) => Promise<NextResponse> | NextResponse;

export function withOptionalUser(handler: NextHandlerWithUser): NextRouteHandler {
  return async (request: NextRequest, arg?: any) =>
    handleErrors(request, async () => {
      let authenticatedUser;
      const apiKey = request.headers.get(API_KEY_HEADER_NAME);
      if (apiKey) {
        authenticatedUser = await getAuthenticatedUserFromApiKey(request, false);
      } else {
        authenticatedUser = await makeAuthedUserFromSessionOrReturnNull();
      }

      (request as RequestOptionalUser).user = authenticatedUser;
      return handler(request as RequestOptionalUser, arg);
    });
}

// ================ MARK: Authenticated User ================

export interface RequestAuthedUser extends NextRequest {
  user: AuthenticatedUser;
}

type NextHandlerWithAuthedUser<T = any> = (request: RequestAuthedUser, arg?: T) => Promise<NextResponse> | NextResponse;

export function withAuthedUser(handler: NextHandlerWithAuthedUser): NextRouteHandler {
  return async (request: NextRequest, arg?: any) =>
    handleErrors(request, async () => {
      let authenticatedUser;
      const apiKey = request.headers.get(API_KEY_HEADER_NAME);
      if (apiKey) {
        authenticatedUser = await getAuthenticatedUserFromApiKey(request, false);
      } else {
        authenticatedUser = await makeAuthedUserFromSessionOrReturnNull();
      }

      if (!authenticatedUser) {
        return NextResponse.json(
          {
            error:
              'This endpoint requires authorization. Specify your API key in the header x-api-key. Your API key is under Settings on neuronpedia.org.',
          },
          { status: 401 },
        );
      }

      (request as RequestAuthedUser).user = authenticatedUser;
      return handler(request as RequestAuthedUser, arg);
    });
}

// ================ MARK: Admin User ================

export interface RequestAuthedAdminUser extends NextRequest {
  user: AuthenticatedUser;
}

type NextHandlerWithAuthedAdminUser<T = any> = (
  request: RequestAuthedAdminUser,
  arg?: T,
) => Promise<NextResponse> | NextResponse;

export async function getAuthedAdminUser(request: NextRequest): Promise<AuthenticatedUser | null> {
  let authenticatedUser;
  const apiKey = request.headers.get(API_KEY_HEADER_NAME);
  if (apiKey) {
    authenticatedUser = await getAuthenticatedUserFromApiKey(request, false);
  } else {
    const user = await makeAuthedUserFromSessionOrReturnNull();
    if (user) {
      authenticatedUser = await getUserById(user.id);
    }
  }
  return authenticatedUser?.admin ? authenticatedUser : null;
}

export function withAuthedAdminUser(handler: NextHandlerWithAuthedAdminUser): NextRouteHandler {
  return async (request: NextRequest, arg?: any) =>
    handleErrors(request, async () => {
      const authenticatedAdminUser = await getAuthedAdminUser(request);
      if (!authenticatedAdminUser) {
        return NextResponse.json(
          {
            error:
              'This endpoint requires authorization and admin access. Specify your API key in the header x-api-key. Your API key is under Settings on neuronpedia.org.',
          },
          { status: 401 },
        );
      }

      (request as RequestAuthedAdminUser).user = authenticatedAdminUser;
      return handler(request as RequestAuthedAdminUser, arg);
    });
}
