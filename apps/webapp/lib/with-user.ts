import { NextRequest, NextResponse } from 'next/server';

import { getAuthenticatedUserFromApiKey, getUserById, makeAuthedUserFromSessionOrReturnNull } from './db/user';
import { API_KEY_HEADER_NAME } from './env';

export type AuthenticatedUser = {
  id: string;
  name: string;
};

type NextRouteHandler = (request: NextRequest, arg?: any) => Promise<NextResponse> | NextResponse;

// ================ MARK: Optionally Authenticated User ================

export interface RequestOptionalUser extends NextRequest {
  user: AuthenticatedUser | null;
}

type NextHandlerWithUser<T = any> = (request: RequestOptionalUser, arg?: T) => Promise<NextResponse> | NextResponse;

export function withOptionalUser(handler: NextHandlerWithUser): NextRouteHandler {
  return async (request: NextRequest, arg?: any) => {
    let authenticatedUser;
    const apiKey = request.headers.get(API_KEY_HEADER_NAME);
    if (apiKey) {
      authenticatedUser = await getAuthenticatedUserFromApiKey(request, false);
    } else {
      authenticatedUser = await makeAuthedUserFromSessionOrReturnNull();
    }

    (request as RequestOptionalUser).user = authenticatedUser;
    return handler(request as RequestOptionalUser, arg);
  };
}

// ================ MARK: Authenticated User ================

export interface RequestAuthedUser extends NextRequest {
  user: AuthenticatedUser;
}

type NextHandlerWithAuthedUser<T = any> = (
  request: RequestAuthedUser,
  arg?: T,
) => Promise<NextResponse> | NextResponse;

export function withAuthedUser(handler: NextHandlerWithAuthedUser): NextRouteHandler {
  return async (request: NextRequest, arg?: any) => {
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
  };
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
  return async (request: NextRequest, arg?: any) => {
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
  };
}
