import { NextRequest, NextResponse } from 'next/server';
// eslint-disable-next-line import/no-cycle
import { getAuthenticatedUserFromApiKey, getUserById, makeAuthedUserFromSessionOrReturnNull } from './db/user';
import { API_KEY_HEADER_NAME } from './env';

export type AuthenticatedUser = {
  id: string;
  name: string;
};

// ================ MARK: Optionally Authenticated User ================

export interface RequestOptionalUser extends NextRequest {
  user: AuthenticatedUser | null;
}

type NextHandlerWithUser<T = any> = (request: RequestOptionalUser, arg?: T) => Promise<NextResponse> | NextResponse;

export function withOptionalUser(handler: NextHandlerWithUser): NextHandlerWithUser {
  return async (request: RequestOptionalUser, response: NextResponse) => {
    let authenticatedUser;
    const apiKey = request.headers.get(API_KEY_HEADER_NAME);
    if (apiKey) {
      authenticatedUser = await getAuthenticatedUserFromApiKey(request, false);
    } else {
      authenticatedUser = await makeAuthedUserFromSessionOrReturnNull();
    }

    request.user = authenticatedUser;
    return handler(request, response);
  };
}

// ================ MARK: Authenticated User ================

export interface RequestAuthedUser extends NextRequest {
  user: AuthenticatedUser;
}

type NextHandlerWithAuthedUser<T = any> = (
  request: RequestAuthedUser,
  arg?: T, // res: NextResponse,
) => Promise<NextResponse> | NextResponse;

export function withAuthedUser(handler: NextHandlerWithAuthedUser): NextHandlerWithAuthedUser {
  return async (request: RequestAuthedUser, response: NextResponse) => {
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

    request.user = authenticatedUser;
    return handler(request, response);
  };
}

// ================ MARK: Admin User ================

export interface RequestAuthedAdminUser extends NextRequest {
  user: AuthenticatedUser;
}

type NextHandlerWithAuthedAdminUser<T = any> = (
  request: RequestAuthedAdminUser,
  arg?: T, // res: NextResponse,
) => Promise<NextResponse> | NextResponse;

export async function getAuthedAdminUser(request: RequestAuthedAdminUser): Promise<AuthenticatedUser | null> {
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

export function withAuthedAdminUser(handler: NextHandlerWithAuthedAdminUser): NextHandlerWithAuthedAdminUser {
  return async (request: RequestAuthedAdminUser, response: NextResponse) => {
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

    request.user = authenticatedAdminUser;
    return handler(request, response);
  };
}
