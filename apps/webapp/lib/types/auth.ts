import { NextRequest } from 'next/server';

export type AuthenticatedUser = {
  id: string;
  name: string;
};

export interface RequestOptionalUser extends NextRequest {
  user: AuthenticatedUser | null;
}

export interface RequestAuthedUser extends NextRequest {
  user: AuthenticatedUser;
}

export interface RequestAuthedAdminUser extends NextRequest {
  user: AuthenticatedUser;
}
