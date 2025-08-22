import { authOptions } from '@/app/api/auth/[...nextauth]/authOptions';
import { API_KEY_HEADER_NAME } from '@/lib/constants';
import { prisma } from '@/lib/db';
import { env } from '@/lib/env';
import { RequestOptionalUser } from '@/lib/types/auth';
import { UserSecretType } from '@prisma/client';
import { getServerSession, Session } from 'next-auth';

export const getAuthenticatedUserFromApiKey = async (request: RequestOptionalUser, throwOnFail = true) => {
  const apiKey = request.headers.get(API_KEY_HEADER_NAME);
  if (!apiKey) {
    throw new Error('API Key missing');
  }
  const userSecret = await prisma.userSecret.findFirst({
    where: {
      value: apiKey,
      type: UserSecretType.NEURONPEDIA,
    },
  });
  if (!userSecret) {
    console.log('Invalid API Key');
    if (throwOnFail) {
      throw new Error('Invalid API Key');
    } else {
      return null;
    }
  }
  const user = await prisma.user.findUnique({
    where: {
      name: userSecret.username,
    },
    select: {
      id: true,
      name: true,
      admin: true,
    },
  });
  if (!user && throwOnFail) {
    throw new Error('Invalid API Key');
  } else {
    return user;
  }
};

// gets the session if not provided, then constructs an AuthenticatedUser
export const makeAuthedUserFromSessionOrReturnNull = async (session: Session | null = null) => {
  if (env.DEMO_MODE) {
    return null;
  }

  // eslint-disable-next-line no-param-reassign
  session = session || (await getServerSession(authOptions));
  if (session) {
    return { id: session.user.id, name: session.user.name };
  }
  return null;
};
