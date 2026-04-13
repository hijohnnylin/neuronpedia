import { PrismaAdapter } from '@next-auth/prisma-adapter';
import { PrismaClient } from '@prisma/client';
import { AdapterUser } from 'next-auth/adapters';
import { generateFromEmail } from 'unique-username-generator';

import { checkUserExistsByEmail, checkUserExistsByName, createUser } from './user';

// https://github.com/nextauthjs/next-auth/discussions/562
/** @return { import("next-auth/adapters").Adapter } */
export default function CustomPrismaAdapterForNextAuth(prisma: PrismaClient) {
  const adapter = PrismaAdapter(prisma);

  adapter.createUser = async (data: any) => {
    const userExist = await checkUserExistsByEmail(data.email);
    if (userExist) {
      return userExist as AdapterUser;
    }

    let username = generateFromEmail(data.email);
    const userNameExists = await checkUserExistsByName(username);
    if (userNameExists) {
      username = generateFromEmail(data.email, 4);
    }

    const createdUser = await createUser(
      data.email,
      username,
      data.image || `https://www.gravatar.com/avatar/${Math.random().toString(36).substring(7)}?d=identicon&r=PG`,
      data.emailVerified,
    );
    return createdUser as AdapterUser;
  };

  return adapter;
}
