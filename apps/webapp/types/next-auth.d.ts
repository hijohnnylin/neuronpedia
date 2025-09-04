// this file augments our session user to include the user id
// ignore this eslint rule because we need to augment the type

/* eslint-disable-next-line unused-imports/no-unused-imports */
import 'next-auth';

declare module 'next-auth' {
  /**
   * Returned by `useSession`, `getSession` and received as a prop on the `SessionProvider` React Context
   */
  interface Session {
    user: {
      id: string;
      email: string;
      image: string;
      name: string;
      country?: string;
    };
  }
}
