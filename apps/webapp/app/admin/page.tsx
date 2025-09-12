import { getUserById } from '@/lib/db/user';
import { env } from '@/lib/env';
import { getServerSession, Session } from 'next-auth';
import { authOptions } from '../api/auth/[...nextauth]/authOptions';
import ManageSourcesPane from './manage-sources-pane';

export default async function Page() {
  const session: Session | null = await getServerSession(authOptions);
  let isAdminUser = false;
  if (session?.user?.id) {
    try {
      const user = await getUserById(session?.user?.id || '');
      isAdminUser = user?.admin;
    } catch (error) {
      console.error('Not admin user', error);
    }
  }
  if (!env.DEMO_MODE && !env.IS_LOCALHOST && !isAdminUser) {
    return <div>Not authorized</div>;
  }
  return <ManageSourcesPane />;
}
