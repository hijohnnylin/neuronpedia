import EditProfileForm from '@/components/edit-profile-form';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { getUserById } from '@/lib/db/user';
import { getServerSession, Session } from 'next-auth';
import { redirect } from 'next/navigation';
import { authOptions } from '../api/auth/[...nextauth]/authOptions';

export default async function Account() {
  const session: Session | null = await getServerSession(authOptions);
  if (!session) {
    redirect('/signin');
  }
  const user = await getUserById(session.user.id);

  return (
    <div className="mb-10 mt-6 w-full max-w-lg px-5 xl:px-0">
      <Card className="bg-white">
        <CardHeader>
          <CardTitle>Account</CardTitle>
        </CardHeader>
        <CardContent>
          <EditProfileForm
            defaultUsername={user.name}
            defaultNewsletterNotifyEmail={user.emailNewsletterNotification}
            defaultUnsubscribeAllEmail={user.emailUnsubscribeAll}
          />
        </CardContent>
      </Card>
    </div>
  );
}
