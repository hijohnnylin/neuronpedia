import { getVectorsForUser } from '@/lib/db/neuron';
import { getUserByName } from '@/lib/db/user';
import { Neuron } from '@prisma/client';
import { getServerSession, Session } from 'next-auth';
import { authOptions } from '../../../api/auth/[...nextauth]/authOptions';
import UserVectors from './user-vectors';

export default async function Page({ params }: { params: { name: string } }) {
  const session: Session | null = await getServerSession(authOptions);

  const user = await getUserByName(params.name);
  if (session?.user.id !== user.id) {
    return <div className="mt-10 text-center text-sm text-slate-600">You are not authorized to view this page.</div>;
  }

  const vectors = await getVectorsForUser(user.id);

  // sort vectors by label name
  vectors.sort((a, b) => {
    if (!a.vectorLabel) return 1;
    if (!b.vectorLabel) return -1;
    return a.vectorLabel.localeCompare(b.vectorLabel);
  });

  return <UserVectors initialVectors={vectors as Neuron[]} />;
}
