import { authOptions } from '@/app/api/auth/[...nextauth]/authOptions';
import { prisma } from '@/lib/db';
import { getProblemNodes } from '@/lib/db/problem';
import { Metadata } from 'next';
import { getServerSession } from 'next-auth';
import ProblemsGraph from './explorer-graph';

export const metadata: Metadata = {
  title: 'Interpretability Field Explorer',
  description:
    'Explore the landscape of open problems, tools, papers, and datasets in mechanistic interpretability research.',
};

export default async function ProblemsPage() {
  const session = await getServerSession(authOptions);

  const initialNodes = await getProblemNodes(false, session?.user?.id);

  let canEdit = false;
  if (session?.user?.id) {
    const dbUser = await prisma.user.findUnique({
      where: { id: session.user.id },
      select: { admin: true, isProblemEditor: true },
    });
    canEdit = dbUser?.admin === true || dbUser?.isProblemEditor === true;
  }

  const editors = await prisma.user.findMany({
    where: { isProblemEditor: true },
    select: { id: true, name: true, image: true },
    orderBy: { name: 'asc' },
  });

  return (
    <ProblemsGraph
      initialNodes={JSON.parse(JSON.stringify(initialNodes))}
      canEdit={canEdit}
      editors={editors}
    />
  );
}
