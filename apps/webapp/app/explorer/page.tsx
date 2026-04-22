import { authOptions } from '@/app/api/auth/[...nextauth]/authOptions';
import { prisma } from '@/lib/db';
import { getProblemNodes, getRecentProblemCreations } from '@/lib/db/problem';
import { Metadata } from 'next';
import { getServerSession } from 'next-auth';
import ProblemsGraph from './explorer-graph';

export const metadata: Metadata = {
  title: 'Interpretability Explorer',
  description: 'Explore the landscape of tools, papers, datasets, and problems in interpretability research.',
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

  const recentLogs = await getRecentProblemCreations(8);

  return (
    <ProblemsGraph
      initialNodes={JSON.parse(JSON.stringify(initialNodes))}
      canEdit={canEdit}
      editors={editors}
      initialRecentLogs={JSON.parse(JSON.stringify(recentLogs))}
    />
  );
}
