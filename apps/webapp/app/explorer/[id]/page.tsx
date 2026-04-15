import { authOptions } from '@/app/api/auth/[...nextauth]/authOptions';
import { prisma } from '@/lib/db';
import { getProblemNodes } from '@/lib/db/problem';
import { Metadata } from 'next';
import { getServerSession } from 'next-auth';
import ProblemsGraph from '../explorer-graph';

export const metadata: Metadata = {
  title: 'Open Problems in Mechanistic Interpretability',
  description:
    'Explore the landscape of open problems, tools, papers, and datasets in mechanistic interpretability research.',
};

export default async function ProblemNodePage(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const session = await getServerSession(authOptions);

  const initialNodes = await getProblemNodes(false);

  let canEdit = false;
  if (session?.user?.id) {
    const dbUser = await prisma.user.findUnique({
      where: { id: session.user.id },
      select: { admin: true, isProblemEditor: true },
    });
    canEdit = dbUser?.admin === true || dbUser?.isProblemEditor === true;
  }

  return (
    <ProblemsGraph
      initialNodes={JSON.parse(JSON.stringify(initialNodes))}
      canEdit={canEdit}
      initialSelectedId={Number(params.id)}
    />
  );
}
