import { prisma } from '@/lib/db';
import { env } from '@/lib/env';
import { ModelWithRelations } from '@/prisma/generated/zod';
import AvailableResourcesTable from './table';

export default async function AvailableResourcesPage() {
  const models = await prisma.model.findMany({
    where: {
      visibility: 'PUBLIC',
    },
    include: {
      sourceSets: {
        where: { visibility: 'PUBLIC', hasDashboards: true },
        include: {
          sources: {
            where: { visibility: 'PUBLIC' },
            orderBy: { id: 'asc' },
            select: { id: true, inferenceEnabled: true },
          },
        },
        orderBy: { name: 'asc' },
      },
    },
    orderBy: {
      id: 'asc',
    },
  });

  return (
    <div className="mt-5 flex w-full flex-col justify-center px-3">
      <h1 className="text-center text-xl font-bold">Available Resources</h1>
      <div className="w-full text-center text-sm text-slate-500">
        The following are the public models and sources that are available for{' '}
        <a
          href={`${env.NEXT_PUBLIC_URL}/api-doc`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sky-700 underline"
        >
          API access
        </a>
        .
      </div>
      <div className="mt-2 max-h-screen w-full overflow-auto text-xs">
        <AvailableResourcesTable models={models as ModelWithRelations[]} />
      </div>
    </div>
  );
}
