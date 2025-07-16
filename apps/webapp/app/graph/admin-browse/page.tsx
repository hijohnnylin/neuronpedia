import { prisma } from '@/lib/db';
import { GRAPH_ADMIN_BROWSE_KEY, NEXT_PUBLIC_URL } from '@/lib/env';
import { notFound } from 'next/navigation';

export default async function Page({
  searchParams,
}: {
  searchParams: {
    key: string;
  };
}) {
  // Check that the browse key is correct
  if (searchParams.key !== GRAPH_ADMIN_BROWSE_KEY) {
    notFound();
  }

  // get all metadatagraphs from the database after May 29th, 9am, 2025 pacific time
  const allGraphMetadatas = await prisma.graphMetadata.findMany({
    where: {
      createdAt: {
        gt: new Date('2025-05-29T16:00:00Z'),
      },
    },
    orderBy: {
      createdAt: 'desc',
    },
  });
  return (
    <div className="mx-auto max-w-screen-xl p-4">
      <h1 className="mb-2 text-2xl font-bold">{allGraphMetadatas.length} Total Graphs Generated</h1>
      <p className="mb-4 text-sm font-bold text-red-600">DO NOT SHARE</p>

      <div className="overflow-x-auto">
        <table className="min-w-full border border-gray-200 bg-white">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                Created At (PT)
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">Slug</th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">Prompt</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {allGraphMetadatas.map((metadata) => (
              <tr key={metadata.id}>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-900">
                  {metadata.createdAt.toLocaleString('en-US', { timeZone: 'America/Los_Angeles' })}
                </td>
                <td className="max-w-48 overflow-x-hidden truncate whitespace-nowrap px-6 py-4 text-sm font-medium text-sky-600">
                  <a
                    href={`${NEXT_PUBLIC_URL}/${metadata.modelId}/graph?slug=${metadata.slug}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline"
                  >
                    {metadata.slug}
                  </a>
                </td>
                <td className="px-6 py-4 text-sm text-gray-900">
                  <div className="flex flex-col">
                    <p className="mb-2">{metadata.prompt}</p>
                    <div className="flex max-w-[600px] flex-wrap gap-x-1 gap-y-[3px]">
                      {metadata.promptTokens?.map((t, idx) => (
                        <span
                          key={`${t}-${idx}`}
                          className="mx-0 rounded bg-slate-200 px-[3px] py-[1px] font-mono text-[10px] text-slate-700"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
