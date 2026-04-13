'use client';

import NewVectorForm from '@/components/new-vector-form';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import { getLayerNumFromSource } from '@/lib/utils/source';
import copy from 'copy-to-clipboard';
import { Copy, ExternalLink, Joystick, Trash } from 'lucide-react';
import Link from 'next/link';
import { NeuronWithPartialRelations } from 'prisma/generated/zod';
import { TableVirtuoso } from 'react-virtuoso';

export default function UserVectors({ initialVectors }: { initialVectors: NeuronWithPartialRelations[] }) {
  const { getInferenceEnabledForModel } = useGlobalContext();

  async function deleteVector(vector: NeuronWithPartialRelations) {
    if (window.confirm('Are you sure you want to delete this vector?')) {
      await fetch(`/api/vector/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelId: vector.modelId,
          source: vector.layer,
          index: vector.index,
        }),
      });
      // reload the page
      window.location.reload();
    }
  }

  return (
    <div className="flex w-full max-w-screen-xl flex-col items-center pt-5">
      <div className="flex w-full max-w-screen-xl flex-col">
        <h1 className="mb-0 px-3 py-1 text-center text-sm font-medium text-slate-500 xl:px-0">My Vectors</h1>
      </div>
      <div className="mt-2 flex w-full max-w-screen-xl flex-col">
        <TableVirtuoso
          id="vectors-table"
          style={{ height: 400, width: '100%' }}
          fixedHeaderContent={() => (
            <tr className="h-8 w-full rounded-md bg-slate-200 text-center text-[11px] uppercase text-slate-400">
              <th className="">Label</th>
              <th className="">Model</th>
              <th className="">Layer</th>
              <th className="">Hook</th>
              <th className="">Steer Strength</th>
              <th className="">Created At</th>
              <th className="">Raw Vector</th>
              <th className="">Actions</th>
            </tr>
          )}
          className="rounded text-xs text-slate-600 odd:bg-slate-100"
          data={initialVectors}
          itemContent={(index, vector) => (
            <>
              <td className="px-3 py-2 text-[13px]">
                <Link
                  prefetch={false}
                  href={`/${vector.modelId}/${vector.layer}/${vector.index}`}
                  className="font-medium text-sky-700 hover:underline"
                >
                  {vector.vectorLabel}
                </Link>
              </td>
              <td className="text-center">{vector.modelId}</td>
              <td className="text-center">{getLayerNumFromSource(vector.layer)}</td>
              <td className="text-center">{vector.hookName}</td>
              <td className="text-center">{vector.vectorDefaultSteerStrength}</td>
              <td className="text-center">{new Date(vector.createdAt).toLocaleDateString()}</td>
              <td>
                <div className="mt-0 flex flex-row items-center justify-center gap-x-2">
                  <input
                    readOnly
                    className="h-8 min-h-8 w-32 min-w-32 rounded border-slate-300 px-2.5 py-1 font-mono text-[10px] leading-tight text-slate-400 outline-none focus:border-slate-400 focus:outline-none focus:ring-0"
                    value={JSON.stringify(vector.vector)}
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => {
                      copy(JSON.stringify(vector.vector));
                      alert('Copied to clipboard.');
                    }}
                    className="h-8 w-8 min-w-8"
                  >
                    <Copy className="h-4 w-4 text-slate-500" />
                  </Button>
                </div>
              </td>
              <td className="flex flex-1 flex-col items-center justify-center py-2">
                <div className="mt-0 flex flex-row items-center justify-center gap-x-1.5">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => {
                      window.open(`/${vector.modelId}/steer`, '_blank');
                    }}
                    aria-label="Steer"
                    className={`h-8 w-8 min-w-8 ${getInferenceEnabledForModel(vector.modelId) ? '' : 'hidden'}`}
                  >
                    <Joystick className="h-4 w-4 text-slate-500" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => {
                      window.open(`/${vector.modelId}/${vector.layer}/${vector.index}`, '_blank');
                    }}
                    aria-label="Open in New Tab"
                    className="h-8 w-8 min-w-8"
                  >
                    <ExternalLink className="h-4 w-4 text-slate-500" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => {
                      deleteVector(vector);
                    }}
                    className="h-8 w-8 border border-red-200 text-red-400 hover:bg-red-50"
                  >
                    <Trash className="h-4 w-4" />
                  </Button>
                </div>
              </td>
            </>
          )}
        />
      </div>
      <div className="mt-3 grid w-full max-w-screen-xl items-start gap-x-2 gap-y-2 px-2 py-3 pb-16 sm:grid-cols-2 md:grid-cols-3">
        <div className="flex grid-cols-1 flex-col rounded-md border bg-white px-5 pb-5 pt-2">
          <div className="mb-2 mt-2 text-center text-sm font-medium text-slate-500">Add Vector</div>
          <NewVectorForm
            callback={() => {
              window.location.reload();
            }}
          />
        </div>
      </div>
    </div>
  );
}
