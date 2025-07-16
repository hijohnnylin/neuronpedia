'use client';

import { GraphMetadataSubgraphWithPartialRelations } from '@/prisma/generated/zod';
import { TrashIcon } from '@radix-ui/react-icons';
import { useSession } from 'next-auth/react';
import { parseGraphClerps, parseGraphSupernodes } from '../utils';

interface SubgraphItemProps {
  subgraph: GraphMetadataSubgraphWithPartialRelations;
  onClick: (subgraph: GraphMetadataSubgraphWithPartialRelations) => void;
  onDelete: () => void;
  fallbackDisplayName?: string;
}

export default function SubgraphItem({ subgraph, onClick, onDelete, fallbackDisplayName }: SubgraphItemProps) {
  const session = useSession();
  return (
    <button
      type="button"
      key={subgraph.id}
      onClick={() => onClick(subgraph)}
      className="flex w-full cursor-pointer flex-row items-center gap-x-2 border-b border-slate-100 p-3 py-2 text-left last:border-b-0 hover:bg-sky-50 focus:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-sky-500"
    >
      <div className="flex w-full flex-col items-start justify-between">
        <div className="mb-1 flex w-full flex-1 flex-row items-center justify-between">
          <p className="text-[13px] font-semibold text-sky-700">
            {subgraph.displayName || fallbackDisplayName || 'Subgraph'}
          </p>
          <p className="text-[11px] text-slate-400">
            {new Date(subgraph.createdAt).toLocaleDateString()} at {new Date(subgraph.createdAt).toLocaleTimeString()}
          </p>
        </div>
        <div className="flex w-full flex-1 flex-col items-start justify-start gap-y-0.5">
          <p className="mt-0.5 text-[11px] text-slate-500">
            {subgraph.pinnedIds.length} pinned nodes,{' '}
            {parseGraphSupernodes(JSON.stringify(subgraph.supernodes)).length || 0} supernodes,{' '}
            {parseGraphClerps(JSON.stringify(subgraph.clerps)).length || 0} custom labels
          </p>
          <p className="text-[11px] text-slate-500">
            {subgraph.pruningThreshold ? `${(subgraph.pruningThreshold * 100).toFixed(0)}%` : 'no'} influence threshold,{' '}
            {subgraph.densityThreshold ? `${(subgraph.densityThreshold * 100).toFixed(0)}%` : 'no'} density threshold
          </p>
          {subgraph.user?.name && session.data?.user?.id !== subgraph.user?.id && (
            <p className="w-full text-right text-[11px] text-slate-400">{subgraph.user?.name}</p>
          )}
        </div>
      </div>
      {session.data?.user?.id === subgraph.user?.id && (
        <button
          type="button"
          onClick={async (e) => {
            e.stopPropagation();
            // eslint-disable-next-line
            if (confirm('Are you sure you want to delete this subgraph?')) {
              try {
                await fetch('/api/graph/subgraph/delete', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({ subgraphId: subgraph.id }),
                });
                onDelete();
              } catch (error) {
                console.error('Failed to delete subgraph:', error);
              }
            }
          }}
          className="flex h-6 min-h-6 w-6 min-w-6 items-center justify-center rounded-md bg-red-100 text-red-700 hover:bg-red-300"
        >
          <TrashIcon className="h-4 w-4" />
        </button>
      )}
    </button>
  );
}
