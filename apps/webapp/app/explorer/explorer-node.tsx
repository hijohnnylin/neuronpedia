'use client';

import { Handle, Position } from '@xyflow/react';
import { ExternalLink } from 'lucide-react';
import { memo } from 'react';
import { NODE_HEIGHT } from './use-layout';

export const ROOT_NODE_WIDTH = 240;
export const DEFAULT_NODE_WIDTH = 250;

export const TYPE_COLORS: Record<
  string,
  {
    border: string;
    selectedBorder: string;
    label: string;
    icon: string;
    handleColor: string;
    selectedHandleColor: string;
  }
> = {
  topic: {
    border: 'border-slate-300 hover:border-slate-600',
    selectedBorder: 'border-slate-600 outline outline-slate-600 !outline-2',
    label: 'text-slate-400',
    icon: 'bg-slate-600',
    handleColor: '!bg-slate-500',
    selectedHandleColor: '!bg-slate-600',
  },
  paper: {
    border: 'border-slate-300 hover:border-emerald-600',
    selectedBorder: 'border-emerald-600 outline outline-emerald-600 outline-2',
    label: 'text-emerald-600',
    icon: 'bg-emerald-600',
    handleColor: '!bg-emerald-500',
    selectedHandleColor: '!bg-emerald-600',
  },
  tool: {
    border: 'border-slate-300 hover:border-indigo-600',
    selectedBorder: 'border-indigo-600 outline outline-indigo-600 outline-2',
    label: 'text-indigo-600',
    icon: 'bg-indigo-600',
    handleColor: '!bg-indigo-500',
    selectedHandleColor: '!bg-indigo-600',
  },
  dataset: {
    border: 'border-slate-300 hover:border-amber-600',
    selectedBorder: 'border-amber-600 outline outline-amber-600 outline-2',
    label: 'text-amber-600',
    icon: 'bg-amber-600',
    handleColor: '!bg-amber-500',
    selectedHandleColor: '!bg-amber-600',
  },
  eval: {
    border: 'border-slate-300 hover:border-rose-600',
    selectedBorder: 'border-rose-600 outline outline-rose-600 outline-2',
    label: 'text-rose-600',
    icon: 'bg-rose-600',
    handleColor: '!bg-rose-500',
    selectedHandleColor: '!bg-rose-600',
  },
  replication: {
    border: 'border-slate-300 hover:border-fuchsia-600',
    selectedBorder: 'border-fuchsia-600 outline outline-fuchsia-600 outline-2',
    label: 'text-fuchsia-600',
    icon: 'bg-fuchsia-600',
    handleColor: '!bg-fuchsia-500',
    selectedHandleColor: '!bg-fuchsia-600',
  },
  model: {
    border: 'border-slate-300 hover:border-stone-600',
    selectedBorder: 'border-stone-600 outline outline-stone-600 outline-2',
    label: 'text-stone-600',
    icon: 'bg-stone-600',
    handleColor: '!bg-stone-500',
    selectedHandleColor: '!bg-stone-600',
  },
};

function ProblemNodeComponent({ data, selected }: { data: any; selected: boolean }) {
  const types: string[] = data.nodeTypes || [data.type || 'topic'];
  const primaryType = types[0] || 'topic';
  const colors = TYPE_COLORS[primaryType] || TYPE_COLORS.topic;
  const highlighted = selected || data.isDraft;
  const border = highlighted ? colors.selectedBorder : colors.border;
  //const hasCollapsedChildren = (data.hiddenChildCount ?? 0) > 0;

  return (
    <div
      className={`group relative rounded-md border ${border} duration-250 flex items-start bg-white px-2 py-[5px] shadow-sm transition-[shadow,opacity,filter] hover:shadow-md`}
      style={{
        height: NODE_HEIGHT,
        ...(data.isRoot ? { width: ROOT_NODE_WIDTH } : { minWidth: DEFAULT_NODE_WIDTH, maxWidth: DEFAULT_NODE_WIDTH }),
        ...(data.dimmed
          ? { opacity: 0.3, filter: 'blur(1.3px) grayscale(0.5)' }
          : data.filterDimmed
            ? { opacity: 0.4 }
            : {}),
      }}
      onMouseEnter={() => {
        data.onHoverNode?.();
      }}
      onMouseLeave={() => {
        data.onHoverLeave?.();
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ width: 0, height: 0, minWidth: 0, minHeight: 0, border: 'none', background: 'transparent', left: 0 }}
      />

      <div className="absolute left-2 top-[3px] flex gap-1">
        {types.map((t) => (
          <span key={t} className={`text-[6.5px] font-medium uppercase ${(TYPE_COLORS[t] || TYPE_COLORS.topic).label}`}>
            {t.toUpperCase()}
          </span>
        ))}
      </div>

      <div className="my-0.5 mb-[0px] mt-[5px] flex w-full flex-col justify-start text-left">
        <div className="mt-[4px] truncate font-sans text-[9.5px] font-normal leading-tight tracking-tight text-slate-800">
          {data.label || '(untitled)'}
        </div>
        {primaryType === 'topic'
          ? data.description && (
              <div className="mt-[3px] truncate text-[8px] leading-tight text-slate-400">{data.description}</div>
            )
          : data.author && (
              <div className="mt-[3px] truncate text-[8px] leading-tight text-slate-400">{data.author}</div>
            )}
      </div>
      {Object.keys((data.descendantTypeCounts || {}) as Record<string, number>).length > 0 && (
        <div className="absolute bottom-1 right-1.5 flex gap-0.5">
          {Object.entries((data.descendantTypeCounts || {}) as Record<string, number>)
            .sort(([typeA], [typeB]) => typeA.localeCompare(typeB))
            .map(([type, count]) => (
              <span
                key={type}
                className={`rounded-[2px] px-1 py-[2px] text-[6px] font-medium uppercase leading-none text-white ${(TYPE_COLORS[type] || TYPE_COLORS.topic).icon}`}
              >
                {count} {type}
                {count !== 1 ? 's' : ''}
              </span>
            ))}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Right}
        style={{ width: 0, height: 0, minWidth: 0, minHeight: 0, border: 'none', background: 'transparent', right: 0 }}
      />
      {data.approvalState === 'PENDING' && (
        <div className="absolute -bottom-1 left-1/2 z-10 -translate-x-1/2 whitespace-nowrap rounded-full border border-slate-300 bg-slate-100 px-2 py-[1px] text-[7px] font-semibold uppercase text-slate-500 shadow-sm">
          Pending Approval
        </div>
      )}
      {data.mainUrl && (
        <a
          href={data.mainUrl}
          target="_blank"
          rel="noopener noreferrer"
          title="Open link in new tab"
          onClick={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
          className="absolute bottom-1.5 left-2 z-10 flex items-center gap-x-0.5 text-[7.5px] font-semibold uppercase leading-none text-sky-600 hover:underline"
        >
          Link
          <ExternalLink size={8} className="-mt-[1px]" />
        </a>
      )}
      {/* {data.mainUrl && (
        <a
          href={data.mainUrl}
          target="_blank"
          rel="noopener noreferrer"
          title="Open link in new tab"
          onClick={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
          className="absolute -bottom-1 right-14 z-10 flex h-5 w-16 translate-x-full items-center justify-center gap-x-1 rounded-full border border-slate-400 bg-white text-[9px] font-semibold text-slate-500 opacity-0 shadow transition-opacity hover:border-slate-600 hover:bg-slate-600 hover:text-white group-hover:opacity-100"
        >
          Source
          <ExternalLink size={10} className="-mt-[1px]" />
        </a>
      )} */}
      {/* {hasCollapsedChildren && !data.isDraft && (
        <button
          type="button"
          title="Expand children"
          className="absolute right-7 top-1/2 z-10 flex h-5 w-14 -translate-y-1/2 translate-x-full items-center justify-center rounded-full border border-slate-400 bg-white text-[7.5px] font-medium text-slate-500 opacity-0 shadow transition-opacity hover:border-slate-600 hover:bg-slate-600 hover:text-white group-hover:opacity-100"
        >
          Expand &#8594;
        </button>
      )} */}
      {data.onAddChild && !data.isDraft && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            data.onAddChild();
          }}
          title="Add child node"
          className="absolute -top-2 right-10 z-10 flex h-4 w-12 translate-x-full items-center justify-center gap-x-1.5 rounded-full border border-sky-600 bg-white text-[6.5px] font-medium leading-none text-sky-600 opacity-0 shadow transition-opacity hover:border-sky-600 hover:bg-sky-600 hover:text-white group-hover:opacity-100"
        >
          + Sub-Item
        </button>
      )}
    </div>
  );
}

export default memo(ProblemNodeComponent);
