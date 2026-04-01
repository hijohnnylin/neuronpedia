'use client';

import { Handle, Position } from '@xyflow/react';
import { Pencil } from 'lucide-react';
import { memo } from 'react';

export const ROOT_NODE_WIDTH = 180;
export const DEFAULT_NODE_WIDTH = 270;

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
    border: 'border-slate-300 hover:border-sky-600',
    selectedBorder: 'border-sky-600 outline outline-sky-600 !outline-2',
    label: 'text-sky-600',
    icon: 'bg-sky-600',
    handleColor: '!bg-sky-500',
    selectedHandleColor: '!bg-sky-600',
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

  return (
    <div
      className={`group relative rounded-md border ${border} flex items-center bg-white px-2 py-[5px] shadow-sm transition-all hover:shadow-md`}
      style={data.isRoot ? { width: ROOT_NODE_WIDTH } : { minWidth: DEFAULT_NODE_WIDTH, maxWidth: DEFAULT_NODE_WIDTH }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ width: 0, height: 0, minWidth: 0, minHeight: 0, border: 'none', background: 'transparent', left: 0 }}
      />

      <div className="absolute left-2 top-[1px] flex gap-1">
        {types.map((t) => (
          <span key={t} className={`text-[6px] font-bold ${(TYPE_COLORS[t] || TYPE_COLORS.topic).label}`}>
            {t.toUpperCase()}
          </span>
        ))}
      </div>

      <div className="my-0.5 mb-[0px] mt-[5px] flex w-full flex-col justify-start text-left">
        <div
          className={`font-sans text-[11px] font-normal leading-tight tracking-tight text-slate-800 ${data.isRoot ? 'whitespace-nowrap' : ''}`}
        >
          {data.label || '(untitled)'}
        </div>
        {data.author && !types.every((t: string) => t === 'topic') && (
          <div className="mt-0.5 truncate text-[9px] leading-tight text-slate-400">{data.author}</div>
        )}
      </div>
      <Handle
        type="source"
        position={Position.Right}
        style={{ width: 0, height: 0, minWidth: 0, minHeight: 0, border: 'none', background: 'transparent', right: 0 }}
      />
      {data.onEditNode && !data.isDraft && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            data.onEditNode();
          }}
          className="absolute -top-3 left-1/2 flex h-5 w-5 -translate-x-1/2 items-center justify-center rounded-full border border-slate-400 bg-white text-slate-400 opacity-0 shadow-sm transition-opacity hover:border-slate-600 hover:text-slate-600 group-hover:opacity-100"
        >
          <Pencil size={10} />
        </button>
      )}
      {data.onAddChild && !data.isDraft && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            data.onAddChild();
          }}
          className="absolute right-2 top-1/2 flex h-5 w-5 -translate-y-1/2 translate-x-full items-center justify-center rounded-full border border-sky-600 bg-white text-xs font-black leading-none text-sky-600 opacity-0 shadow-sm transition-opacity hover:border-sky-600 hover:bg-sky-600 hover:text-white group-hover:opacity-100"
        >
          +
        </button>
      )}
    </div>
  );
}

export default memo(ProblemNodeComponent);
