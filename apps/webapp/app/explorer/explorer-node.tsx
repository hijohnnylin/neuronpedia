'use client';

import { Handle, Position } from '@xyflow/react';
import { ExternalLink } from 'lucide-react';
import { memo, useState } from 'react';
import { TYPE_COLORS as SHARED_TYPE_COLORS } from './explorer-shared';
import { NODE_HEIGHT } from './use-layout';

export const ROOT_NODE_WIDTH = 180;
export const DEFAULT_NODE_WIDTH = 270;

const MAX_PREVIEW_CHILDREN = 8;

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

type ChildPreview = { id: number; title: string | null; nodeTypes: string[]; approvalState: string };

function ProblemNodeComponent({ data, selected }: { data: any; selected: boolean }) {
  const [isHovered, setIsHovered] = useState(false);
  const types: string[] = data.nodeTypes || [data.type || 'topic'];
  const primaryType = types[0] || 'topic';
  const colors = TYPE_COLORS[primaryType] || TYPE_COLORS.topic;
  const highlighted = selected || data.isDraft;
  const border = highlighted ? colors.selectedBorder : colors.border;
  const children: ChildPreview[] = data.children || [];
  const hasCollapsedChildren = (data.hiddenChildCount ?? 0) > 0;
  const showPreview = isHovered && hasCollapsedChildren && !data.hoverDimmed;

  const effectiveOpacity = data.hoverDimmed ? 0.35 : data.dimmed ? 0.2 : undefined;

  return (
    <div
      className={`group relative rounded-md border ${border} duration-250 flex items-start bg-white px-2 py-[5px] shadow-sm transition-[shadow,opacity] hover:shadow-md`}
      style={{
        height: NODE_HEIGHT,
        ...(data.isRoot ? { width: ROOT_NODE_WIDTH } : { minWidth: DEFAULT_NODE_WIDTH, maxWidth: DEFAULT_NODE_WIDTH }),
        opacity: effectiveOpacity ?? 1,
      }}
      onMouseEnter={() => {
        if (hasCollapsedChildren && data.onHoverNode) {
          setIsHovered(true);
          data.onHoverNode();
        }
      }}
      onMouseLeave={() => {
        if (isHovered) {
          setIsHovered(false);
          data.onHoverLeave?.();
        }
      }}
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
        <div className="mt-0.5 truncate font-sans text-[11px] font-normal leading-tight tracking-tight text-slate-800">
          {data.label || '(untitled)'}
        </div>
        {data.author && <div className="mt-1 truncate text-[9px] leading-tight text-slate-400">{data.author}</div>}
      </div>
      {data.hiddenChildCount > 0 && (
        <div className="absolute bottom-1 right-1 rounded bg-slate-400 px-1 py-0.5 text-[7.5px] font-semibold uppercase text-white group-hover:hidden">
          {data.hiddenChildCount} Subnode{data.hiddenChildCount !== 1 ? 's' : ''}
        </div>
      )}
      <Handle
        type="source"
        position={Position.Right}
        style={{ width: 0, height: 0, minWidth: 0, minHeight: 0, border: 'none', background: 'transparent', right: 0 }}
      />
      {data.mainUrl && (
        <a
          href={data.mainUrl}
          target="_blank"
          rel="noopener noreferrer"
          title="Open link in new tab"
          onClick={(e) => e.stopPropagation()}
          className="absolute -top-1 right-14 z-10 flex h-5 w-16 translate-x-full items-center justify-center gap-x-1.5 rounded-full border border-slate-400 bg-white text-[9px] font-semibold text-slate-500 opacity-0 shadow transition-opacity hover:border-slate-600 hover:text-slate-600 group-hover:opacity-100"
        >
          <ExternalLink size={10} />
          Open
        </a>
      )}
      {hasCollapsedChildren && !data.isDraft && (
        <button
          type="button"
          title="Expand children"
          className="absolute right-14 top-1/2 z-10 flex h-5 w-16 -translate-y-1/2 translate-x-full items-center justify-center rounded-full border border-slate-400 bg-white text-[9px] font-semibold text-slate-500 opacity-0 shadow transition-opacity hover:border-slate-600 hover:bg-slate-600 hover:text-white group-hover:opacity-100"
        >
          Expand &#8594;
        </button>
      )}
      {data.onAddChild && !data.isDraft && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            data.onAddChild();
          }}
          title="Add child node"
          className="absolute -bottom-1 right-14 z-10 flex h-5 w-16 translate-x-full items-center justify-center gap-x-1.5 rounded-full border border-sky-600 bg-white text-[9px] font-semibold text-sky-600 opacity-0 shadow transition-opacity hover:border-sky-600 hover:bg-sky-600 hover:text-white group-hover:opacity-100"
        >
          + Subnode
        </button>
      )}

      {showPreview && (
        <div className="absolute left-full top-1/2 ml-3 w-56 -translate-y-1/2 rounded-lg border border-slate-200 bg-white/95 p-2 shadow-xl backdrop-blur-sm">
          {/* <div className="mb-1.5 text-[8px] font-semibold uppercase tracking-wide text-slate-400">
            {children.length} Subnode{children.length !== 1 ? 's' : ''}
          </div> */}
          <div className="flex flex-col gap-0.5">
            {children.slice(0, MAX_PREVIEW_CHILDREN).map((child) => (
              <div
                key={child.id}
                className="flex flex-col items-start gap-0.5 rounded border-b border-slate-200 px-1 py-2 first:pt-0 last:border-b-0 last:pb-0"
              >
                {/* Tag(s) above item title */}
                <div className="mb-0.5 flex shrink-0 gap-0.5">
                  {child.nodeTypes.map((t) => (
                    <span
                      key={t}
                      className={`rounded px-1 py-[1px] text-[6px] font-bold uppercase leading-tight ${SHARED_TYPE_COLORS[t] || 'bg-slate-500 text-white'}`}
                    >
                      {t}
                    </span>
                  ))}
                </div>
                <span
                  className="line-clamp-2 max-w-xs break-words text-[10px] leading-tight text-slate-700"
                  style={{
                    display: '-webkit-box',
                    WebkitBoxOrient: 'vertical',
                    WebkitLineClamp: 2,
                    overflow: 'hidden',
                  }}
                >
                  {child.title || '(untitled)'}
                </span>
              </div>
            ))}
            {children.length > MAX_PREVIEW_CHILDREN && (
              <div className="mt-0.5 text-center text-[9px] text-slate-400">
                +{children.length - MAX_PREVIEW_CHILDREN} more
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default memo(ProblemNodeComponent);
