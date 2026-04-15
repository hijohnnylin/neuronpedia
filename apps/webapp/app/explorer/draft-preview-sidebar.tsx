'use client';

import { type ProblemNodeData, TYPE_COLORS } from './explorer-shared';

export function DraftPreviewSidebar({ draftNode, onCancel }: { draftNode: ProblemNodeData; onCancel: () => void }) {
  return (
    <div className="w-[420px] shrink-0 overflow-y-auto border-l border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          {draftNode.nodeTypes.map((t) => (
            <span
              key={t}
              className={`rounded px-2 py-0.5 text-xs font-semibold uppercase ${TYPE_COLORS[t] || 'bg-slate-100 text-slate-800'}`}
            >
              {t}
            </span>
          ))}
        </div>
        <button type="button" onClick={onCancel} className="text-slate-400 hover:text-slate-600">
          ✕
        </button>
      </div>
      <h2 className="mt-2 text-lg font-semibold text-slate-900">{draftNode.title || '(paste a URL to populate)'}</h2>
      {draftNode.author && <p className="mt-0.5 text-sm text-slate-500">{draftNode.author}</p>}
      {draftNode.description && <p className="mt-1 text-xs leading-relaxed text-slate-600">{draftNode.description}</p>}
      {(draftNode.mainUrl || draftNode.additionalUrls.length > 0) && (
        <div className="mt-3 flex flex-col gap-1">
          {draftNode.mainUrl && (
            <a
              href={draftNode.mainUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="break-all text-sm text-sky-600 hover:underline"
            >
              {draftNode.mainUrl}
            </a>
          )}
          {draftNode.additionalUrls.map((url, i) => (
            <a
              key={i}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="break-all text-sm text-sky-600 hover:underline"
            >
              {url}
            </a>
          ))}
        </div>
      )}
      {draftNode.parent && (
        <div className="mt-3 border-t border-slate-100 pt-3">
          <h3 className="text-xs font-semibold text-slate-500">Parent</h3>
          <div className="mt-1 flex items-center gap-2 rounded-lg border border-slate-100 px-3 py-2 text-sm">
            {draftNode.parent.nodeTypes.map((t) => (
              <span
                key={t}
                className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${TYPE_COLORS[t] || ''}`}
              >
                {t}
              </span>
            ))}
            <span className="text-slate-800">{draftNode.parent.title || '(untitled)'}</span>
          </div>
        </div>
      )}
      <p className="mt-4 text-xs text-slate-400">Paste a URL in the node to auto-populate fields, then click Create.</p>
    </div>
  );
}
