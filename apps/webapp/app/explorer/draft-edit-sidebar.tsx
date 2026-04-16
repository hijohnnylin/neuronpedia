'use client';

import { useCallback, useState } from 'react';
import { TYPE_COLORS as NODE_TYPE_COLORS } from './explorer-node';
import { MAX_TITLE_LENGTH, NODE_TYPES_LIST, type ProblemNodeData } from './explorer-shared';

export function DraftEditSidebar({
  draftNode,
  setDraftNode,
  existingNodes,
  onSaved,
  onCancel,
}: {
  draftNode: ProblemNodeData;
  setDraftNode: (d: ProblemNodeData) => void;
  existingNodes: ProblemNodeData[];
  onSaved: () => Promise<void>;
  onCancel: () => void;
}) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [newAdditionalUrl, setNewAdditionalUrl] = useState('');

  const handleSubmit = useCallback(async () => {
    if (!draftNode.title?.trim()) {
      setError('Title is required');
      return;
    }
    setSubmitting(true);
    setError('');
    try {
      await onSaved();
    } catch {
      setError('Network error');
      setSubmitting(false);
    }
  }, [draftNode, onSaved]);

  const updateDraft = useCallback(
    (fields: Partial<ProblemNodeData>) => {
      setDraftNode({ ...draftNode, ...fields });
    },
    [draftNode, setDraftNode],
  );

  return (
    <div className="w-[420px] shrink-0 overflow-y-auto border-l border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">New Node</h2>
        <button type="button" onClick={onCancel} className="text-slate-400 hover:text-slate-600">
          ✕
        </button>
      </div>

      <div className="mt-4 flex flex-col gap-3">
        {/* Type */}
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Types *</label>
          <div className="flex flex-wrap gap-1">
            {NODE_TYPES_LIST.map((t) => {
              const colors = NODE_TYPE_COLORS[t];
              return (
                <button
                  type="button"
                  key={t}
                  onClick={() => {
                    const cur = draftNode.nodeTypes;
                    const next = cur.includes(t) ? (cur.length > 1 ? cur.filter((x) => x !== t) : cur) : [...cur, t];
                    updateDraft({ nodeTypes: next });
                  }}
                  className={`rounded-full border px-2.5 py-1 text-xs font-bold uppercase transition-colors ${
                    draftNode.nodeTypes.includes(t)
                      ? `${colors ? colors.icon : 'bg-slate-800'} border-transparent text-white`
                      : `${colors ? `${colors.border} ${colors.label}` : 'border-slate-200 text-slate-600'} bg-white hover:bg-slate-50`
                  }`}
                >
                  {t}
                </button>
              );
            })}
          </div>
        </div>

        {/* Title */}
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Title *</label>
          <input
            type="text"
            value={draftNode.title || ''}
            onChange={(e) => updateDraft({ title: e.target.value })}
            maxLength={MAX_TITLE_LENGTH}
            placeholder="e.g. Sparse Autoencoders"
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
          />
        </div>

        {/* Description */}
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Description</label>
          <textarea
            value={draftNode.description || ''}
            onChange={(e) => updateDraft({ description: e.target.value })}
            placeholder="Optional description..."
            rows={2}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
          />
        </div>

        {/* URL */}
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">URL</label>
          <input
            type="url"
            value={draftNode.mainUrl || ''}
            onChange={(e) => updateDraft({ mainUrl: e.target.value })}
            placeholder="https://..."
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
          />
        </div>

        {/* Additional URLs */}
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Additional URLs</label>
          <div className="flex flex-col gap-1">
            {(draftNode.additionalUrls || []).map((url, i) => (
              <div key={i} className="flex items-center gap-1">
                <span className="flex-1 truncate rounded border border-slate-100 px-2 py-1.5 text-xs text-slate-600">
                  {url}
                </span>
                <button
                  type="button"
                  onClick={() => updateDraft({ additionalUrls: draftNode.additionalUrls.filter((_, j) => j !== i) })}
                  className="shrink-0 text-xs text-slate-400 hover:text-red-500"
                >
                  ✕
                </button>
              </div>
            ))}
            <div className="flex gap-1">
              <input
                type="text"
                value={newAdditionalUrl}
                onChange={(e) => setNewAdditionalUrl(e.target.value)}
                placeholder="https://..."
                className="flex-1 rounded-lg border border-slate-200 px-3 py-1.5 text-xs focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && newAdditionalUrl.trim()) {
                    e.preventDefault();
                    updateDraft({
                      additionalUrls: [...(draftNode.additionalUrls || []), newAdditionalUrl.trim()],
                    });
                    setNewAdditionalUrl('');
                  }
                }}
              />
              <button
                type="button"
                disabled={!newAdditionalUrl.trim()}
                onClick={() => {
                  updateDraft({
                    additionalUrls: [...(draftNode.additionalUrls || []), newAdditionalUrl.trim()],
                  });
                  setNewAdditionalUrl('');
                }}
                className="shrink-0 rounded-lg border border-slate-200 px-2 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50"
              >
                Add
              </button>
            </div>
          </div>
        </div>

        {/* Author */}
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Author</label>
          <input
            type="text"
            value={draftNode.author || ''}
            onChange={(e) => updateDraft({ author: e.target.value })}
            placeholder="Auto-populated from metadata"
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
          />
        </div>

        {/* Parent */}
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Parent node</label>
          <select
            value={draftNode.parentId ? String(draftNode.parentId) : ''}
            onChange={(e) => {
              const pid = e.target.value ? Number(e.target.value) : null;
              const parentNode = pid ? existingNodes.find((n) => n.id === pid) : null;
              updateDraft({
                parentId: pid,
                parent: parentNode
                  ? { id: parentNode.id, title: parentNode.title, nodeTypes: parentNode.nodeTypes }
                  : null,
              });
            }}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
          >
            <option value="">None (root node)</option>
            {existingNodes.map((n) => (
              <option key={n.id} value={n.id}>
                [{n.nodeTypes.join(', ')}] {n.title || '(untitled)'}
              </option>
            ))}
          </select>
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}

        <div className="mt-1 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitting}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
              submitting ? 'cursor-not-allowed bg-slate-300 text-slate-500' : 'bg-sky-700 text-white hover:bg-sky-800'
            }`}
          >
            {submitting ? 'Saving...' : 'Create Node'}
          </button>
        </div>
      </div>
    </div>
  );
}
