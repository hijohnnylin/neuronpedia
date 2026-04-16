'use client';

import { useCallback, useState } from 'react';
import { MAX_TITLE_LENGTH } from './explorer-shared';

const NODE_TYPES = ['topic', 'paper', 'tool', 'dataset', 'eval', 'replication', 'model'] as const;

type ExistingNode = {
  id: number;
  title: string | null;
  nodeTypes: string[];
};

export default function CreateNodeModal({
  existingNodes,
  defaultParentId,
  onClose,
  onCreated,
}: {
  existingNodes: ExistingNode[];
  defaultParentId?: number | null;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [nodeTypes, setNodeTypes] = useState<string[]>(['topic']);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [mainUrl, setMainUrl] = useState('');
  const [parentId, setParentId] = useState(defaultParentId ? String(defaultParentId) : '');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = useCallback(async () => {
    if (!title.trim()) {
      setError('Title is required');
      return;
    }
    setSubmitting(true);
    setError('');
    try {
      const res = await fetch('/api/explorer/node/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nodeTypes,
          title: title.trim(),
          description: description.trim() || null,
          mainUrl: mainUrl.trim() || null,
          parentId: parentId || null,
        }),
      });
      if (!res.ok) {
        const data = await res.json();
        setError(data.error || 'Failed to create node');
        return;
      }
      onCreated();
    } catch {
      setError('Network error');
    } finally {
      setSubmitting(false);
    }
  }, [nodeTypes, title, description, mainUrl, parentId, onCreated]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div className="w-full max-w-md rounded-xl bg-white p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">Add Problem Node</h2>
          <button type="button" onClick={onClose} className="text-slate-400 hover:text-slate-600">
            ✕
          </button>
        </div>

        <div className="mt-4 flex flex-col gap-3">
          {/* Type */}
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-600">Types *</label>
            <div className="flex flex-wrap gap-1">
              {NODE_TYPES.map((t) => (
                <button
                  type="button"
                  key={t}
                  onClick={() =>
                    setNodeTypes((prev) =>
                      prev.includes(t) ? (prev.length > 1 ? prev.filter((x) => x !== t) : prev) : [...prev, t],
                    )
                  }
                  className={`rounded-full border px-2.5 py-1 text-xs font-medium capitalize transition-colors ${
                    nodeTypes.includes(t)
                      ? 'border-slate-800 bg-slate-800 text-white'
                      : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Title */}
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-600">Title *</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              maxLength={MAX_TITLE_LENGTH}
              placeholder="e.g. Sparse Autoencoders"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
            />
          </div>

          {/* Description */}
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-600">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description..."
              rows={2}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
            />
          </div>

          {/* URL */}
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-600">
              URL {!nodeTypes.includes('topic') && <span className="text-slate-400">(recommended)</span>}
            </label>
            <input
              type="url"
              value={mainUrl}
              onChange={(e) => setMainUrl(e.target.value)}
              placeholder="https://..."
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
            />
          </div>

          {/* Parent */}
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-600">Parent node</label>
            <select
              value={parentId}
              onChange={(e) => setParentId(e.target.value)}
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
              onClick={onClose}
              className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={submitting}
              className="rounded-lg bg-sky-700 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-800 disabled:opacity-50"
            >
              {submitting ? 'Creating...' : 'Create Node'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
