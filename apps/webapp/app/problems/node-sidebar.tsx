'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { useCallback, useEffect, useState } from 'react';
import { CommentItem } from './comment-item';
import { type DetailNode, type ProblemNodeData, TYPE_COLORS } from './problems-shared';

export function NodeSidebar({
  selectedNode,
  detailNode,
  canEdit,
  session,
  onClose,
  onDelete,
  onUpdated,
  allNodes,
  onSelectNode,
  editOnSelect,
  onEditOnSelectConsumed,
}: {
  selectedNode: ProblemNodeData;
  detailNode: DetailNode | null;
  canEdit: boolean;
  session: any;
  allNodes: ProblemNodeData[];
  onClose: () => void;
  onDelete: () => void;
  onUpdated: () => void;
  onSelectNode: (id: number) => void;
  editOnSelect?: boolean;
  onEditOnSelectConsumed?: () => void;
}) {
  const { setSignInModalOpen, showToastMessage } = useGlobalContext();
  const [editing, setEditing] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editMainUrl, setEditMainUrl] = useState('');
  const [editAuthor, setEditAuthor] = useState('');
  const [editNodeTypes, setEditNodeTypes] = useState<string[]>([]);
  const [editParentId, setEditParentId] = useState<number | null>(null);
  const [editAdditionalUrls, setEditAdditionalUrls] = useState<string[]>([]);
  const [newAdditionalUrl, setNewAdditionalUrl] = useState('');
  const [saving, setSaving] = useState(false);
  const [commentText, setCommentText] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const node = detailNode || selectedNode;

  const startEditing = useCallback(() => {
    setEditTitle(node.title || '');
    setEditDescription(node.description || '');
    setEditMainUrl(node.mainUrl || '');
    setEditAuthor(node.author || '');
    setEditNodeTypes([...node.nodeTypes]);
    setEditParentId(node.parentId ?? null);
    setEditAdditionalUrls([...(node.additionalUrls || [])]);
    setNewAdditionalUrl('');
    setEditing(true);
  }, [node]);

  useEffect(() => {
    if (editOnSelect) {
      startEditing();
      onEditOnSelectConsumed?.();
    } else {
      setEditing(false);
    }
  }, [selectedNode.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const saveEdit = useCallback(async () => {
    setSaving(true);
    try {
      const res = await fetch('/api/problem/node/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: selectedNode.id,
          title: editTitle,
          description: editDescription,
          mainUrl: editMainUrl || null,
          author: editAuthor || null,
          nodeTypes: editNodeTypes,
          additionalUrls: editAdditionalUrls,
          parentId: editParentId,
        }),
      });
      if (res.ok) {
        setEditing(false);
        onUpdated();
      }
    } finally {
      setSaving(false);
    }
  }, [selectedNode.id, editTitle, editDescription, editMainUrl, editAuthor, editNodeTypes, editAdditionalUrls, editParentId, onUpdated]);

  const handleAddComment = useCallback(async () => {
    if (!session?.user) {
      setSignInModalOpen(true);
      return;
    }
    if (!commentText.trim()) return;
    setSubmitting(true);
    try {
      await fetch('/api/problem/comment/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ problemNodeId: selectedNode.id, text: commentText }),
      });
      setCommentText('');
      onUpdated();
    } finally {
      setSubmitting(false);
    }
  }, [session, setSignInModalOpen, commentText, selectedNode.id, onUpdated]);

  const typeOptions = ['topic', 'paper', 'tool', 'dataset', 'eval', 'replication', 'model'];

  return (
    <div className="w-[420px] shrink-0 overflow-y-auto border-l border-slate-200 bg-white p-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          {node.nodeTypes.map((t) => (
            <span
              key={t}
              className={`rounded px-2 py-0.5 text-xs font-semibold uppercase ${TYPE_COLORS[t] || 'bg-slate-100 text-slate-800'}`}
            >
              {t}
            </span>
          ))}
        </div>
        <div className="flex items-center gap-2">
          {canEdit && !editing && (
            <button type="button" onClick={startEditing} className="text-xs text-sky-600 hover:text-sky-700">
              Edit
            </button>
          )}
          <button type="button" onClick={onClose} className="text-slate-400 hover:text-slate-600">
            ✕
          </button>
        </div>
      </div>

      {editing ? (
        /* ── Edit mode ── */
        <div className="mt-3 flex flex-col gap-3">
          <div>
            {/* eslint-disable-next-line jsx-a11y/label-has-associated-control */}
            <label className="text-xs font-medium text-slate-600">Types</label>
            <div className="mt-1 flex flex-wrap gap-1">
              {typeOptions.map((t) => (
                <button
                  type="button"
                  key={t}
                  onClick={() =>
                    setEditNodeTypes((prev) =>
                      prev.includes(t) ? (prev.length > 1 ? prev.filter((x) => x !== t) : prev) : [...prev, t],
                    )
                  }
                  className={`rounded-full border px-2.5 py-1 text-[11px] font-medium capitalize transition-colors ${
                    editNodeTypes.includes(t)
                      ? 'border-slate-800 bg-slate-800 text-white'
                      : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
          <div>
            {/* eslint-disable-next-line jsx-a11y/label-has-associated-control */}
            <label className="text-xs font-medium text-slate-600">Title</label>
            <input
              type="text"
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
            />
          </div>
          <div>
            {/* eslint-disable-next-line jsx-a11y/label-has-associated-control */}
            <label className="text-xs font-medium text-slate-600">Description</label>
            <textarea
              value={editDescription}
              onChange={(e) => setEditDescription(e.target.value)}
              rows={4}
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
            />
          </div>
          <div>
            {/* eslint-disable-next-line jsx-a11y/label-has-associated-control */}
            <label className="text-xs font-medium text-slate-600">Main URL</label>
            <input
              type="text"
              value={editMainUrl}
              onChange={(e) => setEditMainUrl(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
            />
          </div>
          <div>
            {/* eslint-disable-next-line jsx-a11y/label-has-associated-control */}
            <label className="text-xs font-medium text-slate-600">Additional URLs</label>
            <div className="mt-1 flex flex-col gap-1">
              {editAdditionalUrls.map((url, i) => (
                <div key={i} className="flex items-center gap-1">
                  <span className="flex-1 truncate rounded border border-slate-100 px-2 py-1.5 text-xs text-slate-600">
                    {url}
                  </span>
                  <button
                    type="button"
                    onClick={() => setEditAdditionalUrls((prev) => prev.filter((_, j) => j !== i))}
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
                      setEditAdditionalUrls((prev) => [...prev, newAdditionalUrl.trim()]);
                      setNewAdditionalUrl('');
                    }
                  }}
                />
                <button
                  type="button"
                  disabled={!newAdditionalUrl.trim()}
                  onClick={() => {
                    setEditAdditionalUrls((prev) => [...prev, newAdditionalUrl.trim()]);
                    setNewAdditionalUrl('');
                  }}
                  className="shrink-0 rounded-lg border border-slate-200 px-2 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                >
                  Add
                </button>
              </div>
            </div>
          </div>
          <div>
            {/* eslint-disable-next-line jsx-a11y/label-has-associated-control */}
            <label className="text-xs font-medium text-slate-600">Author</label>
            <input
              type="text"
              value={editAuthor}
              onChange={(e) => setEditAuthor(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
            />
          </div>
          <div>
            {/* eslint-disable-next-line jsx-a11y/label-has-associated-control */}
            <label className="text-xs font-medium text-slate-600">Parent node</label>
            <select
              value={editParentId ? String(editParentId) : ''}
              onChange={(e) => setEditParentId(e.target.value ? Number(e.target.value) : null)}
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
            >
              <option value="">None (root node)</option>
              {allNodes
                .filter((n) => n.id !== selectedNode.id)
                .map((n) => (
                  <option key={n.id} value={n.id}>
                    [{n.nodeTypes.join(', ')}] {n.title || '(untitled)'}
                  </option>
                ))}
            </select>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setEditing(false)}
              className="flex-1 rounded-lg border border-slate-200 px-4 py-2 text-xs font-medium text-slate-600 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={saveEdit}
              disabled={saving}
              className="flex-1 rounded-lg bg-sky-700 px-4 py-2 text-xs font-medium text-white hover:bg-sky-800 disabled:opacity-50"
            >
              {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
          {(session?.user?.id === selectedNode.createdBy?.id || canEdit) && (
            <button
              type="button"
              onClick={onDelete}
              className="self-start rounded-lg border border-red-200 px-3 py-2 text-xs font-medium text-red-600 transition-colors hover:bg-red-50"
            >
              Delete
            </button>
          )}
        </div>
      ) : (
        /* ── View mode ── */
        <>
          <h2 className="mt-2 text-lg font-semibold text-slate-900">
            {node.title || '(untitled)'}
            <span className="ml-1.5 text-xs font-normal text-slate-300">&middot;</span>
            <a
              href={`/problems/${selectedNode.id}`}
              onClick={(e) => {
                e.preventDefault();
                const url = `${window.location.origin}/problems/${selectedNode.id}`;
                navigator.clipboard.writeText(url);
                showToastMessage('Link copied to clipboard.');
              }}
              onContextMenu={() => {
                const url = `${window.location.origin}/problems/${selectedNode.id}`;
                navigator.clipboard.writeText(url);
              }}
              className="ml-1 text-xs font-normal text-slate-400 hover:text-slate-600"
            >
              #{selectedNode.id}
            </a>
          </h2>
          {node.author && <p className="mt-0.5 text-sm text-slate-500">{node.author}</p>}
          {node.description && <p className="mt-1 text-xs leading-normal text-slate-600">{node.description}</p>}

          {/* Links */}
          {(node.mainUrl || (node.additionalUrls && node.additionalUrls.length > 0)) && (
            <div className="mt-3 flex flex-col gap-1">
              {node.mainUrl && (
                <a
                  href={node.mainUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="break-all text-xs text-sky-600 hover:underline"
                >
                  {node.mainUrl}
                </a>
              )}
              {node.additionalUrls?.map((url, i) => (
                <a
                  key={i}
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="break-all text-xs text-sky-600 hover:underline"
                >
                  {url}
                </a>
              ))}
            </div>
          )}

          {/* Meta */}
          <div className="mt-3 hidden border-t border-slate-100 pt-3">
            <div className="flex flex-wrap gap-3 text-xs text-slate-400">
              {/* <span>
                Created by <span className="font-medium text-slate-600">{node.createdBy?.name}</span>
              </span> */}
              {node.approvalState === 'PENDING' && (
                <span>
                  Status: <span className="font-medium text-amber-600">{node.approvalState}</span>
                </span>
              )}
              {/* {node.approver && <span>Approved by {node.approver.name}</span>} */}
              {/* {detailNode?.updatedAt && <span>Updated {new Date(detailNode.updatedAt).toLocaleDateString()}</span>} */}
            </div>
          </div>

          {/* Tags */}
          {node.applicationTags && node.applicationTags.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1">
              {node.applicationTags.map((tag) => (
                <span key={tag} className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
                  {tag}
                </span>
              ))}
            </div>
          )}

          {/* Parent */}
          {node.parent && (
            <div className="mt-3">
              <h3 className="text-xs font-semibold text-slate-500">Parent</h3>
              <button
                type="button"
                onClick={() => onSelectNode(node.parent!.id)}
                className="mt-1 flex w-full items-center gap-2 rounded-lg border border-slate-100 px-2 py-2 text-left text-sm hover:bg-slate-50"
              >
                {node.parent.nodeTypes.map((t) => (
                  <span
                    key={t}
                    className={`rounded px-1.5 py-0 text-[9px] font-semibold uppercase ${TYPE_COLORS[t] || ''}`}
                  >
                    {t}
                  </span>
                ))}
                <span className="text-xs text-slate-800">{node.parent.title || '(untitled)'}</span>
              </button>
            </div>
          )}

          {/* Children */}
          {node.children && node.children.length > 0 && (
            <div className="mt-3">
              <h3 className="text-xs font-semibold text-slate-500">Sub-items</h3>
              <div className="mt-1 flex flex-col gap-1">
                {node.children.map((child) => (
                  <button
                    type="button"
                    key={child.id}
                    onClick={() => onSelectNode(child.id)}
                    className="flex w-full items-center gap-2 rounded-lg border border-slate-100 px-2 py-2 text-left text-sm hover:bg-slate-50"
                  >
                    {child.nodeTypes.map((t) => (
                      <span
                        key={t}
                        className={`rounded px-1.5 py-0 text-[9px] font-semibold uppercase ${TYPE_COLORS[t] || ''}`}
                      >
                        {t}
                      </span>
                    ))}
                    <span className="text-xs text-slate-800">{child.title || '(untitled)'}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Related Edges */}
          {detailNode && (detailNode.edgesAsSource.length > 0 || detailNode.edgesAsTarget.length > 0) && (
            <div className="mt-3 border-t border-slate-100 pt-3">
              <h3 className="text-xs font-semibold text-slate-500">Related Nodes</h3>
              <div className="mt-1 flex flex-col gap-1">
                {detailNode.edgesAsSource.map((edge) => (
                  <button
                    type="button"
                    key={edge.id}
                    onClick={() => onSelectNode(edge.targetNode.id)}
                    className="flex w-full items-center gap-2 rounded-lg border border-slate-100 px-3 py-2 text-left text-sm hover:bg-slate-50"
                  >
                    <span className="text-xs text-slate-400">{edge.type}</span>
                    <span className="text-slate-800">{edge.targetNode.title || '(untitled)'}</span>
                  </button>
                ))}
                {detailNode.edgesAsTarget.map((edge) => (
                  <button
                    type="button"
                    key={edge.id}
                    onClick={() => onSelectNode(edge.sourceNode.id)}
                    className="flex w-full items-center gap-2 rounded-lg border border-slate-100 px-3 py-2 text-left text-sm hover:bg-slate-50"
                  >
                    <span className="text-xs text-slate-400">{edge.type}</span>
                    <span className="text-slate-800">{edge.sourceNode.title || '(untitled)'}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Comments */}
          {detailNode && (
            <div className="mt-4 hidden border-t border-slate-200 pt-4">
              <h3 className="text-xs font-semibold text-slate-500">Comments ({detailNode.comments.length})</h3>
              <div className="mt-2 flex gap-2">
                <input
                  type="text"
                  value={commentText}
                  onChange={(e) => setCommentText(e.target.value)}
                  placeholder="Add a comment..."
                  className="flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-sky-300 focus:outline-none focus:ring-1 focus:ring-sky-300"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleAddComment();
                    }
                  }}
                />
                <button
                  type="button"
                  onClick={handleAddComment}
                  disabled={submitting || !commentText.trim()}
                  className="rounded-lg bg-sky-700 px-3 py-2 text-xs font-medium text-white hover:bg-sky-800 disabled:opacity-50"
                >
                  {submitting ? '...' : 'Post'}
                </button>
              </div>
              <div className="mt-3 flex flex-col gap-2">
                {detailNode.comments.map((comment) => (
                  <CommentItem key={comment.id} comment={comment} />
                ))}
              </div>
            </div>
          )}

        </>
      )}
    </div>
  );
}
