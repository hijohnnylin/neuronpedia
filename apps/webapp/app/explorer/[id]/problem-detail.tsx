'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { useSession } from 'next-auth/react';
import Link from 'next/link';
import { useCallback, useState } from 'react';

const TYPE_COLORS: Record<string, string> = {
  topic: 'bg-blue-100 text-blue-800',
  paper: 'bg-emerald-100 text-emerald-800',
  tool: 'bg-indigo-100 text-indigo-800',
  dataset: 'bg-amber-100 text-amber-800',
  eval: 'bg-rose-100 text-rose-800',
  replication: 'bg-fuchsia-100 text-fuchsia-800',
  model: 'bg-stone-100 text-stone-800',
};

type Comment = {
  id: string;
  text: string;
  createdAt: string | Date;
  user: { id: string; name: string; image: string | null };
  replies?: Comment[];
};

type ProblemNodeDetail = {
  id: number;
  nodeTypes: string[];
  title: string | null;
  description: string | null;
  mainUrl: string | null;
  additionalUrls: string[];
  applicationTags: string[];
  approvalState: string;
  parentId: number | null;
  parent: { id: number; title: string | null; nodeTypes: string[] } | null;
  children: { id: number; title: string | null; nodeTypes: string[]; approvalState: string }[];
  createdBy: { id: string; name: string; image: string | null };
  approver: { id: string; name: string; image: string | null } | null;
  comments: Comment[];
  edgesAsSource: { id: string; type: string; targetNode: { id: number; title: string | null; nodeTypes: string[] } }[];
  edgesAsTarget: { id: string; type: string; sourceNode: { id: number; title: string | null; nodeTypes: string[] } }[];
  createdAt: string | Date;
  updatedAt: string | Date;
};

function CommentItem({ comment }: { comment: Comment }) {
  return (
    <div className="rounded-lg border border-slate-100 px-3 py-2">
      <div className="flex items-center gap-2 text-xs text-slate-500">
        <span className="font-medium text-slate-700">{comment.user.name}</span>
        <span>{new Date(comment.createdAt).toLocaleDateString()}</span>
      </div>
      <p className="mt-1 text-sm text-slate-700">{comment.text}</p>
      {comment.replies && comment.replies.length > 0 && (
        <div className="ml-4 mt-2 flex flex-col gap-2 border-l-2 border-slate-100 pl-3">
          {comment.replies.map((reply) => (
            <CommentItem key={reply.id} comment={reply} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function ProblemDetail({ initialNode }: { initialNode: ProblemNodeDetail }) {
  const { data: session } = useSession();
  const { setSignInModalOpen } = useGlobalContext();
  const [node, setNode] = useState(initialNode);
  const [commentText, setCommentText] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleAddComment = useCallback(
    async (parentCommentId?: string) => {
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
          body: JSON.stringify({ problemNodeId: node.id, text: commentText, parentCommentId }),
        });
        // Refresh node data
        const res = await fetch('/api/problem/node/get', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: node.id }),
        });
        const updated = await res.json();
        setNode(updated);
        setCommentText('');
      } catch (err) {
        console.error('Failed to add comment', err);
      } finally {
        setSubmitting(false);
      }
    },
    [session, setSignInModalOpen, commentText, node.id],
  );

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-8">
      {/* Breadcrumb */}
      <div className="mb-4 flex items-center gap-2 text-sm text-slate-500">
        <Link href="/" className="hover:text-sky-600 hover:underline">
          Home
        </Link>
        <span>/</span>
        <Link href="/explorer" className="hover:text-sky-600 hover:underline">
          Explorer
        </Link>
        {node.parent && (
          <>
            <span>/</span>
            <Link href={`/explorer/${node.parent.id}`} className="hover:text-sky-600 hover:underline">
              {node.parent.title || '(untitled)'}
            </Link>
          </>
        )}
        <span>/</span>
        <span className="text-slate-800">{node.title || '(untitled)'}</span>
      </div>

      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="mt-1 flex gap-1">
          {node.nodeTypes.map((t) => (
            <span
              key={t}
              className={`inline-block rounded px-2 py-0.5 text-xs font-semibold uppercase ${TYPE_COLORS[t] || 'bg-slate-100 text-slate-800'}`}
            >
              {t}
            </span>
          ))}
        </div>
        <div>
          <h1 className="text-2xl font-bold text-slate-900">{node.title || '(untitled)'}</h1>
          {node.description && <p className="mt-2 text-sm leading-relaxed text-slate-600">{node.description}</p>}
        </div>
      </div>

      {/* Links */}
      {(node.mainUrl || node.additionalUrls.length > 0) && (
        <div className="mt-4 flex flex-col gap-1">
          {node.mainUrl && (
            <a
              href={node.mainUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-sky-600 hover:underline"
            >
              {node.mainUrl}
            </a>
          )}
          {node.additionalUrls.map((url, i) => (
            <a
              key={i}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-sky-600 hover:underline"
            >
              {url}
            </a>
          ))}
        </div>
      )}

      {/* Meta */}
      <div className="mt-4 flex flex-wrap gap-4 text-xs text-slate-500">
        <span>
          Created by <span className="font-medium text-slate-700">{node.createdBy.name}</span>
        </span>
        <span>
          Status:{' '}
          <span
            className={`font-medium ${
              node.approvalState === 'APPROVED'
                ? 'text-emerald-600'
                : node.approvalState === 'REJECTED'
                  ? 'text-red-600'
                  : 'text-amber-600'
            }`}
          >
            {node.approvalState}
          </span>
        </span>
        {node.approver && <span>Approved by {node.approver.name}</span>}
        <span>Updated {new Date(node.updatedAt).toLocaleDateString()}</span>
      </div>

      {/* Tags */}
      {node.applicationTags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {node.applicationTags.map((tag) => (
            <span key={tag} className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Children */}
      {node.children.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-semibold text-slate-700">Sub-items ({node.children.length})</h3>
          <div className="mt-2 flex flex-col gap-1">
            {node.children.map((child) => (
              <Link
                key={child.id}
                href={`/explorer/${child.id}`}
                className="flex items-center gap-2 rounded-lg border border-slate-100 px-3 py-2 text-sm hover:bg-slate-50"
              >
                {child.nodeTypes.map((t) => (
                  <span
                    key={t}
                    className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${TYPE_COLORS[t] || ''}`}
                  >
                    {t}
                  </span>
                ))}
                <span className="text-slate-800">{child.title || '(untitled)'}</span>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Related Edges */}
      {(node.edgesAsSource.length > 0 || node.edgesAsTarget.length > 0) && (
        <div className="mt-6">
          <h3 className="text-sm font-semibold text-slate-700">Related Nodes</h3>
          <div className="mt-2 flex flex-col gap-1">
            {node.edgesAsSource.map((edge) => (
              <Link
                key={edge.id}
                href={`/explorer/${edge.targetNode.id}`}
                className="flex items-center gap-2 rounded-lg border border-slate-100 px-3 py-2 text-sm hover:bg-slate-50"
              >
                <span className="text-xs text-slate-400">{edge.type}</span>
                <span className="text-slate-800">{edge.targetNode.title || '(untitled)'}</span>
              </Link>
            ))}
            {node.edgesAsTarget.map((edge) => (
              <Link
                key={edge.id}
                href={`/explorer/${edge.sourceNode.id}`}
                className="flex items-center gap-2 rounded-lg border border-slate-100 px-3 py-2 text-sm hover:bg-slate-50"
              >
                <span className="text-xs text-slate-400">{edge.type}</span>
                <span className="text-slate-800">{edge.sourceNode.title || '(untitled)'}</span>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Comments */}
      <div className="mt-8 border-t border-slate-200 pt-6">
        <h3 className="text-sm font-semibold text-slate-700">Comments ({node.comments.length})</h3>

        <div className="mt-4 flex gap-2">
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
            onClick={() => handleAddComment()}
            disabled={submitting || !commentText.trim()}
            className="rounded-lg bg-sky-700 px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-sky-800 disabled:opacity-50"
          >
            {submitting ? '...' : 'Post'}
          </button>
        </div>

        <div className="mt-4 flex flex-col gap-3">
          {node.comments.map((comment) => (
            <CommentItem key={comment.id} comment={comment} />
          ))}
        </div>
      </div>
    </div>
  );
}
