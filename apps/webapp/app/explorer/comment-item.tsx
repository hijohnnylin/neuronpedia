'use client';

import { type Comment } from './explorer-shared';

export function CommentItem({ comment }: { comment: Comment }) {
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
