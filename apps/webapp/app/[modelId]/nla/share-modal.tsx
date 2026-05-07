'use client';

import type { NlaShareDraft } from '@/components/provider/nla-provider';
import { Button } from '@/components/shadcn/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { MAX_COMMENT_LENGTH } from '@/lib/nla-constants';
import copy from 'copy-to-clipboard';
import { LinkIcon, Loader2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

function describeShareTarget(draft: NlaShareDraft | null): string {
  if (!draft) return "(Optional) Describe what is significant about this token's explanation.";
  if (draft.highlightStart !== null && draft.highlightEnd !== null) {
    return '(Optional) Describe what is significant about this highlighted snippet.';
  }
  if (draft.paragraph !== null) {
    return '(Optional) Describe what is significant about this paragraph.';
  }
  return "(Optional) Describe what is significant about this token's explanation.";
}

function draftStableKey(draft: NlaShareDraft | null): string {
  if (!draft) return '';
  return [
    draft.cacheId,
    draft.position ?? '',
    draft.paragraph ?? '',
    draft.highlightStart ?? '',
    draft.highlightEnd ?? '',
    draft.initialComment ?? '',
    draft.existingShareId ?? '',
  ].join(':');
}

export default function ShareModal({
  open,
  onOpenChange,
  shareDraft,
  shareError,
  onCommentCommit,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  shareDraft: NlaShareDraft | null;
  shareError: string | null;
  onCommentCommit?: (comment: string | null) => void;
}) {
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const [comment, setComment] = useState('');
  const [generatedShortUrl, setGeneratedShortUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const draftKey = useMemo(() => draftStableKey(shareDraft), [shareDraft]);

  useEffect(() => {
    if (!open) {
      setCopyFeedback(null);
      setGeneratedShortUrl(null);
      setGenerateError(null);
      setIsGenerating(false);
      return;
    }
    setCopyFeedback(null);
    setGenerateError(null);
    setIsGenerating(false);
    if (!shareDraft) {
      setComment('');
      setGeneratedShortUrl(null);
      return;
    }
    const seed = shareDraft.initialComment ?? '';
    setComment(seed.length > MAX_COMMENT_LENGTH ? seed.slice(0, MAX_COMMENT_LENGTH) : seed);
    if (shareDraft.existingShareId) {
      setGeneratedShortUrl(`${window.location.origin}/nla/${shareDraft.existingShareId}`);
    } else {
      setGeneratedShortUrl(null);
    }
  }, [open, draftKey, shareDraft]);

  const commentDescription = useMemo(() => describeShareTarget(shareDraft), [shareDraft]);

  const handleGenerate = useCallback(async () => {
    if (!shareDraft || shareDraft.existingShareId) return;
    const trimmed = comment.trim();
    const capped = trimmed.length > MAX_COMMENT_LENGTH ? trimmed.slice(0, MAX_COMMENT_LENGTH) : trimmed;

    setIsGenerating(true);
    setGenerateError(null);
    setCopyFeedback(null);

    try {
      const res = await fetch('/api/nla/explain-share', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cacheId: shareDraft.cacheId,
          position: shareDraft.position,
          paragraph: shareDraft.paragraph,
          highlightStart: shareDraft.highlightStart,
          highlightEnd: shareDraft.highlightEnd,
          comment: capped.length > 0 ? capped : null,
        }),
      });

      const data = (await res.json()) as { error?: string; id?: string };

      if (!res.ok) {
        setGenerateError(data.error || `Could not create share link (${res.status})`);
        return;
      }
      if (typeof data.id !== 'string') {
        setGenerateError('Invalid response from server.');
        return;
      }

      if (onCommentCommit) {
        onCommentCommit(capped.length > 0 ? capped : null);
      }

      setGeneratedShortUrl(`${window.location.origin}/nla/${data.id}`);
    } catch {
      setGenerateError('Network error while creating share link.');
    } finally {
      setIsGenerating(false);
    }
  }, [shareDraft, comment, onCommentCommit]);

  const handleCommentChange = (value: string) => {
    if (value.length > MAX_COMMENT_LENGTH) {
      setComment(value.slice(0, MAX_COMMENT_LENGTH));
      return;
    }
    setComment(value);
  };

  const handleCopyOption = (type: 'embed' | 'iframe' | 'normal') => {
    if (!generatedShortUrl) return;

    const url = new URL(generatedShortUrl);

    switch (type) {
      case 'embed': {
        url.searchParams.set('embed', 'true');
        copy(url.toString());
        setCopyFeedback('Embed link copied to clipboard.');
        break;
      }
      case 'iframe': {
        url.searchParams.set('embed', 'true');
        const iframeCode = `<iframe src="${url.toString()}" width="100%" height="600" frameborder="0"></iframe>`;
        copy(iframeCode);
        setCopyFeedback('Iframe code copied to clipboard.');
        break;
      }
      case 'normal': {
        url.searchParams.delete('embed');
        copy(url.toString());
        setCopyFeedback('Link copied to clipboard.');
        break;
      }
      default:
        break;
    }
  };

  const canInteract = Boolean(shareDraft) && !shareError;
  const showResult = Boolean(generatedShortUrl);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-white sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-base text-slate-600">Share NLA Result</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-3 py-2 pt-0">
          {shareError && (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {shareError}
            </div>
          )}
          {generateError && (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {generateError}
            </div>
          )}

          {!shareDraft?.existingShareId && (
            <>
              <div className="flex flex-col gap-1">
                <label htmlFor="nla-share-comment" className="text-[11px] font-medium text-slate-500">
                  {commentDescription}
                </label>
                <textarea
                  id="nla-share-comment"
                  value={comment}
                  onChange={(e) => handleCommentChange(e.target.value)}
                  disabled={!canInteract || isGenerating || showResult}
                  rows={3}
                  maxLength={MAX_COMMENT_LENGTH}
                  placeholder="Add an optional comment to share alongside this link."
                  className="resize-y rounded-md border border-slate-200 bg-white px-3 py-2 text-[12px] leading-relaxed text-slate-700 placeholder:text-slate-300 focus:border-sky-400 focus:outline-none focus:ring-1 focus:ring-sky-400 disabled:cursor-not-allowed disabled:bg-slate-50 disabled:opacity-60"
                />
                <div className="flex items-center justify-end text-[10px] text-slate-400">
                  {comment.length} / {MAX_COMMENT_LENGTH}
                </div>
              </div>

              {!showResult && (
                <Button
                  type="button"
                  className="flex w-full items-center justify-center"
                  onClick={() => void handleGenerate()}
                  disabled={!canInteract || isGenerating}
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating…
                    </>
                  ) : (
                    'Generate Share Link'
                  )}
                </Button>
              )}
            </>
          )}

          {showResult && (
            <>
              <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-600">
                <div className="break-all font-mono text-[9px] leading-normal text-slate-700">{generatedShortUrl}</div>
              </div>

              {copyFeedback && <div className="text-sm text-green-600">{copyFeedback}</div>}

              <Button
                variant="outline"
                className="h-auto justify-start gap-3 py-2"
                onClick={() => handleCopyOption('normal')}
              >
                <LinkIcon className="h-4 w-4" />
                <div className="text-left">
                  <div className="text-[13px] font-medium">Copy Link</div>
                  <div className="text-[10px] leading-snug text-slate-400">Share as a normal URL</div>
                </div>
              </Button>

              {/* <Button
                variant="outline"
                className="h-auto justify-start gap-3 py-2"
                onClick={() => handleCopyOption('embed')}
              >
                <CopyIcon className="h-4 w-4" />
                <div className="text-left">
                  <div className="text-[13px] font-medium">Copy Embed Link</div>
                  <div className="text-[10px] leading-snug text-slate-400">URL optimized for iFrame embed</div>
                </div>
              </Button>

              <Button
                variant="outline"
                className="h-auto justify-start gap-3 py-2"
                onClick={() => handleCopyOption('iframe')}
              >
                <CodeIcon className="h-4 w-4" />
                <div className="text-left">
                  <div className="text-[13px] font-medium">Copy iFrame Code</div>
                  <div className="text-[10px] leading-snug text-slate-400">
                    HTML code snippet to embed this explanation
                  </div>
                </div>
              </Button> */}
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
