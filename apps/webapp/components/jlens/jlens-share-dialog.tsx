'use client';

// Shared "share this run" dialog for both jlens interfaces. Collects an
// optional description, POSTs the run to be re-computed + cached server-side,
// and returns a copyable link. The caller supplies the request body (which
// differs by kind: chat carries `messages`, completion carries `prompt`).

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { MAX_JLENS_SHARE_DESCRIPTION_LENGTH } from '@/lib/utils/jlens-share';
import { Check, Copy, Link2, Loader2, Share2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';

export function JlensShareDialog({
  open,
  onOpenChange,
  buildBody,
  userName = null,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  // Assembles the POST body for `/api/lens/share` given the typed description.
  buildBody: (description: string) => Record<string, unknown>;
  userName?: string | null;
}) {
  const [description, setDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Reset transient state whenever the dialog is (re)opened — including when the
  // parent opens it programmatically (in which case `handleOpenChange` is NOT
  // called by Radix). Without this, a second share would show the previous run's
  // link + description instead of a fresh form.
  useEffect(() => {
    if (open) {
      setError(null);
      setUrl(null);
      setCopied(false);
      setDescription('');
    }
  }, [open]);

  function handleOpenChange(next: boolean) {
    if (submitting) {
      return;
    }
    onOpenChange(next);
  }

  async function handleSubmit() {
    if (submitting) {
      return;
    }
    setSubmitting(true);
    setError(null);
    setUrl(null);
    try {
      const res = await fetch('/api/lens/share', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildBody(description.trim())),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error ?? `Share failed (${res.status})`);
      }
      setUrl(`${window.location.origin}${data.path}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCopy() {
    if (!url) {
      return;
    }
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard can fail (permissions); the URL is still selectable.
    }
  }

  const title = 'Share';

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md bg-white">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription />
        </DialogHeader>

        {url ? (
          <div className="flex flex-col gap-y-2">
            <div className="text-[13px] font-medium text-slate-700">Your link is ready:</div>
            <div className="flex flex-row items-center gap-x-2">
              <div className="flex min-w-0 flex-1 flex-row items-center gap-x-1.5 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-2">
                <Link2 className="h-3.5 w-3.5 shrink-0 text-slate-400" />
                <input
                  type="text"
                  readOnly
                  value={url}
                  onFocus={(e) => e.currentTarget.select()}
                  className="min-w-0 flex-1 truncate border-0 bg-transparent p-0 text-xs text-slate-600 outline-none focus:ring-0"
                />
              </div>
              <button
                type="button"
                onClick={handleCopy}
                className="flex shrink-0 flex-row items-center gap-x-1.5 rounded-md bg-sky-700 px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-sky-800"
              >
                {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-y-3">
            <div className="flex flex-col gap-y-1">
              <label htmlFor="jlens-share-description" className="text-[13px] font-medium text-slate-700">
                Add Description <span className="font-normal text-slate-400">(optional)</span>
              </label>
              <ReactTextareaAutosize
                id="jlens-share-description"
                value={description}
                onChange={(e) => setDescription(e.target.value.slice(0, MAX_JLENS_SHARE_DESCRIPTION_LENGTH))}
                minRows={3}
                maxRows={6}
                disabled={submitting}
                placeholder="Add a note about what this run shows…"
                className="w-full resize-none rounded-md border border-slate-200 px-2.5 py-2 text-[13px] text-slate-800 outline-none focus:border-sky-400 focus:ring-1 focus:ring-sky-400"
              />
              <div className="self-end text-[10px] tabular-nums text-slate-400">
                {description.length}/{MAX_JLENS_SHARE_DESCRIPTION_LENGTH}
              </div>
              {userName && (
                <div className="text-[11px] text-slate-500">
                  Your description will be attributed to <span className="font-medium text-slate-600">@{userName}</span>
                  .
                </div>
              )}
            </div>

            {error && (
              <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-[11px] text-red-700">
                {error}
              </div>
            )}

            <button
              type="button"
              onClick={() => void handleSubmit()}
              disabled={submitting}
              className="flex flex-row items-center justify-center gap-x-2 rounded-md bg-sky-700 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Creating link…
                </>
              ) : (
                <>
                  <Share2 className="h-4 w-4" />
                  Create share link
                </>
              )}
            </button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
