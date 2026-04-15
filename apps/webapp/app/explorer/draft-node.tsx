'use client';

import { detectTypeFromUrl } from '@/lib/problem-url-types';
import { Handle, Position } from '@xyflow/react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { DEFAULT_NODE_WIDTH, TYPE_COLORS } from './explorer-node';
import { MAX_TITLE_LENGTH } from './explorer-shared';

export const DRAFT_NODE_HEIGHT_URL = 50;
export const DRAFT_NODE_HEIGHT_POPULATED = 58;

function DraftNodeComponent({ data }: { data: any }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [urlInput, setUrlInput] = useState('');
  const [fetching, setFetching] = useState(false);

  const types: string[] = data.draftTypes || ['topic'];
  const tc = TYPE_COLORS[types[0]] || TYPE_COLORS.topic;
  const populated = !!data.draftTitle;

  // Keep a ref so callbacks always see the latest data
  const dataRef = useRef(data);
  dataRef.current = data;

  const prevPopulated = useRef(false);
  useEffect(() => {
    if (!populated) {
      setTimeout(() => inputRef.current?.focus(), 100);
    } else if (!prevPopulated.current) {
      // Clear the input when transitioning to populated (first URL loaded)
      setUrlInput('');
    }
    prevPopulated.current = populated;
  }, [populated]);

  const handleTextSubmit = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    const d = dataRef.current;
    d.onUpdateDraft?.({ title: trimmed, nodeTypes: ['topic'] });
    setUrlInput('');
  }, []);

  const handlePaste = useCallback(
    async (url: string) => {
      const trimmed = url.trim();
      if (!trimmed) return;

      if (!/^https?:\/\/.+/i.test(trimmed)) {
        // Not a URL — treat as topic title
        handleTextSubmit(trimmed);
        return;
      }

      const d = dataRef.current;
      const detectedType = detectTypeFromUrl(trimmed);
      const isAdditional = !!d.draftTitle; // check latest data, not stale closure

      if (isAdditional) {
        // Additional URL: add to additionalUrls, merge detected type if new
        const currentAdditional: string[] = d.currentAdditionalUrls || [];
        const mainUrl: string = d.draftUrl || '';
        if (trimmed === mainUrl || currentAdditional.includes(trimmed)) {
          setUrlInput('');
          return;
        }
        const currentTypes: string[] = d.currentNodeTypes || ['topic'];
        const newTypes = currentTypes.includes(detectedType) ? currentTypes : [...currentTypes, detectedType];
        d.onUpdateDraft?.({
          additionalUrls: [...currentAdditional, trimmed],
          nodeTypes: newTypes,
        });
        setUrlInput('');
      } else {
        // First URL: set as mainUrl, detect type, fetch metadata
        d.onUpdateDraft?.({ mainUrl: trimmed, nodeTypes: [detectedType] });

        setFetching(true);
        try {
          const res = await fetch('/api/problem/url-metadata', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: trimmed }),
          });
          if (res.ok) {
            const meta = await res.json();
            if (meta.error) {
              // eslint-disable-next-line no-alert
              alert(`Could not fetch URL: ${meta.error}`);
              dataRef.current.onUpdateDraft?.({ mainUrl: null, nodeTypes: ['topic'] });
              setUrlInput('');
            } else {
              const title = meta.title ? meta.title.slice(0, MAX_TITLE_LENGTH) : null;
              dataRef.current.onUpdateDraft?.({
                mainUrl: trimmed,
                title,
                description: meta.description || null,
                author: meta.author || null,
                nodeTypes: [detectedType],
              });
            }
          } else {
            const err = await res.json().catch(() => null);
            // eslint-disable-next-line no-alert
            alert(`Could not fetch URL: ${err?.error || res.statusText}`);
            dataRef.current.onUpdateDraft?.({ mainUrl: null, nodeTypes: ['topic'] });
            setUrlInput('');
          }
        } catch {
          // eslint-disable-next-line no-alert
          alert('Could not fetch URL. Please check the URL and try again.');
          dataRef.current.onUpdateDraft?.({ mainUrl: null, nodeTypes: ['topic'] });
          setUrlInput('');
        } finally {
          setFetching(false);
        }
      }
    },
    [handleTextSubmit],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handlePaste(urlInput);
      }
      // Prevent React Flow from handling keyboard events
      e.stopPropagation();
    },
    [urlInput, handlePaste],
  );

  const handleInputPaste = useCallback(
    (e: React.ClipboardEvent) => {
      const pasted = e.clipboardData.getData('text');
      if (pasted && /^https?:\/\/.+/i.test(pasted.trim())) {
        e.preventDefault();
        setUrlInput(pasted);
        handlePaste(pasted);
      }
    },
    [handlePaste],
  );

  return (
    <div
      className={`relative rounded-lg border ${tc.selectedBorder} flex flex-col bg-white shadow-md`}
      style={{ minWidth: DEFAULT_NODE_WIDTH, maxWidth: DEFAULT_NODE_WIDTH }}
      onClick={(e) => e.stopPropagation()}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ width: 0, height: 0, minWidth: 0, minHeight: 0, border: 'none', background: 'transparent', left: 0 }}
      />

      {populated ? (
        /* ── Populated: title → URLs → additional URL input → buttons ── */
        <div className="flex flex-col px-3 py-1.5">
          <div className="flex gap-1">
            {types.map((t) => (
              <span key={t} className={`text-[6px] font-bold ${(TYPE_COLORS[t] || TYPE_COLORS.topic).label}`}>
                {t.toUpperCase()}
              </span>
            ))}
          </div>
          <span className="mt-0.5 truncate font-sans text-[11px] font-semibold text-slate-700">{data.draftTitle}</span>
          {data.draftUrl && (
            <a
              href={data.draftUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-0.5 truncate text-[9px] text-sky-600 hover:underline"
            >
              {data.draftUrl}
            </a>
          )}
          {(data.currentAdditionalUrls || []).map((url: string, i: number) => (
            <a
              key={i}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="truncate text-[9px] text-sky-600 hover:underline"
            >
              {url}
            </a>
          ))}
          <div className="mt-1">
            <input
              type="text"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={(e) => {
                const pasted = e.clipboardData.getData('text');
                if (pasted) {
                  setTimeout(() => handlePaste(pasted), 0);
                }
              }}
              placeholder="Paste additional URL..."
              className="nodrag w-full rounded border-0 bg-sky-50 px-2 py-0.5 text-[10px] text-slate-500 placeholder-slate-300 focus:outline-none focus:ring-0"
            />
          </div>
          <div className="mb-1 mt-1.5 flex gap-1">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                // eslint-disable-next-line no-alert
                if (window.confirm('Cancel creating a new node?')) {
                  data.onCancel?.();
                }
              }}
              className="flex-1 rounded border border-slate-200 py-0.5 text-center text-[10px] font-medium text-slate-600 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={data.saving}
              onClick={(e) => {
                e.stopPropagation();
                data.onSubmitDraft?.();
              }}
              className={`flex-1 rounded py-0.5 text-center text-[10px] font-medium ${
                data.saving
                  ? 'cursor-not-allowed bg-slate-300 text-slate-500'
                  : 'bg-sky-700 text-white hover:bg-sky-800'
              }`}
            >
              {data.saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
      ) : (
        /* ── Empty: show URL input ── */
        <div className="flex items-center gap-2 px-2 py-1">
          <input
            ref={inputRef}
            type="text"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handleInputPaste}
            placeholder={fetching ? 'Fetching metadata...' : 'Paste URL here, or enter new topic title.'}
            disabled={fetching}
            className="nodrag w-full rounded border-none bg-sky-50 px-2 py-0.5 text-[11px] text-slate-600 placeholder-slate-400 focus:outline-none focus:ring-0 disabled:opacity-60"
          />
          {fetching && (
            <div className="h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-slate-300 border-t-sky-600" />
          )}
          {/* <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              data.onCancel?.();
            }}
            className="shrink-0 text-[10px] text-slate-400 hover:text-slate-600"
          >
            ✕
          </button> */}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Right}
        style={{ width: 0, height: 0, minWidth: 0, minHeight: 0, border: 'none', background: 'transparent', right: 0 }}
      />
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          // eslint-disable-next-line no-alert
          if (window.confirm('Cancel creating a new node?')) {
            data.onCancel?.();
          }
        }}
        className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full border border-red-400 bg-white text-[9px] font-bold leading-none text-red-500 shadow-sm hover:border-red-500 hover:bg-red-500 hover:text-white"
      >
        ✕
      </button>
    </div>
  );
}

export default memo(DraftNodeComponent);
