'use client';

import CustomTooltip from '@/components/custom-tooltip';
import {
  formatMetadataKey,
  getExplanationBadge,
  hasMetadata,
  orderMetadataEntries,
  safeHttpUrl,
} from '@/lib/utils/explanation-source';
import { ExplanationPartialWithRelations } from '@/prisma/generated/zod';
import * as Popover from '@radix-ui/react-popover';
import { ExternalLink, Info } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

const BADGE_CLASSES = {
  negative: 'bg-rose-50 text-rose-600',
  both: 'bg-sky-50 text-sky-700',
  steer: 'bg-amber-50 text-amber-700',
};

export function ExplanationBadgeView({ explanation }: { explanation: ExplanationPartialWithRelations }) {
  const badge = getExplanationBadge(explanation);
  if (!badge) {
    return null;
  }
  return (
    <CustomTooltip
      trigger={
        <span
          className={`relative -top-px shrink-0 cursor-default whitespace-nowrap rounded px-1 py-px font-sans text-[9px] font-semibold uppercase leading-tight ${
            BADGE_CLASSES[badge.kind]
          }`}
        >
          {badge.label}
        </span>
      }
      side="top"
    >
      <p className="leading-relaxed">{badge.tooltip}</p>
      {badge.kind === 'steer' && explanation.type?.settings && (
        <p className="mt-1.5 leading-relaxed text-slate-400">{explanation.type.settings}</p>
      )}
    </CustomTooltip>
  );
}

function MetadataValue({ value }: { value: unknown }) {
  if (value === null || value === undefined || value === '') {
    return <span className="text-slate-300">-</span>;
  }
  if (typeof value === 'string') {
    if (safeHttpUrl(value)) {
      return (
        <a href={value} target="_blank" rel="noopener noreferrer" className="break-all text-sky-700 hover:underline">
          {value}
        </a>
      );
    }
    return <span className="whitespace-pre-wrap break-words">{value}</span>;
  }
  if (typeof value !== 'object') {
    return <span>{String(value)}</span>;
  }
  if (Array.isArray(value)) {
    if (value.every((v) => v === null || typeof v !== 'object')) {
      return <span className="break-words">{value.map((v) => String(v)).join(', ')}</span>;
    }
    return (
      <div className="flex flex-col gap-y-1.5">
        {value.map((v, i) => (
          <div key={i} className="rounded border border-slate-100 px-2 py-1.5">
            <MetadataValue value={v} />
          </div>
        ))}
      </div>
    );
  }
  return (
    <dl className="flex flex-col gap-y-1.5">
      {orderMetadataEntries(value as Record<string, unknown>).map(([k, v]) => (
        <div key={k}>
          <dt className="text-[10px] font-semibold uppercase text-slate-400">{formatMetadataKey(k)}</dt>
          <dd className="text-slate-600">
            <MetadataValue value={v} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

// A circled i that shows all of an explanation's metadata. Hover to look, click to keep it open.
export function ExplanationMetadataButton({ metadata }: { metadata: unknown }) {
  const [hovered, setHovered] = useState(false);
  const [pinned, setPinned] = useState(false);
  // Wait before closing, so the pointer can cross the gap from the button to the popup.
  const closeTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const startHover = () => {
    clearTimeout(closeTimer.current);
    setHovered(true);
  };
  const endHover = () => {
    clearTimeout(closeTimer.current);
    closeTimer.current = setTimeout(() => setHovered(false), 200);
  };
  useEffect(() => () => clearTimeout(closeTimer.current), []);
  if (!hasMetadata(metadata)) {
    return null;
  }
  return (
    <Popover.Root
      open={hovered || pinned}
      onOpenChange={(open) => {
        if (!open) {
          setPinned(false);
          setHovered(false);
        }
      }}
    >
      <Popover.Trigger asChild>
        <button
          type="button"
          aria-label="Show details"
          className="ml-1 inline-flex -translate-y-px align-middle text-slate-400 hover:text-sky-600"
          onMouseEnter={startHover}
          onMouseLeave={endHover}
          onClick={() => setPinned(!pinned)}
        >
          <Info className="h-3 w-3" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          className="z-50 max-h-[420px] w-[380px] overflow-y-auto rounded-lg border border-slate-200 bg-white px-4 py-3 text-xs shadow-lg"
          sideOffset={4}
          side="bottom"
          align="start"
          onMouseEnter={startHover}
          onMouseLeave={endHover}
          onOpenAutoFocus={(e) => e.preventDefault()}
        >
          <MetadataValue value={metadata} />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

// The credit for an explanation from another site: the source and its author (or contributor),
// linked to the unit there, with the license and citation on hover.
export function ExternalExplanationCredit({ explanation }: { explanation: ExplanationPartialWithRelations }) {
  const { type } = explanation;
  const sourceName = type?.displayName || explanation.typeName || 'External source';
  const siteUrl = safeHttpUrl(type?.url);
  const sourceUrl = safeHttpUrl(explanation.sourceUrl) || siteUrl;
  const licenseUrl = safeHttpUrl(type?.licenseUrl);
  const contributorUrl = safeHttpUrl(explanation.contributorUrl);
  const creditName = explanation.contributorName ? `@${explanation.contributorName}` : type?.creatorName;
  return (
    <CustomTooltip
      trigger={
        <a
          className="flex flex-row items-center gap-x-1 font-sans text-[11px] font-medium text-slate-400 hover:text-sky-600"
          href={sourceUrl}
          target="_blank"
          rel="noopener noreferrer"
        >
          {sourceName}
          {creditName && ` · ${creditName}`}
          <ExternalLink className="h-2.5 w-2.5" />
        </a>
      }
      side="bottom"
      align="start"
      alignOffset={12}
      collisionPadding={12}
    >
      <div className="flex flex-col gap-y-1.5 leading-relaxed">
        <p>
          From{' '}
          {siteUrl ? (
            <a href={siteUrl} target="_blank" rel="noopener noreferrer" className="text-sky-700 hover:underline">
              {sourceName}
            </a>
          ) : (
            sourceName
          )}
          {type?.creatorName ? ` by ${type.creatorName}` : ''}.
          {explanation.contributorName && (
            <>
              {' '}
              Written by{' '}
              {contributorUrl ? (
                <a
                  href={contributorUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sky-700 hover:underline"
                >
                  @{explanation.contributorName}
                </a>
              ) : (
                `@${explanation.contributorName}`
              )}
              .
            </>
          )}
          {explanation.explanationModelName ? ` Labeled by ${explanation.explanationModelName}.` : ''}
        </p>
        {type?.description && <p className="text-slate-500">{type.description}</p>}
        <p>
          License:{' '}
          {type?.license ? (
            licenseUrl ? (
              <a href={licenseUrl} target="_blank" rel="noopener noreferrer" className="text-sky-700 hover:underline">
                {type.license}
              </a>
            ) : (
              type.license
            )
          ) : (
            'used with permission'
          )}
          . Shown here in a different format.
        </p>
        {type?.citation && (
          <p className="text-slate-500">
            <span className="font-semibold text-slate-600">Cite: </span>
            {type.citation}
          </p>
        )}
      </div>
    </CustomTooltip>
  );
}

// The badge, shown above the explanation. Nothing if the explanation has the default pole and aspect.
export function ExplanationTagsRow({ explanation }: { explanation: ExplanationPartialWithRelations }) {
  if (!getExplanationBadge(explanation)) {
    return null;
  }
  return (
    <div className="mb-1 flex flex-row items-center">
      <ExplanationBadgeView explanation={explanation} />
    </div>
  );
}
