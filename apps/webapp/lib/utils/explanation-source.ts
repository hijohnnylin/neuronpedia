import { EXPLANATIONTYPE_NP_MIN_ACT_LOGITS } from '@/lib/utils/autointerp';
import type { ExplanationAspect, ExplanationPole } from '@prisma/client';

export type ExplanationBadgeKind = 'negative' | 'both' | 'steer';

export type ExplanationBadge = {
  kind: ExplanationBadgeKind;
  label: string;
  tooltip: string;
};

type BadgeFields = {
  typeName?: string | null;
  pole?: ExplanationPole | null;
  aspect?: ExplanationAspect | null;
  steerStrength?: number | null;
};

const MINUS = '\u2212';

function formatStrength(strength: number): string {
  const sign = strength < 0 ? MINUS : '+';
  return `${sign}${Number(Math.abs(strength).toFixed(3))}`;
}

// One badge for what differs from the default: a positive pole that describes activations.
export function getExplanationBadge(explanation: BadgeFields): ExplanationBadge | null {
  if (explanation.aspect === 'STEERING') {
    const negative = explanation.pole === 'NEGATIVE';
    const strength =
      explanation.steerStrength !== null && explanation.steerStrength !== undefined
        ? formatStrength(explanation.steerStrength)
        : negative
          ? '\u2193'
          : '\u2191';
    return {
      kind: 'steer',
      label: `Steer ${strength}`,
      tooltip: `This describes what the model does when this unit is pushed ${negative ? 'down' : 'up'}${
        explanation.steerStrength !== null && explanation.steerStrength !== undefined
          ? `, at a strength of ${strength} in the units of the source`
          : ''
      }.`,
    };
  }
  if (explanation.pole === 'NEGATIVE' || explanation.typeName === EXPLANATIONTYPE_NP_MIN_ACT_LOGITS) {
    return {
      kind: 'negative',
      label: 'Neg Acts',
      tooltip:
        'Explanations are based on positive activations by default. This one is based on the most negative activations.',
    };
  }
  if (explanation.pole === 'BOTH') {
    return {
      kind: 'both',
      label: 'Both Signs',
      tooltip: 'This describes what makes the unit fire, for both positive and negative activations.',
    };
  }
  return null;
}

// "knockoutConcepts" -> "Knockout concepts", "base_output" -> "Base output".
export function formatMetadataKey(key: string): string {
  const words = key
    .replace(/_/g, ' ')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .trim()
    .toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

const FIRST_METADATA_KEYS = ['description', 'summary', 'note', 'notes'];
const LONG_TEXT = 120;

// Postgres jsonb does not keep key order. Put the summary first and long text and lists last.
export function orderMetadataEntries(metadata: Record<string, unknown>): [string, unknown][] {
  const rank = ([key, value]: [string, unknown]) => {
    const first = FIRST_METADATA_KEYS.indexOf(key);
    if (first !== -1) {
      return first;
    }
    const long =
      (typeof value === 'string' && value.length > LONG_TEXT) ||
      (Array.isArray(value) && value.some((v) => v !== null && typeof v === 'object'));
    return long ? FIRST_METADATA_KEYS.length + 1 : FIRST_METADATA_KEYS.length;
  };
  return Object.entries(metadata)
    .map((entry, i) => ({ entry, i }))
    .sort((a, b) => rank(a.entry) - rank(b.entry) || a.i - b.i)
    .map(({ entry }) => entry);
}

// Explanations from other sites first, by type name, then Neuronpedia's. Otherwise keeps the order it was given.
export function externalFirst<T extends { sourceUrl?: string | null; typeName?: string | null }>(
  explanations: T[],
): T[] {
  return [...explanations].sort(
    (a, b) =>
      Number(!!b.sourceUrl) - Number(!!a.sourceUrl) ||
      (a.sourceUrl && b.sourceUrl ? (a.typeName || '').localeCompare(b.typeName || '') : 0),
  );
}

// The URL if it is http or https, else undefined. Do not render other schemes as links.
export function safeHttpUrl(url: string | null | undefined): string | undefined {
  return url && /^https?:\/\/\S+$/i.test(url) ? url : undefined;
}

// True if the metadata has at least one value worth showing.
export function hasMetadata(metadata: unknown): metadata is Record<string, unknown> {
  if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
    return false;
  }
  return Object.values(metadata).some((v) => v !== null && v !== undefined && v !== '');
}
