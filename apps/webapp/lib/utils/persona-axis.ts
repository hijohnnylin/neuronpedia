/**
 * A readout axis as a `PersonaAxis` row, and what it becomes on the way to inference.
 *
 * Kept out of `lib/db` so a chart may import the display half without pulling prisma into a client
 * bundle; nothing in here touches the database. The queries that fill these are
 * `lib/db/persona-axis.ts`.
 */
import type { NPVectorRead, SteerVectorReadout } from '@/lib/api/inference-types';

/**
 * One readout axis as a page needs it to draw a wheel: what the poles are called, what they mean,
 * and which row said so. Not the vectors.
 *
 * That omission is the point of a type of its own. A direction is one float per model dimension --
 * 4096 for Llama-3.1-8B, 8192 for the 70B -- so handing a whole row to a client component would
 * serialize a quarter of a megabyte of numbers into the page's HTML to render six labels. The
 * vectors are wanted only where an axis is measured, which is server-side on the way to inference.
 *
 * `id` and `name` are both identifiers and are not interchangeable. `name` is what a request calls
 * the axis (`mit_toxic`) and what a readout comes back keyed by; `id` is the row, which is what a
 * stored reading records so it stays interpretable against the constants that produced it.
 *
 * The poles are nullable because a row need not have any: a probe or a plain steering vector reads
 * to the same number with no named ends, and every display of one already falls back to its title.
 */
export type PersonaAxisDefinition = {
  id: string;
  name: string;
  author: string;
  layer: number;
  polePositive: string | null;
  poleNegative: string | null;
  polePositiveDescription: string | null;
  poleNegativeDescription: string | null;
  displayName: string | null;
  caveat: string | null;
};

/**
 * A definition plus the numbers that measure it: everything inference needs to read the axis off a
 * turn, and nothing it does not. The fit report, publication and ownership stay in the table.
 */
export type PersonaAxisFit = PersonaAxisDefinition & {
  values: number[];
  projectionParams: unknown;
};

/**
 * The parameters a stored blob may carry: the payload's fields, minus the ones a column holds.
 *
 * A runtime list rather than a type, because it is both. `satisfies` ties it to `NPVectorRead`, so
 * an inference-side rename fails `tsc` here, and `ProjectionParams` below is derived from it --
 * naming a field once rather than in a type and again in a copy that can fall behind it.
 *
 * It is a list at all because what the blob holds is not checked anywhere else. `personaAxisToVectorRead`
 * used to spread the blob whole, so a key nobody declared travelled to inference; the day the payload
 * refuses unknown fields, such a key stops being inert and starts refusing every read of that
 * vector. Copying only what is named here means a hand-written row cannot do that.
 */
const PROJECTION_PARAM_KEYS = [
  'read',
  'normalize',
  'center',
  'scalePos',
  'scaleNeg',
  'preNormMean',
  'postNormMean',
  'quantileLevels',
  'quantilesPos',
  'quantilesNeg',
  'render',
] as const satisfies readonly (keyof NPVectorRead)[];

/**
 * Every key optional, because absent means defaulted and which keys a row has depends on how it was
 * fitted: the assistant axis stores a mean pair and nothing else, the MIT six a center and quantile
 * tables.
 *
 * One type, because inference has one read payload. A second would arrive as a union tagged by the
 * blob's own `v`, and `parseProjectionParams` would branch on it; nothing infers a shape from which
 * keys happen to be present, which would make a typo into a different fitting method.
 */
type ProjectionParams = Partial<Pick<NPVectorRead, (typeof PROJECTION_PARAM_KEYS)[number]>>;

/**
 * A stored blob as projection parameters, or an empty set for a row that has no usable blob.
 *
 * Filtered by name, not validated. Inference checks every value -- vector lengths, non-finite
 * entries, a zero divisor, a non-monotone quantile table -- and has to, because a caller may send a
 * vector inline without ever touching this table. A second copy of those rules here would be one to
 * keep in step for no new coverage, so a malformed value is still a bad backfill that fails loudly
 * upstream. What this does stop is a key inference never agreed to.
 *
 * A key present and explicitly null is copied as it stands, so a row saying "no quantile table"
 * still says it. Only absence is absence.
 */
function parseProjectionParams(stored: unknown): ProjectionParams {
  if (typeof stored !== 'object' || stored === null || Array.isArray(stored)) return {};
  const blob = stored as Record<string, unknown>;
  const params: Record<string, unknown> = {};
  for (const key of PROJECTION_PARAM_KEYS) {
    if (key in blob) params[key] = blob[key];
  }
  return params as ProjectionParams;
}

/**
 * The string entries of a stored `templateKwargs` blob.
 *
 * A chat template takes strings, so anything else in there cannot be passed on. Dropped rather
 * than coerced: `date_string: 26` would render as a plausible date and quietly move the fit off
 * the distribution it was measured on, which is the failure this column exists to prevent.
 */
function templateKwargs(stored: unknown): Record<string, string> {
  if (typeof stored !== 'object' || stored === null || Array.isArray(stored)) return {};
  const kwargs: Record<string, string> = {};
  for (const [key, value] of Object.entries(stored)) {
    if (typeof value === 'string') kwargs[key] = value;
  }
  return kwargs;
}

/**
 * A row as the read payload inference takes.
 *
 * Mostly the stored parameters spread verbatim, which is the point of holding them in the payload's
 * own shape: a fitting method inference already understands needs nothing added here to reach it.
 *
 * Sends `id: name`, not the row id: `id` is what the readout comes back under and what the chart
 * matches on, and a cuid there would make every stored reading unreadable to the next request.
 * Which row was used is recorded separately, beside the reading.
 *
 * **No labels.** The payload has no field for the poles, the title, the author or the caveat: they
 * are this table's, they are already in hand wherever a reading is displayed or stored, and sending
 * them to a compute server to have them echoed back made inference the courier for display text it
 * cannot check. `labelReadouts` is what puts them on the response instead.
 */
export function personaAxisToVectorRead(axis: PersonaAxisFit): NPVectorRead {
  const params = parseProjectionParams(axis.projectionParams);
  return {
    ...params,

    // The four the payload requires, defaulted here rather than in every writer of a row. This is
    // what omitting them means: the reading is the bare dot product, in the axis's own units.
    normalize: params.normalize ?? 'none',
    center: params.center ?? 0,
    scalePos: params.scalePos ?? 1,
    scaleNeg: params.scaleNeg ?? 1,

    // Rebuilt rather than passed through, because `templateKwargs` is the one field where a wrong
    // type is worse than a rejection: `date_string: 26` renders as a plausible date and moves the
    // fit off the distribution it was measured on.
    render: {
      blankSystemPrompt: params.render?.blankSystemPrompt ?? false,
      templateKwargs: templateKwargs(params.render?.templateKwargs),
    },

    // Named field by field for the same reason the top level is, since filtering a blob's own keys
    // says nothing about what is nested inside one of them. Absent stays absent: a row that says
    // nothing about how to read it means inference's defaults, which is what every row meant before
    // the spec existed.
    read: params.read ? { site: params.read.site, tokens: params.read.tokens, pool: params.read.pool } : undefined,

    id: axis.name,
    layer: axis.layer,
    direction: axis.values,
  };
}

/**
 * Readouts with this table's labels put back on them, so a reading reads as something.
 *
 * The counterpart of `personaAxisToVectorRead` sending none. Inference answers with the id it was
 * asked for, a placeholder author and the id as a title, and nothing else it could honestly say
 * about a bare direction; everything a display or a stored row wants beside the numbers is here,
 * one row lookup away, and was never worth a round trip.
 *
 * Matched on `name`, which is what the readout's `id` is. A readout with no row -- an axis fetched
 * from a published artifact, where inference did read a manifest -- is left exactly as it came back.
 */
export function labelReadouts(
  readouts: SteerVectorReadout[],
  definitions: PersonaAxisDefinition[],
): SteerVectorReadout[] {
  const byName = new Map(definitions.map((definition) => [definition.name, definition]));
  return readouts.map((readout) => {
    const row = byName.get(readout.id);
    if (!row) return readout;
    return {
      ...readout,
      author: row.author,
      title: row.displayName ?? readout.id,
      caveat: row.caveat ?? undefined,
      polePositive: row.polePositive ?? undefined,
      poleNegative: row.poleNegative ?? undefined,
      polePositiveDescription: row.polePositiveDescription ?? undefined,
      poleNegativeDescription: row.poleNegativeDescription ?? undefined,
    };
  });
}
