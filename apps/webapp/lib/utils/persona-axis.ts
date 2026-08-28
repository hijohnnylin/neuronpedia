/**
 * A readout axis as a `PersonaAxis` row, and what it becomes on the way to inference.
 *
 * Kept out of `lib/db` so a chart may import the display half without pulling prisma into a client
 * bundle; nothing in here touches the database. The queries that fill these are
 * `lib/db/persona-axis.ts`.
 */
import type { NPAxis } from '@/lib/api/inference-types';

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
 */
export type PersonaAxisDefinition = {
  id: string;
  name: string;
  layer: number;
  polePositive: string;
  poleNegative: string;
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
  author: string;
  values: number[];
  projectionParams: unknown;
};

/**
 * The stored `projectionParams` blob: exactly the payload fields, minus the ones a column holds.
 *
 * Derived from `NPAxis` rather than declared, so an inference-side change to any of these fails
 * `tsc` here instead of at runtime, and a new field is one name added to the `Pick` below. Every
 * key optional because absent means defaulted, and which keys a row has depends on how it was
 * fitted -- the PCA axis has almost none of them.
 *
 * One type, because inference has one axis payload. A second would arrive as a union tagged by the
 * blob's own `v`, and `parseProjectionParams` would branch on it; nothing infers a shape from which
 * keys happen to be present, which would make a typo into a different fitting method.
 */
type ProjectionParams = Partial<
  Pick<
    NPAxis,
    | 'normalize'
    | 'center'
    | 'scalePos'
    | 'scaleNeg'
    | 'preNormMean'
    | 'postNormMean'
    | 'quantileLevels'
    | 'quantilesPos'
    | 'quantilesNeg'
    | 'render'
  >
>;

/**
 * A stored blob as projection parameters, or an empty set for a row that has no usable blob.
 *
 * Checked for being an object and not for its fields, deliberately. Inference validates every one
 * of them -- vector lengths, non-finite entries, a zero divisor, a non-monotone quantile table --
 * and has to, because a caller may send an axis inline without ever touching this table. A second
 * copy of those rules here would be one to keep in step for no new coverage, so a malformed blob
 * is a bad backfill that fails loudly upstream rather than something to repair in passing.
 */
function parseProjectionParams(stored: unknown): ProjectionParams {
  if (typeof stored !== 'object' || stored === null || Array.isArray(stored)) return {};
  return stored as ProjectionParams;
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
 * A row as the axis payload inference takes.
 *
 * Mostly the stored parameters spread verbatim, which is the point of holding them in the payload's
 * own shape: a fitting method inference already understands needs nothing added here to reach it.
 *
 * Sends `id: name`, not the row id: `id` is what the readout comes back under and what the chart
 * matches on, and a cuid there would make every stored reading unreadable to the next request.
 * Which row was used is recorded separately, beside the reading.
 */
export function personaAxisToNPAxis(axis: PersonaAxisFit): NPAxis {
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

    id: axis.name,
    author: axis.author,
    layer: axis.layer,
    direction: axis.values,
    polePositive: axis.polePositive,
    poleNegative: axis.poleNegative,
    polePositiveDescription: axis.polePositiveDescription ?? undefined,
    poleNegativeDescription: axis.poleNegativeDescription ?? undefined,
    displayName: axis.displayName ?? undefined,
    caveat: axis.caveat ?? undefined,
  };
}
