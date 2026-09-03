/**
 * Reads and writes the shapes persisted in `SteerOutput`, whose older rows were written under
 * semantics the current wire no longer uses. Keeping those decisions in one file means the storage
 * format is something we state explicitly, rather than a property of whichever codegen template or
 * inference version happened to be current when a row was saved.
 *
 * Axis readouts are stored snake_case in `capMonitorOutput`: the rows predate the wire being
 * camelCase and are not migrated, so reads must accept snake_case and writes must keep producing
 * it. These replace the generated client's `ToJSON`/`FromJSON` helpers, which performed the same
 * conversion as a side effect of its naming scheme.
 *
 * Four generations of that column exist and all four have to read:
 *
 * 1. `{pc_titles, turns:[{pc_values, pc_values_post_cap}]}` -- one fit's components, each turn
 *    holding a map keyed by display title.
 * 2. the same in camelCase, from a window where the writer had stopped converting.
 * 3. `{axes: {<id>: {title, layer, caveat, turns:[{value, value_post_cap}]}}}`, keyed by axis id.
 * 4. the same, with the axis's two poles in place of its title and the `PersonaAxis` row that
 *    measured it recorded beside them -- what is written now.
 *
 * The move to (3) is why this file grew a normalizer rather than a field rename. Keying turn values
 * by display title made a title a database key: rewording one orphaned every stored row, and two
 * axes that happened to share a title collided. Ids are stable and titles became metadata.
 *
 * (4) is what an axis being a database row rather than a file adds: a reading is only interpretable
 * against the constants that produced it, so the row and version are stored with it, and the poles
 * are stored so the reading still reads as something after that row is retired.
 */
import type { SteerVectorReadout } from '@/lib/api/inference-types';
import { STEER_COMPLETION_VERSION } from '@/lib/utils/steer';

/**
 * The snake_case shape persisted in `capMonitorOutput` today.
 *
 * Both readings of a turn are stored. `value` is the measurement, calibrated against the axis's own
 * spread and never clipped, so it passes 1 for a few percent of turns; `percentile` is where that
 * value falls in the axis's calibration corpus, which cannot leave [-1, 1] and is what the chart
 * shows. Storing only the percentile would be lossy in a way no later read could undo: how far past
 * the corpus a turn sat is the evidence that an axis is being read off the distribution it was
 * fitted on. Rows written before percentiles existed simply omit them.
 */
export type StoredAxisTurn = {
  value?: number | null;
  value_post_cap?: number | null;
  percentile?: number | null;
  percentile_post_cap?: number | null;
  snippet?: string | null;
};

export type StoredAxis = {
  /**
   * The `Vector` row this reading was measured with.
   *
   * A percentile is only interpretable against the calibration constants that produced it, and a
   * row is never edited, so this one id is what makes a stored reading readable later: it says
   * which constants, and it lets a cache tell a reading this scheme took from one it did not.
   *
   * Absent for a reading measured from an inference server's own assets, which is every row
   * written before axes became table rows. Rows written by an earlier generation of this scheme
   * also carry an `axis_version`, which nothing reads any more -- a name now has one row for good,
   * so the id says everything the pair used to.
   */
  axis_id?: string | null;
  /**
   * What the axis's two ends were called when this was measured.
   *
   * Stored rather than looked up so a reading stays legible after its axis is retired, and so the
   * deprecated `assistant_axis` view keys a cached conversation exactly as it keys a fresh one.
   */
  pole_positive?: string | null;
  pole_negative?: string | null;
  /** Written by an earlier generation, which had one display string instead of two poles. */
  title?: string | null;
  layer?: number | null;
  caveat?: string | null;
  turns?: StoredAxisTurn[] | null;
};

/**
 * Which row measured each axis, keyed by the axis name a readout comes back under.
 *
 * Passed to a write rather than read off the readout, because inference is told an axis and not
 * where it came from: the id it reports is the name it was asked for. Only the caller that
 * resolved the rows knows which ones they were.
 */
export type AxisProvenance = Record<string, string>;

export type StoredAxisSet = {
  axes: Record<string, StoredAxis>;
  type?: string | null;
};

type LegacyValues = Record<string, number>;

/** A pre-`axes` row, in either casing: rows exist from before and after the wire changed. */
type LegacyTurn = {
  pc_values?: LegacyValues | null;
  pc_values_post_cap?: LegacyValues | null;
  pcValues?: LegacyValues | null;
  pcValuesPostCap?: LegacyValues | null;
  snippet?: string | null;
};

type LegacyRow = {
  pc_titles?: string[] | null;
  pcTitles?: string[] | null;
  turns?: LegacyTurn[] | null;
  type?: string | null;
};

/**
 * Axis ids for the titles legacy rows were keyed by.
 *
 * Only one asset was ever stored in that format -- the Llama-3.3-70B assistant axis -- so this is
 * the whole of it. Without the mapping an old row would read as an axis whose id is its title,
 * which would never satisfy a request for `lu_assistant-axis` and would silently re-run inference
 * for every conversation already in the database.
 */
const LEGACY_TITLE_TO_AXIS_ID: Record<string, string> = {
  '- Role-playing \u2194\ufe0f + Assistant-like': 'lu_assistant-axis',
};

function legacyAxisId(title: string): string {
  return LEGACY_TITLE_TO_AXIS_ID[title] ?? title;
}

/**
 * Who fitted an axis, which is the `<author>_` prefix of its id.
 *
 * Derived rather than stored. The prefix is the only source the server has for it too, so a
 * stored copy would just be a second thing to drift. Empty for an id predating the convention --
 * a legacy row whose title did not map, which was fitted by someone nothing recorded.
 */
export function authorFromAxisId(id: string): string {
  const separator = id.indexOf('_');
  return separator > 0 ? id.slice(0, separator) : '';
}

function isStoredAxisSet(stored: unknown): stored is StoredAxisSet {
  return typeof stored === 'object' && stored !== null && 'axes' in stored;
}

/** Reshape a pre-`axes` row: one component becomes one axis, its title becomes its id's label. */
function readoutsFromLegacy(row: LegacyRow, fallbackType?: string | null): SteerVectorReadout[] {
  const titles = row.pcTitles ?? row.pc_titles ?? [];
  const turns = row.turns ?? [];
  const type = (row.type ?? fallbackType ?? undefined) as SteerVectorReadout['type'];

  return titles.map((title) => {
    const id = legacyAxisId(title);
    return {
      id,
      author: authorFromAxisId(id),
      title,
      type,
      turns: turns.map((turn) => {
        const values = turn.pcValues ?? turn.pc_values ?? undefined;
        const valuesPostCap = turn.pcValuesPostCap ?? turn.pc_values_post_cap ?? undefined;
        return {
          value: values?.[title] ?? undefined,
          valuePostCap: valuesPostCap?.[title] ?? undefined,
          snippet: turn.snippet ?? undefined,
        };
      }),
    };
  });
}

/**
 * Read a stored `capMonitorOutput` blob as axis readouts, whichever generation wrote it.
 *
 * `fallbackType` is the steer type of the row being read. Rows written before the type was stored
 * carry none, and which column a readout belongs to is not recoverable from the payload.
 */
export function axisReadoutsFromStored(stored: unknown, fallbackType?: string | null): SteerVectorReadout[] {
  if (stored === null || stored === undefined) return [];

  if (isStoredAxisSet(stored)) {
    const { axes, type } = stored;
    return Object.entries(axes ?? {}).map(([id, axis]) => ({
      id,
      author: authorFromAxisId(id),
      title: axis.title ?? id,
      type: (type ?? fallbackType ?? undefined) as SteerVectorReadout['type'],
      layer: axis.layer ?? undefined,
      caveat: axis.caveat ?? undefined,
      polePositive: axis.pole_positive ?? undefined,
      poleNegative: axis.pole_negative ?? undefined,
      turns: (axis.turns ?? []).map((turn) => ({
        value: turn.value ?? undefined,
        valuePostCap: turn.value_post_cap ?? undefined,
        percentile: turn.percentile ?? undefined,
        percentilePostCap: turn.percentile_post_cap ?? undefined,
        snippet: turn.snippet ?? undefined,
      })),
    }));
  }

  return readoutsFromLegacy(stored as LegacyRow, fallbackType);
}

/**
 * Write readouts in the stored snake_case shape, keyed by axis id.
 *
 * `measuredWith` names the row behind each axis, where the caller measured with rows at all. No
 * `title`: the poles are stored instead, and a display string assembled from them is something a
 * reader can do and a writer cannot undo.
 */
export function axisReadoutsToStored(
  readouts: SteerVectorReadout[],
  type?: string | null,
  measuredWith?: AxisProvenance,
): StoredAxisSet {
  const axes: Record<string, StoredAxis> = {};
  for (const readout of readouts) {
    axes[readout.id] = {
      axis_id: measuredWith?.[readout.id],
      pole_positive: readout.polePositive ?? undefined,
      pole_negative: readout.poleNegative ?? undefined,
      layer: readout.layer ?? undefined,
      caveat: readout.caveat ?? undefined,
      turns: (readout.turns ?? []).map((turn) => ({
        value: turn.value ?? undefined,
        value_post_cap: turn.valuePostCap ?? undefined,
        percentile: turn.percentile ?? undefined,
        percentile_post_cap: turn.percentilePostCap ?? undefined,
        snippet: turn.snippet ?? undefined,
      })),
    };
  }
  return { axes, type: type ?? readouts[0]?.type ?? undefined };
}

/**
 * The axis ids a stored blob covers.
 *
 * Read directly rather than by reshaping the whole payload, because the caller asking is the cache:
 * it needs to know whether a row answers this request before deciding to reuse it.
 */
export function storedAxisIds(stored: unknown): string[] {
  if (stored === null || stored === undefined) return [];
  if (isStoredAxisSet(stored)) return Object.keys(stored.axes ?? {});
  const row = stored as LegacyRow;
  return (row.pcTitles ?? row.pc_titles ?? []).map(legacyAxisId);
}

/**
 * The `Vector` row each stored reading was measured with, for the axes that recorded one.
 *
 * The caller asking is the cache, which needs to know whether a stored reading came from the row
 * that would answer this request. Axes with no id recorded are absent here rather than reported as
 * some sentinel: they were measured from whatever asset a serving pod had on disk, which is not
 * this fit and often reported no percentile at all.
 */
export function storedAxisRowIds(stored: unknown): Record<string, string> {
  if (!isStoredAxisSet(stored)) return {};
  const rows: Record<string, string> = {};
  for (const [id, axis] of Object.entries(stored.axes ?? {})) {
    if (typeof axis.axis_id === 'string') rows[id] = axis.axis_id;
  }
  return rows;
}

/**
 * Merge freshly measured readouts into what a row already stored, new values winning.
 *
 * A row measured for one axis and later asked about another must end up holding both, or the
 * second request's work is thrown away and the row stays permanently short of what is asked of it.
 *
 * A row already in the current shape is carried across as it stands rather than reshaped through
 * readouts, which is the only way the axes it is not being asked about keep their provenance: a
 * `SteerVectorReadout` has no field for the row that measured it, so a round trip through one would
 * quietly drop `axis_id` from every axis this request did not measure.
 */
export function mergeStoredAxes(
  existing: unknown,
  incoming: SteerVectorReadout[],
  type?: string | null,
  measuredWith?: AxisProvenance,
): StoredAxisSet {
  const kept = isStoredAxisSet(existing)
    ? { axes: existing.axes ?? {}, type: existing.type ?? undefined }
    : axisReadoutsToStored(axisReadoutsFromStored(existing, type), type);
  const fresh = axisReadoutsToStored(incoming, type, measuredWith);
  return { axes: { ...kept.axes, ...fresh.axes }, type: fresh.type ?? kept.type ?? type ?? undefined };
}

/**
 * Whether a stored row's `outputText` already begins with the prompt.
 *
 * Inference used to return prompt + generation for completions, so rows saved below
 * `STEER_COMPLETION_VERSION` have the prompt baked in. The UI now renders the prompt as its own
 * node, so prepending it to one of those rows shows it twice. Chat rows are exempt: they render
 * from `outputTextChatTemplate` and their `outputText` is never displayed on its own.
 */
export function storedOutputTextIncludesPrompt(stored: {
  version: number;
  outputTextChatTemplate: string | null;
}): boolean {
  return !stored.outputTextChatTemplate && stored.version < STEER_COMPLETION_VERSION;
}
