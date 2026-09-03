/**
 * The `assistant_axis` shape `/api/steer-chat` has always returned, rebuilt from axis readouts.
 *
 * `/api/*` has real users and its field names are a contract of their own, so a rename inside
 * inference must not reach them. Inference now returns one `SteerVectorReadout` per axis, keyed by a
 * stable id and holding a bare value per turn; the public field is one entry per steer type, whose
 * turns hold a map keyed by display title. This maps between them explicitly rather than forwarding
 * the new shape through under the old name.
 *
 * Nothing here is lossy in the direction that matters: the readouts a request produces are for one
 * conversation, so grouping them by steer type and keying by title reproduces exactly what the old
 * inference response carried. `layer` and `caveat` have no place in the old shape and are dropped;
 * a caller that wants them reads `axes`. `percentile` is dropped for the same reason and not
 * because it is unimportant -- this view reproduces a shape callers already parse, so growing it
 * would change what a field means for someone who never asked for the new reading.
 *
 * Nothing inside the webapp reads this any more. The assistant-axis page takes its numbers from
 * `axes`, by the id it asked for, so this exists for callers outside the repo and is written to be
 * deletable the day they stop reading it.
 */
import type { SteerVectorReadout } from '@/lib/api/inference-types';
import { LEGACY_AXIS_TITLES } from '@/lib/utils/steer-wire';

export type LegacyAssistantAxisTurn = {
  pcValues?: Record<string, number>;
  pcValuesPostCap?: Record<string, number>;
  snippet?: string;
};

export type LegacyAssistantAxis = {
  pcTitles?: string[];
  turns?: LegacyAssistantAxisTurn[];
  type?: SteerVectorReadout['type'];
};

/**
 * The key an axis's values appear under in this view.
 *
 * A lookup in the table of what was actually written, not a string assembled from the poles. The
 * assembly was an inference -- that a key had always been `- <minus> \u2194\ufe0f + <plus>`, so
 * rebuilding it from two poles would reproduce it -- and it held only while every axis was worded
 * the way that one was. One axis has callers who parse this key, the table names it, and the same
 * table reads it back in `steer-wire.ts`, so a reworded pole cannot move it.
 *
 * Anything else keys by its id, which is what this view can honestly say about an axis nobody was
 * parsing before ids existed.
 */
function legacyTitle(readout: SteerVectorReadout): string {
  return LEGACY_AXIS_TITLES[readout.id] ?? readout.id;
}

/** Group readouts by steer type and re-key their values by title, one entry per type. */
export function axisReadoutsToLegacyAssistantAxis(readouts: SteerVectorReadout[]): LegacyAssistantAxis[] {
  const byType = new Map<string, SteerVectorReadout[]>();
  for (const readout of readouts) {
    const key = readout.type ?? '';
    const group = byType.get(key);
    if (group) {
      group.push(readout);
    } else {
      byType.set(key, [readout]);
    }
  }

  return Array.from(byType.entries()).map(([, group]) => {
    const titles = group.map(legacyTitle);
    // Readouts for one steer type describe the same conversation, so the longest turn list is the
    // conversation's length; an axis that came back short leaves its title out of that turn.
    const nTurns = Math.max(0, ...group.map((readout) => readout.turns?.length ?? 0));

    const turns: LegacyAssistantAxisTurn[] = [];
    for (let index = 0; index < nTurns; index += 1) {
      const pcValues: Record<string, number> = {};
      const pcValuesPostCap: Record<string, number> = {};
      let snippet: string | undefined;
      for (const readout of group) {
        const turn = readout.turns?.[index];
        if (!turn) continue;
        const title = legacyTitle(readout);
        if (turn.value !== null && turn.value !== undefined) pcValues[title] = turn.value;
        if (turn.valuePostCap !== null && turn.valuePostCap !== undefined) {
          pcValuesPostCap[title] = turn.valuePostCap;
        }
        snippet = snippet ?? turn.snippet ?? undefined;
      }
      turns.push({
        pcValues,
        // Absent rather than empty when nothing was steered, since the old field was absent then
        // and callers test for it rather than for its size.
        pcValuesPostCap: Object.keys(pcValuesPostCap).length > 0 ? pcValuesPostCap : undefined,
        snippet,
      });
    }

    return { pcTitles: titles, turns, type: group[0].type };
  });
}
