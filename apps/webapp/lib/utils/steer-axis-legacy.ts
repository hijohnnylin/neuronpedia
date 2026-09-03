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
 * Two axes sharing a display title would collide in the map, which is the reason readouts are keyed
 * by id everywhere else. It cannot happen for the assets shipped today and would only ever affect
 * this compatibility view.
 */
import type { SteerVectorReadout } from '@/lib/api/inference-types';

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
 * The display title of an axis, in the form this view has always used: `- <minus> \u2194 + <plus>`.
 *
 * Assembled from the poles rather than taken from `title`, because `title` no longer means what it
 * did. An axis was once one display string that a reader split on the arrow; it is now two named
 * poles, and a request-supplied axis reports its id there. Rebuilding the old string from the poles
 * is what keeps the keys of this payload the bytes an outside caller already parses -- for the one
 * axis that has such callers, `- Role-playing \u2194\ufe0f + Assistant-like`, exactly.
 *
 * Falls back to `title` for an axis that names no poles, which is all this view ever had.
 */
function legacyTitle(readout: SteerVectorReadout): string {
  const { polePositive, poleNegative } = readout;
  if (!polePositive || !poleNegative) return readout.title;
  return `- ${poleNegative} \u2194\ufe0f + ${polePositive}`;
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
