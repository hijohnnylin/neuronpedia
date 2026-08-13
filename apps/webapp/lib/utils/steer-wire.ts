/**
 * Reads and writes the shapes persisted in `SteerOutput`, whose older rows were written under
 * semantics the current wire no longer uses. Keeping those decisions in one file means the storage
 * format is something we state explicitly, rather than a property of whichever codegen template or
 * inference version happened to be current when a row was saved.
 *
 * Assistant-axis payloads are stored snake_case in `capMonitorOutput`: the rows predate the wire
 * being camelCase and are not migrated, so reads must accept snake_case and writes must keep
 * producing it. Rows written more recently can already be camelCase, so reads accept either and
 * normalize. These replace the generated client's `ToJSON`/`FromJSON` helpers, which performed the
 * same conversion as a side effect of its naming scheme.
 */
import type { SteerAssistantAxis } from '@/lib/api/inference-types';
import { STEER_COMPLETION_VERSION } from '@/lib/utils/steer';

type PcValues = Record<string, number>;

/** The snake_case shape persisted in `capMonitorOutput`. */
type StoredAssistantAxisTurn = {
  pc_values?: PcValues | null;
  pc_values_post_cap?: PcValues | null;
  snippet?: string | null;
};

type StoredAssistantAxis = {
  pc_titles?: string[] | null;
  turns?: StoredAssistantAxisTurn[] | null;
  type?: string | null;
};

/** Either casing, since rows written before and after the wire changed both exist. */
type EitherTurn = {
  pc_values?: PcValues | null;
  pc_values_post_cap?: PcValues | null;
  pcValues?: PcValues | null;
  pcValuesPostCap?: PcValues | null;
  snippet?: string | null;
};

type EitherAxis = {
  pc_titles?: string[] | null;
  pcTitles?: string[] | null;
  turns?: EitherTurn[] | null;
  type?: string | null;
};

/** Read a stored axis, tolerating rows written in either casing. */
export function assistantAxisFromStored(stored: unknown): SteerAssistantAxis {
  const row = (stored ?? {}) as EitherAxis;
  return {
    pcTitles: row.pcTitles ?? row.pc_titles ?? undefined,
    turns: (row.turns ?? []).map((turn) => ({
      pcValues: turn.pcValues ?? turn.pc_values ?? undefined,
      pcValuesPostCap: turn.pcValuesPostCap ?? turn.pc_values_post_cap ?? undefined,
      snippet: turn.snippet ?? undefined,
    })),
    type: (row.type ?? undefined) as SteerAssistantAxis['type'],
  };
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

/** Write an axis in the stored snake_case shape, so existing readers keep working. */
export function assistantAxisToStored(axis: SteerAssistantAxis): StoredAssistantAxis {
  return {
    pc_titles: axis.pcTitles ?? undefined,
    turns: (axis.turns ?? []).map((turn) => ({
      pc_values: turn.pcValues ?? undefined,
      pc_values_post_cap: turn.pcValuesPostCap ?? undefined,
      snippet: turn.snippet ?? undefined,
    })),
    type: axis.type ?? undefined,
  };
}
