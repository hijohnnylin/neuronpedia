/**
 * The `axes` shape `/api/steer-chat` returns, as a type this repository owns.
 *
 * `/api/*` has real users and its field names are a contract of their own, so an inference-side
 * rename must not reach them. This field used to be the inference `SteerVectorReadout[]` assigned
 * straight onto the response, which meant the contract was whatever the pydantic model happened to
 * say that week -- the one arrangement the root `AGENTS.md` names as the way to break a public
 * surface silently. Declaring the shape here and mapping into it field by field means a rename
 * upstream fails `tsc` at the mapper instead of shipping.
 *
 * The names and the nullability match what the endpoint already emitted, so this is a refactor with
 * no wire change. That is also why each field is copied rather than defaulted: `null` and absent are
 * different bytes after `JSON.stringify`, and inference sends both.
 *
 * The deprecated `assistant_axis` view of the same readouts is `steer-axis-legacy.ts`; what gets
 * persisted is `steer-wire.ts`. Three shapes because they have three sets of readers, and only this
 * one is the current public answer.
 */
import type { SteerReadoutTurn, SteerVectorReadout } from '@/lib/api/inference-types';

/** One assistant turn's reading, as a value in the axis's own units and as a percentile. */
export type PublicAxisTurn = {
  value?: number | null;
  valuePostCap?: number | null;
  percentile?: number | null;
  percentilePostCap?: number | null;
  snippet?: string | null;
};

/** One axis read across the assistant's turns, for one steer type. */
export type PublicAxisReadout = {
  id: string;
  author: string;
  title: string;
  type?: SteerVectorReadout['type'];
  layer?: number | null;
  caveat?: string | null;
  polePositive?: string | null;
  poleNegative?: string | null;
  polePositiveDescription?: string | null;
  poleNegativeDescription?: string | null;
  sourceRevision?: string | null;
  turns?: PublicAxisTurn[] | null;
};

function toPublicTurn(turn: SteerReadoutTurn): PublicAxisTurn {
  return {
    value: turn.value,
    valuePostCap: turn.valuePostCap,
    percentile: turn.percentile,
    percentilePostCap: turn.percentilePostCap,
    snippet: turn.snippet,
  };
}

/** Inference readouts as the public `axes` field. */
export function axisReadoutsToPublic(readouts: SteerVectorReadout[]): PublicAxisReadout[] {
  return readouts.map((readout) => ({
    id: readout.id,
    author: readout.author,
    title: readout.title,
    type: readout.type,
    layer: readout.layer,
    caveat: readout.caveat,
    polePositive: readout.polePositive,
    poleNegative: readout.poleNegative,
    polePositiveDescription: readout.polePositiveDescription,
    poleNegativeDescription: readout.poleNegativeDescription,
    sourceRevision: readout.sourceRevision,
    // Mapped rather than passed through: a turn is where a field would be added, and a spread here
    // would carry a new inference field into the public response without anyone deciding to.
    turns: readout.turns ? readout.turns.map(toPublicTurn) : readout.turns,
  }));
}
