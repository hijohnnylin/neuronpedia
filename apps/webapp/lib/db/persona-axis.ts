import { prisma } from '@/lib/db';
import type { PersonaAxisDefinition, PersonaAxisFit } from '@/lib/utils/persona-axis';
import { AuthenticatedUser } from '@/lib/with-user';
import { ProjectionType } from '@prisma/client';
import { AllowUnlistedFor, userCanAccessClause } from './userCanAccess';

/** Exactly the columns of `PersonaAxisDefinition`, so the row a query yields is one. */
const DEFINITION_COLUMNS = {
  id: true,
  name: true,
  layers: true,
  polePositive: true,
  poleNegative: true,
  polePositiveDescription: true,
  poleNegativeDescription: true,
  displayName: true,
  caveat: true,
} as const;

/** The above plus the numbers, which is `PersonaAxisFit`. */
const FIT_COLUMNS = {
  ...DEFINITION_COLUMNS,
  author: true,
  values: true,
  projectionParams: true,
} as const;

/**
 * The rows for these names, subject to what this user may see. One row per name, by constraint.
 *
 * Restricted to `AXIS_PROJECTION` because everything downstream of here -- `personaAxisToNPAxis`,
 * the payload it builds, the chart that reads the result -- assumes that shape. A row of another
 * type would otherwise be read with its keys defaulted rather than refused, and measure something
 * plausible and wrong.
 */
const axisClause = (modelId: string, names: string[], user?: AuthenticatedUser | null) => ({
  where: {
    modelId,
    name: { in: names },
    projectionType: ProjectionType.AXIS_PROJECTION,
    ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
  },
});

/**
 * The rows that can be read as an axis: both poles named, and exactly one site.
 *
 * This is where a `Vector` becomes a `PersonaAxisDefinition`, and it is deliberately the only place
 * that knows the two differ. `Vector` holds directions of every kind, so it names its poles
 * optionally -- one without them is a probe or a plain steering vector, which nothing downstream
 * can label a reading from -- and it holds `layers` as an array, because a probe reads at several.
 * An axis is the narrow case of both, so both are checked once here and everything above this line
 * sees a scalar `layer` and two poles that are certainly there.
 *
 * A row failing either check is dropped with its own line in the log rather than carried on and
 * re-checked at every use. Neither should happen -- the query already asks for AXIS_PROJECTION --
 * so one of these in the log means a bad write, not a case to handle.
 */
type AxisShaped<T> = Omit<T, 'layers'> & { layer: number; polePositive: string; poleNegative: string };

function asAxes<T extends { name: string; layers: number[]; polePositive: string | null; poleNegative: string | null }>(
  rows: T[],
): AxisShaped<T>[] {
  const axes: AxisShaped<T>[] = [];
  for (const { layers, ...rest } of rows) {
    const row = rest as Omit<T, 'layers'> & Pick<T, 'name' | 'polePositive' | 'poleNegative'>;
    if (row.polePositive === null || row.poleNegative === null) {
      console.warn(`Vector ${row.name} names no poles, so it cannot be read as an axis`);
    } else if (layers.length !== 1) {
      console.warn(`Vector ${row.name} reads at ${layers.length} sites, but an axis reads at one`);
    } else {
      axes.push({ ...row, layer: layers[0], polePositive: row.polePositive, poleNegative: row.poleNegative });
    }
  }
  return axes;
}

/**
 * The rows in the order the names were asked for.
 *
 * A name with no row is left out and logged. The caller is then one axis short, which is visible on
 * the page and diagnosable in the log; the alternative -- carrying the name through as a
 * placeholder -- would put an axis on screen that nothing can measure.
 */
function inOrder<T extends { name: string }>(modelId: string, names: string[], rows: T[]): T[] {
  const byName = new Map(rows.map((row) => [row.name, row]));
  const missing = names.filter((name) => !byName.has(name));
  if (missing.length > 0) {
    console.warn(`No Vector row for ${modelId}: ${missing.join(', ')}`);
  }
  return names.map((name) => byName.get(name)).filter((row): row is T => row !== undefined);
}

/** What a page needs to draw the axes named by `names`: the labels, without the vectors. */
export async function getPersonaAxisDefinitions(
  modelId: string,
  names: string[],
  user?: AuthenticatedUser | null,
): Promise<PersonaAxisDefinition[]> {
  if (names.length === 0) return [];
  const rows = await prisma.vector.findMany({
    ...axisClause(modelId, names, user),
    select: DEFINITION_COLUMNS,
  });
  return inOrder(modelId, names, asAxes(rows));
}

/**
 * What inference needs to measure the axes named by `names`: the whole fit, vectors included.
 *
 * Server-side only, and the reason `PersonaAxisFit` is a type of its own -- one of these is a few
 * hundred kilobytes of floats, so it belongs on the path from the route to the inference server
 * and nowhere near a page's props.
 */
export async function getPersonaAxisFits(
  modelId: string,
  names: string[],
  user?: AuthenticatedUser | null,
): Promise<PersonaAxisFit[]> {
  if (names.length === 0) return [];
  const rows = await prisma.vector.findMany({
    ...axisClause(modelId, names, user),
    select: FIT_COLUMNS,
  });
  return inOrder(modelId, names, asAxes(rows));
}
