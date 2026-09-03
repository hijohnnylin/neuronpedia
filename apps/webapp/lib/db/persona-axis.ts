import { prisma } from '@/lib/db';
import type { PersonaAxisDefinition, PersonaAxisFit } from '@/lib/utils/persona-axis';
import { AuthenticatedUser } from '@/lib/with-user';
import { AllowUnlistedFor, userCanAccessClause } from './userCanAccess';

/** Exactly the columns of `PersonaAxisDefinition`, so the row a query yields is one. */
const DEFINITION_COLUMNS = {
  id: true,
  name: true,
  author: true,
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
  values: true,
  projectionParams: true,
} as const;

/**
 * The rows for these names, subject to what this user may see. One row per name, by constraint.
 *
 * Not filtered by `uses`. Every row projects the same way -- subtract a mean, maybe normalize, dot
 * with the direction, scale, look up a percentile -- so a probe or a plain steering vector reads
 * just as well as an axis does, and `uses` says what a row is for rather than whether it can be
 * read. Refusing one here would have been a display convention denying a caller a number that is
 * perfectly well defined.
 */
const axisClause = (modelId: string, names: string[], user?: AuthenticatedUser | null) => ({
  where: {
    modelId,
    name: { in: names },
    ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
  },
});

/**
 * The rows inference can read: those at exactly one site.
 *
 * This is where a `Vector` becomes a `PersonaAxisDefinition`, and it is deliberately the only place
 * that knows the two differ. `Vector` holds `layers` as an array, because a probe that sums across
 * layers is one row reading at several, while the wire carries a scalar `layer` -- so the narrowing
 * happens once here and everything above this line sees one site.
 *
 * The poles are not checked. A row without them reads perfectly well; it just has nothing to label
 * the ends with, which every shape downstream already allows for.
 *
 * A row reading at several sites is dropped with its own line in the log rather than carried on and
 * re-checked at every use. It should not happen while nothing writes such a row, so one of these in
 * the log means a fit arrived that the wire cannot express yet, not a case to handle here.
 */
type AtOneSite<T> = Omit<T, 'layers'> & { layer: number };

function asAtOneSite<T extends { name: string; layers: number[] }>(rows: T[]): AtOneSite<T>[] {
  const readable: AtOneSite<T>[] = [];
  for (const { layers, ...rest } of rows) {
    const row = rest as Omit<T, 'layers'> & Pick<T, 'name'>;
    if (layers.length === 1) {
      readable.push({ ...row, layer: layers[0] });
    } else {
      console.warn(`Vector ${row.name} reads at ${layers.length} sites, and the read payload takes one`);
    }
  }
  return readable;
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
  return inOrder(modelId, names, asAtOneSite(rows));
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
  return inOrder(modelId, names, asAtOneSite(rows));
}
