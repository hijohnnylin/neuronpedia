import { badRequest } from '@/lib/api-error';
import type { Activation } from '@prisma/client';
import { decodeMixedToken } from '../utils/byte-level-tokens';
import { callJev, ChoiceAnswer, ChoiceQuestion, JEV_SCORE_MODEL_NAME } from './autointerp-scorer-jev';
import { decoyScoreActivationsRaw } from './autointerp-scorer-recall-json';

// Intruder detection (Paulo & Belrose 2025, arXiv 2507.08473): is a feature interpretable at all,
// with no explanation in the loop? Each group shows four texts where the feature fires, with the
// firing tokens marked << >>, and one intruder where it does not, with random tokens marked so
// the marks alone give nothing away. Jev picks the odd one out. The score is the share of groups
// it gets right. Chance is 20%.
//
// Computed on demand and not stored: the result is a property of the feature, not of any
// explanation, so it has no row in ExplanationScore.

export const INTRUDER_GROUPS = 5;
export const ACTIVATING_PER_GROUP = 4;
export const GROUP_SIZE = ACTIVATING_PER_GROUP + 1;
// The paper crops every text to 32 tokens. Longer texts add cost and hide the marked tokens.
export const INTRUDER_MAX_TOKENS = 32;
// Same rule as jev_fuzz: on a 0-10 scale, tokens at or above this are the ones the feature fires on.
const MARK_MIN = 3;
// Dense features clear that bar on most tokens, which marks the whole text. Keep the strongest.
export const MAX_MARKS = 8;

type ActivationLike = Pick<Activation, 'tokens' | 'values'>;

export type IntruderExampleKind = 'top' | 'zero' | 'decoy';

export type IntruderExample = {
  kind: IntruderExampleKind;
  // Cropped to the group's window, centered on the max-activating token.
  tokens: string[];
  values: number[];
  marked_text: string;
  is_intruder: boolean;
};

export type IntruderGroup = {
  examples: IntruderExample[];
  intruder_index: number;
};

export type IntruderGroupResult = IntruderGroup & {
  picked_index: number;
  probabilities: number[];
  confidence: number;
  correct: boolean;
};

export type InterpretabilityCheckResult = {
  model: string;
  input_tokens: number;
  score: number;
  correct: number;
  total: number;
  groups: IntruderGroupResult[];
};

// ---- deterministic randomness, so the same feature gets the same groups on every click ----

function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i += 1) {
    h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  }
  return h >>> 0;
}

// mulberry32
function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled<T>(items: T[], rng: () => number): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// ---- pure helpers (exported for tests) ----

// Takes `length` tokens with the anchor as close to the middle as the bounds allow.
export function cropAround(tokens: string[], values: number[], anchor: number, length: number) {
  const n = Math.min(length, tokens.length);
  let start = anchor - Math.floor(n / 2);
  start = Math.max(0, Math.min(start, tokens.length - n));
  return { tokens: tokens.slice(start, start + n), values: values.slice(start, start + n), anchor: anchor - start };
}

function maxIndex(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

export function markTokens(tokens: string[], marked: Set<number>): string {
  return tokens
    .map((t, i) => (marked.has(i) ? `<<${decodeMixedToken(t)}>>` : decodeMixedToken(t)))
    .join('')
    .replace(/\n/g, ' ');
}

// Indices at or above MARK_MIN on a 0-10 scale, at most MAX_MARKS of them, strongest first. The
// max token is always included.
export function firingIndices(values: number[]): Set<number> {
  const max = Math.max(0, ...values);
  if (max <= 0) return new Set();
  const above = values
    .map((v, i) => ({ v, i }))
    .filter(({ v }) => (v * 10) / max >= MARK_MIN)
    .sort((a, b) => b.v - a.v)
    .slice(0, MAX_MARKS);
  return new Set(above.map(({ i }) => i));
}

// One group: four activating texts and one intruder, cropped to a shared window and shuffled.
// The intruder gets as many random marks as the activating texts have on average (at least one).
export function buildGroup(
  activating: ActivationLike[],
  intruder: ActivationLike,
  intruderKind: IntruderExampleKind,
  rng: () => number,
): IntruderGroup {
  const members = [...activating, intruder];
  const window = Math.min(INTRUDER_MAX_TOKENS, ...members.map((m) => m.tokens.length));

  const cropped = activating.map((a) => {
    const c = cropAround(a.tokens, a.values, maxIndex(a.values), window);
    const marks = firingIndices(c.values);
    marks.add(c.anchor);
    return { kind: 'top' as IntruderExampleKind, tokens: c.tokens, values: c.values, marks };
  });

  // A zero-activation text has no anchor, so it is cropped around its middle; a decoy has a
  // made-up max, which serves as well as any other point.
  const intruderMax = Math.max(0, ...intruder.values);
  const intruderAnchor = intruderMax > 0 ? maxIndex(intruder.values) : Math.floor(intruder.tokens.length / 2);
  const ic = cropAround(intruder.tokens, intruder.values, intruderAnchor, window);
  const avgMarks = cropped.reduce((s, c) => s + c.marks.size, 0) / cropped.length;
  const nMarks = Math.max(1, Math.min(ic.tokens.length, Math.floor(avgMarks)));
  const intruderMarks = new Set(shuffled([...ic.tokens.keys()], rng).slice(0, nMarks));

  const examples: IntruderExample[] = [
    ...cropped.map((c) => ({
      kind: c.kind,
      tokens: c.tokens,
      values: c.values,
      marked_text: markTokens(c.tokens, c.marks),
      is_intruder: false,
    })),
    {
      kind: intruderKind,
      tokens: ic.tokens,
      values: ic.values,
      marked_text: markTokens(ic.tokens, intruderMarks),
      is_intruder: true,
    },
  ];
  const order = shuffled(examples, rng);
  return { examples: order, intruder_index: order.findIndex((e) => e.is_intruder) };
}

// Up to INTRUDER_GROUPS groups. Activating texts are dealt out without repeats until they run
// out, then reshuffled and reused. Intruders come from the zero-activation texts when the feature
// has any, else from the fixed decoys.
export function buildGroups(
  topActivations: ActivationLike[],
  zeroActivations: ActivationLike[],
  seed: string,
): IntruderGroup[] {
  if (topActivations.length === 0) {
    throw badRequest('This feature has no activating example texts stored, so there is nothing to check.');
  }
  const rng = makeRng(hashString(seed));
  const intruderKind: IntruderExampleKind = zeroActivations.length > 0 ? 'zero' : 'decoy';
  const intruderPool: ActivationLike[] = zeroActivations.length > 0 ? zeroActivations : decoyScoreActivationsRaw;
  const intruders = shuffled(intruderPool, rng);

  let deck: ActivationLike[] = [];
  const draw = () => {
    if (deck.length === 0) deck = shuffled(topActivations, rng);
    return deck.pop() as ActivationLike;
  };

  const groups: IntruderGroup[] = [];
  for (let g = 0; g < INTRUDER_GROUPS; g += 1) {
    const activating: ActivationLike[] = [];
    for (let k = 0; k < ACTIVATING_PER_GROUP; k += 1) activating.push(draw());
    groups.push(buildGroup(activating, intruders[g % intruders.length], intruderKind, rng));
  }
  return groups;
}

export function buildQuestions(groups: IntruderGroup[]): Record<string, ChoiceQuestion> {
  const criteria: Record<string, string | null> = {};
  for (let i = 0; i < GROUP_SIZE; i += 1) criteria[String(i)] = null;
  const questions: Record<string, ChoiceQuestion> = {};
  groups.forEach((_, g) => {
    questions[`group_${g}`] = {
      type: 'choice',
      instructions:
        `In \`groups[${g}]\`, four of the five texts share one pattern: the words wrapped in << >> ` +
        'are the same kind of thing, or sit in the same kind of context. One text does not fit. ' +
        'Which index is the odd one out?',
      criteria,
    };
  });
  return questions;
}

// ---- the check ----

export async function checkInterpretability(
  topActivations: ActivationLike[],
  zeroActivations: ActivationLike[],
  seed: string,
): Promise<InterpretabilityCheckResult> {
  const groups = buildGroups(topActivations, zeroActivations, seed);
  const state = { groups: groups.map((g) => g.examples.map((e) => e.marked_text)) };
  const resp = await callJev(state, buildQuestions(groups));

  const results: IntruderGroupResult[] = groups.map((group, g) => {
    const answer = resp.answers[`group_${g}`] as ChoiceAnswer;
    const probabilities = Array.from({ length: GROUP_SIZE }, (_, i) => answer.probabilities[String(i)] || 0);
    const picked = Number.parseInt(answer.choice, 10);
    return {
      ...group,
      picked_index: picked,
      probabilities,
      confidence: answer.confidence,
      correct: picked === group.intruder_index,
    };
  });
  const correct = results.filter((r) => r.correct).length;
  return {
    model: resp.model || JEV_SCORE_MODEL_NAME,
    input_tokens: resp.usage.input_tokens,
    score: Math.round((100 * correct) / results.length),
    correct,
    total: results.length,
    groups: results,
  };
}
