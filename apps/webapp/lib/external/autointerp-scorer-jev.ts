import { ApiError, badRequest, upstreamError } from '@/lib/api-error';
import { prisma } from '@/lib/db';
import { TYPESAFE_API_KEY } from '@/lib/env';
import { Activation, Explanation } from '@prisma/client';
import { decodeMixedToken } from '../utils/byte-level-tokens';
import { AuthenticatedUser } from '../with-user';
import { decoyScoreActivationsRaw } from './autointerp-scorer-recall-json';

// Explanation scorers backed by TypeSafe's Jev (https://docs.typesafe.ai). Jev answers typed
// questions about a JSON `state` with calibrated probabilities instead of generated text, and
// evaluates every question in one request in parallel, so one explanation costs one request.
//
// Three score types share this file:
// - jev_detection: one yes/no question per example on the plain text. Value = balanced accuracy.
// - jev_fuzz: the same, but the tokens the feature fires on are wrapped in << >>. Value =
//   balanced accuracy. Catches explanations that are true of any text ("the word 'the'").
// - jev_score: one 5-level rating of the explanation against the top examples. Value = level / 4.
//
// All three also ask whether the feature's top output logits fit the explanation and store the
// answer in jsonDetails as `logit_fit`. It never enters `value`: for many features the top logits
// are byte-pair fragments, and folding that in penalizes a correct explanation.

export const JEV_SCORE_MODEL_NAME = 'jev-latest';
export const JEV_SCORE_TYPE_NAMES = ['jev_fuzz', 'jev_detection', 'jev_score'] as const;
export type JevScoreTypeName = (typeof JEV_SCORE_TYPE_NAMES)[number];
export const isJevScoreType = (name: string): name is JevScoreTypeName =>
  (JEV_SCORE_TYPE_NAMES as readonly string[]).includes(name);

const TYPESAFE_API_URL = 'https://api.typesafe.ai/v1/systemone';
const REQUEST_TIMEOUT_MS = 30_000;
const MAX_ATTEMPTS = 3;
// The route normalizes activations so the max token is 10. Tokens at or above this are the ones
// the fuzz prompt marks; marking every non-zero token drowns the signal in function words.
const FUZZ_MARK_MIN = 3;
export const NOUL_THRESHOLD = 0.5;
const HOLISTIC_MAX_EXAMPLES = 12;
const MAX_TOP_LOGITS = 10;

export const JEV_SCORE_LEVELS = [
  'Does not describe the marked tokens at all.',
  'Describes a few of the marked tokens. Most do not fit.',
  'Describes about half of the marked tokens.',
  'Describes most of the marked tokens, with minor gaps or being too broad.',
  'Describes the marked tokens perfectly and specifically.',
];

// ---- wire types (hand-written: TypeSafe publishes no OpenAPI spec) ----

type NoulQuestion = { type: 'noul'; instructions: string; criteria?: { true: string; false: string } };
type ScoreQuestion = { type: 'score'; instructions: string; criteria: string[] };
type JevQuestion = NoulQuestion | ScoreQuestion;

type NoulAnswer = { type: 'noul'; noul: number };
type ScoreAnswer = {
  type: 'score';
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
};
type JevResponse = {
  model: string;
  answers: Record<string, NoulAnswer | ScoreAnswer>;
  usage: { input_tokens: number; output_tokens: number };
};

// ---- stored jsonDetails shapes. Keys are snake_case and must stay stable: rows are read back. ----

export type JevExampleKind = 'top' | 'zero' | 'decoy';

export type JevLogitFit = { noul: number; top_logits: string[] } | null;

export type JevFuzzDetectionJsonDetails = {
  model: string;
  threshold: number;
  input_tokens: number;
  examples: {
    kind: JevExampleKind;
    text: string;
    marked_text: string;
    str_tokens: string[];
    activations: number[];
    ground_truth: boolean;
    noul: number;
    prediction: boolean;
    correct: boolean;
  }[];
  logit_fit: JevLogitFit;
};

export type JevHolisticJsonDetails = {
  model: string;
  input_tokens: number;
  levels: string[];
  score: number;
  confidence: number;
  probabilities: Record<string, number>;
  examples: {
    text: string;
    marked_text: string;
    str_tokens: string[];
    activations: number[];
  }[];
  logit_fit: JevLogitFit;
};

// ---- pure helpers (exported for tests) ----

export function plainText(tokens: string[]): string {
  return tokens.map(decodeMixedToken).join('').replace(/\n/g, ' ');
}

// Picks one word by a hash of the text, so the choice is stable across runs.
function markOneWord(text: string): string {
  const words = text.split(' ');
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) % 1_000_003;
  }
  const k = hash % words.length;
  words[k] = `<<${words[k]}>>`;
  return words.join(' ');
}

// Wraps the tokens the feature fires on in << >>. A text with no activation still gets one
// marked word, so the marker itself is not the signal that separates positives from negatives.
export function markedText(tokens: string[], values: number[]): string {
  const max = Math.max(0, ...values);
  if (max <= 0) {
    return markOneWord(plainText(tokens));
  }
  return tokens
    .map((token, i) => {
      const decoded = decodeMixedToken(token);
      return ((values[i] || 0) * 10) / max >= FUZZ_MARK_MIN ? `<<${decoded}>>` : decoded;
    })
    .join('')
    .replace(/\n/g, ' ');
}

// Mean of the true-positive rate and the true-negative rate, in [0, 1]. Chance is 0.5.
export function balancedAccuracy(predictions: boolean[], truths: boolean[]): number {
  let tp = 0;
  let tn = 0;
  let pos = 0;
  let neg = 0;
  truths.forEach((truth, i) => {
    if (truth) {
      pos += 1;
      if (predictions[i]) tp += 1;
    } else {
      neg += 1;
      if (!predictions[i]) tn += 1;
    }
  });
  if (pos === 0 || neg === 0) {
    return 0;
  }
  return 0.5 * (tp / pos + tn / neg);
}

// ---- transport ----

const sleep = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

async function callJev(state: unknown, questions: Record<string, JevQuestion>): Promise<JevResponse> {
  if (!TYPESAFE_API_KEY) {
    throw new ApiError(503, 'Jev scoring is not configured on this server.');
  }
  const body = JSON.stringify({ state, model: JEV_SCORE_MODEL_NAME, questions });
  for (let attempt = 0; ; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(TYPESAFE_API_URL, {
        method: 'POST',
        headers: { Authorization: `Bearer ${TYPESAFE_API_KEY}`, 'Content-Type': 'application/json' },
        body,
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
    } catch (e) {
      throw upstreamError('typesafe', e);
    }
    if (res.ok) {
      return (await res.json()) as JevResponse;
    }
    // The docs ask for backoff on 429 and 529; anything else is not going to change on retry.
    if ((res.status === 429 || res.status === 529) && attempt < MAX_ATTEMPTS - 1) {
      await sleep(1000 * 2 ** attempt);
      continue;
    }
    throw upstreamError('typesafe', { status: res.status, body: await res.text() });
  }
}

// Secondary signal, stored but never part of `value`. Best-effort: a failure here must not
// throw away a score that already succeeded.
async function logitFit(explanation: string, topLogits: string[]): Promise<JevLogitFit> {
  const logits = topLogits
    .slice(0, MAX_TOP_LOGITS)
    .map(decodeMixedToken)
    .filter((t) => t.trim().length > 0);
  if (logits.length === 0) {
    return null;
  }
  try {
    const resp = await callJev(
      { explanation, top_output_tokens: logits },
      {
        fit: {
          type: 'noul',
          instructions:
            '`top_output_tokens` are the next-token predictions a feature promotes. ' +
            'Do most of the tokens in `top_output_tokens` fit the topic described in `explanation`?',
        },
      },
    );
    return { noul: (resp.answers.fit as NoulAnswer).noul, top_logits: logits };
  } catch (e) {
    console.error('jev logit_fit failed', e);
    return null;
  }
}

// ---- scorers ----

async function scoreFuzzOrDetection(
  type: 'jev_fuzz' | 'jev_detection',
  topActivations: Activation[],
  zeroActivations: Activation[],
  explanation: Explanation,
  topLogits: string[],
  user: AuthenticatedUser,
) {
  if (topActivations.length === 0) {
    throw badRequest('Scoring with this method needs activating example texts, and this feature has none stored.');
  }
  const isFuzz = type === 'jev_fuzz';
  const examples = [
    ...topActivations.map((a) => ({ kind: 'top' as JevExampleKind, tokens: a.tokens, values: a.values, truth: true })),
    ...zeroActivations.map((a) => ({
      kind: 'zero' as JevExampleKind,
      tokens: a.tokens,
      values: a.values,
      truth: false,
    })),
    ...decoyScoreActivationsRaw.map((a) => ({
      kind: 'decoy' as JevExampleKind,
      tokens: a.tokens,
      values: a.values,
      truth: false,
    })),
  ].map((e) => ({ ...e, text: plainText(e.tokens), marked_text: markedText(e.tokens, e.values) }));

  const instructions = isFuzz
    ? 'Do the words wrapped in << >> in `examples[{i}]` match the description in `explanation`?'
    : 'Does `examples[{i}]` contain any word or phrase that matches the neuron explanation in `explanation`?';
  const questions: Record<string, JevQuestion> = {};
  examples.forEach((_, i) => {
    questions[`ex_${i}`] = { type: 'noul', instructions: instructions.replace('{i}', String(i)) };
  });
  const state = {
    explanation: explanation.description,
    examples: examples.map((e) => (isFuzz ? e.marked_text : e.text)),
  };
  const resp = await callJev(state, questions);

  const scored = examples.map((e, i) => {
    const { noul } = resp.answers[`ex_${i}`] as NoulAnswer;
    const prediction = noul > NOUL_THRESHOLD;
    return {
      kind: e.kind,
      text: e.text,
      marked_text: e.marked_text,
      str_tokens: e.tokens,
      activations: e.values,
      ground_truth: e.truth,
      noul,
      prediction,
      correct: prediction === e.truth,
    };
  });
  const value = balancedAccuracy(
    scored.map((s) => s.prediction),
    scored.map((s) => s.ground_truth),
  );

  const jsonDetails: JevFuzzDetectionJsonDetails = {
    model: resp.model,
    threshold: NOUL_THRESHOLD,
    input_tokens: resp.usage.input_tokens,
    examples: scored,
    logit_fit: await logitFit(explanation.description, topLogits),
  };

  return prisma.explanationScore.create({
    data: {
      value,
      explanationId: explanation.id,
      explanationScoreTypeName: type,
      explanationScoreModelName: JEV_SCORE_MODEL_NAME,
      initiatedByUserId: user.id,
      jsonDetails: JSON.stringify(jsonDetails),
    },
  });
}

async function scoreHolistic(
  topActivations: Activation[],
  explanation: Explanation,
  topLogits: string[],
  user: AuthenticatedUser,
) {
  if (topActivations.length === 0) {
    throw badRequest('Scoring with this method needs activating example texts, and this feature has none stored.');
  }
  const examples = topActivations.slice(0, HOLISTIC_MAX_EXAMPLES).map((a) => ({
    text: plainText(a.tokens),
    marked_text: markedText(a.tokens, a.values),
    str_tokens: a.tokens,
    activations: a.values,
  }));
  const resp = await callJev(
    { explanation: explanation.description, examples: examples.map((e) => e.marked_text) },
    {
      rating: {
        type: 'score',
        instructions:
          'In each string in `examples`, the tokens wrapped in << >> are where one neural network feature ' +
          'fires. How well does `explanation` describe the marked tokens across `examples`?',
        criteria: JEV_SCORE_LEVELS,
      },
    },
  );
  const answer = resp.answers.rating as ScoreAnswer;
  const value = answer.score / (JEV_SCORE_LEVELS.length - 1);

  const jsonDetails: JevHolisticJsonDetails = {
    model: resp.model,
    input_tokens: resp.usage.input_tokens,
    levels: JEV_SCORE_LEVELS,
    score: answer.score,
    confidence: answer.confidence,
    probabilities: answer.probabilities,
    examples,
    logit_fit: await logitFit(explanation.description, topLogits),
  };

  return prisma.explanationScore.create({
    data: {
      value,
      explanationId: explanation.id,
      explanationScoreTypeName: 'jev_score',
      explanationScoreModelName: JEV_SCORE_MODEL_NAME,
      initiatedByUserId: user.id,
      jsonDetails: JSON.stringify(jsonDetails),
    },
  });
}

export const generateScoreJev = async (
  type: JevScoreTypeName,
  topActivations: Activation[],
  zeroActivations: Activation[],
  explanation: Explanation,
  topLogits: string[],
  user: AuthenticatedUser,
) => {
  if (type === 'jev_score') {
    return scoreHolistic(topActivations, explanation, topLogits, user);
  }
  return scoreFuzzOrDetection(type, topActivations, zeroActivations, explanation, topLogits, user);
};
