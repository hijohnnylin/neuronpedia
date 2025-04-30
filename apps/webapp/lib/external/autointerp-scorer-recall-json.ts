import { prisma } from '@/lib/db';
import { ActivationPartial } from '@/prisma/generated/zod';
import { Activation, Explanation, ExplanationScoreModel, UserSecretType } from '@prisma/client';
import OpenAI from 'openai';
import { ERROR_RECALL_ALT_FAILED, OPENROUTER_BASE_URL } from '../utils/autointerp';
import { AuthenticatedUser } from '../with-user';
import { makeOaiMessage } from './autointerp-shared';

const ERROR_ON_RECALL_REQUEST = 'Error';

type ScoreMatchAndRawQuery = {
  match: boolean;
  reason: string;
  messages: string;
};

const decoyScoreActivationsRaw = [
  {
    tokens: [
      'Sources',
      ':',
      ' Cowboys',
      "'",
      ' Dak',
      ' Prescott',
      ' agrees',
      ' to',
      ' four',
      '-year',
      ',',
      ' $',
      '136',
      'M',
      ' deal',
    ],
    values: [0, 0, 10, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
  },
  {
    tokens: [
      ' The',
      ' second',
      ' element',
      ' is',
      ' the',
      ' ',
      '1',
      '0',
      '-',
      'residue',
      ' MT',
      'ase',
      '-',
      'Rd',
      'RP',
      ' linker',
      ' that',
      ' overall',
      ' exhibits',
      ' low',
    ],
    values: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0],
  },
  {
    tokens: [
      'The',
      ' effect',
      ' of',
      ' a',
      ' good',
      ' metabolic',
      ' control',
      ' in',
      ' the',
      ' natural',
      ' history',
      ' of',
      ' diabetic',
      ' retin',
      'opathy',
      ' is',
      ' discussed',
    ],
    values: [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  },
  {
    tokens: [
      'Category',
      ':',
      'Military',
      ' history',
      ' of',
      ' the',
      ' Soviet',
      ' Union',
      ' during',
      ' World',
      ' War',
      ' II',
    ],
    values: [0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0],
  },
  {
    tokens: [
      ' Wells',
      ' Fargo',
      ' issued',
      ' three',
      ' Forms',
      ' ',
      '1',
      '0',
      '9',
      '9',
      '-',
      'C',
      ',',
      ' Cancellation',
      ' of',
      ' Debt',
      ', ',
      ' to',
    ],
    values: [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  },
];

const makeJsonActivationAndExplanation = (act: ActivationPartial, explanation: string) =>
  JSON.stringify({
    activations: act.tokens?.map((token, i) => ({
      token: token.replaceAll('▁', ' '),
      activation: act.values && act.values[i],
    })),
    explanation,
  });

const activationMatchesExplanation = async (
  activation: ActivationPartial,
  explanation: string,
  scorerModel: string,
  explanationScoreModelOpenRouterId: string | null,
  scorerKeyType: UserSecretType,
  scorerKey: string,
): Promise<ScoreMatchAndRawQuery> => {
  const systemPromptScore = `We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates, and tell me if the explanation accurately explains what the neuron is looking for.
  
The activation format is in JSON. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.

Only respond with in JSON with the format {"match": true, "reason": [reason why it matches]} or {"match": false, "reason": [reason why it doesn't match] }.`;

  const firstUserMessage = makeJsonActivationAndExplanation(
    {
      tokens: [' in', ' the', ' art', ' of', ' impossible', ' disappearing', '/', 're', 'app', 'earing', ' acts'],
      values: [0, 0, 0, 0, 0, 6, 0, 5, 8, 8, 0],
    },
    'words about disappearing and reappearing',
  );
  const firstAssistantMessage = `{"match": true, "reason": "the tokens disappearing and the combined tokens re, app, earing, which form the word reappearing, match the explanation."}`;
  const secondUserMessage = makeJsonActivationAndExplanation(
    {
      tokens: [' with', ' handheld', ' cameras', ' and', ' apparently', ' no', ' movie', ' lights', ' by'],
      values: [0, 0, 10, 0, 0, 0, 0, 0, 0],
    },
    'terms related to cats',
  );
  const secondAssistantMessage = `{"match": false, "reason": "cameras has nothing to do with cats}`;
  const thirdUserMessage = makeJsonActivationAndExplanation(activation, explanation);

  // console.log("scorerKeyType: " + scorerKeyType);
  // console.log(
  //   "explanationScoreModelOpenRouterId: " + explanationScoreModelOpenRouterId,
  // );
  const openai = new OpenAI({
    baseURL: scorerKeyType === UserSecretType.OPENROUTER ? OPENROUTER_BASE_URL : undefined,
    apiKey: scorerKey,
  });
  const messages = [
    makeOaiMessage('system', systemPromptScore),
    makeOaiMessage('user', firstUserMessage),
    makeOaiMessage('assistant', firstAssistantMessage),
    makeOaiMessage('user', secondUserMessage),
    makeOaiMessage('assistant', secondAssistantMessage),
    makeOaiMessage('user', thirdUserMessage),
  ];
  const rawMessages = JSON.stringify(messages);
  try {
    const chatCompletion = await openai.chat.completions.create({
      messages,
      model: explanationScoreModelOpenRouterId || scorerModel,
      max_tokens: 5000,
      response_format: { type: 'json_object' },
      temperature: 0,
    });
    const matchResponse = chatCompletion.choices[0].message.content;
    // console.log("Tokens: " + activation.tokens?.join(""));
    // console.log("Explanation: " + explanation);
    // console.log(matchResponse);
    if (!matchResponse) {
      return { match: false, reason: 'Errored', messages: rawMessages };
    }
    try {
      const matchResponseParsed = JSON.parse(matchResponse) as {
        match: boolean;
        reason: string;
      };
      if (matchResponseParsed) {
        return {
          match: matchResponseParsed.match,
          reason: matchResponseParsed.reason,
          messages: rawMessages,
        };
      }
      // invalid json response, consider this a failure
      return {
        match: false,
        reason: 'Invalid JSON response format',
        messages: rawMessages,
      };
    } catch {
      // invalid json response, consider this a failure
      return {
        match: false,
        reason: 'Invalid JSON response',
        messages: rawMessages,
      };
    }
  } catch (e) {
    console.log(e);
    return {
      match: false,
      reason: ERROR_ON_RECALL_REQUEST,
      messages: rawMessages,
    };
  }
};

export const generateScoreRecallAlt = async (
  topActivations: Activation[],
  zeroActivations: Activation[],
  explanation: Explanation,
  scorerModel: ExplanationScoreModel,
  explanationScoreModelOpenRouterId: string | null,
  user: AuthenticatedUser,
  scorerKeyType: UserSecretType,
  scorerKey: string,
) => {
  const explanationScoreTypeName = 'recall_alt';

  const totalSamples = topActivations.length + zeroActivations.length + decoyScoreActivationsRaw.length;
  let totalCorrectAnswers = 0;
  const topActPromises: Promise<ScoreMatchAndRawQuery>[] = [];
  // do all topActivations. these should all return "yes"
  topActivations.forEach((topAct) => {
    const matchPromise = activationMatchesExplanation(
      topAct,
      explanation.description,
      scorerModel.name,
      explanationScoreModelOpenRouterId,
      scorerKeyType,
      scorerKey,
    );
    topActPromises.push(matchPromise);
  });
  const topActResults = await Promise.all(topActPromises);
  // check if all are "Error" -
  const allErrors = topActResults.every(
    (result) =>
      typeof result === 'object' &&
      result !== null &&
      result.match === false &&
      result.reason === ERROR_ON_RECALL_REQUEST,
  );

  if (allErrors) {
    throw new Error(ERROR_RECALL_ALT_FAILED);
  }

  totalCorrectAnswers += topActResults.filter((res) => res.match).length;

  // do all zero and decoy activations. these should all return "no"
  const zeroActivationsPromises: Promise<ScoreMatchAndRawQuery>[] = [];
  zeroActivations.forEach((zeroAct) => {
    const matchPromise = activationMatchesExplanation(
      zeroAct,
      explanation.description,
      scorerModel.name,
      explanationScoreModelOpenRouterId,
      scorerKeyType,
      scorerKey,
    );
    zeroActivationsPromises.push(matchPromise);
  });
  const decoyActivationPromises: Promise<ScoreMatchAndRawQuery>[] = [];
  decoyScoreActivationsRaw.forEach((decoyAct) => {
    const decoyPromise = activationMatchesExplanation(
      decoyAct,
      explanation.description,
      scorerModel.name,
      explanationScoreModelOpenRouterId,
      scorerKeyType,
      scorerKey,
    );
    decoyActivationPromises.push(decoyPromise);
  });

  const noActResults = await Promise.all(zeroActivationsPromises.concat(decoyActivationPromises));
  totalCorrectAnswers += noActResults.filter((res) => !res.match).length;

  const jsonDetails = {
    top: topActivations.map((topAct, i) => ({
      id: topAct.id,
      tokens: topAct.tokens,
      values: topAct.values,
      match: topActResults[i].match,
      reason: topActResults[i].reason,
      messages: topActResults[i].messages,
    })),
    zero: zeroActivations.map((act, i) => ({
      id: act.id,
      tokens: act.tokens,
      values: act.values,
      match: noActResults[i].match,
      reason: noActResults[i].reason,
      messages: noActResults[i].messages,
    })),
    decoy: decoyScoreActivationsRaw.map((act, i) => ({
      id: i,
      tokens: act.tokens,
      values: act.values,
      match: noActResults[zeroActivations.length + i].match,
      reason: noActResults[zeroActivations.length + i].reason,
      messages: noActResults[zeroActivations.length + i].messages,
    })),
  };

  // calculate score out of 100
  const score = totalCorrectAnswers / totalSamples;

  // save to DB
  const savedScore = await prisma.explanationScore.create({
    data: {
      value: score,
      explanationId: explanation.id,
      explanationScoreTypeName,
      explanationScoreModelName: scorerModel.name,
      initiatedByUserId: user.id,
      jsonDetails: JSON.stringify(jsonDetails),
    },
  });
  return savedScore;
};
