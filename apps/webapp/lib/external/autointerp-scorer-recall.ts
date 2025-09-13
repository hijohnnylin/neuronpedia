/* eslint-disable no-param-reassign */

import { prisma } from '@/lib/db';
import { ActivationPartial } from '@/prisma/generated/zod';
import { Activation, Explanation } from '@prisma/client';
import OpenAI from 'openai';
import { AutoInterpModelType, getAutoInterpModelTypeFromModelId } from '../utils/autointerp';
import { AuthenticatedUser } from '../with-user';
import { makeOaiMessage } from './autointerp-shared';

const openai = new OpenAI();

// only submit tokens within 8 tokens of the top activating token
const TRIM_TO_RANGE_OF_TOP_ACT = 100;

// this is recall-scoring but without JSON. it is currently not used.

const makeFormattedNeuronFromActivation = (act: ActivationPartial) => {
  let toReturn = `

Neuron Activation:
<start>
`;
  act.tokens?.forEach((token, i) => {
    if (act.values && act.values.length >= i - 1) {
      toReturn = `${toReturn + token.replaceAll('▁', ' ')}\t${act.values[i]}\n`;
    }
  });
  toReturn += `<end>

`;
  return toReturn;
};

const activationMatchesExplanation = async (
  activation: ActivationPartial,
  explanation: string,
  scorerModel: string,
) => {
  const systemPromptScore = `We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates, and tell me if the explanation accurately explains what the neuron is looking for.
  
The activation format is token<tab>activation. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.

Tell me if the explanation accurately explains what the neuron is looking for, and why.`;

  const firstUserMessage = `

Neuron Activation:
<start>
 in	0
 the	0
 art	0
 of	0
 impossible	0
 disappearing	6
/	0
re	0
app	0
earing	10
 acts	0
<end>

Explanation: words about disappearing and reappearing
`;
  const firstAssistantMessage =
    'yes, because the tokens disappearing and the combined tokens re, app, earing match the explanation.';
  const secondUserMessage = `

Neuron Activation:
<start>
 with	0
 handheld	0
 cameras	10
 and	0
 apparently	0
 no	0
 movie	0
 lights	0
 by	0
<end>

Explanation: terms related to cats
`;
  const secondAssistantMessage = 'no, because cameras have nothing to do with cats';
  let thirdUserMessage = makeFormattedNeuronFromActivation(activation);
  thirdUserMessage = `${thirdUserMessage}Explanation: ${explanation}\n`;

  if (getAutoInterpModelTypeFromModelId(scorerModel) === AutoInterpModelType.OPENAI) {
    const messages = [
      makeOaiMessage('system', systemPromptScore),
      makeOaiMessage('user', firstUserMessage),
      makeOaiMessage('assistant', firstAssistantMessage),
      makeOaiMessage('user', secondUserMessage),
      makeOaiMessage('assistant', secondAssistantMessage),
      makeOaiMessage('user', thirdUserMessage),
    ];
    const chatCompletion = await openai.chat.completions.create({
      messages,
      model: scorerModel,
      max_tokens: 50,
      temperature: 0,
    });
    let yesOrNoString = chatCompletion.choices[0].message.content;
    console.log(`Tokens: ${activation.tokens?.join('')}`);
    console.log(`Explanation: ${explanation}`);
    console.log(yesOrNoString);
    if (!yesOrNoString) {
      return false;
    }
    yesOrNoString = yesOrNoString.split(',')[0].trim().toLowerCase();
    if (yesOrNoString === 'yes') {
      return true;
    }
    return false;
  }
  throw new Error('Unsupported right now');
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

export const generateScoreRecall = async (
  topActivations: Activation[],
  zeroActivations: Activation[],
  explanation: Explanation,
  scorerModel: string,
  user: AuthenticatedUser,
) => {
  const explanationScoreTypeName = `recall_${scorerModel}_2024v1`;

  // Trim activations to tokens around the max value
  const trimActivation = (activation: Activation) => {
    const maxValueIndex = activation.values.indexOf(Math.max(...activation.values));
    const start = Math.max(0, maxValueIndex - TRIM_TO_RANGE_OF_TOP_ACT / 2);
    const end = Math.min(activation.values.length, maxValueIndex + TRIM_TO_RANGE_OF_TOP_ACT / 2);

    return {
      ...activation,
      values: activation.values.slice(start, end),
      tokens: activation.tokens.slice(start, end),
    };
  };
  topActivations = topActivations.map(trimActivation);
  zeroActivations = zeroActivations.map(trimActivation);

  const totalSamples = topActivations.length + zeroActivations.length + decoyScoreActivationsRaw.length;
  let totalCorrectAnswers = 0;
  const topActPromises: Promise<boolean>[] = [];
  // do all topActivations. these should all return "yes"
  topActivations.forEach((topAct) => {
    const matchPromise = activationMatchesExplanation(topAct, explanation.description, scorerModel);
    topActPromises.push(matchPromise);
  });
  const topActResults = await Promise.all(topActPromises);
  totalCorrectAnswers += topActResults.filter((res) => res).length;

  // do all zero and decoy activations. these should all return "no"
  const zeroActivationsPromises: Promise<boolean>[] = [];
  zeroActivations.forEach((zeroAct) => {
    const matchPromise = activationMatchesExplanation(zeroAct, explanation.description, scorerModel);
    zeroActivationsPromises.push(matchPromise);
  });
  const decoyActivationPromises: Promise<boolean>[] = [];
  decoyScoreActivationsRaw.forEach((decoyAct) => {
    const decoyPromise = activationMatchesExplanation(decoyAct, explanation.description, scorerModel);
    decoyActivationPromises.push(decoyPromise);
  });

  const noActResults = await Promise.all(zeroActivationsPromises.concat(decoyActivationPromises));
  totalCorrectAnswers += noActResults.filter((res) => !res).length;

  const jsonDetails = {
    top: topActivations.map((topAct, i) => ({
      id: topAct.id,
      tokens: topAct.tokens,
      values: topAct.values,
      match: topActResults[i],
    })),
    zero: zeroActivations.map((act, i) => ({
      id: act.id,
      tokens: act.tokens,
      values: act.values,
      match: noActResults[i],
    })),
    decoy: decoyScoreActivationsRaw.map((act, i) => ({
      id: i,
      tokens: act.tokens,
      values: act.values,
      match: noActResults[zeroActivations.length + i],
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
      explanationScoreModelName: scorerModel,
      initiatedByUserId: user.id,
      jsonDetails: JSON.stringify(jsonDetails),
    },
  });
  return savedScore;
};
