import { prisma } from '@/lib/db';
import { Activation, Explanation, ExplanationModelType } from '@prisma/client';
import { AuthenticatedUser } from '../types/auth';
import { AUTOINTERP_SERVER_API } from '../utils/autointerp';

const ELEUTHER_EMBEDDING_MODEL_NAME = 'stella_en_400m_v5';

export const generateScoreEleuther = async (
  type: 'fuzz' | 'detection' | 'embedding',
  activations: Activation[],
  zeroActivations: Activation[],
  explanation: Explanation,
  explanationModel: ExplanationModelType,
  user: AuthenticatedUser,
  explainerKey: string,
) => {
  if (!explanationModel.openRouterModelId) {
    throw new Error('Explaining using np-auto-interp requires an OpenRouter model id.');
  }
  const bareActivations = activations
    .concat(zeroActivations)
    .map((act) => ({ tokens: act.tokens, values: act.values }));

  if (type === 'fuzz' || type === 'detection') {
    const explanationScoreTypeName = `eleuther_${type === 'fuzz' ? 'fuzz' : 'recall'}`;
    const data = await AUTOINTERP_SERVER_API.scoreFuzzDetectionPost({
      scoreFuzzDetectionPostRequest: {
        type: type === 'fuzz' ? 'FUZZ' : 'DETECTION',
        activations: bareActivations,
        explanation: explanation.description || '',
        model: explanationModel.openRouterModelId,
        openrouterKey: explainerKey,
      },
    });

    const converted = data.breakdown.map((b) => ({
      text: b.strTokens?.join('') || '',
      distance: b.distance,
      ground_truth: b.groundTruth,
      prediction: b.prediction,
      highlighted: b.highlighted,
      correct: b.correct,
      probability: b.probability,
      conditional_probability: b.probability,
      activations: b.activations,
      str_tokens: b.strTokens,
    }));

    // save to DB
    const savedScore = await prisma.explanationScore.create({
      data: {
        value: data.score,
        explanationId: explanation.id,
        explanationScoreTypeName,
        explanationScoreModelName: explanationModel.name,
        initiatedByUserId: user.id,
        jsonDetails: JSON.stringify(converted),
      },
    });
    return savedScore;
  }
  if (type === 'embedding') {
    const explanationScoreTypeName = 'eleuther_embedding';
    const data = await AUTOINTERP_SERVER_API.scoreEmbeddingPost({
      scoreEmbeddingPostRequest: {
        activations: bareActivations,
        explanation: explanation.description || '',
      },
    });

    // for each breakdown, find the corresponding text
    // convert to the format we expect
    const converted = data.breakdown.map((b) => {
      // find the matching text
      let matchingAct = bareActivations.find((a) => a.tokens.join('') === b.text);
      if (!matchingAct) {
        matchingAct = { tokens: [], values: [] };
      }
      return {
        text: b.text,
        distance: b.distance,
        similarity: b.similarity,
        activations: matchingAct.values,
        str_tokens: matchingAct.tokens,
      };
    });

    // save to DB
    const savedScore = await prisma.explanationScore.create({
      data: {
        value: data.score,
        explanationId: explanation.id,
        explanationScoreTypeName,
        explanationScoreModelName: ELEUTHER_EMBEDDING_MODEL_NAME,
        initiatedByUserId: user.id,
        jsonDetails: JSON.stringify(converted),
      },
    });
    return savedScore;
  }
  throw new Error('Invalid Eleuther type');
};
