import { Activation, ExplanationModelType } from '@prisma/client';
import { AUTOINTERP_SERVER_API, unwrapAutointerpResponse } from '../utils/autointerp';

export const generateExplanationEleutherActsTop20 = async (
  activations: Activation[],
  explanationModel: ExplanationModelType,
  explainerKey: string,
) => {
  if (!explanationModel.openRouterModelId) {
    throw new Error('Explaining using np-auto-interp requires an OpenRouter model id.');
  }

  const result = await unwrapAutointerpResponse(
    AUTOINTERP_SERVER_API.POST('/v1/explain/default', {
      body: {
        activations: activations.map((act) => ({ tokens: act.tokens, values: act.values })),
        openrouterKey: explainerKey,
        model: explanationModel.openRouterModelId,
      },
    }),
  );

  return result.explanation;
};
