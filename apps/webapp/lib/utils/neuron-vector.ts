import { NeuronPartial } from '@/prisma/generated/zod';

export const NEURONPEDIA_SOURCE_SET_ID = '-neuronpedia';

export const VECTOR_VALID_HOOK_TYPE_TO_DETAILS: Record<string, { displayName: string; sourceSetSuffix: string }> = {
  hook_resid_pre: {
    displayName: 'Residual Stream Pre',
    sourceSetSuffix: 'resid-pre',
  },
  hook_resid_post: {
    displayName: 'Residual Stream Post',
    sourceSetSuffix: 'resid-post',
  },
};

// for not in own source, we don't want to show the feature selector at the top or the breadcrumbs
export function shouldHideBreadcrumbsAndSelectorForNeuronVector(neuron: NeuronPartial | undefined) {
  return neuron?.layer?.includes(NEURONPEDIA_SOURCE_SET_ID);
}

// for features that have vector in the database, show the extra feature panel
export function neuronHasVectorInDatabase(neuron: NeuronPartial | undefined) {
  return neuron?.vector && neuron.vector.length > 0;
}
