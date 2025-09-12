import { createNeurons } from '@/lib/db/neuron';
import { RequestAuthedAdminUser } from '@/lib/types/auth';
import { withAuthedAdminUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

type SkippedData = {
  model_id: string;
  layer: string;
  sae_set: string;
  sae_id: string;
  skipped_indexes: number[];
};

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const data = (await request.json()) as SkippedData;

  const modelId = data.model_id || '';
  const layer = `${data.layer}-${data.sae_set}`;
  const sourceSetName = data.sae_set;
  const skippedIndexes: number[] = data.skipped_indexes;
  const createDate = new Date();

  const featuresToCreate = [];
  // eslint-disable-next-line no-restricted-syntax
  for (const index of skippedIndexes) {
    featuresToCreate.push({
      modelId,
      layer,
      index: index.toString(),
      creatorId: request.user.id,
      createdAt: createDate,

      sourceSetName,
      maxActApprox: 0,

      neg_str: [],
      neg_values: [],
      pos_str: [],
      pos_values: [],
      frac_nonzero: undefined,
      freq_hist_data_bar_heights: [],
      freq_hist_data_bar_values: [],

      logits_hist_data_bar_heights: [],
      logits_hist_data_bar_values: [],

      num_prompts: undefined,

      neuron_alignment_indices: [],
      neuron_alignment_values: [],
      neuron_alignment_l1: [],
      correlated_neurons_indices: [],
      correlated_neurons_pearson: [],
      correlated_neurons_l1: [],
      correlated_features_indices: [],
      correlated_features_pearson: [],
      correlated_features_l1: [],
    });
  }

  await createNeurons(modelId, layer, featuresToCreate, request.user);

  console.log('Features created');

  return NextResponse.json({ message: 'success' });
});
