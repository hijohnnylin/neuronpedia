import { getNeuronOptimized, neuronExistsAndUserHasAccess } from '@/lib/db/neuron';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/db/user';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { Metadata } from 'next';
import { notFound, redirect } from 'next/navigation';
import FeatureDashboard from './feature-dashboard';

export async function generateMetadata({
  params,
}: {
  params: { modelId: string; layer: string; index: string };
}): Promise<Metadata> {
  let title = `${params.modelId.toUpperCase()} · ${params.layer.toUpperCase()} · ${params.index.toUpperCase()}`;
  const feat = await neuronExistsAndUserHasAccess(params.modelId, params.layer, params.index);
  if (!feat) {
    title = '';
  }
  return {
    title,
    description: '',
    openGraph: {
      title,
      description: '',
      url: `/${params.modelId}/${params.layer}/${params.index}`,
    },
  };
}

export default async function Page({
  params,
  searchParams,
}: {
  params: { modelId: string; layer: string; index: string };
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const embed = searchParams.embed === 'true';
  const embedPlots = searchParams.embedplots !== 'false'; // default embed plots
  const embedExplanation = searchParams.embedexplanation !== 'false'; // default embed auto interp
  const embedTest = searchParams.embedtest !== 'false'; // default embed test
  const defaultTestText = searchParams.defaulttesttext ? (searchParams.defaulttesttext as string) : undefined;

  // TODO: this is a temporary map since there is a bug in our lesswrong plugin that breaks when dots are in modelIds for hoverover links
  const REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT = {
    'llama31-8b-it': 'llama3.1-8b-it',
    'qwen25-7b-it': 'qwen2.5-7b-it',
  };
  if (params.modelId in REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT) {
    // redirect to the new model id
    const queryString = new URLSearchParams(searchParams as Record<string, string>).toString();
    const redirectUrl = `/${REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT[params.modelId as keyof typeof REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT]}/${params.layer}/${params.index}${queryString ? `?${queryString}` : ''}`;
    redirect(redirectUrl);
  }

  const { modelId } = params;
  let neuron: NeuronWithPartialRelations | null = null;

  try {
    neuron = await getNeuronOptimized(
      modelId,
      params.layer,
      params.index,
      await makeAuthedUserFromSessionOrReturnNull(),
    );

    if (!neuron) {
      if (params.layer.match(/^-?\d+$/)) {
        neuron = {
          modelId,
          layer: params.layer,
          index: params.index,
          sourceSetName: 'neurons',
          creatorId: '',
          createdAt: new Date(),
          maxActApprox: 0.01,
          neuron_alignment_indices: [],
          neuron_alignment_values: [],
          neuron_alignment_l1: [],
          correlated_neurons_indices: [],
          correlated_neurons_pearson: [],
          correlated_neurons_l1: [],
          correlated_features_indices: [],
          correlated_features_pearson: [],
          correlated_features_l1: [],
          neg_str: [],
          neg_values: [],
          pos_str: [],
          pos_values: [],
          frac_nonzero: 0,
          freq_hist_data_bar_heights: [],
          freq_hist_data_bar_values: [],
          logits_hist_data_bar_heights: [],
          logits_hist_data_bar_values: [],
          decoder_weights_dist: [],
          umap_cluster: null,
          umap_log_feature_sparsity: null,
          umap_x: null,
          umap_y: null,
          topkCosSimIndices: [],
          topkCosSimValues: [],
          vector: [],
          hasVector: false,
          vectorDefaultSteerStrength: null,
          hookName: null,
          vectorLabel: null,
        };
      } else {
        notFound();
      }
    }
  } catch (e) {
    notFound();
  }

  return (
    <div className="h-[calc(100vh-110px)] w-full pt-0 sm:h-auto">
      <FeatureDashboard
        initialNeuron={neuron}
        embed={embed}
        embedPlots={embedPlots}
        embedTest={embedTest}
        defaultTestText={defaultTestText}
        embedExplanation={embedExplanation}
      />
    </div>
  );
}
