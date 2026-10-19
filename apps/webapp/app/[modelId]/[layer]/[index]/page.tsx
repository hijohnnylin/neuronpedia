import { getNeuronOptimized, neuronExistsAndUserHasAccess } from '@/lib/db/neuron';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/db/user';
import { userCanAccessModelAndSourceSet } from '@/lib/db/userCanAccess';
import { getNeuronLayerTitle, isNeuronLayerSource, NEURONS_SOURCESET } from '@/lib/utils/source';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import FeatureDashboard from './feature-dashboard';

export async function generateMetadata(props: {
  params: Promise<{ modelId: string; layer: string; index: string }>;
}): Promise<Metadata> {
  const params = await props.params;
  const isNeuronLayer = isNeuronLayerSource(params.layer);
  const layerTitle = isNeuronLayer ? getNeuronLayerTitle(params.layer) : params.layer.toUpperCase();
  let title = `${params.modelId.toUpperCase()} · ${layerTitle} · ${params.index.toUpperCase()}`;
  // A neuron layer page renders without a Neuron row, so check the source set instead.
  const visible =
    (await neuronExistsAndUserHasAccess(params.modelId, params.layer, params.index)) ||
    (isNeuronLayer &&
      /^\d+$/.test(params.index) &&
      (await userCanAccessModelAndSourceSet(params.modelId, NEURONS_SOURCESET, null, true)));
  if (!visible) {
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

export default async function Page(props: {
  params: Promise<{ modelId: string; layer: string; index: string }>;
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  const searchParams = await props.searchParams;
  const params = await props.params;
  const embed = searchParams.embed === 'true';
  const embedPlots = searchParams.embedplots !== 'false'; // default embed plots
  const embedExplanation = searchParams.embedexplanation !== 'false'; // default embed auto interp
  const embedTest = searchParams.embedtest !== 'false'; // default embed test
  const embedActivations = searchParams.embedactivations !== 'false'; // default embed activations
  const embedLink = searchParams.embedlink !== 'false'; // default embed link (link to the dashboard)
  const embedSteer = searchParams.embedsteer !== 'false'; // default embed steer
  const embedTestField = searchParams.embedtestfield !== 'false'; // default embed test field
  const defaultTestText = searchParams.defaulttesttext ? (searchParams.defaulttesttext as string) : undefined;

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
        embedTestField={embedTestField}
        embedExplanation={embedExplanation}
        embedActivations={embedActivations}
        embedLink={embedLink}
        embedSteer={embedSteer}
      />
    </div>
  );
}
