import { getNeuronOptimized, neuronExistsAndUserHasAccess } from '@/lib/db/neuron';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/db/user';
import { userCanAccessModelAndSourceSet } from '@/lib/db/userCanAccess';
import { emptyNeuron } from '@/lib/utils/neuron-dashboards';
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
        neuron = emptyNeuron(modelId, params.layer, params.index, NEURONS_SOURCESET);
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
