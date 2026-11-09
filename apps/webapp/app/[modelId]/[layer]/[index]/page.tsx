import { getNeuronOptimized, neuronExistsAndUserHasAccess } from '@/lib/db/neuron';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/db/user';
import { userCanAccessModelAndSourceSet } from '@/lib/db/userCanAccess';
import { COMPARE_PARAM, parseCompareValue } from '@/lib/utils/feature-compare';
import { emptyNeuron } from '@/lib/utils/neuron-dashboards';
import { getNeuronLayerTitle, isNeuronLayerSource, NEURONS_SOURCESET } from '@/lib/utils/source';
import { AuthenticatedUser } from '@/lib/with-user';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import FeatureCompare from './feature-compare';
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

// Returns null when the feature does not exist or the user cannot see it.
async function loadNeuron(
  modelId: string,
  layer: string,
  index: string,
  user: AuthenticatedUser | null,
): Promise<NeuronWithPartialRelations | null> {
  try {
    const neuron = await getNeuronOptimized(modelId, layer, index, user);
    if (neuron) {
      return neuron;
    }
    // A neuron layer page renders without a Neuron row.
    return layer.match(/^-?\d+$/) ? emptyNeuron(modelId, layer, index, NEURONS_SOURCESET) : null;
  } catch (e) {
    return null;
  }
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

  const compareId = embed ? null : parseCompareValue(searchParams[COMPARE_PARAM]);
  const user = await makeAuthedUserFromSessionOrReturnNull();
  const [neuron, compareNeuron] = await Promise.all([
    loadNeuron(params.modelId, params.layer, params.index, user),
    compareId ? loadNeuron(compareId.modelId, compareId.layer, compareId.index, user) : null,
  ]);
  if (!neuron) {
    notFound();
  }

  if (compareId) {
    return <FeatureCompare leftNeuron={neuron} rightNeuron={compareNeuron} rightId={compareId} />;
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
