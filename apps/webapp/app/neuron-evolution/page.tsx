import { Metadata } from 'next';
import NeuronEvolutionExplorer from './neuron-evolution-explorer';

export async function generateMetadata(): Promise<Metadata> {
  const title = 'Neuron Evolution';
  const description =
    'How MLP neurons of the Pythia suite change across training: LLM explanations at every checkpoint, chained into concept lineages.';
  return {
    title,
    description,
    openGraph: {
      title,
      description,
      url: '/neuron-evolution',
    },
  };
}

export default async function Page(props: {
  searchParams: Promise<{ model?: string; neuron?: string; step?: string }>;
}) {
  const searchParams = await props.searchParams;
  const step = searchParams.step ? Number(searchParams.step) : NaN;
  return (
    <div className="flex h-full w-full flex-col items-center overflow-y-scroll bg-white">
      <NeuronEvolutionExplorer
        initialModelId={searchParams.model}
        initialNeuron={searchParams.neuron}
        initialStep={Number.isFinite(step) ? step : undefined}
      />
    </div>
  );
}
