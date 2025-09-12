// TODO: better error handling

import UmapProvider from '@/components/provider/umap-provider';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/auth/utils';
import { getNeuronsForQuickList } from '@/lib/db/neuron';
import { NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import { notFound } from 'next/navigation';
import QuickList from './quick-list';

export default async function Page({
  searchParams,
}: {
  searchParams?: {
    name?: string;
    features?: string; // array of objects { model, layer, index }
  };
}) {
  if (!searchParams || !searchParams.name || !searchParams.features) {
    notFound();
  }

  const { name } = searchParams;
  const featuresParam = searchParams.features;

  const featuresArray = JSON.parse(featuresParam) as NeuronIdentifier[];
  if (featuresArray.length === 0) {
    notFound();
  }

  if (featuresArray.length > 150) {
    notFound();
  }

  const features = await getNeuronsForQuickList(featuresArray, await makeAuthedUserFromSessionOrReturnNull());

  const featuresOriginalOrder = featuresArray.map((feature) => {
    const foundFeature = features.find(
      (f) => f.modelId === feature.modelId && f.layer === feature.layer && f.index === feature.index,
    );
    if (foundFeature) {
      return foundFeature;
    }
    throw new Error('Feature not found');
  });

  return (
    <UmapProvider>
      <QuickList name={name} features={featuresOriginalOrder} />
    </UmapProvider>
  );
}
