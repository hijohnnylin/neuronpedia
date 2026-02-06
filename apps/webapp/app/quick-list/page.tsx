// TODO: better error handling

import UmapProvider from '@/components/provider/umap-provider';
import { getNeuronsForQuickList } from '@/lib/db/neuron';
import { NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import { notFound } from 'next/navigation';
import { makeAuthedUserFromSessionOrReturnNull } from '../../lib/db/user';
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
  let featuresArray: NeuronIdentifier[] = [];
  const parsedFeatures = JSON.parse(featuresParam);

  // Handle edge case where index is a stringified array of arrays
  if (Array.isArray(parsedFeatures) && parsedFeatures.length > 0) {
    for (const feature of parsedFeatures) {
      if (typeof feature.index === 'string' && feature.index.startsWith('[[')) {
        // Parse the nested array structure
        const nestedIndices = JSON.parse(feature.index) as number[][];
        // Create a NeuronIdentifier for each number in each inner array
        for (const indexArray of nestedIndices) {
          for (const index of indexArray) {
            featuresArray.push(new NeuronIdentifier(feature.modelId, feature.layer, String(index)));
          }
        }
      } else {
        featuresArray.push(new NeuronIdentifier(feature.modelId, feature.layer, feature.index));
      }
    }
  } else {
    featuresArray = parsedFeatures as NeuronIdentifier[];
  }

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
    throw new Error(`Feature not found: ${feature.modelId}-${feature.layer}-${feature.index}`);
  });

  return (
    <UmapProvider>
      <QuickList name={name} features={featuresOriginalOrder} />
    </UmapProvider>
  );
}
