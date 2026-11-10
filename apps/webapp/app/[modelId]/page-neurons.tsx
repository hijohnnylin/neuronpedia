import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import ReleasesDropdown from '@/components/nav/releases-dropdown';
import { BreadcrumbLink, BreadcrumbPage } from '@/components/shadcn/breadcrumbs';
import { prisma } from '@/lib/db';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/db/user';
import { userCanAccessClause } from '@/lib/db/userCanAccess';
import { getLayerNumFromSource, NEURONS_SOURCESET } from '@/lib/utils/source';
import { SourceReleaseWithPartialRelations } from '@/prisma/generated/zod';
import NeuronsHome, { NeuronModelRow } from './neurons-home';

export default async function PageNeurons({ release }: { release: SourceReleaseWithPartialRelations }) {
  const user = await makeAuthedUserFromSessionOrReturnNull();
  const sourceSets = await prisma.sourceSet.findMany({
    where: {
      name: NEURONS_SOURCESET,
      hasDashboards: true,
      ...userCanAccessClause(user),
      model: userCanAccessClause(user),
    },
    include: {
      model: { select: { neuronsPerLayer: true, inferenceEnabled: true } },
      sources: {
        where: { hasDashboards: true, ...userCanAccessClause(user) },
        select: { id: true, inferenceEnabled: true },
      },
    },
  });

  const models: NeuronModelRow[] = sourceSets
    .filter((ss) => ss.sources.length > 0)
    .map((ss) => ({
      modelId: ss.modelId,
      layers: ss.sources.map((s) => s.id).toSorted((a, b) => getLayerNumFromSource(a) - getLayerNumFromSource(b)),
      neuronsPerLayer: ss.model.neuronsPerLayer,
      searchable: ss.model.inferenceEnabled && ss.allowInferenceSearch && ss.sources.some((s) => s.inferenceEnabled),
    }))
    .toSorted((a, b) => a.modelId.localeCompare(b.modelId));

  return (
    <div className="flex w-full flex-col items-center pb-10">
      <BreadcrumbsComponent
        crumbsArray={[
          <BreadcrumbPage key={0}>
            <ReleasesDropdown breadcrumb />
          </BreadcrumbPage>,
          <BreadcrumbLink href={`/${release.name}`} key={1}>
            {release.description}
          </BreadcrumbLink>,
        ]}
      />
      <NeuronsHome release={release} models={models} />
    </div>
  );
}
