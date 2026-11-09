'use client';

import ExplanationScoreDetailDialog from '@/components/explanation-score-detail-dialog';
import FeatureSelector from '@/components/feature-selector/feature-selector';
import SelectorLinkButton from '@/components/feature-selector/selector-link-button';
import { FeatureId, getCompareLinks } from '@/lib/utils/feature-compare';
import { getFeaturePath } from '@/lib/utils/neuron-identifier';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { X } from 'lucide-react';
import Link from 'next/link';
import FeatureDashboard from './feature-dashboard';

// `force-mobile` gives each column the mobile layout (see tailwind.config.js).
const COLUMN_CLASSES = 'force-mobile relative h-full min-w-0 flex-1 basis-0 overflow-y-auto overflow-x-hidden';

function featureKey(feature: FeatureId) {
  return `${feature.modelId}/${feature.layer}/${feature.index}`;
}

// Two feature dashboards side by side. On a narrow screen, only the left one shows.
export default function FeatureCompare({
  leftNeuron,
  rightNeuron,
  rightId,
}: {
  leftNeuron: NeuronWithPartialRelations;
  rightNeuron: NeuronWithPartialRelations | null;
  rightId: FeatureId;
}) {
  const links = getCompareLinks(leftNeuron, rightId);

  return (
    <div className="flex h-[calc(100dvh-48px)] w-full flex-col border-t border-slate-200">
      <ExplanationScoreDetailDialog />

      <div className="flex w-full shrink-0 flex-row items-center justify-between gap-x-3 border-b border-slate-200 bg-amber-50 px-3 py-1.5 text-[11px] text-slate-600 sm:hidden">
        <span>Side-by-side compare needs a wider screen.</span>
        <div className="flex shrink-0 flex-row gap-x-3">
          <Link href={getFeaturePath(rightId.modelId, rightId.layer, rightId.index)} className="text-sky-700">
            Open other feature
          </Link>
          <Link href={links.right.closeHref} className="text-sky-700">
            Close compare
          </Link>
        </div>
      </div>

      <div className="flex min-h-0 w-full flex-1 flex-row">
        <div className={COLUMN_CLASSES}>
          <FeatureDashboard key={featureKey(leftNeuron)} initialNeuron={leftNeuron} compare={links.left} />
        </div>
        <div className="hidden w-1 shrink-0 bg-slate-300 sm:block" />
        <div className={`${COLUMN_CLASSES} hidden sm:block`}>
          {rightNeuron ? (
            <FeatureDashboard key={featureKey(rightNeuron)} initialNeuron={rightNeuron} compare={links.right} />
          ) : (
            <div className="flex flex-col items-center gap-y-4 px-3 pt-10 text-center">
              <div className="text-sm font-medium text-slate-600">
                This feature could not be loaded: {featureKey(rightId)}
              </div>
              <FeatureSelector
                filterToPublic
                defaultModelId={leftNeuron.modelId}
                defaultSourceSet={leftNeuron.sourceSetName || undefined}
                defaultSource={leftNeuron.layer}
                defaultIndex={leftNeuron.index}
                openInNewTab={false}
                filterToFeaturedReleases={false}
                autoFocus={false}
                getHref={links.right.hrefFor}
                trailing={
                  <SelectorLinkButton
                    href={links.right.closeHref}
                    title="Close this column"
                    label="Close"
                    icon={<X className="h-4 w-4" />}
                  />
                }
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
