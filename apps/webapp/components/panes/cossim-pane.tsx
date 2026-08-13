import CustomTooltip from '@/components/custom-tooltip';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import { UtilSaeTopkByDecoderCossimResponse } from '@/lib/api/inference-types';
import { getSourceSet } from '@/lib/db/source';
import { NeuronWithPartialRelations, SourceWithPartialRelations } from '@/prisma/generated/zod';
import * as Tooltip from '@radix-ui/react-tooltip';
import { HelpCircle } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { LoadingSquare } from '../svg/loading-square';

const MAX_TOPK_COS_SIM_FEATURES_TO_FETCH = 50;
const MAX_TOPK_TO_SHOW_INITIALLY = 10;
const MAX_COSSIM_FEATURE_DESC_LENGTH = 128;

export default function CosSimPane({ currentNeuron }: { currentNeuron: NeuronWithPartialRelations | undefined }) {
  const { setFeatureModalFeature, setFeatureModalOpen, getSource } = useGlobalContext();
  const [topkCosSimFeatures, setTopkCosSimFeatures] = useState<NeuronWithPartialRelations[]>([]);
  const [numTopkToShow, setNumTopkToShow] = useState(MAX_TOPK_TO_SHOW_INITIALLY);
  const [topkMatchedCosSimFeatures, setTopkMatchedCosSimFeatures] = useState<NeuronWithPartialRelations[]>([]);
  const [topkMatchedCosSimValues, setTopkMatchedCosSimValues] = useState<number[]>([]);
  const [matchingSource, setMatchingSource] = useState<SourceWithPartialRelations | undefined>(undefined);

  useEffect(() => {
    if (currentNeuron && currentNeuron.topkCosSimIndices && currentNeuron.topkCosSimIndices.length > 0) {
      // make the features to fetch
      const features = currentNeuron.topkCosSimIndices
        .filter(
          (idx, pos) => idx.toString() !== currentNeuron.index.toString() && pos <= MAX_TOPK_COS_SIM_FEATURES_TO_FETCH,
        )
        .map((idx) => ({
          modelId: currentNeuron.modelId,
          layer: currentNeuron.layer,
          index: idx,
          maxActsToReturn: 3,
        }));
      fetch(`/api/features`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(features),
      })
        .then((response) => response.json())
        .then((featuresResponse: NeuronWithPartialRelations[]) => {
          setTopkCosSimFeatures(featuresResponse);
        });
    }
    if (currentNeuron && currentNeuron.layer) {
      const source = getSource(currentNeuron.modelId, currentNeuron.layer);
      if (source && source.cosSimMatchSourceId && source.cosSimMatchModelId) {
        setMatchingSource(getSource(source.cosSimMatchModelId, source.cosSimMatchSourceId));
        fetch('/api/cossim', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            modelId: currentNeuron.modelId,
            source: currentNeuron.layer,
            index: currentNeuron.index,
            targetModelId: source.cosSimMatchModelId,
            targetSourceId: source.cosSimMatchSourceId,
          }),
        })
          .then((response) => response.json())
          .then((cosSimResponse: UtilSaeTopkByDecoderCossimResponse) => {
            const featureIndexes: number[] = [];
            if (cosSimResponse.topkDecoderCossimFeatures) {
              cosSimResponse.topkDecoderCossimFeatures.forEach((feature) => {
                if (feature.feature?.index) {
                  featureIndexes.push(feature.feature?.index);
                }
              });
            }
            const featuresToFetch = featureIndexes.map((index) => ({
              modelId: source.cosSimMatchModelId,
              layer: source.cosSimMatchSourceId,
              index: index.toString(),
              maxActsToReturn: 3,
            }));
            setTopkMatchedCosSimValues(
              cosSimResponse.topkDecoderCossimFeatures?.map((f) => f.cosineSimilarity || 0) || [],
            );
            return fetch(`/api/features`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(featuresToFetch),
            });
          })
          .then((response) => response.json())
          .then((featuresResponse: NeuronWithPartialRelations[]) => {
            setTopkMatchedCosSimFeatures(featuresResponse);
          })
          .catch((error) => {
            console.error('Error fetching matched cosine similarity features:', error);
          });
      }
    }
  }, [currentNeuron, getSourceSet]);

  return (
    <div
      className={`mt-2 hidden flex-col gap-x-2 overflow-hidden rounded-lg border bg-white px-3 pb-4 pt-2 text-xs shadow transition-all sm:mt-3 ${
        (currentNeuron?.topkCosSimIndices && currentNeuron?.topkCosSimIndices.length > 0) || matchingSource
          ? 'sm:flex'
          : 'sm:hidden'
      }`}
    >
      <div className="mb-1.5 flex w-full flex-row items-center justify-center gap-x-1 text-[10px] font-normal uppercase text-slate-400">
        Top Features by Cosine Similarity
        <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
          <Tooltip.Root>
            <Tooltip.Trigger asChild>
              <button type="button">
                <HelpCircle className="h-2.5 w-2.5" />
              </button>
            </Tooltip.Trigger>
            <Tooltip.Portal>
              <Tooltip.Content className="rounded bg-slate-500 px-3 py-2 text-xs text-white" sideOffset={5}>
                Top features by cosine similarity to this feature.
                <Tooltip.Arrow className="fill-slate-500" />
              </Tooltip.Content>
            </Tooltip.Portal>
          </Tooltip.Root>
        </Tooltip.Provider>
      </div>
      {currentNeuron?.topkCosSimIndices &&
        currentNeuron?.topkCosSimValues &&
        currentNeuron?.topkCosSimIndices.length > 0 && (
          <div className="relative flex-col">
            <div className="mb-1 w-full text-center text-[10px] font-normal uppercase text-slate-400">
              Comparing With{' '}
              <Link
                prefetch={false}
                href={`/${currentNeuron.modelId}/${currentNeuron.layer}`}
                className="text-[10px] text-sky-700 hover:underline"
              >
                {currentNeuron.modelId.toUpperCase()} @ {currentNeuron.layer}
              </Link>
            </div>
            <div className="relative flex w-full flex-col gap-y-0.5 text-slate-600">
              {topkCosSimFeatures.length > 0 ? (
                topkCosSimFeatures
                  .filter((idx, pos) => pos <= numTopkToShow)
                  .map((item, j) => (
                    <CustomTooltip
                      key={item.index}
                      trigger={
                        <div
                          key={item.index}
                          className="mx-0 flex w-full cursor-pointer flex-row items-center gap-x-4 py-0.5"
                        >
                          <button
                            type="button"
                            onClick={() => {
                              setFeatureModalFeature({
                                modelId: item.modelId || '',
                                layer: item.layer || '',
                                index: item.index.toString(),
                              } as NeuronWithPartialRelations);
                              setFeatureModalOpen(true);
                            }}
                            className="cursor-pointer text-left text-[11px] leading-snug text-sky-700 hover:cursor-pointer hover:underline"
                          >
                            {(() => {
                              if (item.explanations && item.explanations.length > 0) {
                                if (
                                  item.explanations?.[0]?.description &&
                                  item.explanations?.[0]?.description?.length > MAX_COSSIM_FEATURE_DESC_LENGTH
                                ) {
                                  return `${item.explanations?.[0].description?.slice(
                                    0,
                                    MAX_COSSIM_FEATURE_DESC_LENGTH,
                                  )}...`;
                                }
                                return item.explanations?.[0]?.description;
                              }
                              return `${item.modelId.toUpperCase()} | ${item.layer.toUpperCase()} | Index ${
                                item.index
                              }`;
                            })()}
                          </button>
                          <div className="flex-1 text-right font-mono text-[10px] font-medium text-slate-400">
                            {currentNeuron?.topkCosSimValues?.[j + 1]?.toFixed(2)}
                          </div>
                        </div>
                      }
                    >
                      {item.modelId.toUpperCase()}
                      <br />
                      {item.layer.toUpperCase()}
                      <br />
                      Index {item.index}
                    </CustomTooltip>
                  ))
              ) : (
                <div className="flex h-20 flex-col items-center justify-center text-xs font-bold text-slate-300 sm:h-16 sm:text-sm">
                  <LoadingSquare size={18} className="text-sky-700" />
                </div>
              )}
              {topkCosSimFeatures.length > numTopkToShow && (
                <div className="mt-1.5 flex w-full flex-row items-center justify-center gap-x-1.5">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-6 text-[10px] text-slate-400"
                    onClick={() => setNumTopkToShow(MAX_TOPK_COS_SIM_FEATURES_TO_FETCH)}
                  >
                    Show More
                  </Button>
                </div>
              )}
            </div>
          </div>
        )}
      {matchingSource && (
        <div className="relative flex-col">
          <div className="mb-1 w-full text-center text-[10px] font-normal uppercase text-slate-400">
            Comparing With{' '}
            <Link
              prefetch={false}
              href={`/${matchingSource.modelId}/${matchingSource.id}`}
              className="text-[10px] text-sky-700 hover:underline"
            >
              {matchingSource.modelId.toUpperCase()} @ {matchingSource.id}
            </Link>
          </div>
          <div className="relative flex w-full flex-col gap-y-0.5 text-slate-600">
            {topkMatchedCosSimFeatures.length > 0 ? (
              topkMatchedCosSimFeatures
                .filter((idx, pos) => pos <= numTopkToShow)
                .map((item, j) => (
                  <CustomTooltip
                    key={item.index}
                    trigger={
                      <div
                        key={item.index}
                        className="mx-0 flex w-full cursor-pointer flex-row items-center gap-x-4 py-0.5"
                      >
                        <button
                          type="button"
                          onClick={() => {
                            setFeatureModalFeature({
                              modelId: item.modelId || '',
                              layer: item.layer || '',
                              index: item.index.toString(),
                            } as NeuronWithPartialRelations);
                            setFeatureModalOpen(true);
                          }}
                          className="cursor-pointer text-left text-[11px] leading-snug text-sky-700 hover:cursor-pointer hover:underline"
                        >
                          {(() => {
                            if (item.explanations && item.explanations.length > 0) {
                              if (
                                item.explanations?.[0]?.description &&
                                item.explanations?.[0]?.description?.length > MAX_COSSIM_FEATURE_DESC_LENGTH
                              ) {
                                return `${item.explanations?.[0].description?.slice(
                                  0,
                                  MAX_COSSIM_FEATURE_DESC_LENGTH,
                                )}...`;
                              }
                              return item.explanations?.[0]?.description;
                            }
                            return `${item.modelId.toUpperCase()} | ${item.layer.toUpperCase()} | Index ${item.index}`;
                          })()}
                        </button>
                        <div className="flex-1 text-right font-mono text-[10px] font-medium text-slate-400">
                          {topkMatchedCosSimValues?.[j]?.toFixed(2)}
                        </div>
                      </div>
                    }
                  >
                    {item.modelId.toUpperCase()}
                    <br />
                    {item.layer.toUpperCase()}
                    <br />
                    Index {item.index}
                  </CustomTooltip>
                ))
            ) : (
              <div className="flex h-20 flex-col items-center justify-center text-xs font-bold text-slate-300 sm:h-16 sm:text-sm">
                <LoadingSquare size={18} className="text-sky-700" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
