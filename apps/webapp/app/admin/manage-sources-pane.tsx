'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { useImportContext } from '@/components/provider/import-provider';
import { Button } from '@/components/shadcn/button';
import { LoadingSquare } from '@/components/svg/loading-square';
import { getS3ModelsToSources } from '@/lib/utils/s3';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { useRouter } from '@bprogress/next';
import { Fragment, useEffect, useState } from 'react';
import DownloadAllButton from './download-all-button';
import DownloadButton from './download-button';
import ResetButton from './reset-button';
import SyncConfigButton from './sync-config-button';

export default function ManageSourcesPane() {
  const [s3ModelsToSources, setS3ModelsToSources] = useState<Record<string, string[]>>({});
  const [expandedSets, setExpandedSets] = useState<Record<string, boolean>>({});
  const [isLoading, setIsLoading] = useState(true);
  const { getSource } = useGlobalContext();
  const { isAnyDownloadRunning } = useImportContext();
  const router = useRouter();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getS3ModelsToSources();
        setS3ModelsToSources(data);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  const toggleSetExpansion = (setKey: string) => {
    setExpandedSets((prev) => ({
      ...prev,
      [setKey]: !prev[setKey],
    }));
  };

  const expandSet = (setKey: string) => {
    setExpandedSets((prev) => ({
      ...prev,
      [setKey]: true,
    }));
  };

  function getDownloadedCountForModelAndSourceSet(modelId: string, sourceSetName: string) {
    const sources = s3ModelsToSources[modelId];
    if (!sources) {
      return 0;
    }
    return sources.filter(
      (source) => getSource(modelId, source) && getSourceSetNameFromSource(source) === sourceSetName,
    ).length;
  }

  return (
    <div className="flex w-full max-w-screen-xl flex-col items-center justify-center pt-8">
      <div className="flex w-full flex-col items-center justify-center overflow-hidden rounded-lg border border-slate-200 bg-white text-sm">
        {isLoading ? (
          <div className="flex w-full justify-center py-8">
            <LoadingSquare />
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs text-slate-400">
                <th className="whitespace-nowrap px-6 py-3 font-medium">Model</th>
                <th className="whitespace-nowrap px-6 py-3 font-medium">Set</th>
                <th className="min-w-[263px] whitespace-nowrap px-6 py-3 font-medium">Source/SAE</th>
                <th className="w-full px-6 py-3 font-medium" aria-label="Download Button">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(s3ModelsToSources).map(([model]) => {
                const sourcesBySet: Record<string, string[]> = {};
                s3ModelsToSources[model].forEach((source) => {
                  const setName = getSourceSetNameFromSource(source || '');
                  if (!sourcesBySet[setName]) {
                    sourcesBySet[setName] = [];
                  }
                  sourcesBySet[setName].push(source);
                });

                return Object.entries(sourcesBySet).map(([setName, sources], setIndex) => {
                  const setKey = `${model}-${setName}`;
                  const isSetExpanded = expandedSets[setKey] || false;

                  return (
                    <Fragment key={setKey}>
                      <tr
                        onClick={() => toggleSetExpansion(setKey)}
                        className="h-12 cursor-pointer border-b border-slate-100 text-xs text-slate-700 hover:bg-sky-50"
                      >
                        <td className="whitespace-nowrap px-6 font-mono">
                          {setIndex === 0 && <div className="flex h-6 items-center font-medium">{model}</div>}
                        </td>
                        <td className="whitespace-nowrap px-6 font-mono">
                          <div className="flex h-8 w-full flex-row items-center justify-between gap-2 font-medium">
                            {setName}
                          </div>
                        </td>
                        <td className="whitespace-nowrap px-6 font-mono">
                          <div className="flex h-8 items-center">
                            <span className="font-sans text-xs">
                              {getDownloadedCountForModelAndSourceSet(model, setName)} of {sources.length} Downloaded
                            </span>
                          </div>
                        </td>
                        <td className="flex h-12 w-full flex-row items-center justify-between gap-x-2 px-6">
                          <div className="flex items-center gap-x-2">
                            {getDownloadedCountForModelAndSourceSet(model, setName) > 0 && (
                              <Button
                                size="sm"
                                variant="emerald"
                                onClick={(e) => {
                                  e.stopPropagation(); // Prevent row click when clicking button
                                  if (isAnyDownloadRunning) {
                                    window.open(`/${model}/${setName}`, '_blank');
                                  } else {
                                    router.push(`/${model}/${setName}`);
                                  }
                                }}
                                className="h-7"
                              >
                                Browse
                              </Button>
                            )}
                            <DownloadAllButton
                              modelId={model}
                              sourceSetName={setName}
                              onDownloadStart={() => expandSet(`${model}-${setName}`)}
                            />
                          </div>

                          <span
                            className="text-xl text-slate-600 transition-transform duration-200"
                            style={{ transform: `rotate(${isSetExpanded ? '90deg' : '0deg'})` }}
                          >
                            ▸
                          </span>
                        </td>
                      </tr>
                      {isSetExpanded &&
                        sources.map((source) => (
                          <tr
                            key={`${model}-${source}`}
                            className="h-10 border-b border-slate-50 text-xs text-slate-600 hover:bg-slate-50"
                          >
                            <td className="whitespace-nowrap px-6 font-mono" aria-label={source} />
                            <td className="whitespace-nowrap px-6 font-mono" aria-label={source} />
                            <td className="whitespace-nowrap px-6 font-mono" aria-label={source}>
                              <div className="flex h-8 items-center">{source}</div>
                            </td>
                            <td className="w-full px-6">
                              <div className="flex h-8 items-center gap-x-2">
                                {getSource(model, source) && (
                                  <Button
                                    size="sm"
                                    variant="emerald"
                                    onClick={() => {
                                      if (isAnyDownloadRunning) {
                                        window.open(`/${model}/${source}`, '_blank');
                                      } else {
                                        router.push(`/${model}/${source}`);
                                      }
                                    }}
                                    className="h-7"
                                  >
                                    Browse
                                  </Button>
                                )}
                                <DownloadButton modelId={model} sourceId={source} />
                              </div>
                            </td>
                          </tr>
                        ))}
                    </Fragment>
                  );
                });
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="mt-5 flex w-full flex-row justify-end gap-x-2">
        <SyncConfigButton />
        <ResetButton />
      </div>
    </div>
  );
}
