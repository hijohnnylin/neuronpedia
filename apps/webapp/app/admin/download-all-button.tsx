'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { useImportContext } from '@/components/provider/import-provider';
import { Button } from '@/components/shadcn/button';
import { useEffect, useState } from 'react';

interface DownloadAllButtonProps {
  modelId: string;
  sourceSetName: string;
  onDownloadStart?: () => void;
}

export default function DownloadAllButton({ modelId, sourceSetName, onDownloadStart }: DownloadAllButtonProps) {
  const [isDownloading, setIsDownloading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [total, setTotal] = useState(0);
  const [sourcesToDownload, setSourcesToDownload] = useState<string[]>([]);
  const { getSource } = useGlobalContext();
  const { downloadSource, isAnyDownloadRunning, getSourcesForModelAndSet, fetchSourcesIfNeeded, isSourcesFetched } =
    useImportContext();

  useEffect(() => {
    const updateSourcesToDownload = () => {
      const sources = getSourcesForModelAndSet(modelId, sourceSetName);
      if (sources) {
        // Filter out sources that are already downloaded
        const needsDownload = sources.filter((source) => !getSource(modelId, source));
        setSourcesToDownload(needsDownload);
      }
    };

    if (isSourcesFetched) {
      updateSourcesToDownload();
    } else {
      fetchSourcesIfNeeded().then(() => {
        updateSourcesToDownload();
      });
    }
  }, [modelId, sourceSetName, getSource, getSourcesForModelAndSet, isSourcesFetched, fetchSourcesIfNeeded]);

  const handleDownloadAll = async () => {
    try {
      setIsDownloading(true);
      setTotal(sourcesToDownload.length);
      setProgress(0);

      // Call onDownloadStart callback if provided
      onDownloadStart?.();

      // Download sources sequentially
      for (const source of sourcesToDownload) {
        try {
          await downloadSource(modelId, source);
        } catch (error) {
          console.error(`Failed to download ${source}:`, error);
          // Continue with next source even if one fails
        }
        setProgress((prev) => prev + 1);
      }
    } catch (error) {
      console.error('Error downloading all sources:', error);
      alert('Failed to download all sources');
    } finally {
      setIsDownloading(false);
      setProgress(0);
      setTotal(0);
    }
  };

  return (
    <Button
      variant="default"
      size="sm"
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (
          // eslint-disable-next-line no-alert
          confirm(
            'Are you sure you want to download all sources? This may take a long time and if aborted, you will need to manually re-sync the interrupted download.',
          )
        ) {
          handleDownloadAll();
        }
      }}
      disabled={isDownloading || sourcesToDownload.length === 0 || isAnyDownloadRunning}
      className={`h-7 font-sans ${!isDownloading && sourcesToDownload.length === 0 ? 'hidden' : ''}`}
    >
      {isDownloading ? `Downloading ${progress}/${total}...` : sourcesToDownload.length > 0 ? `Download All` : ''}
    </Button>
  );
}
