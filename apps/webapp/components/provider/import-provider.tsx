'use client';

import { describeImportError } from '@/lib/utils/import-error';
import { getS3ModelsToSources } from '@/lib/utils/s3';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { createContext, useContext, useMemo, useState } from 'react';
import { useGlobalContext } from './global-provider';

interface ImportState {
  isDownloading: boolean;
  error: string;
  progress: number;
  progressText: string;
}

export interface ImportContextType {
  downloadSource: (modelId: string, sourceId: string) => Promise<void>;
  getImportState: (modelId: string, sourceId: string) => ImportState;
  isAnyDownloadRunning: boolean;
  getSourcesForModelAndSet: (modelId: string, sourceSetName: string) => string[] | undefined;
  fetchSourcesIfNeeded: () => Promise<void>;
  isSourcesFetched: boolean;
}

const ImportContext = createContext<ImportContextType | undefined>(undefined);

export function useImportContext() {
  const context = useContext(ImportContext);
  if (!context) {
    throw new Error('useImportContext must be used within an ImportProvider');
  }
  return context;
}

export function ImportProvider({ children }: { children: React.ReactNode }) {
  const { refreshGlobal } = useGlobalContext();
  const [importStates, setImportStates] = useState<Record<string, ImportState>>({});
  const [allModelSources, setAllModelSources] = useState<Record<string, string[]>>({});
  const [isSourcesFetched, setIsSourcesFetched] = useState(false);
  const isAnyDownloadRunning = useMemo(
    () => Object.values(importStates).some((state) => state.isDownloading),
    [importStates],
  );

  const getImportState = useMemo(
    () => (modelId: string, sourceId: string) => {
      const key = `${modelId}-${sourceId}`;
      return (
        importStates[key] || {
          isDownloading: false,
          error: '',
          progress: 0,
          progressText: '',
        }
      );
    },
    [importStates],
  );

  const updateImportState = useMemo(
    () => (modelId: string, sourceId: string, update: Partial<ImportState>) => {
      const key = `${modelId}-${sourceId}`;
      setImportStates((prev) => ({
        ...prev,
        [key]: {
          ...prev[key],
          ...update,
        },
      }));
    },
    [],
  );

  const downloadSource = async (modelId: string, sourceId: string) => {
    updateImportState(modelId, sourceId, {
      isDownloading: true,
      error: '',
      progress: 0,
      progressText: '',
    });

    const eventSource = new EventSource(`/api/admin/import?modelId=${modelId}&sourceId=${sourceId}`);

    try {
      await new Promise<void>((resolve, reject) => {
        eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.progress) {
            updateImportState(modelId, sourceId, {
              progress: data.progress,
              progressText: data.progressText || '',
            });
          }
        };

        eventSource.onerror = (e) => {
          eventSource.close();
          reject(e);
        };

        eventSource.addEventListener('complete', () => {
          eventSource.close();
          refreshGlobal();
          resolve();
        });
      });

      refreshGlobal();
    } catch (e) {
      updateImportState(modelId, sourceId, {
        error: `Error: ${describeImportError(e)}`,
      });
      throw e;
    } finally {
      eventSource.close();
      updateImportState(modelId, sourceId, {
        isDownloading: false,
        progress: 0,
        progressText: '',
      });
    }
  };

  const fetchSourcesIfNeeded = async () => {
    if (!isSourcesFetched) {
      const sources = await getS3ModelsToSources();
      setAllModelSources(sources);
      setIsSourcesFetched(true);
    }
  };

  const getSourcesForModelAndSet = (modelId: string, sourceSetName: string) => {
    if (!allModelSources[modelId]) return undefined;
    return allModelSources[modelId].filter((source) => getSourceSetNameFromSource(source) === sourceSetName);
  };

  const value = useMemo(
    () => ({
      downloadSource,
      getImportState,
      isAnyDownloadRunning,
      getSourcesForModelAndSet,
      fetchSourcesIfNeeded,
      isSourcesFetched,
    }),
    [
      downloadSource,
      getImportState,
      isAnyDownloadRunning,
      getSourcesForModelAndSet,
      fetchSourcesIfNeeded,
      isSourcesFetched,
    ],
  );

  return <ImportContext.Provider value={value}>{children}</ImportContext.Provider>;
}
