'use client';

/* eslint-disable @typescript-eslint/no-shadow */

import { env } from '@/lib/env';
import { formatToGlobalModels } from '@/lib/utils/general';
import { getSourceSetNameFromSource, NEURONS_SOURCESET } from '@/lib/utils/source';
import {
  Bookmark,
  ExplanationModelType,
  ExplanationScoreModel,
  ExplanationScoreType,
  ExplanationType,
  User,
  Visibility,
} from '@prisma/client';
import * as Toast from '@radix-ui/react-toast';
import { usePathname } from 'next/navigation';
import {
  ModelWithPartialRelations,
  NeuronWithPartialRelations,
  SourceReleasePartialWithRelations,
  SourceReleaseWithPartialRelations,
  SourceSetWithPartialRelations,
  SourceWithPartialRelations,
  UserWithPartialRelations,
} from 'prisma/generated/zod';
import { Dispatch, ReactNode, SetStateAction, useEffect, useMemo, useState } from 'react';
import createContextWrapper from './provider-util';

// TODO: this file should be refactored - don't use this weird data structure, and move the utility functions out of the provider
export const [GlobalContext, useGlobalContext] = createContextWrapper<{
  refreshGlobal: () => void;
  globalModels: {
    [key: string]: ModelWithPartialRelations;
  };
  setGlobalModels: React.Dispatch<
    React.SetStateAction<{
      [key: string]: ModelWithPartialRelations;
    }>
  >;
  releases: SourceReleasePartialWithRelations[];
  getDefaultModel: (publicOnly?: boolean) => ModelWithPartialRelations | undefined;
  getInferenceEnabledForModel: (modelId: string) => boolean;
  getFirstInferenceEnabledModel: () => string | undefined;
  getInferenceEnabledModels: () => string[];
  getInferenceEnabledSourcesForModel: (modelId: string) => string[];
  getSource: (modelId: string, sourceId: string) => SourceWithPartialRelations | undefined;
  getSourceSet: (modelId: string, sourceSet: string) => SourceSetWithPartialRelations | undefined;
  getSourceSetForSource: (modelId: string, sourceId: string) => SourceSetWithPartialRelations | undefined;
  getSourceSetsForModelId: (modelId: string, publicOnly?: boolean) => SourceSetWithPartialRelations[];
  getReleaseForSourceSet: (modelId: string, sourceSet: string) => SourceReleaseWithPartialRelations | undefined;
  getFirstSourceForSourceSet: (modelId: string, sourceSet: string) => string;
  getSourcesForSourceSet: (
    modelId: string,
    sourceSet: string,
    returnUnlisted?: boolean,
    onlyInferenceEnabled?: boolean,
    includeNoDashboards?: boolean,
  ) => string[];
  getHasGraphsSourceSetsForModelId: (modelId: string) => SourceSetWithPartialRelations[];
  isGraphEnabledForSourceSet: (modelId: string, sourceSet: string) => boolean;
  isGraphEnabledForSource: (modelId: string, source: string) => boolean;
  explanationTypes: ExplanationType[];
  explanationModels: ExplanationModelType[];
  explanationScoreTypes: ExplanationScoreType[];
  explanationScoreModelTypes: ExplanationScoreModel[];

  // User
  user: UserWithPartialRelations | undefined;
  setUser: (user: UserWithPartialRelations) => void;
  refreshUser: () => void;
  bookmarks: Bookmark[];
  setBookmarks: (bookmarks: Bookmark[]) => void;

  // UI - User Popover
  showUserPopover: boolean;
  setShowUserPopover: React.Dispatch<React.SetStateAction<boolean>>;

  // UI - Sign In Modal
  signInModalOpen: boolean;
  setSignInModalOpen: Dispatch<SetStateAction<boolean>>;

  // UI - Toast
  showToastServerError: () => void;
  toastMessage: ReactNode;
  showToastMessage: (message: ReactNode) => void;
  toastOpen: boolean;
  setToastOpen: (open: boolean) => void;

  // UI - Feature Modal
  featureModalFeature: NeuronWithPartialRelations | undefined;
  setFeatureModalFeature: (feature: NeuronWithPartialRelations) => void;
  featureModalopen: boolean;
  setFeatureModalOpen: (open: boolean) => void;
}>('GlobalContext');

export default function GlobalProvider({
  children,
  initialModels,
  initialReleases,
  initialExplanationTypes,
  initialExplanationModels,
  initialExplanationScoreTypes,
  initialExplanationScoreModelTypes,
}: {
  children: React.ReactNode;
  initialModels: {
    [key: string]: ModelWithPartialRelations;
  };
  initialReleases: SourceReleaseWithPartialRelations[];
  initialExplanationTypes: ExplanationType[];
  initialExplanationModels: ExplanationModelType[];
  initialExplanationScoreTypes: ExplanationScoreType[];
  initialExplanationScoreModelTypes: ExplanationScoreModel[];
}) {
  const pathname = usePathname();
  const [globalModels, setGlobalModels] = useState<{
    [key: string]: ModelWithPartialRelations;
  }>(initialModels);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [releases, setReleases] = useState<SourceReleaseWithPartialRelations[]>(initialReleases);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [explanationTypes, setExplanationTypes] = useState<ExplanationType[]>(initialExplanationTypes);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [explanationModels, setExplanationModels] = useState<ExplanationModelType[]>(initialExplanationModels);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [explanationScoreTypes, setExplanationScoreTypes] =
    useState<ExplanationScoreType[]>(initialExplanationScoreTypes);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [explanationScoreModelTypes, setExplanationScoreModelTypes] = useState<ExplanationScoreModel[]>(
    initialExplanationScoreModelTypes,
  );
  const [signInModalOpen, setSignInModalOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState<ReactNode>('');
  const [toastOpen, setToastOpen] = useState(false);
  const [user, setUser] = useState<UserWithPartialRelations>();
  const [showUserPopover, setShowUserPopover] = useState(false);
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [featureModalFeature, setFeatureModalFeature] = useState<NeuronWithPartialRelations>();
  const [featureModalOpen, setFeatureModalOpen] = useState(false);

  useEffect(() => {
    if (featureModalFeature && !featureModalFeature?.activations) {
      // load activations since we don't have them
      fetch(`/api/feature/${featureModalFeature.modelId}/${featureModalFeature.layer}/${featureModalFeature.index}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      })
        .then((response) => response.json())
        .then((n: NeuronWithPartialRelations) => {
          setFeatureModalFeature(n);
        })
        .catch((error) => {
          console.error(`error submitting getting rest of neuron: ${error}`);
        });
    }
  }, [featureModalFeature]);

  const refreshUser = () => {
    fetch(`/api/user/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
      .then((response) => response.json())
      .then((userUpdated: User) => {
        setUser(userUpdated);
      });
  };

  const showToastMessage = (message: ReactNode) => {
    setToastOpen(false);
    setTimeout(() => {
      setToastMessage(message);
      setToastOpen(true);
    }, 100);
  };

  const showToastServerError = () => {
    setToastOpen(false);
    setTimeout(() => {
      setToastMessage(
        <div className="flex flex-col items-center justify-center">
          <div className="mb-1.5 rounded bg-red-500 px-2 py-0.5 text-xs text-white">Server Error</div>
          <div className="text-xs">
            Check{' '}
            <a
              href="https://status.neuronpedia.org"
              target="_blank"
              rel="noreferrer"
              className="text-sky-600 underline"
            >
              Neuronpedia Status
            </a>{' '}
            or submit an error report at{' '}
            <a
              href={`mailto:support@neuronpedia.org?subject=Error%20Report&body=Hey%2C%0A%0AI%20got%20a%20server%20error%20on%20this%20page%3A%20${pathname}%0A%0APlease%20look%20into%20it.%0A%0AThanks!`}
              target="_blank"
              rel="noreferrer"
              className="text-sky-600 underline"
            >
              support@neuronpedia.org
            </a>
            .
          </div>
        </div>,
      );
      setToastOpen(true);
      setTimeout(() => {
        setToastOpen(false);
      }, 5000);
    }, 100);
  };

  const getInferenceEnabledForModel = (modelId: string) => {
    if (env.NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS.includes(modelId)) {
      return true;
    }
    const model = globalModels[modelId];
    if (!model) {
      return false;
    }
    return model.inferenceEnabled;
  };

  const getInferenceEnabledModels = () =>
    Object.keys(globalModels).filter((modelId) => getInferenceEnabledForModel(modelId));

  const getFirstInferenceEnabledModel = () => {
    // eslint-disable-next-line no-restricted-syntax
    for (const modelId in globalModels) {
      if (getInferenceEnabledForModel(modelId)) {
        return modelId;
      }
    }
    return undefined;
  };

  const getReleaseForSourceSet = (modelId: string, sourceSet: string) =>
    releases.find((r) => r.sourceSets?.some((ss) => ss.modelId === modelId && ss.name === sourceSet));

  const getSource = (modelId: string, sourceId: string) => {
    if (!globalModels[modelId]) {
      return undefined;
    }
    const sourceSet = globalModels[modelId].sourceSets?.find((s) =>
      s.sources?.find((source) => source.id === sourceId),
    );
    if (sourceSet) {
      return sourceSet.sources?.find((source) => source.id === sourceId) as SourceWithPartialRelations;
    }
    return undefined;
  };

  const getSourceSetsForModelId = (modelId: string, publicOnly?: boolean) => {
    if (Object.keys(globalModels).includes(modelId)) {
      const ss = globalModels[modelId].sourceSets as SourceSetWithPartialRelations[];
      [...ss].sort((a, b) => {
        if (b.name === NEURONS_SOURCESET) {
          return -1;
        }
        return 1;
      });
      return ss.filter((s) => !publicOnly || s.visibility === Visibility.PUBLIC) || [];
    }
    return [];
  };

  const getInferenceEnabledSourcesForModel = (modelId: string) => {
    const ss = getSourceSetsForModelId(modelId);
    const toReturn: string[] = [];
    ss.forEach((s) => {
      toReturn.push(
        ...(s.sources?.filter((source) => source.inferenceEnabled).map((source) => source.id as string) || []),
      );
    });
    return toReturn;
  };

  const getDefaultModel = (publicOnly?: boolean) => {
    if (Object.keys(globalModels).length > 0) {
      if (publicOnly) {
        return Object.values(globalModels).find((m) => m.visibility === Visibility.PUBLIC);
      }
      return globalModels[Object.keys(globalModels)[0]];
    }
    return undefined;
  };

  const getSourcesForSourceSet = (
    modelId: string,
    sourceSet: string,
    returnUnlisted?: boolean,
    onlyInferenceEnabled?: boolean,
    includeNoDashboards?: boolean,
  ) => {
    const ss = getSourceSetsForModelId(modelId);
    let toReturn: string[] = [];
    ss.forEach((s) => {
      if (s.name === sourceSet) {
        toReturn =
          s.sources
            ?.filter((source) => {
              if (!returnUnlisted && source.visibility === Visibility.UNLISTED) {
                return false;
              }
              return true;
            })
            .filter((source) => {
              if (onlyInferenceEnabled) {
                return source.inferenceEnabled;
              }
              return true;
            })
            .filter((source) => {
              if (includeNoDashboards) {
                return true;
              }
              return source.hasDashboards;
            })
            .map((source) => source.id as string) || [];
      }
    });
    // natural sort
    toReturn.sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));
    return toReturn;
  };

  const getFirstSourceForSourceSet = (modelId: string, sourceSet: string) =>
    getSourcesForSourceSet(modelId, sourceSet, true)?.[0];

  const getSourceSetForSource = (modelId: string, sourceId: string) => {
    const ss = getSourceSetsForModelId(modelId);
    let toReturn: SourceSetWithPartialRelations | undefined;
    ss.forEach((s) => {
      s.sources?.forEach((source) => {
        if (source.id === sourceId) {
          toReturn = s;
        }
      });
    });
    return toReturn;
  };

  const getSourceSet = (modelId: string, sourceSet: string) => {
    const sss = getSourceSetsForModelId(modelId);
    let toReturn: SourceSetWithPartialRelations | undefined;
    sss.forEach((ss) => {
      if (ss.name === sourceSet) {
        toReturn = ss;
      }
    });
    return toReturn;
  };

  const getHasGraphsSourceSetsForModelId = (modelId: string) => {
    const ss = getSourceSetsForModelId(modelId);
    return ss.filter((s) => s.hasGraphs);
  };

  const isGraphEnabledForSourceSet = (modelId: string, sourceSet: string) => {
    // for transcoders / graphs we have a different steering method/server, so we don't show it here
    const sourceSetObj = getSourceSet(modelId, sourceSet);
    if (!sourceSetObj) {
      return false;
    }
    return sourceSetObj.graphEnabled;
  };

  const isGraphEnabledForSource = (modelId: string, source: string) => {
    // for transcoders / graphs we have a different steering method/server, so we don't show it here
    const sourceSet = getSourceSet(modelId, getSourceSetNameFromSource(source) || '');
    if (!sourceSet) {
      return false;
    }
    return sourceSet.graphEnabled;
  };

  const refreshGlobal = () => {
    fetch('/api/global', {
      method: 'POST',
      body: JSON.stringify({}),
    })
      .then((response) => response.json())
      .then(
        ({
          models,
          releases,
          explanationTypes,
          explanationModels,
          explanationScoreTypes,
          explanationScoreModelTypes,
        }) => {
          setGlobalModels(formatToGlobalModels(models));
          for (const release of releases) {
            if (release.createdAt) {
              release.createdAt = new Date(release.createdAt);
            }
          }
          setReleases(releases);
          setExplanationTypes(explanationTypes);
          setExplanationModels(explanationModels);
          setExplanationScoreTypes(explanationScoreTypes);
          setExplanationScoreModelTypes(explanationScoreModelTypes);
        },
      );
  };

  return (
    <GlobalContext.Provider
      value={useMemo(
        () => ({
          refreshGlobal,
          globalModels,
          setGlobalModels,
          releases,
          getDefaultModel,
          getInferenceEnabledForModel,
          getInferenceEnabledModels,
          getFirstInferenceEnabledModel,
          getReleaseForSourceSet,
          getSource,
          getSourceSetForSource,
          getSourceSet,
          getSourcesForSourceSet,
          getFirstSourceForSourceSet,
          getSourceSetsForModelId,
          getInferenceEnabledSourcesForModel,
          getHasGraphsSourceSetsForModelId,
          isGraphEnabledForSourceSet,
          isGraphEnabledForSource,
          explanationTypes,
          explanationModels,
          explanationScoreTypes,
          explanationScoreModelTypes,
          signInModalOpen,
          setSignInModalOpen,
          showToastServerError,
          toastMessage,
          showToastMessage,
          toastOpen,
          setToastOpen,
          user,
          setUser,
          refreshUser,
          bookmarks,
          setBookmarks,
          showUserPopover,
          setShowUserPopover,
          featureModalFeature,
          setFeatureModalFeature,
          featureModalopen: featureModalOpen,
          setFeatureModalOpen,
        }),
        [
          bookmarks,
          explanationModels,
          explanationScoreModelTypes,
          explanationScoreTypes,
          explanationTypes,
          getDefaultModel,
          getInferenceEnabledForModel,
          getInferenceEnabledModels,
          getFirstInferenceEnabledModel,
          getInferenceEnabledSourcesForModel,
          getFirstSourceForSourceSet,
          getReleaseForSourceSet,
          getSource,
          getSourceSet,
          getSourceSetForSource,
          getSourceSetsForModelId,
          getSourcesForSourceSet,
          getHasGraphsSourceSetsForModelId,
          isGraphEnabledForSourceSet,
          isGraphEnabledForSource,
          globalModels,
          featureModalFeature,
          featureModalOpen,
          releases,
          showToastServerError,
          showUserPopover,
          signInModalOpen,
          toastMessage,
          toastOpen,
          user,
        ],
      )}
    >
      <Toast.Provider swipeDirection="up" duration={3000} swipeThreshold={10}>
        {children}
      </Toast.Provider>
    </GlobalContext.Provider>
  );
}
