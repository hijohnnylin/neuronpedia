// this is used to delimit the source set name in the layer string. for example: "3-res-jb__blah" = layer 3, sourceSet: res-jb, and BLAH can be anything
// this allows us to have SourceIds that are not exactly the suffix of the Source Set name

import {
  ModelWithPartialRelations,
  SourceReleaseWithPartialRelations,
  SourceSetWithPartialRelations,
  SourceWithPartialRelations,
} from '@/prisma/generated/zod';
import { Visibility } from '@prisma/client';

// if this delimiter doesn't exist, then there is no additional info
export const DELIMITER_BETWEEN_SOURCESET_AND_ADDITIONAL_INFO = '__';

export const EMBED_LAYER_STRING = 'e'; // embed for pythia-70m-deduped is "e-res-sm"
export const NEURONS_SOURCESET = 'neurons';

export function isNeuronLayerSource(source: string) {
  // this just means if it's an integer
  return /^-?\d+$/.test(source);
}

export function getLayerNumFromSource(source: string) {
  if (isNeuronLayerSource(source)) {
    return parseInt(source, 10);
  }
  const firstPart = source.split('-')[0];
  if (firstPart === EMBED_LAYER_STRING) {
    return 0;
  }
  return parseInt(source.split('-')[0], 10);
}
export function getLayerNumAsStringFromSource(source: string) {
  if (isNeuronLayerSource(source)) {
    return source;
  }
  const firstPart = source.split('-')[0];
  if (firstPart === EMBED_LAYER_STRING) {
    return 'Embed';
  }
  return source.split('-')[0];
}

export function isCanonicalSource(source: string) {
  return source.indexOf(DELIMITER_BETWEEN_SOURCESET_AND_ADDITIONAL_INFO) === -1;
}

export function getAdditionalInfoFromSource(source: string) {
  if (isCanonicalSource(source)) {
    return '';
  }
  return source.substring(
    source.indexOf(DELIMITER_BETWEEN_SOURCESET_AND_ADDITIONAL_INFO) +
      DELIMITER_BETWEEN_SOURCESET_AND_ADDITIONAL_INFO.length,
  );
}

export function getSourceSetNameFromSource(source: string) {
  if (isNeuronLayerSource(source)) {
    return NEURONS_SOURCESET;
  }
  if (source && source.indexOf('-') !== -1) {
    return source.substring(source.indexOf('-') + 1).split(DELIMITER_BETWEEN_SOURCESET_AND_ADDITIONAL_INFO)[0];
  }

  // this shouldn't happen
  return source;
}

export function getFirstLayerForSourceSet(sourceSet: SourceSetWithPartialRelations) {
  return sourceSet.sources?.sort((a, b) => getLayerNumFromSource(a.id || '') - getLayerNumFromSource(b.id || ''))[0];
}

export function getFirstSourceSetForRelease(
  release: SourceReleaseWithPartialRelations,
  visibility?: Visibility,
  onlyInferenceEnabled?: boolean,
  includeNoDashboards?: boolean,
) {
  return (
    (release.sourceSets
      ?.filter((ss) => {
        if (visibility) {
          if (visibility === Visibility.UNLISTED) {
            return ss.visibility === Visibility.UNLISTED || ss.visibility === Visibility.PUBLIC;
          }
          return ss.visibility === visibility;
        }
        return true;
      })
      .filter((ss) => {
        if (onlyInferenceEnabled) {
          // at least once of its sources needs to allow activation testing
          return ss.sources?.some((s) => s.inferenceEnabled);
        }
        return true;
      })
      .filter((ss) => {
        if (includeNoDashboards) {
          return true;
        }
        return ss.hasDashboards;
      })
      .sort((a, b) => a.name?.localeCompare(b.name || '') || 0)[0] as SourceSetWithPartialRelations) || undefined
  );
}

export function getDefaultSourceSetAndSourceForRelease(release: SourceReleaseWithPartialRelations) {
  let defaultSourceSet: SourceSetWithPartialRelations | undefined;
  let defaultSource: SourceWithPartialRelations | undefined;

  if (release.defaultSourceSetName) {
    defaultSourceSet = release.sourceSets?.find(
      (ss) => ss.name === release.defaultSourceSetName,
    ) as SourceSetWithPartialRelations;
    if (release.defaultSourceId) {
      defaultSource = defaultSourceSet?.sources?.find(
        (s) => s.id === release.defaultSourceId,
      ) as SourceWithPartialRelations;
    } else {
      defaultSource = getFirstLayerForSourceSet(
        defaultSourceSet as SourceSetWithPartialRelations,
      ) as SourceWithPartialRelations;
    }
  } else {
    const firstSourceSet = getFirstSourceSetForRelease(release);
    if (firstSourceSet) {
      defaultSourceSet = firstSourceSet as SourceSetWithPartialRelations;
      defaultSource = getFirstLayerForSourceSet(
        firstSourceSet as SourceSetWithPartialRelations,
      ) as SourceWithPartialRelations;
    }
  }

  return {
    defaultSourceSet: defaultSourceSet || undefined,
    defaultSource: defaultSource || undefined,
  };
}

export function getFirstSourceForRelease(
  release: SourceReleaseWithPartialRelations,
  visibility?: Visibility,
  onlyInferenceEnabled?: boolean,
  includeNoDashboards?: boolean,
) {
  return (
    getFirstSourceSetForRelease(release, visibility, onlyInferenceEnabled, includeNoDashboards)
      ?.sources?.sort((a, b) => a?.id?.localeCompare(b?.id || '') || 0)
      ?.filter((s) => {
        if (includeNoDashboards) {
          return true;
        }
        return s.hasDashboards;
      })
      ?.find((s) => {
        if (visibility) {
          if (visibility === Visibility.UNLISTED) {
            return s.visibility === Visibility.UNLISTED || s.visibility === Visibility.PUBLIC;
          }
          return s.visibility === visibility;
        }
        return true;
      }) || undefined
  );
}

export function getFirstSourceSetForModel(
  model: ModelWithPartialRelations,
  visibility?: Visibility,
  onlyInferenceEnabled?: boolean,
  includeNoDashboards?: boolean,
  onlyInferenceSearchEnabled?: boolean,
) {
  return (
    model.sourceSets
      ?.filter((ss) => {
        if (visibility) {
          if (visibility === Visibility.UNLISTED) {
            return ss.visibility === Visibility.UNLISTED || ss.visibility === Visibility.PUBLIC;
          }
          return ss.visibility === visibility;
        }
        return true;
      })
      .filter((ss) => {
        if (onlyInferenceEnabled) {
          // at least once of its sources needs to allow activation testing
          return ss.sources?.some((s) => s.inferenceEnabled);
        }
        return true;
      })
      .filter((ss) => {
        // axbench allows inference but not inference search
        if (onlyInferenceSearchEnabled) {
          return ss.allowInferenceSearch;
        }
        return true;
      })
      .filter((ss) => {
        if (includeNoDashboards) {
          return true;
        }
        return ss.hasDashboards;
      })
      .sort((a, b) => a?.name?.localeCompare(b?.name || '') || 0)[0] || undefined
  );
}

export function getFirstSourceForModel(
  model: ModelWithPartialRelations,
  visibility?: Visibility,
  onlyInferenceEnabled?: boolean,
  includeNoDashboards?: boolean,
) {
  return (
    getFirstSourceSetForModel(model, visibility, onlyInferenceEnabled, includeNoDashboards)
      ?.sources?.sort((a, b) => a?.id?.localeCompare(b?.id || '') || 0)
      ?.filter((s) => {
        if (includeNoDashboards) {
          return true;
        }
        return s.hasDashboards;
      })
      ?.find((s) => {
        if (visibility) {
          if (visibility === Visibility.UNLISTED) {
            return s.visibility === Visibility.UNLISTED || s.visibility === Visibility.PUBLIC;
          }
          return s.visibility === visibility;
        }
        return true;
      }) || undefined
  );
}
