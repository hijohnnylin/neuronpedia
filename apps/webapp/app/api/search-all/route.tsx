// TODO: rewrite this entirely. has a bunch of lazy var things going on.

import {
  InferenceActivationAllResponse,
  InferenceActivationAllResult,
} from '@/components/provider/inference-activation-all-provider';
import { badRequest } from '@/lib/api-error';
import { ActivationAllBatchResponse, ActivationAllResponse } from '@/lib/api/inference-types';
import { prisma } from '@/lib/db';
import { createInferenceActivationsAndReturn } from '@/lib/db/activation';
import { getNeuronsForSearcher } from '@/lib/db/neuron';
import { assertUserCanAccessModelAndSourceSet } from '@/lib/db/userCanAccess';
import {
  DEMO_MODE,
  INFERENCE_ACTIVATION_USER_ID_DO_NOT_INCLUDE_IN_PUBLIC_ACTIVATIONS,
  PUBLIC_ACTIVATIONS_USER_IDS,
} from '@/lib/env';
import { InferenceServerError, runInferenceActivationAll } from '@/lib/utils/inference';
import { Prisma } from '@prisma/client';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// Hobby plans don't support > 60 seconds
// export const maxDuration = 120;

const NUMBER_TOP_RESULTS = 50;

/**
 * Run the search, and retry once without the sort indexes if inference rejects them.
 *
 * A saved search can outlive the source set it was made against, so a bookmarked URL can
 * carry token indexes that no longer address anything. Inference answers that with a 5xx,
 * which would otherwise surface as a dead search page rather than as the unsorted results
 * the caller can still use. Returns the indexes actually used so the response and the cache
 * write agree with what was run.
 */
async function runInferenceActivationAllWithSortFallback(
  modelId: string,
  sourceSetName: string,
  text: string | string[],
  numResults: number,
  selectedLayers: string[],
  sortIndexes: number[],
  ignoreBos: boolean,
  user: RequestOptionalUser['user'],
): Promise<{
  result: ActivationAllBatchResponse | ActivationAllResponse;
  sortIndexes: number[];
}> {
  const run = (indexes: number[]) =>
    runInferenceActivationAll(
      modelId,
      sourceSetName,
      text,
      numResults,
      selectedLayers,
      indexes,
      ignoreBos,
      user,
    ) as Promise<ActivationAllBatchResponse | ActivationAllResponse>;

  try {
    return { result: await run(sortIndexes), sortIndexes };
  } catch (error) {
    // Only a server-side failure with indexes to drop is worth retrying. A 4xx means the
    // request was wrong in some other way, and re-sending it without the sort would just
    // fail again more slowly.
    if (!(error instanceof InferenceServerError) || sortIndexes.length === 0 || error.status < 500) {
      throw error;
    }

    console.warn('Retrying search-all without stale sort indexes:', sortIndexes, error.message);
    return { result: await run([]), sortIndexes: [] };
  }
}
const DEFAULT_DENSITY_THRESHOLD = -1;

/**
 * @swagger
 * /api/search-all:
 *   post:
 *     summary: Top Features for Entire Text
 *     description: Returns the top features for a given text input. Equivalent to the https://neuronpedia.org/search functionality. Contact us to increase your rate limit for free if you hit it.
 *     tags:
 *       - Search via Inference
 *     security:
 *       - apiKey: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               modelId:
 *                 description: The model to search.
 *                 type: string
 *                 required: true
 *                 default: gpt2-small
 *               sourceSet:
 *                 description: The SAE set to search.
 *                 type: string
 *                 required: true
 *                 default: res-jb
 *               text:
 *                 oneOf:
 *                   - type: string
 *                   - type: array
 *                     items:
 *                       type: string
 *                 description: The custom text to run through the model. Either a single string or an array of strings.
 *                 required: true
 *                 default: hello world
 *               selectedLayers:
 *                 description: The SAE IDs to search. Use [] to search all SAEs in this SAE set.
 *                 type: array
 *                 items:
 *                   type: string
 *                 default:
 *                   - 6-res-jb
 *               sortIndexes:
 *                 description: The token(s) to sort by. Specify multiple to sort by the sum of the selected tokens. Use [] to sort by max activation of any token (default behavior). In this "hello world" example, a <|endoftext|> is automatically prepended, so sorting by index 1 means we sort by token " hello".
 *                 type: array
 *                 items:
 *                   type: number
 *                 default:
 *                   - 1
 *               ignoreBos:
 *                 description: Don't return results where the top token activation is the BOS token.
 *                 type: boolean
 *                 default: true
 *               densityThreshold:
 *                 description: Don't return features with a density greater than this threshold. Should be between 0 and 1. -1 means no threshold (default).
 *                 type: number
 *                 default: -1
 *               numResults:
 *                 description: The max number of results to return. May return fewer if density threshold is used. Max is 100.
 *                 type: number
 *                 default: 50
 *     responses:
 *       200:
 *         description: Successful search with results
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 tokens:
 *                   type: array
 *                   items:
 *                     type: string
 *                 result:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       modelId:
 *                         type: string
 *                       layer:
 *                         type: string
 *                       index:
 *                         type: string
 *                       values:
 *                         type: array
 *                         items:
 *                           type: number
 *                       maxValue:
 *                         type: number
 *                       maxValueIndex:
 *                         type: number
 *       400:
 *         description: Invalid request body or missing search text.
 *       500:
 *         description: Internal server error during the search process.
 */

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  if (body.text === undefined || body.text === null || body.text === '') {
    throw badRequest('Missing search text.');
  }

  console.log(body);

  const { modelId } = body;
  const selectedLayers = ((body.selectedLayers || []) as string[]).sort();
  const requestedSortIndexes = ((body.sortIndexes || []) as number[]).sort();
  const sourceSetName = body.sourceSet;
  // What the search actually ran with, which the fallback above may narrow to [].
  let effectiveSortIndexes = requestedSortIndexes;

  const numResults = body.numResults || NUMBER_TOP_RESULTS;
  if (numResults < 1) {
    throw badRequest('numResults must be greater than 0.');
  } else if (numResults > 100) {
    throw badRequest('numResults must be less than 100.');
  }

  const densityThreshold = body.densityThreshold || DEFAULT_DENSITY_THRESHOLD;
  if (densityThreshold !== DEFAULT_DENSITY_THRESHOLD && (densityThreshold <= 0 || densityThreshold >= 1)) {
    throw badRequest('densityThreshold must be between 0 and 1.');
  }

  // Throws a 404 ApiError, which the wrapper renders. Catching it here to return 500 was
  // reporting a caller's typo as a Neuronpedia outage.
  await assertUserCanAccessModelAndSourceSet(modelId, sourceSetName, request.user);

  // if it's a batch search, we don't need to check savedSearch or fetch the feature
  if (Array.isArray(body.text)) {
    const { result: batchResult, sortIndexes: batchSortIndexes } = await runInferenceActivationAllWithSortFallback(
      modelId,
      sourceSetName,
      body.text,
      numResults,
      selectedLayers,
      requestedSortIndexes,
      body.ignoreBos,
      request.user,
    );
    const resultsBatch = batchResult as ActivationAllBatchResponse;

    const batchResults: InferenceActivationAllResponse[] = [];
    resultsBatch.results.forEach((promptSearchAllResult) => {
      const result: InferenceActivationAllResponse = {
        tokens: promptSearchAllResult.tokens,
        result: promptSearchAllResult.activations.map((activation) => ({
          modelId,
          layer: activation.source,
          index: activation.index.toString(),
          maxValue: activation.maxValue,
          maxValueIndex: activation.maxValueIndex,
          values: activation.values,
          neuron: undefined,
          dfaValues: activation.dfaValues ?? undefined,
          dfaTargetIndex: activation.dfaTargetIndex ?? undefined,
          dfaMaxValue: activation.dfaMaxValue ?? undefined,
        })),
        sortIndexes: batchSortIndexes,
      };
      batchResults.push(result);
    });

    return NextResponse.json({ results: batchResults });
  }
  // see if we found this before
  const savedSearch = await prisma.savedSearch.findUnique({
    where: {
      modelId_query: {
        modelId,
        query: body.text,
        selectedLayers,
        sortByIndexes: requestedSortIndexes,
        sourceSet: sourceSetName,
        ignoreBos: body.ignoreBos,
        numResults,
        densityThreshold,
      },
    },
    include: {
      activations: {
        orderBy: {
          order: 'asc',
        },
        include: {
          activation: {
            include: {
              neuron: {
                include: {
                  activations: {
                    orderBy: {
                      maxValue: 'desc',
                    },
                    take: 1,
                    where: {
                      creatorId: {
                        in: PUBLIC_ACTIVATIONS_USER_IDS,
                      },
                    },
                  },
                  explanations: {
                    // include: {
                    //   author: {
                    //     select: {
                    //       name: true,
                    //     },
                    //   },
                    //   votes: true,
                    // },
                    orderBy: [{ scoreV2: 'desc' }, { scoreV1: 'desc' }],
                  },
                  model: true,
                },
              },
            },
          },
        },
      },
    },
  });

  let hasMissingNeuron = false;

  if (savedSearch) {
    console.log('found saved search');
    const activations = savedSearch.activations.map((a) => a.activation);
    // eslint-disable-next-line
    var { tokens } = savedSearch;
    // eslint-disable-next-line
    var searchResults: InferenceActivationAllResult[] = [];
    activations.forEach((activation) => {
      searchResults.push({
        modelId,
        layer: activation.layer,
        index: activation.neuron.index,
        maxValue: activation.maxValue,
        maxValueIndex: activation.maxValueTokenIndex,
        values: activation.values,
        neuron: activation.neuron,
        dfaValues: activation.dfaValues ?? undefined,
        dfaMaxValue: activation.dfaMaxValue !== null ? activation.dfaMaxValue : undefined,
        dfaTargetIndex: activation.dfaTargetIndex !== null ? activation.dfaTargetIndex : undefined,
      });
    });
  } else {
    console.log('no saved search found');
    const { result: inferenceResult, sortIndexes: usedSortIndexes } =
      await runInferenceActivationAllWithSortFallback(
        modelId,
        sourceSetName,
        body.text,
        numResults,
        selectedLayers,
        requestedSortIndexes,
        body.ignoreBos,
        request.user,
      );
    const result = inferenceResult as ActivationAllResponse;
    effectiveSortIndexes = usedSortIndexes;

    console.log('got activations: ', result.activations.length);
    console.log('got tokens: ', result.tokens.length);

    const neuronData = await getNeuronsForSearcher(modelId, sourceSetName, result, request.user);
    console.log('got neurons');
    // var so that it can be accessed later in the outer scope
    // eslint-disable-next-line
    var { tokens } = result;
    const { activations } = result;

    // create searchresults
    // eslint-disable-next-line
    var searchResults: InferenceActivationAllResult[] = [];
    activations.forEach((activation) => {
      const neuron = neuronData.find(
        (neuronDataNeuron) =>
          neuronDataNeuron.index === activation.index.toString() && neuronDataNeuron.layer === activation.source,
      );
      if (!neuron) {
        console.log(`couldnt find neuron for activation: ${activation.index}`);
        hasMissingNeuron = true;

        searchResults.push({
          modelId,
          layer: activation.source,
          index: activation.index.toString(),
          maxValue: activation.maxValue,
          maxValueIndex: activation.maxValueIndex,
          values: activation.values,
          neuron: undefined,
          dfaValues: activation.dfaValues ?? undefined,
          dfaTargetIndex: activation.dfaTargetIndex ?? undefined,
          dfaMaxValue: activation.dfaMaxValue ?? undefined,
        });
        return;
      }
      if (densityThreshold !== DEFAULT_DENSITY_THRESHOLD && neuron.frac_nonzero > densityThreshold) {
        // don't save this because it's too dense
        console.log(`skipping dense neuron: ${neuron.index}`);
        return;
      }
      if (
        (effectiveSortIndexes.length === 0 && activation.maxValue > 0) ||
        (effectiveSortIndexes.length > 0 && activation.sumValues != null && activation.sumValues > 0)
      ) {
        searchResults.push({
          modelId,
          layer: neuron.layer,
          index: neuron.index,
          maxValue: activation.maxValue,
          maxValueIndex: activation.maxValueIndex,
          values: activation.values,
          neuron,
          dfaValues: activation.dfaValues ?? undefined,
          dfaTargetIndex: activation.dfaTargetIndex ?? undefined,
          dfaMaxValue: activation.dfaMaxValue ?? undefined,
        });
      }
    });
  }

  const toReturn: InferenceActivationAllResponse = {
    tokens,

    result: searchResults,
    sortIndexes: effectiveSortIndexes,
  };

  // if not a cached retrieval, make savedsearch
  if (!savedSearch && !hasMissingNeuron) {
    // look up the userid to use for creating search
    if (request.user) {
      // eslint-disable-next-line
      var userIdForSearch = request.user.id;
    } else {
      // eslint-disable-next-line
      var userIdForSearch = INFERENCE_ACTIVATION_USER_ID_DO_NOT_INCLUDE_IN_PUBLIC_ACTIVATIONS;
    }

    // create all the activations first
    // then connect all
    const toCreateMany: {
      tokens: string[];
      index: string;
      layer: string;
      modelId: string;
      maxValue: number;
      maxValueTokenIndex: number;
      minValue: number;
      values: number[];
      dfaValues: number[] | undefined;
      dfaTargetIndex: number | undefined;
      dfaMaxValue: number | undefined;
      creatorId: string;
    }[] = [];

    console.log(`creating: ${searchResults.length}`);

    searchResults.forEach((searchResult) => {
      if (searchResult.neuron) {
        toCreateMany.push({
          tokens,
          index: searchResult.index,
          layer: searchResult.layer,
          modelId,
          maxValue: searchResult.maxValue,
          maxValueTokenIndex: searchResult.maxValueIndex,
          minValue: Math.min(...searchResult.values),
          values: searchResult.values,
          dfaValues: searchResult.dfaValues,
          dfaTargetIndex: searchResult.dfaTargetIndex,
          dfaMaxValue: searchResult.dfaMaxValue,

          creatorId: userIdForSearch,
        });
      }
    });

    const actIds = await createInferenceActivationsAndReturn(modelId, sourceSetName, toCreateMany, request.user);

    console.log(`${actIds.length} activations created`);

    // make actIds the same order as searchResult
    const matchingActIds: {
      id: string;
      modelId: string;
      layer: string;
      index: string;
    }[] = [];
    toReturn.result.forEach((r) => {
      actIds.forEach((actId: any) => {
        if (actId.modelId === r.modelId && actId.layer === r.layer && actId.index === r.index) {
          matchingActIds.push(actId);
        }
      });
    });

    if (DEMO_MODE) {
      console.log('skipping saved search creation in demo mode');
    } else {
      // The cache lookup above keyed on the REQUESTED indexes; this writes the EFFECTIVE
      // ones. When the fallback narrows them to [], that row may already exist from an
      // earlier unsorted search of the same text, so the unique constraint can fire on a
      // path the lookup could not have predicted. The search itself already succeeded, and
      // the cache is an optimization, so a duplicate is a skip rather than a failure.
      try {
        const savedSearch = await prisma.savedSearch.create({
          data: {
            modelId,
            query: body.text,
            selectedLayers,
            sortByIndexes: effectiveSortIndexes,

            tokens,

            userId: userIdForSearch,
            sourceSet: sourceSetName,
            ignoreBos: body.ignoreBos,
            numResults,
            densityThreshold,
          },
        });

        console.log('savedSearchCreated');

        // create the connections
        const toConnectNew: {
          savedSearchId: string;
          activationId: string;
          order: number;
        }[] = [];

        matchingActIds.forEach((item, i) => {
          toConnectNew.push({
            order: i,
            savedSearchId: savedSearch.id,
            activationId: item.id,
          });
        });

        await prisma.savedSearchActivation.createMany({
          data: toConnectNew,
        });
        console.log(`connections created: ${toConnectNew.length}`);
      } catch (error) {
        if (!(error instanceof Prisma.PrismaClientKnownRequestError) || error.code !== 'P2002') {
          throw error;
        }
        console.warn('Skipping duplicate savedSearch cache write for search-all fallback');
      }
    }
  }

  return NextResponse.json(toReturn);
});
