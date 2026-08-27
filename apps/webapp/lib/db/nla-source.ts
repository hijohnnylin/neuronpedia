import { ComputeService } from '@prisma/client';
import { computeFetch } from './compute-host';

/**
 * POST to the NLA server for a source, with failover across its hosts.
 *
 * The identifiers are optional because most NLA routes thread them through as
 * optional query parameters; without both there is nothing to resolve.
 *
 * `init.body` must be replayable across attempts, so a string or a Uint8Array
 * rather than a ReadableStream.
 */
export const nlaFetch = (
  modelId: string | undefined,
  nlaSourceId: string | undefined,
  path: string,
  init: Omit<RequestInit, 'headers'> & { headers?: HeadersInit } = {},
) => {
  if (!modelId || !nlaSourceId) {
    throw new Error(`NLA request needs both a model and a source (got model "${modelId}", source "${nlaSourceId}")`);
  }
  return computeFetch({ service: ComputeService.NLA, modelId, nlaSourceId }, path, init);
};
