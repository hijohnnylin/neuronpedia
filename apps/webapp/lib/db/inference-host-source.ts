import { prisma } from '@/lib/db';
import { InferenceEngine, InferenceHostSource, InferenceHostSourceOnSource } from '@prisma/client';
import { IS_DOCKER_COMPOSE, USE_LOCALHOST_INFERENCE } from '../env';
import { AuthenticatedUser } from '../with-user';
import { getSourceInferenceHosts } from './source';
import { userCanAccessModelAndSourceSet } from './userCanAccess';

export const LOCALHOST_INFERENCE_HOST = IS_DOCKER_COMPOSE ? 'http://inference:5002' : 'http://127.0.0.1:5002';

export const createInferenceHostSource = async (input: InferenceHostSource) =>
  prisma.inferenceHostSource.create({
    data: {
      ...input,
    },
  });

export const getInferenceHostSourceById = async (id: string) =>
  prisma.inferenceHostSource.findUnique({ where: { id } });

export const createInferenceHostSourceOnSource = async (input: InferenceHostSourceOnSource) =>
  prisma.inferenceHostSourceOnSource.create({
    data: { ...input },
  });

export const getAllServerHostsForSourceSet = async (
  modelId: string,
  sourceSetName: string,
  engine: InferenceEngine = InferenceEngine.TRANSFORMER_LENS,
) => {
  const sources = await prisma.source.findMany({
    where: {
      modelId,
      setName: sourceSetName,
    },
    include: {
      inferenceHosts: { where: { inferenceHost: { engine } }, include: { inferenceHost: true } },
    },
  });

  // Flatten the array of arrays into a single array of unique host URLs
  const allHosts = sources.flatMap((source) => source.inferenceHosts.map((host) => host.inferenceHost.hostUrl));
  return allHosts;
};

export const getAllServerHostsForModel = async (
  modelId: string,
  engine: InferenceEngine = InferenceEngine.TRANSFORMER_LENS,
) => {
  const sources = await prisma.source.findMany({
    where: {
      modelId,
    },
    include: {
      inferenceHosts: { where: { inferenceHost: { engine } }, include: { inferenceHost: true } },
    },
  });

  // Flatten the array of arrays into a single array of unique host URLs
  const allHosts = sources.flatMap((source) => source.inferenceHosts.map((host) => host.inferenceHost.hostUrl));
  return allHosts;
};

export const getOneRandomServerHostForSourceSet = async (
  modelId: string,
  sourceSetName: string,
  user: AuthenticatedUser | null = null,
  engine: InferenceEngine = InferenceEngine.TRANSFORMER_LENS,
) => {
  const canAccess = await userCanAccessModelAndSourceSet(modelId, sourceSetName, user, true);
  if (!canAccess) {
    return null;
  }

  if (USE_LOCALHOST_INFERENCE) {
    return LOCALHOST_INFERENCE_HOST;
  }

  // TODO: we don't currently support search-all on different instances, so we assume these instances are all the same
  const hosts = await getAllServerHostsForSourceSet(modelId, sourceSetName, engine);
  if (hosts.length === 0) {
    return null;
  }

  // pick a random one
  const randomIndex = Math.floor(Math.random() * hosts.length);
  return hosts[randomIndex];
};

export const getOneRandomServerHostForSource = async (
  modelId: string,
  sourceId: string,
  user: AuthenticatedUser | null = null,
  engine: InferenceEngine = InferenceEngine.TRANSFORMER_LENS,
) => {
  if (USE_LOCALHOST_INFERENCE) {
    return LOCALHOST_INFERENCE_HOST;
  }

  const hosts = await getSourceInferenceHosts(modelId, sourceId, user, engine);
  if (!hosts) {
    throw new Error('Source not found.');
  }

  const randomIndex = Math.floor(Math.random() * hosts.length);
  return hosts[randomIndex].inferenceHost.hostUrl;
};

export const getOneRandomServerHostForModel = async (
  modelId: string,
  engine: InferenceEngine = InferenceEngine.TRANSFORMER_LENS,
) => {
  if (USE_LOCALHOST_INFERENCE) {
    return LOCALHOST_INFERENCE_HOST;
  }

  let hosts = await getAllServerHostsForModel(modelId, engine);
  if (hosts.length === 0) {
    throw new Error('No hosts found.');
  }

  // unique the hosts
  hosts = [...new Set(hosts)];

  return hosts[0];
};

export const getTwoRandomServerHostsForModel = async (
  modelId: string,
  engine: InferenceEngine = InferenceEngine.TRANSFORMER_LENS,
) => {
  if (USE_LOCALHOST_INFERENCE) {
    return [LOCALHOST_INFERENCE_HOST, LOCALHOST_INFERENCE_HOST];
  }

  let hosts = await getAllServerHostsForModel(modelId, engine);
  if (hosts.length === 0) {
    throw new Error('No hosts found.');
  }

  // unique the hosts
  hosts = [...new Set(hosts)];

  if (hosts.length < 2) {
    return [hosts[0], hosts[0]];
  }
  // pick two random, different ones
  const randomIndex = Math.floor(Math.random() * hosts.length);
  let randomIndex2 = Math.floor(Math.random() * hosts.length);
  while (randomIndex2 === randomIndex) {
    randomIndex2 = Math.floor(Math.random() * hosts.length);
  }

  return [hosts[randomIndex], hosts[randomIndex2]];
};

export const getTwoRandomServerHostsForSourceSet = async (
  modelId: string,
  sourceSetName: string,
  user: AuthenticatedUser | null = null,
  engine: InferenceEngine = InferenceEngine.TRANSFORMER_LENS,
) => {
  if (USE_LOCALHOST_INFERENCE) {
    return [LOCALHOST_INFERENCE_HOST, LOCALHOST_INFERENCE_HOST];
  }

  // ensure we can access the sourceSet
  const canAccess = await userCanAccessModelAndSourceSet(modelId, sourceSetName, user, true);
  if (!canAccess) {
    throw new Error('Source set not found.');
  }

  // TODO: we don't currently support search-all on different instances, so we assume these instances are all the same
  let hosts = await getAllServerHostsForSourceSet(modelId, sourceSetName, engine);
  if (hosts.length === 0) {
    throw new Error('No hosts found.');
  }

  // unique the hosts
  hosts = [...new Set(hosts)];

  if (hosts.length < 2) {
    return [hosts[0], hosts[0]];
  }
  // pick two random, different ones
  const randomIndex = Math.floor(Math.random() * hosts.length);
  let randomIndex2 = Math.floor(Math.random() * hosts.length);
  while (randomIndex2 === randomIndex) {
    randomIndex2 = Math.floor(Math.random() * hosts.length);
  }

  return [hosts[randomIndex], hosts[randomIndex2]];
};
