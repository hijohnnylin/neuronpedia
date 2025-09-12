import { env } from '@/lib/env';
import { AuthenticatedUser } from '@/lib/types/auth';
import { prisma } from '../db';
import { userCanAccessModelAndSourceSet } from './userCanAccess';

export const LOCALHOST_GRAPH_HOST = env.IS_DOCKER_COMPOSE ? 'http://graph:5004' : 'http://127.0.0.1:5004';

export const getSourceSetGraphHosts = async (
  modelId: string,
  sourceSetName: string,
  user: AuthenticatedUser | null = null,
) => {
  const canAccess = await userCanAccessModelAndSourceSet(modelId, sourceSetName, user, true);
  if (!canAccess) {
    throw new Error('SourceSet not found');
  }
  return prisma.graphHostSourceOnSourceSet.findMany({
    where: {
      sourceSetModelId: modelId,
      sourceSetName,
    },
    include: {
      graphHostSource: true,
    },
  });
};

export const getGraphServerHostForSourceSet = async (
  modelId: string,
  sourceSetName: string,
  user: AuthenticatedUser | null = null,
) => {
  const hosts = await getSourceSetGraphHosts(modelId, sourceSetName, user);
  if (!hosts || hosts.length === 0) {
    throw new Error('SourceSet not found.');
  }
  const randomIndex = Math.floor(Math.random() * hosts.length);
  return hosts[randomIndex].graphHostSource.hostUrl;
};

export const getGraphServerRunpodHostForSourceSet = async (
  modelId: string,
  sourceSetName: string,
  user: AuthenticatedUser | null = null,
) => {
  const hosts = await getSourceSetGraphHosts(modelId, sourceSetName, user);
  // find first host that has runpodServerlessUrl
  const host = hosts.find((h) => h.graphHostSource.runpodServerlessUrl);
  if (!host) {
    throw new Error('No runpod serverless host found.');
  }
  return host.graphHostSource.runpodServerlessUrl;
};

export const getAuthHeaderForGraphServerRequest = () => {
  if (env.USE_RUNPOD_GRAPH) {
    return {
      Authorization: `Bearer ${env.GRAPH_RUNPOD_SECRET}`,
      'x-secret-key': '',
    };
  }
  return {
    Authorization: '',
    'x-secret-key': env.GRAPH_SERVER_SECRET,
  };
};

export const wrapRequestBodyForRunpodIfNeeded = (body: any) => {
  if (env.USE_RUNPOD_GRAPH) {
    return {
      input: body,
    };
  }
  return body;
};

export const getGraphServerRequestUrlForSourceSet = async (modelId: string, sourceSetName: string, action: string) => {
  if (env.USE_LOCALHOST_GRAPH) {
    return `${LOCALHOST_GRAPH_HOST}/${action}`;
  }
  if (env.USE_RUNPOD_GRAPH) {
    // for runpod the action is in the body
    return `${await getGraphServerRunpodHostForSourceSet(modelId, sourceSetName)}/runsync`;
  }
  return `${await getGraphServerHostForSourceSet(modelId, sourceSetName)}/${action}`;
};
