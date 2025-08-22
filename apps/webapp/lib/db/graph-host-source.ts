import { env } from '../env';

export function getGraphServerUrlForModel(modelId: string) {
  if (modelId === 'qwen3-4b') {
    return env.GRAPH_SERVER_QWEN3_4B;
  }
  if (modelId === 'gemma-2-2b') {
    return env.GRAPH_SERVER;
  }
  throw new Error(`No graph server url found for model ${modelId}`);
}

export function getGraphRunpodServerUrlForModel(modelId: string) {
  if (modelId === 'qwen3-4b') {
    return env.GRAPH_RUNPOD_SERVER_QWEN3_4B;
  }
  if (modelId === 'gemma-2-2b') {
    return env.GRAPH_RUNPOD_SERVER;
  }
  throw new Error(`No graph runpod server url found for model ${modelId}`);
}
