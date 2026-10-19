import { fetchNeuronDashboard, NeuronDashboard } from '@/lib/utils/neuron-dashboards';
import { useEffect, useState } from 'react';

export type NeuronDashboardState =
  | { status: 'none' }
  | { status: 'loading' }
  | { status: 'ready'; dashboard: NeuronDashboard }
  | { status: 'error'; error: string };

// Loads a feature's dashboard from its source set's externalDashboardsUrl. Does nothing when the URL is unset.
export default function useNeuronDashboard(
  baseUrl: string | null | undefined,
  modelId: string | undefined,
  layer: string | undefined,
  index: string | undefined,
): NeuronDashboardState {
  const key = baseUrl && modelId && layer && index ? `${baseUrl}|${modelId}|${layer}|${index}` : null;
  const [result, setResult] = useState<{ key: string; state: NeuronDashboardState } | null>(null);

  useEffect(() => {
    if (!key || !baseUrl || !modelId || !layer || !index) {
      return undefined;
    }
    const controller = new AbortController();
    fetchNeuronDashboard(baseUrl, { modelId, layer, index }, controller.signal)
      .then((dashboard) => setResult({ key, state: { status: 'ready', dashboard } }))
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          console.error(`error loading neuron dashboard ${key}:`, error);
          setResult({ key, state: { status: 'error', error: error instanceof Error ? error.message : String(error) } });
        }
      });
    return () => controller.abort();
  }, [key, baseUrl, modelId, layer, index]);

  if (!key) {
    return { status: 'none' };
  }
  // A result for another feature is stale; the effect is already loading this one.
  return result?.key === key ? result.state : { status: 'loading' };
}
