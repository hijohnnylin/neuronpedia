import { CLTGraph } from './graph-types';

export async function computeGraphScoresInWorker(
  graph: CLTGraph,
  pinnedIds: string[] = [],
): Promise<{ replacementScore: number; completenessScore: number }> {
  const { nodes, links } = graph;

  // Check if Worker is available
  if (typeof Worker === 'undefined') {
    throw new Error('Web Workers are not supported in this environment');
  }

  // ensure we have multiple cores
  if (navigator.hardwareConcurrency <= 1) {
    throw new Error('This browser does not support multiple cores');
  }

  console.log('Creating worker...');
  const worker = new Worker(new URL('./score.worker.ts', import.meta.url), { type: 'module' });

  console.log('Worker created successfully');

  return new Promise((resolve, reject) => {
    // Add timeout to prevent hanging if worker never responds with valid data
    const timeout = setTimeout(() => {
      // @ts-ignore
      // eslint-disable-next-line @typescript-eslint/no-use-before-define
      worker.removeEventListener('message', handleMessage as EventListener);
      // eslint-disable-next-line @typescript-eslint/no-use-before-define
      worker.removeEventListener('error', handleError as EventListener);
      worker.terminate();
      reject(new Error('Worker timeout: No valid response received within 30 seconds'));
    }, 30000);

    function handleMessage(ev: MessageEvent) {
      console.log('Client received message from worker:', ev.data);
      const { data } = ev;

      // Handle case where ev.data is null or undefined
      if (!data) {
        console.error(
          'worker error: Received null or undefined data from worker. This may be a Chrome initialization message.',
        );
        // Don't reject immediately - this might be a Chrome initialization message
        // Just return and wait for the actual data message
        return;
      }

      // Clear timeout since we got a valid response
      clearTimeout(timeout);

      // @ts-ignore
      worker.removeEventListener('message', handleMessage as EventListener);
      // eslint-disable-next-line @typescript-eslint/no-use-before-define
      worker.removeEventListener('error', handleError as EventListener);
      worker.terminate();

      if (data.error) {
        console.error('worker error', data.error);
        reject(new Error(data.error));
        return;
      }

      console.log('got scores');
      resolve({
        replacementScore: data.replacementScore || 0,
        completenessScore: data.completenessScore || 0,
      });
    }
    function handleError(err: ErrorEvent) {
      console.error('Worker error event:', err);
      clearTimeout(timeout);
      // @ts-ignore
      worker.removeEventListener('message', handleMessage as EventListener);
      // @ts-ignore
      worker.removeEventListener('error', handleError as EventListener);
      worker.terminate();
      reject(err.error || err);
    }
    worker.addEventListener('message', handleMessage as EventListener);
    worker.addEventListener('error', handleError as EventListener);

    // Send the graph data directly
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    sendGraphData();

    function sendGraphData() {
      try {
        console.log('sending graph data to worker');

        // Clean the data to remove circular references
        const cleanNodes = nodes.map((node) => ({
          node_id: node.node_id,
          feature_type: node.feature_type,
          layer: node.layer,
          ctx_idx: node.ctx_idx,
          feature: node.feature,
          token_prob: node.token_prob,
          // Remove any circular references like targetLinks, sourceLinks, etc.
          // Only keep the essential data needed for computation
        }));

        const cleanLinks = links.map((link) => ({
          // @ts-ignore
          source: typeof link.source === 'string' ? link.source : link.source.node_id,
          // @ts-ignore
          target: typeof link.target === 'string' ? link.target : link.target.node_id,
          weight: link.weight,
          // Remove any circular references like sourceNode, targetNode, etc.
        }));

        const messageData = {
          requestId: 1,
          graph: { nodes: cleanNodes, links: cleanLinks },
          pinnedIds,
        };

        // Test if the cleaned data can be serialized
        try {
          JSON.stringify(messageData);
          console.log('Message data serialization test: OK');
        } catch (serializationErr) {
          console.error('Message serialization failed:', serializationErr);
          throw new Error(`Cannot serialize graph data for worker: ${(serializationErr as Error).message}`);
        }

        worker.postMessage(messageData);
        console.log('message posted to worker successfully');
      } catch (err) {
        console.error('Error posting message to worker:', err);
        clearTimeout(timeout);
        // @ts-ignore
        worker.removeEventListener('message', handleMessage as EventListener);
        worker.removeEventListener('error', handleError as EventListener);
        worker.terminate();
        reject(err);
      }
    }
  });
}
