import {
  fetchNeuronDashboardDocText,
  fetchNeuronDashboardDocTokens,
  fetchNeuronDashboardVocab,
  NeuronDashboardDocRef,
} from '@/lib/utils/neuron-dashboards';
import { createContext, useContext } from 'react';

export type NeuronDashboardDocs = { baseUrl: string; docRefs: Record<string, NeuronDashboardDocRef> };

// Set by a page that shows a file-based neuron dashboard, so each activation row can load its document.
export const NeuronDashboardDocsContext = createContext<NeuronDashboardDocs | null>(null);

// Document tokens before and after an activation's window, as display strings.
export type NeuronDashboardDocContext = { before: string[]; after: string[] };

// Loaders for the document behind an activation row, or null when the row has none. Nothing is fetched until called.
export default function useNeuronDashboardDoc(activationId: string | undefined) {
  const docs = useContext(NeuronDashboardDocsContext);
  const ref = activationId && docs ? docs.docRefs[activationId] : undefined;
  if (!docs || !ref) {
    return null;
  }
  return {
    loadContext: async (): Promise<NeuronDashboardDocContext> => {
      const [tokens, vocab] = await Promise.all([
        fetchNeuronDashboardDocTokens(docs.baseUrl, ref.doc),
        fetchNeuronDashboardVocab(docs.baseUrl),
      ]);
      const strings = tokens.map((t) => vocab[t] ?? `<${t}>`);
      return { before: strings.slice(0, ref.start), after: strings.slice(ref.start + ref.length) };
    },
    loadText: () => fetchNeuronDashboardDocText(docs.baseUrl, ref.doc),
  };
}
