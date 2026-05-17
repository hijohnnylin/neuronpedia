import { SAEFormat } from '@/lib/utils/saelens';

/**
 * Per-(model, layer) allowlist for NLA-compatible SAEs. Each entry maps a
 * model id to the layers we have an NLA-compatible SAE for, along with the
 * on-disk format of that SAE so callers know which safetensors filename
 * and tensor key to read.
 *
 * Consumers:
 *   - `components/panes/explanations-pane.tsx` — gates the "Explain with
 *     NLA" UI button and passes the format through to /explain-saelens.
 *   - `lib/external/autointerp-scorer-nla.ts` — server-side NLA scoring
 *     uses this to build the safetensors path and decoder tensor key.
 *
 * Add new entries here when shipping a new NLA-compatible SAE. Note that
 * an entry here is necessary but not sufficient — there must also be a
 * matching `NlaSource` row in Postgres so we know which GPU servers to
 * forward to.
 */
export const NLA_MODELS_AND_LAYERS: Record<string, Record<number, SAEFormat>> = {
  'gemma-3-27b-it': { 41: SAEFormat.GemmaScope2 },
};

/**
 * Returns the on-disk SAE format registered for the given (model, layer)
 * pair, or `undefined` if NLA is not supported for it. Callers that need
 * a default (e.g. legacy SAELens-formatted SAEs that aren't NLA-gated)
 * should fall back to `SAEFormat.SAELens` themselves.
 */
export function getNlaSaeFormat(modelId: string, layerNum: number): SAEFormat | undefined {
  const layerToFormat = NLA_MODELS_AND_LAYERS[modelId];
  if (!layerToFormat) return undefined;
  return layerToFormat[layerNum];
}
