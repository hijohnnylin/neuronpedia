// Dev helper for fast UI iteration on the jlens panels. The chat / completion
// interfaces can export their current run (meta + the full per-position token
// stream) to a JSON file. Drop that file into `/public` and load it back via
// the fixture bar at the top of the panel to re-render the exact same data
// without hitting the inference server.

import { JlensShareSteer, JlensShareUiState } from '@/lib/utils/jlens-share';
import { LensChatMessage, LensChatTool, LensMetaMessage, LensTokenMessage, LensTokenSpan } from '@/lib/utils/lens';

// Extract the per-token chat-span metadata (parallel to a token stream) so a
// share can persist grouping. Shares re-run inference server-side over the exact
// token ids (no generation), which yields no spans; the client sends these so
// the route can overlay them back onto the recomputed tokens by position.
export function tokenSpansOf(tokens: LensTokenMessage[]): LensTokenSpan[] {
  return tokens.map((t) => ({
    message_index: t.message_index ?? null,
    role: t.role ?? null,
    channel: t.channel ?? null,
    section: t.section ?? null,
  }));
}

export type ChatRole = 'user' | 'assistant';

// A steered run saved alongside the main run in a share's S3 blob: the steer
// config plus its own (heavy) token stream + meta. Re-computed server-side at
// share time (forced decode over the steered token ids, no generation) so the
// stored data is trusted + reproducible.
export interface JlensExportSteer {
  config: JlensShareSteer;
  meta: LensMetaMessage | null;
  tokens: LensTokenMessage[];
}

// One oracle read of the main run, as the sharer saw it, with the signature
// `/api/lens/oracle` gave it (so a share of a share can store it again).
export interface JlensExportOracleRead {
  position: number;
  layer: number;
  adapter: string;
  bullets: string[];
  text: string;
  finish: string;
  sig: string;
}

// Oracle reads saved with a share. A read is not repeatable, so a share keeps
// the text instead of reading again. The `/api/lens/share` body sends the same shape.
export interface JlensExportOracle {
  maxBullets: number;
  reads: JlensExportOracleRead[];
}

interface JlensExportBase {
  // Schema version so we can evolve the format without silently mis-loading.
  version: 1;
  modelId: string;
  exportedAt: string;
  meta: LensMetaMessage | null;
  tokens: LensTokenMessage[];
  // Optional UI-restore state, populated when loading a shared link (merged in
  // from the `JlensShare` DB row). Absent for plain fixture exports.
  uiState?: JlensShareUiState;
  // Optional steered run saved with the share. Absent when the share had no
  // active steered run (and for plain fixture exports).
  steer?: JlensExportSteer;
  // Absent when the run has no oracle reads.
  oracle?: JlensExportOracle;
  // The sidebar layer range [first, last]. Absent when it is the default.
  layerRange?: [number, number];
}

export interface JlensExportCompletion extends JlensExportBase {
  kind: 'completion';
  prompt: string;
}

export interface JlensExportChat extends JlensExportBase {
  kind: 'chat';
  messages: LensChatMessage[];
  // Tool definitions the chat template rendered into the prompt, if any.
  tools?: LensChatTool[];
}

export type JlensExport = JlensExportCompletion | JlensExportChat;

// The steer payload sent in the `/api/lens/share` request body: the steer
// config plus the full token-id sequence of the steered run, so the server can
// reproduce its read-outs (forced decode, no generation).
export interface JlensShareSteerRequest extends JlensShareSteer {
  inputTokenIds: number[];
  // Per-token chat spans (parallel to `inputTokenIds`), overlaid back onto the
  // server-recomputed steered tokens so the shared steered transcript groups.
  spans?: LensTokenSpan[];
}

// Assemble the share-request steer payload from the active steer config + its
// streamed token results. Returns `undefined` (so the share carries no steer)
// when there is no active steer, no selected layers, or the steered tokens lack
// stable ids (can't be reproduced).
export function buildSteerShareBody(
  steer: JlensShareSteer | null,
  steerTokens: LensTokenMessage[],
): JlensShareSteerRequest | undefined {
  if (!steer || steer.layers.length === 0 || steerTokens.length === 0) {
    return undefined;
  }
  const ids = steerTokens.map((t) => t.id);
  if (!ids.every((id) => typeof id === 'number')) {
    return undefined;
  }
  return {
    token: steer.token,
    type: steer.type,
    layers: steer.layers,
    strength: steer.strength,
    ablate: steer.ablate,
    mode: steer.mode ?? 'steer',
    swapToken: steer.swapToken ?? '',
    steerGenerated: steer.steerGenerated ?? false,
    inputTokenIds: ids as number[],
    spans: tokenSpansOf(steerTokens),
  };
}

// Trigger a browser download of `data` as pretty-printed JSON.
export function downloadJson(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.json') ? filename : `${filename}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Build a filesystem-friendly default filename for an export.
export function defaultExportFilename(kind: JlensExport['kind'], modelId: string): string {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const safeModel = (modelId || 'model').replace(/[^a-zA-Z0-9-_]/g, '_');
  return `jlens-${kind}-${safeModel}-${stamp}.json`;
}

// Fetch + validate a fixture from the public folder (or any same-origin path).
// Accepts a bare filename ("foo.json"), a "/foo.json" path, or a "public/..."
// path and normalizes it to a same-origin request.
export async function loadFixture(rawPath: string): Promise<JlensExport> {
  const trimmed = rawPath.trim();
  if (!trimmed) {
    throw new Error('Enter a JSON path (e.g. my-fixture.json).');
  }

  let path = trimmed.replace(/^public\//, '').replace(/^\/?public\//, '');
  if (!path.startsWith('/') && !/^https?:\/\//.test(path)) {
    path = `/${path}`;
  }

  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Could not load "${path}" (${res.status}). Is it in /public?`);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new Error(`"${path}" is not valid JSON.`);
  }

  return parseFixture(data);
}

// Validate the loosely-typed JSON into a JlensExport, throwing on bad shapes.
export function parseFixture(data: unknown): JlensExport {
  if (!data || typeof data !== 'object') {
    throw new Error('Fixture must be a JSON object.');
  }
  const obj = data as Record<string, unknown>;
  if (obj.kind !== 'chat' && obj.kind !== 'completion') {
    throw new Error('Fixture is missing a valid "kind" ("chat" or "completion").');
  }
  if (!Array.isArray(obj.tokens)) {
    throw new Error('Fixture is missing a "tokens" array.');
  }
  // A bad oracle or layer range field costs only that field, not the run.
  const out = { ...obj };
  const oracle = obj.oracle as Record<string, unknown> | undefined;
  if (oracle !== undefined && (typeof oracle?.maxBullets !== 'number' || !Array.isArray(oracle?.reads))) {
    out.oracle = undefined;
  }
  if (obj.layerRange !== undefined && !isLayerRange(obj.layerRange)) {
    out.layerRange = undefined;
  }
  return out as unknown as JlensExport;
}

// Keep a restored layer range inside the model's layers.
export function clampLayerRange(
  range: [number, number] | null,
  bounds: [number, number] | null,
): [number, number] | null {
  if (!range || !bounds) {
    return range;
  }
  const first = Math.min(Math.max(range[0], bounds[0]), bounds[1]);
  return [first, Math.min(Math.max(range[1], first), bounds[1])];
}

export function isLayerRange(v: unknown): v is [number, number] {
  return (
    Array.isArray(v) &&
    v.length === 2 &&
    v.every((n) => Number.isInteger(n) && n >= 0) &&
    (v[0] as number) <= (v[1] as number)
  );
}
