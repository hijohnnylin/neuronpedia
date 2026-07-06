// Dev helper for fast UI iteration on the jlens panels. The chat / completion
// interfaces can export their current run (meta + the full per-position token
// stream) to a JSON file. Drop that file into `/public` and load it back via
// the fixture bar at the top of the panel to re-render the exact same data
// without hitting the inference server.

import { JlensShareSteer, JlensShareUiState } from '@/lib/utils/jlens-share';
import { LensMetaMessage, LensTokenMessage } from '@/lib/utils/lens';

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
}

export interface JlensExportCompletion extends JlensExportBase {
  kind: 'completion';
  prompt: string;
}

export interface JlensExportChat extends JlensExportBase {
  kind: 'chat';
  messages: { role: ChatRole; content: string }[];
}

export type JlensExport = JlensExportCompletion | JlensExportChat;

// The steer payload sent in the `/api/lens/share` request body: the steer
// config plus the full token-id sequence of the steered run, so the server can
// reproduce its read-outs (forced decode, no generation).
export interface JlensShareSteerRequest extends JlensShareSteer {
  inputTokenIds: number[];
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
  return data as JlensExport;
}
